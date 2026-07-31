"""NVIDIA NIM(openai/gpt-oss-20b)으로 마크다운을 문단 배치 단위로 순차 번역한다.

사용법:
    NVIDIA_API_KEY=nvapi-... python src/translate.py --input paper.md --output paper.ko.md
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml
from openai import OpenAI

PARAGRAPH_DELIMITER = "<<<P>>>"

TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
NO_LETTERS_RE = re.compile(r"^[^a-zA-Z가-힣]*$")  # 숫자/기호로만 된 셀(예: "0.993 / 0.99")은 번역할 게 없다

HANGUL_RE = re.compile(r"[가-힣]")
UNTRANSLATED_CHECK_MIN_LEN = 100
# 링크 텍스트 안에 연도가 있어야만 "인용"으로 본다. Marker는 "Eq. 1", "§3" 같은 순수
# 내부 상호참조(수식·절·그림 번호)도 똑같은 마크다운 링크로 뽑아내는데, 이런 링크까지
# 세면 각주·수식 참조가 몇 개만 있어도 본문 문단이 "참고문헌"으로 오탐돼 미번역 문단이
# 검증 없이 그대로 통과해버린다 (실제 발견됨).
CITATION_LINK_RE = re.compile(r"\[[^\]]*(?:19|20)\d{2}[^\]]*\]\([^)]+\)")
CITATION_YEAR_RE = re.compile(r"\((?:19|20)\d{2}[a-z]?\)")
MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
# 모델이 가끔 복잡한 수식(특히 행렬 곱 등)에서 짧은 토큰을 수백 번씩 그대로 반복하는
# 폭주(runaway repetition) 응답을 낸다 (실제로 발견: "\mathbf{M}_{r} \\"가 500번 이상
# 반복돼 16000자 넘는 깨진 수식 블록이 그대로 발행된 사례). 글자가 있는 토큰이 15번
# 이상 연속 반복되면 이걸로 본다 - 표 구분선("---")처럼 글자 없는 반복은 정상이라 제외.
RUNAWAY_REPEAT_RE = re.compile(r"(.{3,80}?)(?:\1){14,}")
ENGLISH_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
UNTRANSLATED_MIN_ENGLISH_WORDS = 5  # 수식 위주 문단은 LaTeX 명령어 몇 개 빼면 실제 영어 단어가 거의 없다
UNTRANSLATED_MAX_ATTEMPTS = 5  # 이제 배치 전체가 아니라 문제 문단만 개별 재시도라 비용이 작아, 여유를 좀 더 둠
UNTRANSLATED_RETRY_WAIT = 2.0

# 문단 전체는 한글이 충분히 섞여 있어 _looks_untranslated를 통과하더라도, 그 안의 문장
# 하나(또는 절 제목 뒤 문단 전체)만 원문 그대로 남는 경우가 실제로 발견됨(예: 수식 정의
# 문장 하나만 영어 그대로, 혹은 소제목만 번역되고 그 뒤 문단 전체가 원문 그대로 발행됨).
# 한글이 전혀 없는 구간이 길고 실제 영어 단어도 많으면 그 구간만 미번역으로 본다. 표
# 셀이나 참고문헌의 URL/doi가 섞인 구간은 영어 고유명사·링크가 정상이므로 제외한다.
HANGUL_RUN_RE = re.compile(r"[가-힣]+")
UNTRANSLATED_RUN_MIN_CHARS = 100
UNTRANSLATED_RUN_MIN_ENGLISH_WORDS = 10
URL_OR_DOI_RE = re.compile(r"https?://|doi:|arxiv:", re.IGNORECASE)
# 저자·참가자 명단("Lei Xiong, Jiahao Wang, ...")처럼 원래 번역 대상이 아닌 고유명사
# 나열은 길고 영어 단어 수도 많아 위 두 조건만으로는 오탐한다(실제 확인됨). 실제
# 문장이라면 the/and/of 같은 기능어가 반드시 여러 번 나오지만, 이름 나열에는 전혀
# 없다는 점으로 구분한다.
ENGLISH_FUNCTION_WORD_RE = re.compile(
    r"\b(?:the|and|of|is|are|in|to|with|for|on|that|this|from|by|as|we|our|which|its|be|can|not)\b",
    re.IGNORECASE,
)
UNTRANSLATED_RUN_MIN_FUNCTION_WORDS = 3

# 모델이 실제 빈 줄 대신 "(blank line)"이라는 문구를 그 자리에 그대로 출력하는 경우가
# 있다(실제 발견: 절 제목 바로 다음 줄에 "(blank line)"이 텍스트 그대로 발행됨). 형식을
# 말로 설명한 것이지 내용이 아니므로 방치하면 그 문구 자체가 페이지에 그대로 노출된다.
LITERAL_BLANK_LINE_RE = re.compile(r"^\s*[\(\[]\s*blank\s*line\s*[\)\]]\s*$", re.IGNORECASE | re.MULTILINE)

# 모델이 실제 줄바꿈 대신 리터럴 "\n" 두 글자를 그대로 출력하는 경우가 있다. 처음엔
# 헤더 앞뒤에서만 발견돼(예: "\n## 4 구성요소 (Components)\n\n") 헤더 줄에만 대응하는
# 좁은 정규식으로 고쳤는데, 나중에 저자·소속·링크 목록 같은 일반 본문 줄에서도 똑같이
# 나오는 게 발견됨(예: "앱: ...\n 홈페이지: ...\n"). 그래서 헤더 여부와 무관하게 코드
# 블록 밖 전체에서 리터럴 "\n"을 실제 줄바꿈으로 바꾼다. 코드 블록 안의 "\n"은 실제
# 코드(예: 파이썬 문자열 리터럴)일 수 있으므로 건드리지 않는다.
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def fix_escaped_newlines(text: str) -> tuple[str, int]:
    """코드 블록 밖에서 리터럴 "\\n"을 실제 줄바꿈으로 치환한다. (치환된 텍스트, 치환 횟수)를
    반환한다 - 치환 횟수는 실행 요약 로그에 쓴다."""
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    count = 0
    for i in range(0, len(parts), 2):
        count += parts[i].count("\\n")
        parts[i] = parts[i].replace("\\n", "\n")
    return "".join(parts), count


# 청크 하나씩 독립적으로 번역하다 보니, 같은 용어를 "처음 등장할 때만 원어 병기"하라는
# 프롬프트 규칙을 청크마다 새로 지켜서, 문서 전체로 보면 같은 용어의 원어 병기가
# 청크 수만큼 반복된다(실제 발견: "핵심 단계(key-step)"가 한 문서에 4번). 프롬프트만으로는
# 청크 간 상태 공유가 안 되니, 문서 전체를 다 모은 뒤 이 후처리로 두 번째부터 걷어낸다.
GLOSS_PAREN_RE = re.compile(r"(?<=[가-힣])( ?)\(([^()\n]{1,80})\)")
CITATION_PAREN_RE = re.compile(r"(?:19|20)\d{2}[a-z]?|et al", re.IGNORECASE)


def dedup_term_glosses(text: str) -> tuple[str, list[dict]]:
    """"한국어(English)" 형태의 용어 원어 병기가 문서 전체에서 같은 용어로 두 번째 이상
    나오면 괄호부를 제거한다. 헤더 줄(제목 병기)과 코드 블록은 건드리지 않고 매번
    유지한다 - 제목 병기는 용어 병기와 성격이 다르다. "(Lightman et al., 2024)" 같은
    인용 괄호는 원어 병기가 아니라 서지 정보라 절대 건드리지 않는다.

    (치환된 텍스트, 실제로 제거가 일어난 용어별 통계)를 반환한다 - 통계는 실행 요약
    로그에 "이 병기가 왜 사라졌는지" 텍스트 blob 없이 바로 보이게 하려는 용도라,
    텍스트 전체가 아니라 (용어, 원래 등장 횟수, 제거 횟수)만 담는다."""
    lines = text.split("\n")
    seen: set[str] = set()
    stats: dict[str, dict] = {}
    in_code_fence = False
    out_lines = []
    for line in lines:
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            out_lines.append(line)
            continue
        if in_code_fence or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        def _dedup(m: re.Match) -> str:
            term = m.group(2).strip()
            # 원어 병기는 정의상 영문 표기(ASCII 알파벳 2글자 이상)라, 그 조건이 없는
            # 괄호는 전부 다른 용도다 - "수식 (1)"의 수식 번호 참조, "그림 (a)"의
            # 서브그림 라벨(둘 다 문서 전체에서 반복 등장하는 게 정상이라 지우면 안 됨),
            # "(우리가 제안한 방식)" 같은 순한글 서술 삽입구까지 이 조건 하나로 걸러진다
            # (실제로 "(1)"이 두 번째 나올 때 통째로 사라지는 사고가 있었음 - 실측 확인).
            if not term or len(term) < 2 or not re.search(r"[a-zA-Z]", term):
                return m.group(0)
            if CITATION_PAREN_RE.search(term) or any(c in term for c in "$\\`"):
                return m.group(0)  # 인용·수식·코드로 보이면 원어 병기가 아니므로 그대로 둔다
            key = term.lower()
            entry = stats.setdefault(key, {"term": term, "occurrences": 0, "removed": 0})
            entry["occurrences"] += 1
            if key in seen:
                entry["removed"] += 1
                return ""  # 이미 나온 용어면 괄호부(앞 공백 포함)를 통째로 제거
            seen.add(key)
            return m.group(0)

        out_lines.append(GLOSS_PAREN_RE.sub(_dedup, line))
    ops = [v for v in stats.values() if v["removed"] > 0]
    return "\n".join(out_lines), ops

# 논문 안에 인용된 프롬프트/지시문(예: LLM-as-a-Judge 평가 템플릿)을 모델이 자기한테
# 내려진 지시로 착각해서, 번역 대신 안전 거부 응답을 그대로 내놓는 경우가 있다.
# 길이나 참고문헌 여부와 무관하게 항상 걸러내야 한다.
REFUSAL_PATTERNS = (
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i am sorry, but i can't",
    "i am sorry, but i cannot",
    "i can't help with that",
    "i cannot help with that",
    "i can't help with this",
    "i cannot assist with",
    "as an ai language model",
    "i'm not able to help",
    "i am not able to help",
)


LIST_ITEM_START_RE = re.compile(r"^[-*]\s")
# Marker가 참고문헌 목록을 항목 사이 빈 줄 없이 하나의 큰 문단으로 뽑을 때도 있고, 항목마다
# 빈 줄을 넣어 하나씩 별도 문단으로 쪼갤 때도 있다. 후자의 경우 문단 분할·재시도 비섹션을
# 거치면 결국 참고문헌 항목 "하나"만 담긴 문단이 개별적으로 검사받게 되는데, 그 안엔 인용
# 연도·링크가 1개뿐이라 아래 3개 이상 요건을 못 채워 정상적으로 미번역 처리돼야 할 항목이
# "번역 안 됨" 실패로 오판되어 매번 헛되이 재시도된다 (실제 발견됨: 참고문헌 항목 2개가
# 각각 5회씩, 전체 호출의 34%를 낭비). "[숫자]"로 시작하는 번호식 서지 표기는 일반 산문이
# 문단 맨 앞자리에서 그렇게 시작하는 일이 사실상 없으므로, 항목 하나뿐이어도 안전하게
# 참고문헌으로 판별할 수 있다.
REFERENCE_ENTRY_START_RE = re.compile(r"^[-*]?\s*\[\d+\]")


def _looks_like_references(paragraph: str) -> bool:
    """실제 참고문헌 "목록" 항목(Marker가 각 서지 항목을 "- [n] ..." 리스트로 뽑아낸 것)만
    미번역 검사에서 제외한다. 관련 연구·논의 등 일반 본문 문단은 인용을 여러 개 걸치더라도
    실제로는 번역이 필요한 산문이며, 리스트 항목이 아니면 인용 개수가 아무리 많아도 절대
    제외하지 않는다. (예전엔 인용 개수만 보고 판단해서, 인용이 빽빽한 본문 문단이 통째로
    영어 원문 그대로 방치되거나 심지어 중국어로 오역된 채 검증 없이 그대로 발행된 사례가
    실제 발견됨 — 리스트 마커 유무로 39개 파일 전수 검사해 회귀 없음을 확인.)"""
    first_line = paragraph.lstrip().split("\n", 1)[0]
    if REFERENCE_ENTRY_START_RE.match(first_line) and (
        CITATION_YEAR_RE.search(paragraph)
        or CITATION_LINK_RE.search(paragraph)
        or URL_OR_DOI_RE.search(paragraph)
    ):
        return True
    if not LIST_ITEM_START_RE.match(first_line):
        return False
    return len(CITATION_LINK_RE.findall(paragraph)) >= 3 or len(CITATION_YEAR_RE.findall(paragraph)) >= 3


def _looks_like_refusal(translated: str) -> bool:
    """번역 결과가 실제 번역이 아니라 모델의 안전 거부 응답인지 확인한다."""
    lowered = translated.lower().replace("’", "'")
    return any(pat in lowered for pat in REFUSAL_PATTERNS)


def _dollar_count_mismatch(original: str, translated: str) -> bool:
    """모델이 수식 안의 \\$ 기호를 하나 빠뜨리거나 더하면 \\$...\\$ 짝이 어긋나서, 그
    문단뿐 아니라 그 뒤로 문서 전체의 수식 렌더링까지 줄줄이 밀려 깨지는 심각한
    문제가 생긴다 (실제로 발견됨). 원문과 번역문의 \\$ 개수가 다르면 실패로 본다."""
    return original.count("$") != translated.count("$")


def _has_runaway_repetition(translated: str) -> bool:
    """짧은 토큰(글자 포함)이 15번 이상 연속 반복되면 폭주 응답으로 본다."""
    for m in RUNAWAY_REPEAT_RE.finditer(translated):
        if re.search(r"[A-Za-z가-힣]", m.group(1)):
            return True
    return False


def _looks_untranslated(original: str, translated: str) -> bool:
    """구분자 개수는 맞는데 모델이 번역 대신 원문을 그대로 돌려주거나(특히 수식·참고문헌이
    아닌 일반 산문에서), 논문에 인용된 프롬프트에 헷갈려 안전 거부 응답을 내놓는 경우가
    있다. 거부 응답은 길이·내용과 무관하게 항상 실패로 본다. 그 외에는, 원문이 충분히
    긴 산문인데 번역 결과에 한글이 사실상 없으면 번역이 안 된 것으로 본다. 참고문헌처럼
    원래 한글이 거의 없는 게 정상인 짧은/인용 밀집 항목이나, 수식($...$/$$...$$)이
    대부분이라 실제 번역 대상 영어 산문이 거의 없는 문단까지 오탐하지 않도록 수식을
    걷어낸 뒤 진짜 영어 단어가 충분히 남아있을 때만 판단한다."""
    if _looks_like_refusal(translated):
        return True
    non_math_original = MATH_SPAN_RE.sub(" ", original)
    if len(non_math_original) < UNTRANSLATED_CHECK_MIN_LEN:
        return False
    if _looks_like_references(original):
        return False
    non_math_translated = MATH_SPAN_RE.sub(" ", translated)
    if len(ENGLISH_WORD_RE.findall(non_math_translated)) < UNTRANSLATED_MIN_ENGLISH_WORDS:
        return False
    return len(HANGUL_RE.findall(non_math_translated)) < 5


def _has_untranslated_run(original: str, translated: str) -> bool:
    """문단 전체는 한글이 충분해 _looks_untranslated를 통과하더라도, 그 안의 문장 하나나
    소제목 뒤 문단 전체만 원문 그대로 남는 경우를 잡는다. 표·참고문헌 항목은 영어 고유
    명사·URL이 정상이므로 통째로 제외하고, 그 외 구간은 URL/doi가 섞여 있지 않으면서
    한글 없이 충분히 길고 실제 문장 형태(the/and 같은 기능어 포함)일 때만 미번역으로 본다
    (저자·참가자 명단처럼 고유명사만 나열된 경우는 기능어가 없어 걸러짐)."""
    if is_table(original) or _looks_like_references(original):
        return False
    non_math = MATH_SPAN_RE.sub(" ", translated)
    for run in HANGUL_RUN_RE.split(non_math):
        if URL_OR_DOI_RE.search(run):
            continue
        if (
            len(run) >= UNTRANSLATED_RUN_MIN_CHARS
            and len(ENGLISH_WORD_RE.findall(run)) >= UNTRANSLATED_RUN_MIN_ENGLISH_WORDS
            and len(ENGLISH_FUNCTION_WORD_RE.findall(run)) >= UNTRANSLATED_RUN_MIN_FUNCTION_WORDS
        ):
            return True
    return False


def _has_literal_blank_line_marker(translated: str) -> bool:
    """모델이 실제 빈 줄 대신 "(blank line)" 문구를 그 자리에 그대로 출력했는지 확인한다."""
    return bool(LITERAL_BLANK_LINE_RE.search(translated))


def load_system_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def is_table(paragraph: str) -> bool:
    """문단 전체가 마크다운 표(헤더 행 + 구분 행 + 데이터 행)로만 이루어져 있는지 확인한다."""
    lines = [line for line in paragraph.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    if not all(TABLE_ROW_RE.match(line) for line in lines):
        return False
    return bool(TABLE_SEP_RE.match(lines[1]))


def _merge_fenced_paragraphs(paragraphs: list[str]) -> list[str]:
    """코드 펜스(```) 블록 안에 빈 줄이 있으면 "\\n\\n" 기준 문단 분리가 펜스를 반으로
    갈라 서로 다른 배치로 보내버린다 - 그러면 모델은 안 닫힌 펜스를 받고, 배치 단위로
    도는 fix_escaped_newlines의 펜스 보호도 그 배치에서만 무력화돼 코드 안의 리터럴
    "\\n"까지 실제 줄바꿈으로 바뀐다(실측 확인). 펜스가 닫히지 않은 채 끝나는 문단은
    닫힐 때까지 다음 문단들과 다시 합쳐 하나의 문단으로 되돌린다."""
    merged: list[str] = []
    buffer: list[str] = []
    open_fence = False
    for para in paragraphs:
        buffer.append(para)
        if para.count("```") % 2 == 1:
            open_fence = not open_fence
        if not open_fence:
            merged.append("\n\n".join(buffer))
            buffer = []
    if buffer:  # 펜스가 안 닫힌 채 문서가 끝남 - 있는 그대로 마지막 덩어리로 둔다
        merged.append("\n\n".join(buffer))
    return merged


def chunk_markdown(text: str, max_chars: int = 4000) -> list[list[str]]:
    """문단(빈 줄 구분)을 max_chars 예산 안에서 묶어 배치로 만든다 (문단 자체는 쪼개지 않음).
    각 배치는 문단 문자열의 리스트로 반환되며, 실제 API 호출 시 고유 구분자로 이어붙인다.
    표 문단은 셀 단위로 따로 번역해야 열이 밀리지 않으므로, 항상 단독 배치로 분리한다."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraphs = _merge_fenced_paragraphs(paragraphs)
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if is_table(para):
            if current:
                batches.append(current)
                current = []
                current_len = 0
            batches.append([para])
            continue
        para_len = len(para) + len(PARAGRAPH_DELIMITER)
        if current and current_len + para_len > max_chars:
            batches.append(current)
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len
    if current:
        batches.append(current)
    return batches


_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 2.0  # NVIDIA 무료 티어 40rpm 한도에 여유를 두고 최대 30rpm으로 제한


def _call_model(client: OpenAI, model: str, system_prompt: str, temperature: float, content: str) -> tuple[str, str]:
    """API를 한 번 호출해 (본문, finish_reason)을 반환한다. 요청 간 최소 간격을 지킨다."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        # thinking을 켜두면 답변마다 긴 추론 과정을 먼저 생성해 훨씬 느려진다.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason


class TranslationCache:
    """모델 원본 응답만 캐시한다 - 후처리·검증은 캐시 히트 여부와 무관하게 항상 새로
    돈다. 키는 청크 텍스트 + 시스템 프롬프트 내용 해시(수동 버전 번호 아님, 자동 계산) +
    모델명 + temperature로 구성해서, 하네스 코드(후처리·검증·청커 등)가 바뀌어도 무효화가
    필요 없다 - 청커가 바뀌면 청크 텍스트 자체가 달라져 키가 자동으로 갈리고, 프롬프트가
    바뀌면 해시가 자동으로 갈린다. 검증을 통과한 응답만 쓰기 때문에(자가 치유), 예전에
    캐시된 응답이 새 검증 규칙에 걸리면 다음에 성공한 실제 호출이 같은 키를 덮어써서
    무효화 로직 없이 저절로 고쳐진다."""

    def __init__(self, db_path: Path):
        import sqlite3

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                finish_reason TEXT NOT NULL,
                written_at TEXT NOT NULL,
                git_sha TEXT
            )"""
        )
        self._conn.commit()

    @staticmethod
    def make_key(content: str, prompt_hash: str, model: str, temperature: float) -> str:
        raw = f"{content}\x00{prompt_hash}\x00{model}\x00{temperature}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[str, str] | None:
        row = self._conn.execute("SELECT content, finish_reason FROM cache WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, key: str, content: str, finish_reason: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, content, finish_reason, written_at, git_sha) VALUES (?, ?, ?, ?, ?)",
            (key, content, finish_reason, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), _git_sha()),
        )
        self._conn.commit()

    def seed(self, key: str, content: str, finish_reason: str, git_sha: str | None) -> None:
        """로그에서 이미 검증을 통과한 걸로 확인된 응답을 그대로 캐시에 심는다 -
        시딩 시점을 기록해두려고 put()과 별도로 git_sha를 인자로 받는다(원래 실행 당시의
        SHA를 남겨야 "이 항목이 언제 만들어졌는지"가 뒤섞이지 않는다)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, content, finish_reason, written_at, git_sha) VALUES (?, ?, ?, ?, ?)",
            (key, content, finish_reason, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_sha),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _call_model_cached(
    cache: "TranslationCache | None",
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    content: str,
    prompt_hash: str,
) -> tuple[str, str, bool]:
    """캐시를 먼저 조회한다 - 히트면 네트워크도, 레이트리밋 대기도 없이 즉시 반환한다.
    미스일 때만 _call_model로 실제 호출한다(그 안의 레이트리밋 대기는 실제 호출에만
    걸린다). 캐시에 쓰는 건 검증을 마친 호출부의 책임이라 여기서는 조회만 한다."""
    if cache is not None:
        key = cache.make_key(content, prompt_hash, model, temperature)
        hit = cache.get(key)
        if hit is not None:
            return hit[0], hit[1], True
    content_out, finish_reason = _call_model(client, model, system_prompt, temperature, content)
    return content_out, finish_reason, False


def translate_paragraphs(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    paragraphs: list[str],
    max_retries: int = 12,
    *,
    logger: "RunLogger | None" = None,
    prompt_hash: str = "",
    batch_index: int = 0,
    path: str = "",
    cache: "TranslationCache | None" = None,
) -> str:
    """문단 여러 개를 번역해 빈 줄로 이어붙인 문자열로 돌려준다 (일반 본문 배치용)."""
    parts = _translate_parts(
        client, model, system_prompt, temperature, paragraphs, max_retries,
        logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, path=path or str(batch_index),
        cache=cache,
    )
    return "\n\n".join(parts)


def _translate_parts(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    paragraphs: list[str],
    max_retries: int = 12,
    *,
    logger: "RunLogger | None" = None,
    prompt_hash: str = "",
    batch_index: int = 0,
    path: str = "",
    cache: "TranslationCache | None" = None,
) -> list[str]:
    """문단(또는 표 셀) 여러 개를 고유 구분자로 묶어 한 번에 번역하고, 번역된 조각을
    입력과 같은 개수의 리스트로 돌려준다. 응답에서 구분자 개수가 보낸 개수와 정확히
    일치하는지 코드로 검증하고, 안 맞으면(모델이 항목을 합쳐버렸으면) 절반으로 나눠
    재귀적으로 재시도한다 -- 프롬프트 지시만으로는 구조 보존을 신뢰할 수 없어서 항상
    코드로 검증한다."""
    path = path or str(batch_index)
    if len(paragraphs) == 1:
        return [
            _translate_single(
                client, model, system_prompt, temperature, paragraphs[0], max_retries,
                logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, item_index=None, path=path,
                cache=cache,
            )
        ]

    joined = f"\n\n{PARAGRAPH_DELIMITER}\n\n".join(paragraphs)
    # 캐시된 응답이 새 검증에 걸리면 use_cache를 꺼서 이후 이 청크는 항상 실제 호출로
    # 넘어간다 - 같은 키를 다시 조회해봐야 아직 아무도 새 값을 쓰지 않았으니 똑같은
    # 낡은 응답만 되돌아온다. 이 폴백은 API를 안 쓴 시도라 attempt를 소모하지 않는다
    # (아래 attempt += 1을 건너뛰고 continue).
    use_cache = cache is not None
    attempt = 0
    while attempt < max_retries:
        call_start = time.monotonic()
        try:
            content, finish_reason, cache_hit = _call_model_cached(
                cache if use_cache else None, client, model, system_prompt, temperature, joined, prompt_hash,
            )
            elapsed = time.monotonic() - call_start
            if finish_reason == "length":
                if cache_hit:
                    use_cache = False
                    continue
                print(
                    f"번역 응답이 컨텍스트 한도로 잘림 ({len(paragraphs)}개 항목 배치) -- 절반으로 나눠 재번역",
                    file=sys.stderr,
                )
                if logger:
                    logger.call(
                        batch_index=batch_index, path=path, call_kind="batch", item_index=None, attempt=attempt + 1,
                        model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                        input_text=joined, raw_response=content, finish_reason=finish_reason,
                        items_in=len(paragraphs), items_out=0, validation_fail_reason="length_truncated",
                        cache_hit=cache_hit,
                    )
                return _bisect_and_translate(
                    client, model, system_prompt, temperature, paragraphs, max_retries,
                    logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, path=path, cache=cache,
                )

            parts = [p.strip() for p in re.split(re.escape(PARAGRAPH_DELIMITER), content) if p.strip()]
            if len(parts) != len(paragraphs):
                if cache_hit:
                    use_cache = False
                    continue
                print(
                    f"번역 응답의 항목 수가 안 맞음 (보냄 {len(paragraphs)} / 받음 {len(parts)}) "
                    "-- 절반으로 나눠 재번역",
                    file=sys.stderr,
                )
                if logger:
                    logger.call(
                        batch_index=batch_index, path=path, call_kind="batch", item_index=None, attempt=attempt + 1,
                        model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                        input_text=joined, raw_response=content, finish_reason=finish_reason,
                        items_in=len(paragraphs), items_out=len(parts), validation_fail_reason="item_count_mismatch",
                        cache_hit=cache_hit,
                    )
                return _bisect_and_translate(
                    client, model, system_prompt, temperature, paragraphs, max_retries,
                    logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, path=path, cache=cache,
                )

            # 항목 수가 맞으면 원문-번역 짝이 인덱스로 정확히 매칭되므로, 문제 있는
            # 항목만 개별 재번역한다. 배치 전체를 버리고 절반씩 다시 보내면, 이미 잘
            # 번역된 나머지 항목들까지 매번 헛되이 재요청하게 되어 시간이 크게 낭비된다
            # (실제로 33개 항목 중 1개 때문에 33->16->8->4->2로 반복 재번역되는 사례 확인).
            bad_indices = [
                i
                for i, (src, out) in enumerate(zip(paragraphs, parts))
                if _looks_untranslated(src, out)
                or _has_untranslated_run(src, out)
                or _dollar_count_mismatch(src, out)
                or _has_runaway_repetition(out)
                or _has_literal_blank_line_marker(out)
            ]
            # bad_indices가 있어도 캐시 히트를 통째로 버리지 않는다 - 이건 구조적 실패가
            # 아니라 항목별 콘텐츠 문제라, 원래도 배치 전체를 버리지 않고 문제 있는
            # 항목만 개별 재번역하는 설계다(바로 아래). 캐시 경로에서만 이 원칙을 깨고
            # 통째로 재요청하면, 캐시 히트분의 멀쩡한 나머지 항목까지 매번 헛되이 다시
            # 부르게 된다 (실측으로 확인: 8개 중 7개가 멀쩡한데 1개 때문에 8개 전부
            # 재호출되던 사고).

            if logger:
                logger.call(
                    batch_index=batch_index, path=path, call_kind="batch", item_index=None, attempt=attempt + 1,
                    model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                    input_text=joined, raw_response=content, finish_reason=finish_reason,
                    items_in=len(paragraphs), items_out=len(parts),
                    validation_fail_reason=(f"partial_bad:{len(bad_indices)}" if bad_indices else None),
                    cache_hit=cache_hit,
                )
            if cache is not None and not cache_hit:
                cache.put(cache.make_key(joined, prompt_hash, model, temperature), content, finish_reason)
            if bad_indices:
                print(
                    f"번역 응답 중 {len(bad_indices)}개 항목만 문제 있음 ({len(paragraphs)}개 항목 배치) "
                    "-- 해당 항목만 개별 재번역",
                    file=sys.stderr,
                )
                for i in bad_indices:
                    parts[i] = _translate_single(
                        client, model, system_prompt, temperature, paragraphs[i], max_retries,
                        logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, item_index=i, path=path,
                        cache=cache,
                    )
            return parts
        except Exception as e:
            elapsed = time.monotonic() - call_start
            wait = min(60, 2**attempt)
            cause = e.__cause__ or e.__context__
            detail = f"{type(e).__name__}: {e}" + (f" | 원인: {type(cause).__name__}: {cause}" if cause else "")
            print(f"번역 실패 (시도 {attempt + 1}/{max_retries}): {detail} -- {wait}초 후 재시도", file=sys.stderr)
            if logger:
                logger.call(
                    batch_index=batch_index, path=path, call_kind="batch", item_index=None, attempt=attempt + 1,
                    model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                    input_text=joined, raw_response=None, finish_reason=None,
                    items_in=len(paragraphs), items_out=0, validation_fail_reason=f"exception:{type(e).__name__}",
                    cache_hit=False,
                )
            time.sleep(wait)
        attempt += 1
    raise RuntimeError("번역 재시도 한도를 초과했습니다")


def _bisect_and_translate(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    paragraphs: list[str],
    max_retries: int,
    *,
    logger: "RunLogger | None" = None,
    prompt_hash: str = "",
    batch_index: int = 0,
    path: str = "",
    cache: "TranslationCache | None" = None,
) -> list[str]:
    path = path or str(batch_index)
    mid = len(paragraphs) // 2
    return _translate_parts(
        client, model, system_prompt, temperature, paragraphs[:mid], max_retries,
        logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, path=f"{path}.L", cache=cache,
    ) + _translate_parts(
        client, model, system_prompt, temperature, paragraphs[mid:], max_retries,
        logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, path=f"{path}.R", cache=cache,
    )


def _split_table_row(row: str) -> list[str]:
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _join_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def translate_table(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    table_text: str,
    max_retries: int = 12,
    *,
    logger: "RunLogger | None" = None,
    prompt_hash: str = "",
    batch_index: int = 0,
    path: str = "",
    cache: "TranslationCache | None" = None,
) -> str:
    """표 전체를 통째로 프롬프트에 넣으면 셀이 밀리거나 열 개수가 바뀌는 사고가 나므로,
    번역이 필요한 셀 텍스트만 뽑아 문단 번역과 같은 구분자 검증 방식으로 번역하고,
    파이프(|) 구조와 각 행의 열 개수는 원본 그대로 코드로 재조립한다. 구분 행(---)과
    숫자·기호만으로 된 셀은 번역 없이 그대로 둔다."""
    lines = [line for line in table_text.split("\n") if line.strip()]
    header, sep, data_rows = lines[0], lines[1], lines[2:]
    rows = [_split_table_row(header)] + [_split_table_row(row) for row in data_rows]

    cell_positions: list[tuple[int, int]] = []
    cell_texts: list[str] = []
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            if cell and not NO_LETTERS_RE.match(cell):
                cell_positions.append((ri, ci))
                cell_texts.append(cell)

    if cell_texts:
        translated_cells = _translate_parts(
            client, model, system_prompt, temperature, cell_texts, max_retries,
            logger=logger, prompt_hash=prompt_hash, batch_index=batch_index, path=path or str(batch_index),
            cache=cache,
        )
        for (ri, ci), translated in zip(cell_positions, translated_cells):
            rows[ri][ci] = translated

    out_lines = [_join_table_row(rows[0]), sep] + [_join_table_row(row) for row in rows[1:]]
    return "\n".join(out_lines)


def _translate_single(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    paragraph: str,
    max_retries: int,
    *,
    logger: "RunLogger | None" = None,
    prompt_hash: str = "",
    batch_index: int = 0,
    item_index: int | None = None,
    path: str = "",
    cache: "TranslationCache | None" = None,
) -> str:
    path = path or str(batch_index)
    last_bad_content: str | None = None
    never_adopt_seen = False  # 이게 한 번이라도 걸리면, 마지막에 절대 그 응답을 채택하면 안 된다
    bad_attempts = 0
    # 캐시된 응답이 새 검증에 걸리면 use_cache를 꺼서 실제 호출로 넘어간다 - API를 안 쓴
    # 시도라 attempt/bad_attempts 둘 다 소모하지 않는다(continue로 카운터 증가를 건너뜀).
    use_cache = cache is not None
    attempt = 0
    while attempt < max_retries:
        call_start = time.monotonic()
        try:
            content, finish_reason, cache_hit = _call_model_cached(
                cache if use_cache else None, client, model, system_prompt, temperature, paragraph, prompt_hash,
            )
            elapsed = time.monotonic() - call_start
            if finish_reason == "length":
                if cache_hit:
                    use_cache = False
                    continue
                if logger:
                    logger.call(
                        batch_index=batch_index, path=path, call_kind="single", item_index=item_index, attempt=attempt + 1,
                        model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                        input_text=paragraph, raw_response=content, finish_reason=finish_reason,
                        items_in=1, items_out=0, validation_fail_reason="length_truncated", cache_hit=cache_hit,
                    )
                # 문단 하나가 그 자체로 모델 컨텍스트 한도를 넘김 -- 더 이상 쪼갤 단위가 없어
                # "축소·요약 금지" 원칙을 지킬 방법이 없으므로 조용히 넘어가지 않고 실패시킨다.
                raise RuntimeError("응답이 컨텍스트 한도로 잘렸는데 더 이상 쪼갤 수 없는 단일 문단입니다")

            fail_code: str | None = None
            if _looks_untranslated(paragraph, content):
                reason = "번역 결과에 한글이 거의 없음(번역 안 되고 원문이 그대로 돌아온 것으로 보임)"
                fail_code = "untranslated"
            elif _has_untranslated_run(paragraph, content):
                reason = "번역 결과 중 일부 구간이 통째로 원문 영어 그대로 남음(문단 전체는 번역됐지만 문장 일부 누락)"
                fail_code = "untranslated_run"
            elif _dollar_count_mismatch(paragraph, content):
                reason = (
                    f"수식 \\$ 개수가 원문과 다름 (원문 {paragraph.count('$')}개 / 번역 {content.count('$')}개) "
                    "-- 방치하면 뒤쪽 수식 렌더링까지 다 깨짐"
                )
                fail_code = "dollar_mismatch"
                never_adopt_seen = True
            elif _has_runaway_repetition(content):
                reason = "번역 응답이 같은 토큰을 수십~수백 번 반복하는 폭주 상태로 보임"
                fail_code = "runaway_repetition"
                never_adopt_seen = True
            elif _has_literal_blank_line_marker(content):
                reason = '번역 결과에 실제 빈 줄 대신 "(blank line)" 같은 문구가 그대로 남음'
                fail_code = "blank_line_marker"
                never_adopt_seen = True
            else:
                if cache is not None and not cache_hit:
                    cache.put(cache.make_key(paragraph, prompt_hash, model, temperature), content, finish_reason)
                if logger:
                    logger.call(
                        batch_index=batch_index, path=path, call_kind="single", item_index=item_index, attempt=attempt + 1,
                        model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                        input_text=paragraph, raw_response=content, finish_reason=finish_reason,
                        items_in=1, items_out=1, validation_fail_reason=None, cache_hit=cache_hit,
                    )
                return content

            if cache_hit:
                use_cache = False
                continue

            if logger:
                logger.call(
                    batch_index=batch_index, path=path, call_kind="single", item_index=item_index, attempt=attempt + 1,
                    model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                    input_text=paragraph, raw_response=content, finish_reason=finish_reason,
                    items_in=1, items_out=0, validation_fail_reason=fail_code, cache_hit=cache_hit,
                )
            last_bad_content = content
            bad_attempts += 1
            # 이건 서버 오류가 아니라 내용 문제라, 길게(최대 60초씩) 여러 번 재시도해봐야
            # 잘 안 바뀐다 -- 낭비를 줄이려고 훨씬 적은 횟수·짧은 대기로 따로 제한한다.
            if bad_attempts >= UNTRANSLATED_MAX_ATTEMPTS:
                break
            print(f"{reason} -- {UNTRANSLATED_RETRY_WAIT}초 후 재시도 ({bad_attempts}/{UNTRANSLATED_MAX_ATTEMPTS})", file=sys.stderr)
            time.sleep(UNTRANSLATED_RETRY_WAIT)
        except Exception as e:
            elapsed = time.monotonic() - call_start
            wait = min(60, 2**attempt)
            cause = e.__cause__ or e.__context__
            detail = f"{type(e).__name__}: {e}" + (f" | 원인: {type(cause).__name__}: {cause}" if cause else "")
            print(f"번역 실패 (시도 {attempt + 1}/{max_retries}): {detail} -- {wait}초 후 재시도", file=sys.stderr)
            if logger:
                logger.call(
                    batch_index=batch_index, path=path, call_kind="single", item_index=item_index, attempt=attempt + 1,
                    model=model, prompt_hash=prompt_hash, temperature=temperature, elapsed_s=elapsed,
                    input_text=paragraph, raw_response=None, finish_reason=None,
                    items_in=1, items_out=0, validation_fail_reason=f"exception:{type(e).__name__}", cache_hit=False,
                )
            time.sleep(wait)
        attempt += 1
    if never_adopt_seen:
        # \$ 짝이 안 맞거나 토큰이 폭주한 응답을 그대로 쓰면 이 문단뿐 아니라 문서 전체
        # 렌더링까지 밀려서 깨지므로, 절대 채택하지 않고 원문을 그대로 둔다.
        print(
            "경고: 여러 번 재시도해도 수식 깨짐/폭주 응답이 반복돼서, 문서 전체 렌더링이 깨지는 걸 막기 위해 "
            "이 문단은 번역하지 않고 원문 그대로 둡니다",
            file=sys.stderr,
        )
        return paragraph
    if last_bad_content is not None:
        # 여러 번 재시도해도 계속 원문 그대로 돌아온다 -- 참고문헌처럼 의도적으로 번역
        # 대상이 아닐 수도 있으므로, 전체 파이프라인을 실패시키는 대신 경고만 남기고
        # 마지막 응답을 그대로 채택한다.
        print(
            "경고: 여러 번 재시도했지만 번역되지 않은 것으로 보이는 문단을 원문 그대로 둡니다 "
            "(참고문헌 등 의도적 미번역 대상일 수 있음)",
            file=sys.stderr,
        )
        return last_bad_content
    raise RuntimeError("번역 재시도 한도를 초과했습니다")


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def _prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


class RunLogger:
    """모델 호출마다 원본 요청·응답·검증 결과를 JSONL로 남긴다. 논문 원문이 그대로
    들어가므로 이 파일은 반드시 .gitignore 대상이어야 한다 (커밋 금지)."""

    def __init__(self, log_path: Path, run_id: str, paper: str):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.run_id = run_id
        self.paper = paper
        self._f = open(log_path, "a", encoding="utf-8")
        self._fail_counts: dict[str, int] = {}
        self._model_calls = 0
        self._cache_hits = 0

    def call(
        self,
        *,
        batch_index: int,
        path: str,
        call_kind: str,  # "batch" | "single"
        item_index: int | None,
        attempt: int,
        model: str,
        prompt_hash: str,
        temperature: float,
        elapsed_s: float,
        input_text: str | None,
        raw_response: str | None,
        finish_reason: str | None,
        items_in: int,
        items_out: int,
        validation_fail_reason: str | None,
        cache_hit: bool = False,
    ) -> None:
        self._model_calls += 1
        if cache_hit:
            self._cache_hits += 1
        if validation_fail_reason:
            self._fail_counts[validation_fail_reason] = self._fail_counts.get(validation_fail_reason, 0) + 1
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "paper": self.paper,
            "call_kind": call_kind,
            "batch_index": batch_index,
            "path": path,
            "item_index": item_index,
            "attempt": attempt,
            "model": model,
            "prompt_hash": prompt_hash,
            "temperature": temperature,
            "elapsed_s": round(elapsed_s, 3),
            "input": input_text,
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "items_in": items_in,
            "items_out": items_out,
            "validation_fail_reason": validation_fail_reason,
            "cache_hit": cache_hit,
        }
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def summary(
        self,
        *,
        batches: int,
        elapsed_s: float,
        escaped_newline_fixes: int,
        output_path: str,
        dedup_ops: list[dict] | None = None,
        seeded_from: str | None = None,
    ) -> None:
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "paper": self.paper,
            "call_kind": "summary",
            "batches": batches,
            "model_calls": self._model_calls,
            "cache_hits": self._cache_hits,
            "validation_failures": self._fail_counts,
            "escaped_newline_fixes": escaped_newline_fixes,
            "dedup_ops": dedup_ops or [],
            "elapsed_s": round(elapsed_s, 3),
            "git_sha": _git_sha(),
            "output_path": output_path,
            "seeded_from": seeded_from,  # 이 run_id가 캐시-재생본이면 원본 run_id를 남긴다
        }
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    # 한때 4000 -> 2000으로 줄였다가(300초 타임아웃 경계 완화 목적) 되돌렸다. 실제
    # 전체 실행 시간으로 비교해보니(같은 논문, 비슷한 시간대: 4000자 배치 149분 vs
    # 2000자 배치 236분) 요청 자체에 상당한 고정 지연이 있어서, 배치를 잘게 쪼개
    # 요청 수를 늘리는 쪽이 오히려 총 시간을 더 늘렸다. 타임아웃 쪽(아래 300 -> 480)을
    # 늘려서 대응하는 게 낫다.
    parser.add_argument("--max-chars", type=int, default=4000, help="배치당 최대 문자 수")
    parser.add_argument("--log-dir", default="logs", help="청크 단위 JSONL 로그를 남길 디렉터리")
    parser.add_argument(
        "--system-prompt-file", default=None,
        help="config.yaml의 system_prompt_file을 덮어쓴다 (캐시 재생 시 프롬프트 버전을 고정할 때 사용)",
    )
    parser.add_argument(
        "--expect-prompt-hash", default=None,
        help="로드한 프롬프트의 해시가 이 값과 다르면 즉시 종료한다 - 캐시 재생 때 프롬프트가 "
        "조용히 바뀌어 전 항목이 미스되는 것을 막는다",
    )
    parser.add_argument("--cache-db", default=None, help="캐시 sqlite DB 경로 (지정하면 캐시 활성화)")
    parser.add_argument("--no-cache", action="store_true", help="--cache-db가 있어도 캐시를 끈다")
    parser.add_argument("--seeded-from", default=None, help="이 실행이 특정 run_id의 로그로 캐시를 시딩한 재생본이면 그 run_id")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    config = full_config["translation"]

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 NVIDIA_API_KEY가 설정되어 있지 않습니다")

    # 90초는 너무 짧아서 정상적으로 느린(수 분 걸리는) 응답까지 타임아웃으로 죽였다.
    # 300초로 늘렸었는데, 실측해보니(로컬 반복 테스트) 4000자대 배치 중 정상적으로
    # 성공하는 것도 230~293초씩 걸려 300초 경계에 바짝 붙어 있었다. 배치를 잘게
    # 쪼개 요청 수를 늘리는 대신(고정 지연이 커서 총 시간이 더 늘어남, 실측 확인)
    # 480초로 더 늘려 여유를 준다. SDK 자체 재시도(기본 max_retries=2)는 꺼서 아래
    # 재시도 루프와 중첩되어 한 배치가 몇 시간씩 멎는 것만 막는다.
    # keep-alive 연결 재사용을 꺼서, 배치 사이 유휴 시간에 서버(프록시)가 먼저 끊어버린
    # 연결을 재사용하다 "Server disconnected without sending a response"가 나는 걸 막는다.
    http_client = httpx.Client(limits=httpx.Limits(max_keepalive_connections=0))
    client = OpenAI(
        base_url=config["base_url"], api_key=api_key, timeout=480.0, max_retries=0, http_client=http_client
    )
    system_prompt = load_system_prompt(args.system_prompt_file or config["system_prompt_file"])
    prompt_hash = _prompt_hash(system_prompt)
    if args.expect_prompt_hash and prompt_hash != args.expect_prompt_hash:
        # 조용한 캐시 미스가 이 세션의 출발점이었던 조용한 스킵과 같은 종류라, 여기서도
        # 조용히 넘어가지 않고 즉시 종료한다.
        raise SystemExit(
            f"프롬프트 해시 불일치: 기대 {args.expect_prompt_hash}, 실제 {prompt_hash} "
            "-- 캐시 시딩 당시와 다른 프롬프트라 전 항목이 미스됩니다"
        )

    cache = None
    if args.cache_db and not args.no_cache:
        cache = TranslationCache(Path(args.cache_db))

    text = Path(args.input).read_text(encoding="utf-8")
    batches = chunk_markdown(text, args.max_chars)
    print(f"{len(batches)}개 배치로 분할, 순차 번역 시작", file=sys.stderr)

    paper_stem = Path(args.input).stem
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    log_path = Path(args.log_dir) / f"translate_{paper_stem}_{run_id}.jsonl"
    logger = RunLogger(log_path, run_id, paper_stem)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    escaped_newline_fix_total = 0
    dedup_ops: list[dict] = []
    total_elapsed = 0.0
    try:
        with open(args.output, "w", encoding="utf-8") as out_f:
            for i, batch in enumerate(batches):
                batch_len = sum(len(p) for p in batch)
                batch_start = time.monotonic()
                if len(batch) == 1 and is_table(batch[0]):
                    print(f"[{i + 1}/{len(batches)}] 표 번역 중 ({batch_len}자)", file=sys.stderr)
                    translated = translate_table(
                        client, config["model"], system_prompt, config["temperature"], batch[0],
                        logger=logger, prompt_hash=prompt_hash, batch_index=i, cache=cache,
                    )
                else:
                    print(f"[{i + 1}/{len(batches)}] 번역 중 (문단 {len(batch)}개, {batch_len}자)", file=sys.stderr)
                    translated = translate_paragraphs(
                        client, config["model"], system_prompt, config["temperature"], batch,
                        logger=logger, prompt_hash=prompt_hash, batch_index=i, cache=cache,
                    )
                fixed, fix_count = fix_escaped_newlines(translated.strip())
                escaped_newline_fix_total += fix_count
                out_f.write(fixed + "\n\n")
                out_f.flush()
                print(f"[{i + 1}/{len(batches)}] {time.monotonic() - batch_start:.1f}초 걸림", file=sys.stderr)

        # dedup_term_glosses는 "이미 나온 용어인지"를 문서 전체 기준으로 판단해야 해서
        # 배치별 스트리밍 저장 중에는 적용할 수 없다 - 전 배치가 파일에 다 쓰인 뒤,
        # 한 번에 다시 읽어서 최종 패스로 돌린다. 배치별 스트리밍 저장(진행 상황 표시,
        # 중간에 죽어도 그때까지 결과는 남게 하는 용도)은 그대로 두고 이 패스만 덧붙인다.
        final_text = Path(args.output).read_text(encoding="utf-8")
        deduped_text, dedup_ops = dedup_term_glosses(final_text)
        Path(args.output).write_text(deduped_text, encoding="utf-8")
    finally:
        # 재시도 한도 초과 등으로 루프 도중 예외가 나도, 그때까지의 실패 집계가 로그
        # 요약에 남아야 한다 - 실패율을 재려고 만든 로그인데 정작 실패한 실행에
        # 요약이 없으면 그 목적을 잃는다.
        total_elapsed = time.monotonic() - run_start
        logger.summary(
            batches=len(batches), elapsed_s=total_elapsed,
            escaped_newline_fixes=escaped_newline_fix_total, output_path=args.output,
            dedup_ops=dedup_ops, seeded_from=args.seeded_from,
        )
        if cache is not None:
            cache.close()
    print(f"번역 완료: {args.output} (총 {total_elapsed:.1f}초, 로그: {log_path})", file=sys.stderr)


if __name__ == "__main__":
    main()
