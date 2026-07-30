"""NVIDIA NIM(openai/gpt-oss-20b)으로 마크다운을 문단 배치 단위로 순차 번역한다.

사용법:
    NVIDIA_API_KEY=nvapi-... python src/translate.py --input paper.md --output paper.ko.md
"""
import argparse
import os
import re
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

# 모델이 실제 줄바꿈 대신 리터럴 "\n" 두 글자를 그대로 출력하는 경우가 있다. 처음엔
# 헤더 앞뒤에서만 발견돼(예: "\n## 4 구성요소 (Components)\n\n") 헤더 줄에만 대응하는
# 좁은 정규식으로 고쳤는데, 나중에 저자·소속·링크 목록 같은 일반 본문 줄에서도 똑같이
# 나오는 게 발견됨(예: "앱: ...\n 홈페이지: ...\n"). 그래서 헤더 여부와 무관하게 코드
# 블록 밖 전체에서 리터럴 "\n"을 실제 줄바꿈으로 바꾼다. 코드 블록 안의 "\n"은 실제
# 코드(예: 파이썬 문자열 리터럴)일 수 있으므로 건드리지 않는다.
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def fix_escaped_newlines(text: str) -> str:
    """코드 블록 밖에서 리터럴 "\\n"을 실제 줄바꿈으로 치환한다."""
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("\\n", "\n")
    return "".join(parts)

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


def _looks_like_references(paragraph: str) -> bool:
    """실제 참고문헌 "목록" 항목(Marker가 각 서지 항목을 "- [n] ..." 리스트로 뽑아낸 것)만
    미번역 검사에서 제외한다. 관련 연구·논의 등 일반 본문 문단은 인용을 여러 개 걸치더라도
    실제로는 번역이 필요한 산문이며, 리스트 항목이 아니면 인용 개수가 아무리 많아도 절대
    제외하지 않는다. (예전엔 인용 개수만 보고 판단해서, 인용이 빽빽한 본문 문단이 통째로
    영어 원문 그대로 방치되거나 심지어 중국어로 오역된 채 검증 없이 그대로 발행된 사례가
    실제 발견됨 — 리스트 마커 유무로 39개 파일 전수 검사해 회귀 없음을 확인.)"""
    first_line = paragraph.lstrip().split("\n", 1)[0]
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


def chunk_markdown(text: str, max_chars: int = 4000) -> list[list[str]]:
    """문단(빈 줄 구분)을 max_chars 예산 안에서 묶어 배치로 만든다 (문단 자체는 쪼개지 않음).
    각 배치는 문단 문자열의 리스트로 반환되며, 실제 API 호출 시 고유 구분자로 이어붙인다.
    표 문단은 셀 단위로 따로 번역해야 열이 밀리지 않으므로, 항상 단독 배치로 분리한다."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
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


def translate_paragraphs(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    paragraphs: list[str],
    max_retries: int = 12,
) -> str:
    """문단 여러 개를 번역해 빈 줄로 이어붙인 문자열로 돌려준다 (일반 본문 배치용)."""
    return "\n\n".join(_translate_parts(client, model, system_prompt, temperature, paragraphs, max_retries))


def _translate_parts(
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    paragraphs: list[str],
    max_retries: int = 12,
) -> list[str]:
    """문단(또는 표 셀) 여러 개를 고유 구분자로 묶어 한 번에 번역하고, 번역된 조각을
    입력과 같은 개수의 리스트로 돌려준다. 응답에서 구분자 개수가 보낸 개수와 정확히
    일치하는지 코드로 검증하고, 안 맞으면(모델이 항목을 합쳐버렸으면) 절반으로 나눠
    재귀적으로 재시도한다 -- 프롬프트 지시만으로는 구조 보존을 신뢰할 수 없어서 항상
    코드로 검증한다."""
    if len(paragraphs) == 1:
        return [_translate_single(client, model, system_prompt, temperature, paragraphs[0], max_retries)]

    joined = f"\n\n{PARAGRAPH_DELIMITER}\n\n".join(paragraphs)
    for attempt in range(max_retries):
        try:
            content, finish_reason = _call_model(client, model, system_prompt, temperature, joined)
            if finish_reason == "length":
                print(
                    f"번역 응답이 컨텍스트 한도로 잘림 ({len(paragraphs)}개 항목 배치) -- 절반으로 나눠 재번역",
                    file=sys.stderr,
                )
                return _bisect_and_translate(client, model, system_prompt, temperature, paragraphs, max_retries)

            parts = [p.strip() for p in re.split(re.escape(PARAGRAPH_DELIMITER), content) if p.strip()]
            if len(parts) != len(paragraphs):
                print(
                    f"번역 응답의 항목 수가 안 맞음 (보냄 {len(paragraphs)} / 받음 {len(parts)}) "
                    "-- 절반으로 나눠 재번역",
                    file=sys.stderr,
                )
                return _bisect_and_translate(client, model, system_prompt, temperature, paragraphs, max_retries)

            # 항목 수가 맞으면 원문-번역 짝이 인덱스로 정확히 매칭되므로, 문제 있는
            # 항목만 개별 재번역한다. 배치 전체를 버리고 절반씩 다시 보내면, 이미 잘
            # 번역된 나머지 항목들까지 매번 헛되이 재요청하게 되어 시간이 크게 낭비된다
            # (실제로 33개 항목 중 1개 때문에 33->16->8->4->2로 반복 재번역되는 사례 확인).
            bad_indices = [
                i
                for i, (src, out) in enumerate(zip(paragraphs, parts))
                if _looks_untranslated(src, out) or _dollar_count_mismatch(src, out)
            ]
            if bad_indices:
                print(
                    f"번역 응답 중 {len(bad_indices)}개 항목만 문제 있음 ({len(paragraphs)}개 항목 배치) "
                    "-- 해당 항목만 개별 재번역",
                    file=sys.stderr,
                )
                for i in bad_indices:
                    parts[i] = _translate_single(
                        client, model, system_prompt, temperature, paragraphs[i], max_retries
                    )
            return parts
        except Exception as e:
            wait = min(60, 2**attempt)
            cause = e.__cause__ or e.__context__
            detail = f"{type(e).__name__}: {e}" + (f" | 원인: {type(cause).__name__}: {cause}" if cause else "")
            print(f"번역 실패 (시도 {attempt + 1}/{max_retries}): {detail} -- {wait}초 후 재시도", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("번역 재시도 한도를 초과했습니다")


def _bisect_and_translate(
    client: OpenAI, model: str, system_prompt: str, temperature: float, paragraphs: list[str], max_retries: int
) -> list[str]:
    mid = len(paragraphs) // 2
    return _translate_parts(
        client, model, system_prompt, temperature, paragraphs[:mid], max_retries
    ) + _translate_parts(client, model, system_prompt, temperature, paragraphs[mid:], max_retries)


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
        translated_cells = _translate_parts(client, model, system_prompt, temperature, cell_texts, max_retries)
        for (ri, ci), translated in zip(cell_positions, translated_cells):
            rows[ri][ci] = translated

    out_lines = [_join_table_row(rows[0]), sep] + [_join_table_row(row) for row in rows[1:]]
    return "\n".join(out_lines)


def _translate_single(
    client: OpenAI, model: str, system_prompt: str, temperature: float, paragraph: str, max_retries: int
) -> str:
    last_bad_content: str | None = None
    never_adopt_seen = False  # 이게 한 번이라도 걸리면, 마지막에 절대 그 응답을 채택하면 안 된다
    bad_attempts = 0
    for attempt in range(max_retries):
        try:
            content, finish_reason = _call_model(client, model, system_prompt, temperature, paragraph)
            if finish_reason == "length":
                # 문단 하나가 그 자체로 모델 컨텍스트 한도를 넘김 -- 더 이상 쪼갤 단위가 없어
                # "축소·요약 금지" 원칙을 지킬 방법이 없으므로 조용히 넘어가지 않고 실패시킨다.
                raise RuntimeError("응답이 컨텍스트 한도로 잘렸는데 더 이상 쪼갤 수 없는 단일 문단입니다")

            if _looks_untranslated(paragraph, content):
                reason = "번역 결과에 한글이 거의 없음(번역 안 되고 원문이 그대로 돌아온 것으로 보임)"
            elif _dollar_count_mismatch(paragraph, content):
                reason = (
                    f"수식 \\$ 개수가 원문과 다름 (원문 {paragraph.count('$')}개 / 번역 {content.count('$')}개) "
                    "-- 방치하면 뒤쪽 수식 렌더링까지 다 깨짐"
                )
                never_adopt_seen = True
            elif _has_runaway_repetition(content):
                reason = "번역 응답이 같은 토큰을 수십~수백 번 반복하는 폭주 상태로 보임"
                never_adopt_seen = True
            else:
                return content

            last_bad_content = content
            bad_attempts += 1
            # 이건 서버 오류가 아니라 내용 문제라, 길게(최대 60초씩) 여러 번 재시도해봐야
            # 잘 안 바뀐다 -- 낭비를 줄이려고 훨씬 적은 횟수·짧은 대기로 따로 제한한다.
            if bad_attempts >= UNTRANSLATED_MAX_ATTEMPTS:
                break
            print(f"{reason} -- {UNTRANSLATED_RETRY_WAIT}초 후 재시도 ({bad_attempts}/{UNTRANSLATED_MAX_ATTEMPTS})", file=sys.stderr)
            time.sleep(UNTRANSLATED_RETRY_WAIT)
        except Exception as e:
            wait = min(60, 2**attempt)
            cause = e.__cause__ or e.__context__
            detail = f"{type(e).__name__}: {e}" + (f" | 원인: {type(cause).__name__}: {cause}" if cause else "")
            print(f"번역 실패 (시도 {attempt + 1}/{max_retries}): {detail} -- {wait}초 후 재시도", file=sys.stderr)
            time.sleep(wait)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    # 4000자대 배치는 생성에 4~5분씩 걸려 300초 타임아웃 경계에 자주 걸렸다(로컬
    # 반복 테스트로 실측: 3700~4100자 배치 5개 중 3개가 정확히 300초에서 타임아웃,
    # 2183자 배치는 109초로 여유 있게 성공). 배치를 작게 나눠 요청당 생성 시간을
    # 줄여서 타임아웃에 걸릴 여지를 줄인다.
    parser.add_argument("--max-chars", type=int, default=2000, help="배치당 최대 문자 수")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    config = full_config["translation"]

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 NVIDIA_API_KEY가 설정되어 있지 않습니다")

    # 90초는 너무 짧아서 정상적으로 느린(수 분 걸리는) 응답까지 타임아웃으로 죽였다.
    # 5분으로 늘리되, SDK 자체 재시도(기본 max_retries=2)는 꺼서 아래 재시도 루프와
    # 중첩되어 한 배치가 몇 시간씩 멎는 것만 막는다.
    # keep-alive 연결 재사용을 꺼서, 배치 사이 유휴 시간에 서버(프록시)가 먼저 끊어버린
    # 연결을 재사용하다 "Server disconnected without sending a response"가 나는 걸 막는다.
    http_client = httpx.Client(limits=httpx.Limits(max_keepalive_connections=0))
    client = OpenAI(
        base_url=config["base_url"], api_key=api_key, timeout=300.0, max_retries=0, http_client=http_client
    )
    system_prompt = load_system_prompt(config["system_prompt_file"])

    text = Path(args.input).read_text(encoding="utf-8")
    batches = chunk_markdown(text, args.max_chars)
    print(f"{len(batches)}개 배치로 분할, 순차 번역 시작", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, batch in enumerate(batches):
            batch_len = sum(len(p) for p in batch)
            batch_start = time.monotonic()
            if len(batch) == 1 and is_table(batch[0]):
                print(f"[{i + 1}/{len(batches)}] 표 번역 중 ({batch_len}자)", file=sys.stderr)
                translated = translate_table(client, config["model"], system_prompt, config["temperature"], batch[0])
            else:
                print(f"[{i + 1}/{len(batches)}] 번역 중 (문단 {len(batch)}개, {batch_len}자)", file=sys.stderr)
                translated = translate_paragraphs(
                    client, config["model"], system_prompt, config["temperature"], batch
                )
            out_f.write(fix_escaped_newlines(translated.strip()) + "\n\n")
            out_f.flush()
            print(f"[{i + 1}/{len(batches)}] {time.monotonic() - batch_start:.1f}초 걸림", file=sys.stderr)

    print(f"번역 완료: {args.output} (총 {time.monotonic() - run_start:.1f}초)", file=sys.stderr)


if __name__ == "__main__":
    main()
