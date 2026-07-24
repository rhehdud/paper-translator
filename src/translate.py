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
    for attempt in range(max_retries):
        try:
            content, finish_reason = _call_model(client, model, system_prompt, temperature, paragraph)
            if finish_reason == "length":
                # 문단 하나가 그 자체로 모델 컨텍스트 한도를 넘김 -- 더 이상 쪼갤 단위가 없어
                # "축소·요약 금지" 원칙을 지킬 방법이 없으므로 조용히 넘어가지 않고 실패시킨다.
                raise RuntimeError("응답이 컨텍스트 한도로 잘렸는데 더 이상 쪼갤 수 없는 단일 문단입니다")
            return content
        except Exception as e:
            wait = min(60, 2**attempt)
            cause = e.__cause__ or e.__context__
            detail = f"{type(e).__name__}: {e}" + (f" | 원인: {type(cause).__name__}: {cause}" if cause else "")
            print(f"번역 실패 (시도 {attempt + 1}/{max_retries}): {detail} -- {wait}초 후 재시도", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("번역 재시도 한도를 초과했습니다")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-chars", type=int, default=4000, help="배치당 최대 문자 수")
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
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, batch in enumerate(batches):
            batch_len = sum(len(p) for p in batch)
            if len(batch) == 1 and is_table(batch[0]):
                print(f"[{i + 1}/{len(batches)}] 표 번역 중 ({batch_len}자)", file=sys.stderr)
                translated = translate_table(client, config["model"], system_prompt, config["temperature"], batch[0])
            else:
                print(f"[{i + 1}/{len(batches)}] 번역 중 (문단 {len(batch)}개, {batch_len}자)", file=sys.stderr)
                translated = translate_paragraphs(
                    client, config["model"], system_prompt, config["temperature"], batch
                )
            out_f.write(translated.strip() + "\n\n")
            out_f.flush()

    print(f"번역 완료: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
