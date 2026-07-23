"""NVIDIA NIM(google/gemma-4-31b-it)으로 마크다운을 청크 단위 순차 번역한다.

사용법:
    NVIDIA_API_KEY=nvapi-... python src/translate.py --input paper.md --output paper.ko.md
"""
import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import yaml
from openai import OpenAI


def load_system_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def chunk_markdown(text: str, max_chars: int) -> list[str]:
    """빈 줄 단위 문단을 max_chars 예산 안에서 묶어 청크로 만든다 (문단 자체는 쪼개지 않음)."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_in_half(chunk: str) -> tuple[str, str] | None:
    """문단(빈 줄) 경계를 기준으로 청크를 절반씩 나눈다. 문단이 하나뿐이면 나눌 수 없다."""
    paragraphs = chunk.split("\n\n")
    if len(paragraphs) < 2:
        return None
    mid = len(paragraphs) // 2
    return "\n\n".join(paragraphs[:mid]), "\n\n".join(paragraphs[mid:])


def translate_chunk(
    client: OpenAI, model: str, system_prompt: str, temperature: float, chunk: str, max_retries: int = 12
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=temperature,
                # thinking을 켜두면 답변마다 긴 추론 과정을 먼저 생성해 훨씬 느려진다.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = resp.choices[0].message.content or ""
            if resp.choices[0].finish_reason == "length":
                # 모델 컨텍스트 한도에 걸려 응답이 중간에 잘림 -- 그대로 쓰면 "축소·요약 금지"
                # 원칙이 조용히 깨지므로, 절대 그대로 반환하지 않고 청크를 반으로 쪼개 재번역한다.
                halves = _split_in_half(chunk)
                if halves is None:
                    raise RuntimeError("응답이 컨텍스트 한도로 잘렸는데 더 이상 쪼갤 수 없는 단일 문단입니다")
                print(
                    f"번역 응답이 컨텍스트 한도로 잘림 ({len(chunk)}자 청크) -- 절반으로 나눠 재번역",
                    file=sys.stderr,
                )
                first, second = halves
                return (
                    translate_chunk(client, model, system_prompt, temperature, first, max_retries)
                    + "\n\n"
                    + translate_chunk(client, model, system_prompt, temperature, second, max_retries)
                )
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
    parser.add_argument("--max-chars", type=int, default=12000, help="청크당 최대 문자 수")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    config = full_config["translation"]

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 NVIDIA_API_KEY가 설정되어 있지 않습니다")

    # 90초는 너무 짧아서 정상적으로 느린(수 분 걸리는) 응답까지 타임아웃으로 죽였다.
    # 5분으로 늘리되, SDK 자체 재시도(기본 max_retries=2)는 꺼서 아래 translate_chunk의
    # 재시도 루프와 중첩되어 한 청크가 몇 시간씩 멎는 것만 막는다.
    # keep-alive 연결 재사용을 꺼서, 청크 사이 유휴 시간에 서버(프록시)가 먼저 끊어버린
    # 연결을 재사용하다 "Server disconnected without sending a response"가 나는 걸 막는다.
    http_client = httpx.Client(limits=httpx.Limits(max_keepalive_connections=0))
    client = OpenAI(
        base_url=config["base_url"], api_key=api_key, timeout=300.0, max_retries=0, http_client=http_client
    )
    system_prompt = load_system_prompt(config["system_prompt_file"])

    text = Path(args.input).read_text(encoding="utf-8")
    chunks = chunk_markdown(text, args.max_chars)
    print(f"{len(chunks)}개 청크로 분할, 순차 번역 시작", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks):
            print(f"[{i + 1}/{len(chunks)}] 번역 중 ({len(chunk)}자)", file=sys.stderr)
            translated = translate_chunk(client, config["model"], system_prompt, config["temperature"], chunk)
            out_f.write(translated.strip() + "\n\n")
            out_f.flush()

    print(f"번역 완료: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
