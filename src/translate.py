"""NVIDIA NIM(google/gemma-4-31b-it)으로 마크다운을 청크 단위 순차 번역한다.

사용법:
    NVIDIA_API_KEY=nvapi-... python src/translate.py --input paper.md --output paper.ko.md
"""
import argparse
import os
import sys
import time
from pathlib import Path

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


def translate_chunk(
    client: OpenAI, model: str, system_prompt: str, temperature: float, chunk: str, max_retries: int = 5
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
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            wait = min(60, 2**attempt)
            print(f"번역 실패 (시도 {attempt + 1}/{max_retries}): {e} -- {wait}초 후 재시도", file=sys.stderr)
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

    client = OpenAI(base_url=config["base_url"], api_key=api_key)
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
