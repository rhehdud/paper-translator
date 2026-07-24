"""선정된 논문의 PDF를 내려받아 Marker로 구조화된 마크다운을 추출한다.

사용법:
    python src/extract.py --selected selected_cs_CL.json --work-dir work/cs_CL
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml
from pypdf import PdfReader


def _marker_single_binary() -> str:
    """pip/conda 환경의 bin 디렉터리에 설치된 marker_single을 찾는다 (PATH에 없어도 동작하도록)."""
    candidate = Path(sys.executable).parent / "marker_single"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("marker_single")
    if found:
        return found
    raise FileNotFoundError("marker_single 실행 파일을 찾을 수 없습니다 (marker-pdf가 설치됐는지 확인하세요)")


def download_pdf(pdf_url: str, dest: Path) -> None:
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "paper-translator-bot/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def count_pages(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def normalize_table_spacing(markdown_text: str) -> str:
    """Marker가 표 캡션·본문 바로 다음 줄에 빈 줄 없이 표를 이어붙이면, 마크다운 표
    파서가 표의 시작을 인식하지 못해 파이프(|) 문자가 렌더링되지 않고 그대로 텍스트로
    노출되는 버그가 생긴다. 표로 보이는 줄(파이프로 시작·끝)의 앞뒤에 빈 줄이 없으면
    강제로 넣어 항상 별도 블록으로 분리한다."""
    lines = markdown_text.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        is_table_row = bool(TABLE_ROW_RE.match(line))
        prev_is_table_row = bool(fixed) and bool(TABLE_ROW_RE.match(fixed[-1]))
        if is_table_row and fixed and fixed[-1].strip() != "" and not prev_is_table_row:
            fixed.append("")
        fixed.append(line)
        if is_table_row:
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            next_is_table_row = next_line is not None and bool(TABLE_ROW_RE.match(next_line))
            if next_line is not None and next_line.strip() != "" and not next_is_table_row:
                fixed.append("")
    return "\n".join(fixed)


def normalize_anchor_spacing(markdown_text: str) -> str:
    """Marker가 <span id="..."></span> 앵커 태그 바로 다음 줄에 빈 줄 없이 내용을 붙여두면,
    마크다운이 그 뒤에 오는 $$ 수식을 같은 문단으로 묶어버려 표시(display) 수식이 인라인으로
    잘못 처리되면서 '$' 기호가 그대로 남는 렌더링 버그가 생긴다. 앵커 단독 줄 뒤에 빈 줄을
    강제로 넣어 항상 별도 문단으로 분리한다."""
    lines = markdown_text.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        fixed.append(line)
        if re.fullmatch(r'\s*<span id="[^"]*"></span>\s*', line):
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if next_line is not None and next_line.strip() != "":
                fixed.append("")
    return "\n".join(fixed)


EQUATION_LINE_RE = re.compile(r"^\s*\$\$.*\$\$\s*$")
# 수식 바로 뒤에 "(2)"처럼 같은 줄에 붙은 수식 번호 (괄호 안 10자 이내로 제한해 오탐 방지)
EQUATION_TRAILING_LABEL_RE = re.compile(r"^(\s*\$\$.*\$\$)\s*(\([^()\n]{1,10}\))\s*$")


def normalize_equation_spacing(markdown_text: str) -> str:
    """Marker가 디스플레이 수식($$...$$) 바로 옆(같은 줄 끝 또는 바로 다음 줄)에 빈 줄
    없이 "(2)" 같은 수식 번호를 붙여두면, 마크다운이 수식과 번호를 하나의 문단으로
    묶어버려 표시 수식이 인라인으로 잘못 처리되는 렌더링 버그가 생긴다 (앵커 태그와
    같은 클래스의 문제). 같은 줄에 붙은 번호는 먼저 별도 줄로 떼어내고, 수식 줄
    앞뒤에 빈 줄이 없으면 강제로 넣어 항상 별도 블록으로 분리한다."""
    lines = markdown_text.split("\n")

    split_lines: list[str] = []
    for line in lines:
        m = EQUATION_TRAILING_LABEL_RE.match(line)
        if m:
            split_lines.append(m.group(1))
            split_lines.append(m.group(2))
        else:
            split_lines.append(line)
    lines = split_lines

    fixed: list[str] = []
    for i, line in enumerate(lines):
        is_eq = bool(EQUATION_LINE_RE.match(line))
        prev_is_eq = bool(fixed) and bool(EQUATION_LINE_RE.match(fixed[-1]))
        if is_eq and fixed and fixed[-1].strip() != "" and not prev_is_eq:
            fixed.append("")
        fixed.append(line)
        if is_eq:
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if next_line is not None and next_line.strip() != "":
                fixed.append("")
    return "\n".join(fixed)


def run_marker(pdf_path: Path, output_dir: Path) -> Path:
    subprocess.run(
        [
            _marker_single_binary(),
            str(pdf_path),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "markdown",
        ],
        check=True,
    )
    stem = pdf_path.stem
    md_path = output_dir / stem / f"{stem}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Marker 출력을 찾을 수 없습니다: {md_path}")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True, help="collect.py가 만든 논문 1편의 JSON")
    parser.add_argument("--work-dir", required=True, help="PDF·마커 출력물을 둘 작업 디렉터리")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-pages", type=int, default=None, help="지정 안 하면 config.yaml의 extraction.max_pages를 씀")
    parser.add_argument("--out-dir", required=True, help="마크다운+이미지를 저장할 디렉터리")
    args = parser.parse_args()

    if args.max_pages is None:
        with open(args.config, encoding="utf-8") as f:
            args.max_pages = yaml.safe_load(f)["extraction"]["max_pages"]

    with open(args.selected, encoding="utf-8") as f:
        paper = json.load(f)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = work_dir / f"{paper['id']}.pdf"
    download_pdf(paper["pdf_url"], pdf_path)

    num_pages = count_pages(pdf_path)
    if num_pages > args.max_pages:
        print(
            f"건너뜀: {pdf_path.name}은 {num_pages}페이지로 max_pages({args.max_pages})를 초과합니다. "
            "이 논문은 이번 주 번역 대상에서 제외합니다.",
            file=sys.stderr,
        )
        return

    md_path = run_marker(pdf_path, work_dir)
    marker_output_dir = md_path.parent  # Marker가 마크다운과 이미지를 같이 뽑아둔 폴더

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in marker_output_dir.iterdir():
        if item.suffix == ".json":
            continue  # _meta.json은 필요 없음
        shutil.copy(item, out_dir / item.name)

    # 마크다운 파일 이름을 다운스트림에서 예측 가능하게 고정하면서, 앵커 태그 뒤 수식이
    # 인라인으로 잘못 처리되는 걸 막기 위해 빈 줄 정규화를 적용한다.
    raw_md = (out_dir / md_path.name).read_text(encoding="utf-8")
    (out_dir / md_path.name).unlink()
    normalized_md = normalize_equation_spacing(normalize_anchor_spacing(normalize_table_spacing(raw_md)))
    (out_dir / "extracted.md").write_text(normalized_md, encoding="utf-8")

    print(f"추출 완료: {out_dir} (이미지 포함)", file=sys.stderr)


if __name__ == "__main__":
    main()
