"""선정된 논문의 PDF를 내려받아 Marker로 구조화된 마크다운을 추출한다.

사용법:
    python src/extract.py --selected selected_cs_CL.json --work-dir work/cs_CL
"""
import argparse
import json
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

    # 마크다운 파일 이름을 다운스트림에서 예측 가능하게 고정
    (out_dir / md_path.name).rename(out_dir / "extracted.md")

    print(f"추출 완료: {out_dir} (이미지 포함)", file=sys.stderr)


if __name__ == "__main__":
    main()
