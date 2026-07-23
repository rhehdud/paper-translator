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


def check_page_count(pdf_path: Path, max_pages: int) -> int:
    num_pages = len(PdfReader(str(pdf_path)).pages)
    if num_pages > max_pages:
        print(
            f"경고: {pdf_path.name}은 {num_pages}페이지로 max_pages({max_pages})를 초과합니다. "
            "잡 시간이 오래 걸릴 수 있습니다.",
            file=sys.stderr,
        )
    return num_pages


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
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--out", required=True, help="최종 마크다운을 저장할 경로")
    args = parser.parse_args()

    with open(args.selected, encoding="utf-8") as f:
        paper = json.load(f)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = work_dir / f"{paper['id']}.pdf"
    download_pdf(paper["pdf_url"], pdf_path)

    check_page_count(pdf_path, args.max_pages)

    md_path = run_marker(pdf_path, work_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(md_path, args.out)
    print(f"추출 완료: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
