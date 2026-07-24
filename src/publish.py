"""번역된 마크다운과 논문 메타데이터를 MkDocs 문서 페이지로 합친다.

사용법:
    python src/publish.py --selected selected_cs_CL.json --translated paper.ko.md --docs-dir docs
"""
import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

CATEGORY_DIR_NAMES = {
    "cs.CL": "nlp",
    "cs.CV": "vision",
    "cs.LG": "ml",
    "cs.AI": "ai",
    "cs.RO": "robotics",
}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".gif", ".webp", ".svg"}
IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

NAV_TITLE_MAX_LEN = 40
_HEADING_LINE_RE = re.compile(r'^\s*#\s+(?:<span[^>]*></span>\s*)?(.*)$')


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len].rstrip("-")


def truncate_title(title: str, max_len: int = NAV_TITLE_MAX_LEN) -> str:
    """사이드바 내비게이션에는 제목이 너무 길면 잘라서 보여준다."""
    if len(title) <= max_len:
        return title
    return title[:max_len].rstrip() + "…"


def extract_translated_title(translated_body: str, fallback: str) -> str:
    """번역된 본문 첫 줄(원 논문 제목의 번역)에서 제목만 뽑아낸다. Marker가 제목·저자·
    다음 섹션 헤더를 빈 줄 없이 한 줄에 붙여 놓는 경우가 있어, 다음 마크다운 헤더
    마커(## ~ ######) 앞까지만 제목으로 본다. 뽑아내지 못하면 원문(영어) 제목을 쓴다."""
    first_line = translated_body.split("\n", 1)[0]
    match = _HEADING_LINE_RE.match(first_line)
    if not match:
        return fallback
    text = match.group(1).strip()
    cut = re.search(r"\s#{2,6}\s", text)
    if cut:
        text = text[: cut.start()]
    text = text.replace("**", "").strip()
    return text or fallback


def copy_images_and_rewrite_refs(translated_body: str, source_dir: Path, page_images_dir: Path) -> str:
    """source_dir에 있던 이미지 파일들을 page_images_dir로 복사하고, 마크다운의 이미지
    참조 경로를 그 위치를 가리키도록 다시 쓴다 (다른 논문과 파일명이 겹쳐도 충돌 없게)."""
    copied: set[str] = set()

    def _rewrite(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        if "/" in path or path.startswith(("http://", "https://")):
            return match.group(0)  # 이미 경로가 있거나 외부 URL이면 손대지 않음
        src = source_dir / path
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS or not src.exists():
            return match.group(0)
        if path not in copied:
            page_images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, page_images_dir / path)
            copied.add(path)
        return f"![{alt}]({page_images_dir.name}/{path})"

    return IMAGE_REF_RE.sub(_rewrite, translated_body)


def build_page(paper: dict, translated_body: str) -> str:
    today = date.today().isoformat()
    translated_title = extract_translated_title(translated_body, paper["title"])
    nav_title = truncate_title(translated_title)
    frontmatter = (
        "---\n"
        f"title: \"{nav_title}\"\n"
        f"arxiv_id: {paper['id']}\n"
        f"category: {paper['category']}\n"
        f"published: {paper['published']}\n"
        f"translated: {today}\n"
        f"selection_reason: \"{paper['selection_reason']}\"\n"
        "---\n\n"
    )
    header = (
        f"# {translated_title}\n\n"
        f"- 원제: {paper['title']}\n"
        f"- 원문: [{paper['abs_url']}]({paper['abs_url']})\n"
        f"- PDF: [{paper['pdf_url']}]({paper['pdf_url']})\n"
        f"- arXiv ID: `{paper['id']}`\n\n"
        "---\n\n"
    )
    return frontmatter + header + translated_body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True)
    parser.add_argument("--translated", required=True)
    parser.add_argument("--source-dir", default=None, help="번역본과 같이 있던 이미지 파일들의 원본 위치")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    with open(args.selected, encoding="utf-8") as f:
        paper = json.load(f)
    translated_body = Path(args.translated).read_text(encoding="utf-8")

    category_dir = CATEGORY_DIR_NAMES.get(paper["category"], paper["category"].replace(".", "_"))
    out_dir = Path(args.docs_dir) / category_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    page_stem = f"{today}-{paper['id']}-{slugify(paper['title'])}"

    if args.source_dir:
        page_images_dir = out_dir / page_stem
        translated_body = copy_images_and_rewrite_refs(translated_body, Path(args.source_dir), page_images_dir)

    out_path = out_dir / f"{page_stem}.md"
    out_path.write_text(build_page(paper, translated_body), encoding="utf-8")

    print(f"발행 완료: {out_path}")


if __name__ == "__main__":
    main()
