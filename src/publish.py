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


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len].rstrip("-")


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
    frontmatter = (
        "---\n"
        f"title: \"{paper['title']}\"\n"
        f"arxiv_id: {paper['id']}\n"
        f"category: {paper['category']}\n"
        f"published: {paper['published']}\n"
        f"translated: {today}\n"
        f"selection_reason: \"{paper['selection_reason']}\"\n"
        "---\n\n"
    )
    header = (
        f"# {paper['title']}\n\n"
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
