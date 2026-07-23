"""번역된 마크다운과 논문 메타데이터를 MkDocs 문서 페이지로 합친다.

사용법:
    python src/publish.py --selected selected_cs_CL.json --translated paper.ko.md --docs-dir docs
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

CATEGORY_DIR_NAMES = {
    "cs.CL": "nlp",
    "cs.CV": "vision",
    "cs.LG": "ml",
    "cs.AI": "ai",
    "cs.RO": "robotics",
}


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len].rstrip("-")


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
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    with open(args.selected, encoding="utf-8") as f:
        paper = json.load(f)
    translated_body = Path(args.translated).read_text(encoding="utf-8")

    category_dir = CATEGORY_DIR_NAMES.get(paper["category"], paper["category"].replace(".", "_"))
    out_dir = Path(args.docs_dir) / category_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    filename = f"{today}-{paper['id']}-{slugify(paper['title'])}.md"
    out_path = out_dir / filename
    out_path.write_text(build_page(paper, translated_body), encoding="utf-8")

    print(f"발행 완료: {out_path}")


if __name__ == "__main__":
    main()
