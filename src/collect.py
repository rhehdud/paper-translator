"""분야별 후보 논문을 arXiv에서 모으고, HF Daily Papers 업보트로 1편을 선별한다.

사용법:
    python src/collect.py --category cs.CL --out selected.json
"""
import argparse
import datetime
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from pypdf import PdfReader

ARXIV_API = "https://export.arxiv.org/api/query"
HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "paper-translator-bot/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


_last_arxiv_request_time = 0.0


def _arxiv_get(url: str, max_retries: int = 8) -> bytes:
    """arXiv는 요청 사이 최소 3초 간격을 권장한다. 429든 타임아웃이든 실패하면 backoff 후 재시도한다."""
    global _last_arxiv_request_time
    for attempt in range(max_retries):
        elapsed = time.monotonic() - _last_arxiv_request_time
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        try:
            data = _http_get(url)
            _last_arxiv_request_time = time.monotonic()
            return data
        except Exception as e:
            _last_arxiv_request_time = time.monotonic()
            if attempt == max_retries - 1:
                raise
            wait = min(60, 2 ** (attempt + 2))
            print(f"arXiv 요청 실패({type(e).__name__}: {e}), {wait}초 후 재시도: {url}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("arXiv 요청 재시도 한도를 초과했습니다")


def fetch_arxiv_candidates(category: str, window_days: int, page_size: int) -> list[dict]:
    """window_days 전체를 커버할 때까지 최신순으로 페이지를 계속 넘겨서 후보를 모은다.

    예전엔 max_results=pool_size(=20)로 딱 한 번만 조회했는데, cs.CL/cs.AI처럼
    투고량이 많은 카테고리는 몇 시간 안에 20편이 다 채워져서 candidate_window_days가
    사실상 무시됐다(실측: 상위 20편이 하루도 안 되는 기간에서 나옴). 결과가 이미
    submittedDate 내림차순이므로, cutoff보다 오래된 항목을 만나는 순간 이후 페이지는
    전부 더 오래된 것들뿐이라 그 즉시 멈출 수 있다.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=window_days)
    candidates = []
    start = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "search_query": f"cat:{category}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": page_size,
            }
        )
        xml_bytes = _arxiv_get(f"{ARXIV_API}?{query}")
        root = ET.fromstring(xml_bytes)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break

        reached_cutoff = False
        for entry in entries:
            arxiv_url = entry.find("atom:id", ATOM_NS).text.strip()
            arxiv_id = re.sub(r"v\d+$", "", arxiv_url.rsplit("/", 1)[-1])
            published = datetime.datetime.fromisoformat(
                entry.find("atom:published", ATOM_NS).text.replace("Z", "+00:00")
            )
            if published < cutoff:
                reached_cutoff = True
                break
            title = " ".join(entry.find("atom:title", ATOM_NS).text.split())
            summary = " ".join(entry.find("atom:summary", ATOM_NS).text.split())
            pdf_url = None
            for link in entry.findall("atom:link", ATOM_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
            candidates.append(
                {
                    "id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                    "published": published.isoformat(),
                }
            )

        if reached_cutoff or len(entries) < page_size:
            break
        start += page_size

    return candidates


def fetch_hf_upvotes(window_days: int) -> dict[str, int]:
    upvotes: dict[str, int] = {}
    today = datetime.date.today()
    for offset in range(window_days + 1):
        date_str = (today - datetime.timedelta(days=offset)).isoformat()
        try:
            raw = _http_get(f"{HF_DAILY_PAPERS_API}?date={date_str}")
        except Exception:
            continue
        for item in json.loads(raw):
            paper = item.get("paper", {})
            pid = paper.get("id")
            if not pid:
                continue
            upvotes[pid] = max(upvotes.get(pid, 0), paper.get("upvotes", 0) or 0)
        time.sleep(0.2)  # HF에 과도한 요청 방지
    return upvotes


def get_page_count(pdf_url: str) -> int:
    data = _arxiv_get(pdf_url)
    return len(PdfReader(io.BytesIO(data)).pages)


def select_paper(
    candidates: list[dict], upvotes: dict[str, int], exclude_ids: set[str], max_pages: int
) -> dict | None:
    """제외 목록을 뺀 후보 중 1등을 고르되, 페이지 상한을 넘으면 다음 후보로 넘어간다."""
    pool = [c for c in candidates if c["id"] not in exclude_ids]

    while pool:
        matched = [c for c in pool if c["id"] in upvotes]
        if matched:
            best = max(matched, key=lambda c: upvotes[c["id"]])
            reason = f"hf_daily_papers_upvotes={upvotes[best['id']]}"
        else:
            # HF Daily Papers에 없으면 최신 제출 논문으로 폴백 (candidates는 이미 최신순 정렬)
            best = pool[0]
            reason = "fallback_most_recent"

        try:
            num_pages = get_page_count(best["pdf_url"])
        except Exception as e:
            print(f"페이지 수 확인 실패, 다음 후보로: {best['id']} ({e})", file=sys.stderr)
            pool = [c for c in pool if c["id"] != best["id"]]
            continue

        if num_pages > max_pages:
            print(
                f"건너뜀(선정 단계): {best['id']}은 {num_pages}페이지로 상한({max_pages}) 초과, 다음 후보로",
                file=sys.stderr,
            )
            pool = [c for c in pool if c["id"] != best["id"]]
            continue

        best["selection_reason"] = reason
        best["num_pages"] = num_pages
        return best

    return None


def load_published_ids(docs_dir: str) -> set[str]:
    """지난주까지 이미 발행된 논문의 arXiv ID를 docs/ 프런트매터에서 읽어온다 (주 간 중복 방지)."""
    published: set[str] = set()
    root = Path(docs_dir)
    if not root.exists():
        return published
    for md_file in root.rglob("*.md"):
        with open(md_file, encoding="utf-8") as f:
            frontmatter_dashes = 0
            for line in f:
                line = line.strip()
                if line == "---":
                    frontmatter_dashes += 1
                    if frontmatter_dashes >= 2:
                        break
                    continue
                if line.startswith("arxiv_id:"):
                    published.add(line.split(":", 1)[1].strip())
                    break
    return published


def select_all_categories(config: dict) -> dict[str, dict]:
    """분야별로 1편씩, 이미 다른 분야/지난주에 뽑힌 논문은 제외하고 선정한다."""
    window_days = config["candidate_window_days"]
    page_size = config["candidate_page_size"]
    upvotes = fetch_hf_upvotes(window_days)

    docs_dir = config.get("publish", {}).get("docs_dir", "docs")
    chosen_ids: set[str] = load_published_ids(docs_dir)
    print(f"기존 발행 논문 {len(chosen_ids)}편을 제외 대상으로 로드", file=sys.stderr)

    max_pages = config["extraction"]["max_pages"]

    results: dict[str, dict] = {}
    for cat in config["categories"]:
        code = cat["code"]
        # arXiv 쪽이 일시적으로 불안정해 이 분야 요청이 재시도까지 다 소진해도, 다른
        # 분야까지 통째로 날리지 않고 이 분야만 건너뛰고 계속 진행한다 (translate.yml의
        # 분야별 격리와 같은 원칙).
        try:
            candidates = fetch_arxiv_candidates(code, window_days, page_size)
        except Exception as e:
            print(f"[{code}] arXiv 후보 조회 실패(재시도 소진), 이번 주는 건너뜀: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        selected = select_paper(candidates, upvotes, chosen_ids, max_pages)
        if selected is None:
            print(f"[{code}] 후보 없음 (전부 중복/페이지 초과이거나 기간 내 제출이 없음)", file=sys.stderr)
            continue
        selected["category"] = code
        chosen_ids.add(selected["id"])
        results[code] = selected
        print(f"[{code}] 선정: {selected['title']} ({selected['selection_reason']})", file=sys.stderr)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", required=True, help="분야별 선정 결과를 JSON으로 저장할 디렉터리")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results = select_all_categories(config)

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    for code, selected in results.items():
        safe_name = code.replace(".", "_")
        with open(f"{args.out_dir}/selected_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
