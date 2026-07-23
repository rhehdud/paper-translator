"""분야별 후보 논문을 arXiv에서 모으고, HF Daily Papers 업보트로 1편을 선별한다.

사용법:
    python src/collect.py --category cs.CL --out selected.json
"""
import argparse
import datetime
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import yaml

ARXIV_API = "https://export.arxiv.org/api/query"
HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "paper-translator-bot/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_arxiv_candidates(category: str, window_days: int, pool_size: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": pool_size,
        }
    )
    xml_bytes = _http_get(f"{ARXIV_API}?{query}")
    root = ET.fromstring(xml_bytes)

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=window_days)
    candidates = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_url = entry.find("atom:id", ATOM_NS).text.strip()
        arxiv_id = re.sub(r"v\d+$", "", arxiv_url.rsplit("/", 1)[-1])
        published = datetime.datetime.fromisoformat(
            entry.find("atom:published", ATOM_NS).text.replace("Z", "+00:00")
        )
        if published < cutoff:
            continue
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


def select_paper(candidates: list[dict], upvotes: dict[str, int], exclude_ids: set[str]) -> dict | None:
    pool = [c for c in candidates if c["id"] not in exclude_ids]
    if not pool:
        return None

    matched = [c for c in pool if c["id"] in upvotes]
    if matched:
        best = max(matched, key=lambda c: upvotes[c["id"]])
        best["selection_reason"] = f"hf_daily_papers_upvotes={upvotes[best['id']]}"
        return best

    # HF Daily Papers에 없으면 최신 제출 논문으로 폴백 (candidates는 이미 최신순 정렬)
    best = pool[0]
    best["selection_reason"] = "fallback_most_recent"
    return best


def select_all_categories(config: dict) -> dict[str, dict]:
    """분야별로 1편씩, 이미 다른 분야에서 뽑힌 논문(cross-list 중복)은 제외하고 선정한다."""
    window_days = config["candidate_window_days"]
    pool_size = config["candidate_pool_size"]
    upvotes = fetch_hf_upvotes(window_days)

    results: dict[str, dict] = {}
    chosen_ids: set[str] = set()
    for cat in config["categories"]:
        code = cat["code"]
        candidates = fetch_arxiv_candidates(code, window_days, pool_size)
        selected = select_paper(candidates, upvotes, chosen_ids)
        if selected is None:
            print(f"[{code}] 후보 없음 (전부 다른 분야와 중복되었거나 기간 내 제출이 없음)", file=sys.stderr)
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
