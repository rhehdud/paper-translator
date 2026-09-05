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
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from pypdf import PdfReader

# 후보 수집은 검색 API(/api/query)가 아니라 OAI-PMH를 쓴다. 검색 API는 sortBy=submittedDate
# 정렬 질의가 서버에 비싸서 IP 단위로 공격적으로 429를 뱉는데, GitHub Actions의 예약 실행은
# 정각 cron에 수천 개 리포와 이그레스 IP를 공유하느라 그 429를 정면으로 맞는다(실측: 2026-08-01,
# 08-08, 08-29, 09-05 네 번의 예약 실행이 전부 5개 분야 재시도를 다 소진하고 빈손으로 끝남 -
# 같은 시각 로컬에서도 동일 쿼리가 429였으므로 러너 탓이 아니라 엔드포인트 자체가 불안정한 것).
# OAI-PMH는 arXiv가 대량 조회용으로 권장하는 인터페이스라 날짜 범위가 네이티브고, 과부하 시
# 429가 아니라 503 + Retry-After로 "언제 다시 오라"를 알려준다.
ARXIV_OAI = "https://export.arxiv.org/oai2"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}


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


def _oai_set_for(category: str) -> str:
    """arXiv 카테고리 코드를 OAI-PMH setSpec으로 바꾼다: "cs.AI" -> "cs:cs:AI".

    이 set은 주 카테고리뿐 아니라 교차등재(cross-list)된 논문도 포함한다 - 예전
    검색 API의 `cat:cs.AI`와 같은 의미라 후보 풀이 달라지지 않는다(실측 확인).
    """
    archive, subject = category.split(".", 1)
    return f"{archive}:{archive}:{subject}"


def _oai_get(url: str, max_retries: int = 5, max_flow_control_waits: int = 20) -> bytes:
    """OAI-PMH 요청. 503 + Retry-After는 에러가 아니라 정상적인 흐름 제어라 별도로 센다.

    검색 API의 429는 "왜 막혔는지, 언제 풀리는지" 정보가 없어 눈먼 지수 backoff밖에
    못 했지만, OAI-PMH의 503은 서버가 대기 시간을 직접 알려준다. 그 값을 그대로
    따르면 되므로 재시도 예산과 분리해서 관리한다.
    """
    flow_waits = 0
    attempt = 0
    while True:
        req = urllib.request.Request(url, headers={"User-Agent": "paper-translator-bot/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                flow_waits += 1
                if flow_waits > max_flow_control_waits:
                    raise
                try:
                    wait = max(1, min(300, int(e.headers.get("Retry-After", "10"))))
                except ValueError:
                    wait = 10
                print(f"OAI-PMH 흐름 제어(503), {wait}초 대기 후 재개", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            wait = min(60, 2 ** (attempt + 1))
            print(f"OAI-PMH 요청 실패({type(e).__name__}: {e}), {wait}초 후 재시도", file=sys.stderr)
            time.sleep(wait)


def _oai_check_error(root: ET.Element) -> str | None:
    """OAI-PMH는 오류도 HTTP 200으로 주고 본문에 <error code=...>를 담는다."""
    err = root.find("oai:error", OAI_NS)
    if err is None:
        return None
    return err.get("code") or "unknown"


def _oai_text(meta: ET.Element, tag: str) -> str:
    el = meta.find(f"arxiv:{tag}", OAI_NS)
    if el is None or not el.text:
        return ""
    return " ".join(el.text.split())


def fetch_arxiv_candidates(
    category: str, window_days: int, reference_dt: datetime.datetime | None = None
) -> list[dict]:
    """OAI-PMH로 최근 window_days 안에 "최초 제출"된 이 분야 논문을 전부 모은다.

    반환 리스트는 최신순(제출일 내림차순)이다 - select_paper가 HF 업보트에 걸리는
    후보가 없을 때 pool[0]을 "가장 최신"으로 보고 폴백하기 때문에 이 순서가 계약이다.

    reference_dt를 지정하면(디버깅용 재현 실행) 그 날짜를 "오늘"로 놓는다.
    """
    reference_dt = reference_dt or datetime.datetime.now(datetime.timezone.utc)
    reference_date = reference_dt.date()
    cutoff_date = reference_date - datetime.timedelta(days=window_days)

    params = urllib.parse.urlencode(
        {
            "verb": "ListRecords",
            "set": _oai_set_for(category),
            "metadataPrefix": "arXiv",
            "from": cutoff_date.isoformat(),
            "until": reference_date.isoformat(),
        }
    )
    url = f"{ARXIV_OAI}?{params}"

    candidates: list[dict] = []
    seen: set[str] = set()
    pages = 0
    while url:
        root = ET.fromstring(_oai_get(url))
        code = _oai_check_error(root)
        if code == "noRecordsMatch":
            break
        if code:
            raise RuntimeError(f"OAI-PMH 오류 응답: {code}")
        pages += 1

        for rec in root.findall("oai:ListRecords/oai:record", OAI_NS):
            header = rec.find("oai:header", OAI_NS)
            if header is not None and header.get("status") == "deleted":
                continue
            meta = rec.find("oai:metadata/arxiv:arXiv", OAI_NS)
            if meta is None:
                continue

            arxiv_id = _oai_text(meta, "id")
            created = _oai_text(meta, "created")
            if not arxiv_id or not created:
                continue
            try:
                created_date = datetime.date.fromisoformat(created)
            except ValueError:
                continue

            # from/until은 "수정일(datestamp)" 기준이라 옛 논문의 개정판이 같이 딸려온다.
            # 최초 제출일(created)로 다시 걸러야 "최근 N일 신규 제출"이 된다
            # (실측: 하루치 1067편 중 69편(6%)이 옛 논문 개정판이었다).
            if created_date < cutoff_date or created_date > reference_date:
                continue
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)

            candidates.append(
                {
                    "id": arxiv_id,
                    "title": _oai_text(meta, "title"),
                    "summary": _oai_text(meta, "abstract"),
                    "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"{ARXIV_PDF_BASE}/{arxiv_id}",
                    "published": created_date.isoformat(),
                }
            )

        token = root.find("oai:ListRecords/oai:resumptionToken", OAI_NS)
        token_text = (token.text or "").strip() if token is not None else ""
        if not token_text:
            break
        # resumptionToken을 넘길 때는 다른 인자를 같이 보내면 안 된다(OAI-PMH 규약).
        url = f"{ARXIV_OAI}?verb=ListRecords&resumptionToken={urllib.parse.quote(token_text)}"
        time.sleep(1)

    # created가 날짜 단위라 같은 날 제출분은 순서가 갈리지 않는다. arXiv ID가 제출 순서대로
    # 커지므로 2차 정렬 키로 써서 하루 안에서도 최신이 앞에 오게 한다.
    candidates.sort(key=lambda c: (c["published"], c["id"]), reverse=True)
    print(f"[{category}] OAI-PMH {pages}페이지에서 후보 {len(candidates)}편 수집", file=sys.stderr)
    return candidates


def fetch_hf_upvotes(window_days: int, reference_date: datetime.date | None = None) -> dict[str, int]:
    upvotes: dict[str, int] = {}
    reference_date = reference_date or datetime.date.today()
    for offset in range(window_days + 1):
        date_str = (reference_date - datetime.timedelta(days=offset)).isoformat()
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


def select_all_categories(config: dict, reference_date: datetime.date | None = None) -> dict[str, dict]:
    """분야별로 1편씩, 이미 다른 분야/지난주에 뽑힌 논문은 제외하고 선정한다.

    reference_date를 지정하면 그 날짜를 "오늘"로 놓고 후보 조회·업보트 조회를 전부
    그 시점 기준으로 재현한다(디버깅/검증 목적 - 과거 특정 실행과 같은 선정 결과를
    다시 얻고 싶을 때. 프로덕션 주간 실행은 항상 생략해서 실제 오늘 기준으로 돈다)."""
    window_days = config["candidate_window_days"]
    reference_dt = (
        datetime.datetime.combine(reference_date, datetime.time(23, 59, 59), tzinfo=datetime.timezone.utc)
        if reference_date
        else None
    )
    upvotes = fetch_hf_upvotes(window_days, reference_date)

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
            candidates = fetch_arxiv_candidates(code, window_days, reference_dt)
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
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="YYYY-MM-DD. 지정하면 이 날짜를 '오늘'로 놓고 과거 선정 결과를 재현한다 (디버깅용, 생략하면 실제 오늘)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reference_date = datetime.date.fromisoformat(args.as_of_date) if args.as_of_date else None
    results = select_all_categories(config, reference_date)

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    for code, selected in results.items():
        safe_name = code.replace(".", "_")
        with open(f"{args.out_dir}/selected_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
