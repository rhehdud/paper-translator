"""이번 주 발행된 논문 목록을 Discord 웹훅으로 알린다.

사용법:
    DISCORD_WEBHOOK_URL=... python src/notify.py --selected selected_cs_CL.json selected_cs_CV.json ... --site-url https://user.github.io/paper-translator
"""
import argparse
import datetime
import json
import os
import sys
import urllib.request

from publish import CATEGORY_DIR_NAMES, slugify


def paper_url(paper: dict, site_url: str, published_date: str | None) -> str:
    """publish.py가 실제로 만든 페이지 경로(카테고리 폴더 + 발행일-arxivID-슬러그)를
    그대로 재현해, 홈이 아니라 그 논문 번역 페이지로 바로 연결되는 링크를 만든다.
    발행일을 모르면(예: last_run 정보 없음) 안전하게 사이트 홈으로 대체한다."""
    base = site_url.rstrip("/")
    if not published_date:
        return base
    category_dir = CATEGORY_DIR_NAMES.get(paper["category"], paper["category"].replace(".", "_"))
    page_stem = f"{published_date}-{paper['id']}-{slugify(paper['title'])}"
    return f"{base}/{category_dir}/{page_stem}/"


def check_staleness(generated_at_file: str | None, max_age_hours: int = 48) -> str | None:
    """처리 워크플로가 제때 못 끝났는지 확인한다. 문제가 있으면 경고 문구를, 없으면 None을 반환."""
    if not generated_at_file or not os.path.exists(generated_at_file):
        return "⚠️ 이번 주 처리 결과를 찾지 못했습니다 (처리 워크플로가 아직 안 끝났을 수 있습니다)."

    with open(generated_at_file, encoding="utf-8") as f:
        generated_at = datetime.datetime.fromisoformat(f.read().strip().replace("Z", "+00:00"))
    age = datetime.datetime.now(datetime.timezone.utc) - generated_at
    if age > datetime.timedelta(hours=max_age_hours):
        return f"⚠️ 마지막 처리가 {age.days}일 {age.seconds // 3600}시간 전이라 오래됐습니다. 이번 주 처리가 지연되었을 수 있습니다."
    return None


def build_message(
    papers: list[dict], site_url: str, warning: str | None = None, published_date: str | None = None
) -> dict:
    lines = []
    if warning:
        lines.append(warning)
    lines.append(f"**이번 주 번역된 논문 {len(papers)}편**")
    for p in papers:
        lines.append(f"- [{p['category']}] {p['title']} — {paper_url(p, site_url, published_date)} (arXiv: {p['id']})")
    return {"content": "\n".join(lines)}


def send_discord(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        # 기본 User-Agent("Python-urllib/...")는 Discord 앞단 Cloudflare가 봇으로 보고
        # 403으로 막는 경우가 많아 일반적인 값으로 바꿔준다.
        headers={"Content-Type": "application/json", "User-Agent": "paper-translator-bot/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def _read_published_date(generated_at_file: str | None) -> str | None:
    """publish.py가 페이지 파일명에 쓴 발행일(YYYY-MM-DD)을, 같은 시점에 기록된
    generated_at.txt 타임스탬프에서 읽어온다. 논문마다 개별 저장은 안 해서, 알림이
    번역이 끝난 당일에 바로 전송된다는 전제(지금 파이프라인 구조)로 이 값을 재사용한다."""
    if not generated_at_file or not os.path.exists(generated_at_file):
        return None
    with open(generated_at_file, encoding="utf-8") as f:
        raw = f.read().strip()
    return raw.split("T", 1)[0] if raw else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", nargs="+", required=True, help="발행된 논문들의 선정 JSON 파일 목록")
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--generated-at-file", default=None, help="처리 워크플로가 남긴 타임스탬프 파일 (신선도 확인·발행일 계산용)")
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise SystemExit("환경변수 DISCORD_WEBHOOK_URL이 설정되어 있지 않습니다")

    warning = check_staleness(args.generated_at_file)
    published_date = _read_published_date(args.generated_at_file)

    papers = []
    for path in args.selected:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                papers.append(json.load(f))

    payload = build_message(papers, args.site_url, warning, published_date)
    send_discord(webhook_url, payload)
    print(f"Discord 알림 전송 완료 ({len(papers)}편)", file=sys.stderr)


if __name__ == "__main__":
    main()
