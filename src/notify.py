"""이번 주 발행된 논문 목록을 Discord 웹훅으로 알린다.

사용법:
    DISCORD_WEBHOOK_URL=... python src/notify.py --selected selected_cs_CL.json selected_cs_CV.json ... --site-url https://user.github.io/paper-translator
"""
import argparse
import json
import os
import sys
import urllib.request


def build_message(papers: list[dict], site_url: str) -> dict:
    lines = [f"**이번 주 번역된 논문 {len(papers)}편**"]
    for p in papers:
        lines.append(f"- [{p['category']}] {p['title']} — {site_url.rstrip('/')} (arXiv: {p['id']})")
    return {"content": "\n".join(lines)}


def send_discord(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", nargs="+", required=True, help="발행된 논문들의 선정 JSON 파일 목록")
    parser.add_argument("--site-url", required=True)
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise SystemExit("환경변수 DISCORD_WEBHOOK_URL이 설정되어 있지 않습니다")

    papers = []
    for path in args.selected:
        with open(path, encoding="utf-8") as f:
            papers.append(json.load(f))

    payload = build_message(papers, args.site_url)
    send_discord(webhook_url, payload)
    print(f"Discord 알림 전송 완료 ({len(papers)}편)", file=sys.stderr)


if __name__ == "__main__":
    main()
