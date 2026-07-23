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


def build_message(papers: list[dict], site_url: str, warning: str | None = None) -> dict:
    lines = []
    if warning:
        lines.append(warning)
    lines.append(f"**이번 주 번역된 논문 {len(papers)}편**")
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
    parser.add_argument("--generated-at-file", default=None, help="처리 워크플로가 남긴 타임스탬프 파일 (신선도 확인용)")
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise SystemExit("환경변수 DISCORD_WEBHOOK_URL이 설정되어 있지 않습니다")

    warning = check_staleness(args.generated_at_file)

    papers = []
    for path in args.selected:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                papers.append(json.load(f))

    payload = build_message(papers, args.site_url, warning)
    send_discord(webhook_url, payload)
    print(f"Discord 알림 전송 완료 ({len(papers)}편)", file=sys.stderr)


if __name__ == "__main__":
    main()
