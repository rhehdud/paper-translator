"""청크 단위 JSONL 로그(RunLogger가 남긴 것)에서 검증을 통과한 호출만 골라
번역 캐시(TranslationCache)를 시딩한다. 하네스 코드(후처리·검증 등)가 그 사이
바뀌었어도, 검증 통과분의 원본 응답은 그대로 재사용 가능하다 - 캐시 경계가
_call_model 자리에 그어져 있어서 후처리·검증은 재생 때 항상 새로 돈다.

사용법:
    python src/seed_cache.py logs/translate_sample_....jsonl --cache-db cache/translate_cache.db
    python src/seed_cache.py logs/translate_sample_....jsonl --dry-run   # 시딩 없이 미스 개수만 확인
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from translate import TranslationCache  # noqa: E402

_SEEDABLE_KINDS = ("batch", "single")


def _is_seedable(reason: str | None) -> bool:
    """None(완전 성공)이거나 partial_bad:N(배치 응답 자체는 채택, 일부 항목만 개별
    재번역)이면 그 raw_response가 실제로 최종 결과에 쓰인 응답이다. 그 외 실패
    사유는 검증에서 이미 버려진 응답이라, 심으면 캐시의 "검증 통과분만 write"
    자가 치유 설계를 어기게 된다."""
    return reason is None or (reason or "").startswith("partial_bad:")


def seed_from_log(log_path: Path, cache: TranslationCache) -> tuple[int, int]:
    seeded = 0
    skipped = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("call_kind") not in _SEEDABLE_KINDS:
                continue
            if not _is_seedable(rec.get("validation_fail_reason")):
                skipped += 1
                continue
            key = cache.make_key(rec["input"], rec["prompt_hash"], rec["model"], rec["temperature"])
            cache.seed(key, rec["raw_response"], rec["finish_reason"], rec.get("git_sha"))
            seeded += 1
    return seeded, skipped


def count_unseedable(log_path: Path) -> list[str]:
    """캐시로 시딩되지 않는(=재생 시 실제 API 호출이 필요한) 고유 입력 목록을
    돌려준다 - 같은 input 텍스트에 대한 모든 attempt 중 단 하나도 검증을 통과한
    적이 없는 경우다."""
    by_input: dict[str, list[str | None]] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("call_kind") not in _SEEDABLE_KINDS:
                continue
            by_input.setdefault(rec["input"], []).append(rec.get("validation_fail_reason"))
    return [text for text, reasons in by_input.items() if not any(_is_seedable(r) for r in reasons)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--cache-db", default=None, help="시딩할 캐시 DB 경로 (--dry-run이면 생략 가능)")
    parser.add_argument("--dry-run", action="store_true", help="심지 않고 미스 개수만 보고한다")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    unseedable = count_unseedable(log_path)
    print(f"재생 시 실제 API 호출이 필요한 고유 입력: {len(unseedable)}개", file=sys.stderr)
    for u in unseedable:
        preview = u[:70].replace("\n", " ")
        print(f"  - {len(u)}자: {preview!r}...", file=sys.stderr)

    if args.dry_run:
        return

    if not args.cache_db:
        raise SystemExit("--cache-db가 필요합니다 (--dry-run이 아니면)")
    cache = TranslationCache(Path(args.cache_db))
    seeded, skipped = seed_from_log(log_path, cache)
    cache.close()
    print(f"시딩 완료: {seeded}개 캐시 항목 기록, {skipped}개 실패 레코드는 제외", file=sys.stderr)


if __name__ == "__main__":
    main()
