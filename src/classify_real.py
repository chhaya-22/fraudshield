import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from l3_classify import classify
from l4_fusion import fuse

_TRANSCRIPT_DIR = _SRC.parent / "data" / "audio" / "transcripts"


def main() -> None:
    txt_files = sorted(_TRANSCRIPT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {_TRANSCRIPT_DIR}")
        return

    print(f"Classifying {len(txt_files)} transcript(s)...\n")

    rows = []
    for path in txt_files:
        transcript = path.read_text(encoding="utf-8")
        cl = classify(transcript)
        fu = fuse(cl)
        rows.append(
            {
                "name": path.stem[:50],
                "verdict": fu["verdict"],
                "stage": cl["detected_stage"],
                "flags": ", ".join(cl["matched_red_flags"]) or "—",
                "reason": fu["reason"],
            }
        )
        print(f"  [{path.stem[:40]}]  {fu['verdict']}")
        time.sleep(2)

    # ── Table ───────────────────────────────────────────────────────────────────
    col = {"name": 52, "verdict": 11, "stage": 7, "flags": 30}
    header = (
        f"{'File':<{col['name']}}  {'Verdict':<{col['verdict']}}"
        f"  {'Stage':<{col['stage']}}  {'Red Flags':<{col['flags']}}  Reason"
    )
    sep = "-" * (sum(col.values()) + len(col) * 2 + 10)
    print(f"\n=== Real Transcript Results ===\n{header}\n{sep}")
    for r in rows:
        print(
            f"{r['name']:<{col['name']}}  {r['verdict']:<{col['verdict']}}"
            f"  {r['stage']:<{col['stage']}}  {r['flags']:<{col['flags']}}  {r['reason']}"
        )


if __name__ == "__main__":
    main()
