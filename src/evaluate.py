import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from l3_classify import classify
from l4_fusion import fuse

_DATA_DIR = _SRC.parent / "data"
_SCAM_POSITIVE = {"HIGH-RISK", "CAUTION"}


def _pct(n: int, d: int) -> str:
    return f"{n/d:.2f}" if d else "N/A"


def main() -> None:
    with open(_DATA_DIR / "eval_transcripts.json") as f:
        data = json.load(f)

    transcripts = data["transcripts"]
    results = []

    print("Running classifier...\n")
    for item in transcripts:
        cl = classify(item["transcript"])
        fu = fuse(cl)
        results.append(
            {
                "id": item["id"],
                "is_scam": item["is_scam"],
                "max_stage": item["max_stage"],
                "pred_stage": cl["detected_stage"],
                "pred_flags": cl["matched_red_flags"],
                "verdict": fu["verdict"],
                "reason": fu["reason"],
            }
        )
        verdict_tag = fu["verdict"]
        print(f"  [{item['id']}] {verdict_tag}")
        time.sleep(2)

    # ── Binary metrics ──────────────────────────────────────────────────────────
    tp = sum(1 for r in results if r["is_scam"] and r["verdict"] in _SCAM_POSITIVE)
    fp = sum(1 for r in results if not r["is_scam"] and r["verdict"] in _SCAM_POSITIVE)
    fn = sum(1 for r in results if r["is_scam"] and r["verdict"] not in _SCAM_POSITIVE)
    tn = sum(1 for r in results if not r["is_scam"] and r["verdict"] not in _SCAM_POSITIVE)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    benign = [r for r in results if not r["is_scam"]]
    fp_rate = fp / len(benign) if benign else 0.0

    scam_cases = [r for r in results if r["is_scam"]]
    stage_exact = sum(1 for r in scam_cases if r["pred_stage"] == r["max_stage"])
    stage_acc = stage_exact / len(scam_cases) if scam_cases else 0.0

    print("\n=== Binary Classification Metrics ===")
    print(f"  Precision : {precision:.2f}  (TP={tp}, FP={fp})")
    print(f"  Recall    : {recall:.2f}  (TP={tp}, FN={fn})")
    print(f"  F1        : {f1:.2f}")
    print(f"  FP Rate   : {fp_rate:.2f}  ({fp}/{len(benign)} benign cases flagged as scam)")
    print(f"  Stage Acc : {stage_acc:.2f}  ({stage_exact}/{len(scam_cases)} exact stage matches on scam cases)")

    # ── Per-case table ──────────────────────────────────────────────────────────
    col = {"id": 12, "truth": 7, "verdict": 11, "ts": 6, "ps": 6}
    header = (
        f"{'ID':<{col['id']}}  {'Truth':<{col['truth']}}  {'Verdict':<{col['verdict']}}"
        f"  {'T-Stg':<{col['ts']}}  {'P-Stg':<{col['ps']}}  Reason"
    )
    print(f"\n=== Per-Case Results ===\n{header}\n{'-' * 90}")
    for r in results:
        truth = "SCAM" if r["is_scam"] else "BENIGN"
        ts = str(r["max_stage"]) if r["max_stage"] is not None else "—"
        print(
            f"{r['id']:<{col['id']}}  {truth:<{col['truth']}}  {r['verdict']:<{col['verdict']}}"
            f"  {ts:<{col['ts']}}  {r['pred_stage']:<{col['ps']}}  {r['reason']}"
        )


if __name__ == "__main__":
    main()
