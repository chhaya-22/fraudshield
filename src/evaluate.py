import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from l3_classify import classify as llm_classify
from l4_fusion import fuse
import baseline_keyword

_DEFAULT_EVAL = _SRC.parent / "data" / "eval_transcripts.json"
_SCAM_POSITIVE = {"HIGH-RISK", "CAUTION"}


def _run_llm(transcript: str) -> dict:
    cl = llm_classify(transcript)
    fu = fuse(cl)
    return {
        "pred_stage": cl["detected_stage"],
        "verdict": fu["verdict"],
        "reason": fu["reason"],
    }


def _run_baseline(transcript: str) -> dict:
    result = baseline_keyword.classify(transcript)
    return {
        "pred_stage": None,
        "verdict": result["verdict"],
        "reason": result["reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the digital-arrest scam classifier.")
    parser.add_argument(
        "eval_file",
        nargs="?",
        default=str(_DEFAULT_EVAL),
        help="Path to eval transcripts JSON (default: data/eval_transcripts.json)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use the keyword baseline instead of the LLM classifier",
    )
    args = parser.parse_args()

    classify_fn = _run_baseline if args.baseline else _run_llm
    mode_label = "keyword baseline" if args.baseline else "LLM (L3+L4)"

    with open(args.eval_file) as f:
        data = json.load(f)

    transcripts = data["transcripts"]
    results = []

    print(f"Running {mode_label} on {args.eval_file}...\n")
    for item in transcripts:
        out = classify_fn(item["transcript"])
        results.append(
            {
                "id": item["id"],
                "is_scam": item["is_scam"],
                "max_stage": item["max_stage"],
                "pred_stage": out["pred_stage"],
                "verdict": out["verdict"],
                "reason": out["reason"],
            }
        )
        print(f"  [{item['id']}] {out['verdict']}")
        if not args.baseline:
            time.sleep(2)

    # ── Binary metrics ──────────────────────────────────────────────────────────
    tp = sum(1 for r in results if r["is_scam"] and r["verdict"] in _SCAM_POSITIVE)
    fp = sum(1 for r in results if not r["is_scam"] and r["verdict"] in _SCAM_POSITIVE)
    fn = sum(1 for r in results if r["is_scam"] and r["verdict"] not in _SCAM_POSITIVE)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    benign = [r for r in results if not r["is_scam"]]
    fp_rate = fp / len(benign) if benign else 0.0

    scam_cases = [r for r in results if r["is_scam"]]
    if args.baseline:
        stage_acc_str = "N/A (baseline has no stage detection)"
    else:
        stage_exact = sum(1 for r in scam_cases if r["pred_stage"] == r["max_stage"])
        stage_acc_str = f"{stage_exact/len(scam_cases):.2f}  ({stage_exact}/{len(scam_cases)} exact matches)"

    print(f"\n=== Binary Classification Metrics  [{mode_label}] ===")
    print(f"  Precision : {precision:.2f}  (TP={tp}, FP={fp})")
    print(f"  Recall    : {recall:.2f}  (TP={tp}, FN={fn})")
    print(f"  F1        : {f1:.2f}")
    print(f"  FP Rate   : {fp_rate:.2f}  ({fp}/{len(benign)} benign cases flagged as scam)")
    print(f"  Stage Acc : {stage_acc_str}")

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
        ps = str(r["pred_stage"]) if r["pred_stage"] is not None else "—"
        print(
            f"{r['id']:<{col['id']}}  {truth:<{col['truth']}}  {r['verdict']:<{col['verdict']}}"
            f"  {ts:<{col['ts']}}  {ps:<{col['ps']}}  {r['reason']}"
        )


if __name__ == "__main__":
    main()
