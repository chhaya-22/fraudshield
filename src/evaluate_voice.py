"""
Evaluate the deepfake voice detector on labelled clips.

Expected layout:
    data/audio/eval/real/   — genuine human recordings (any format librosa reads)
    data/audio/eval/fake/   — AI-synthesised / cloned voice clips

Prints per-clip scores, AUC, and EER.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from l2_voice import synthetic_probability

_EVAL_DIR = _SRC.parent / "data" / "audio" / "eval"
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def _files_in(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in _AUDIO_EXTS)


def _eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    real_dir = _EVAL_DIR / "real"
    fake_dir = _EVAL_DIR / "fake"

    missing = [d for d in (real_dir, fake_dir) if not d.exists()]
    if missing:
        for d in missing:
            print(f"Missing: {d}")
        print(
            "\nCreate data/audio/eval/real/ and data/audio/eval/fake/ "
            "with labelled audio clips, then re-run."
        )
        return

    real_files = _files_in(real_dir)
    fake_files = _files_in(fake_dir)

    if not real_files and not fake_files:
        print("No audio files found in real/ or fake/.")
        return

    print(f"Found {len(real_files)} real, {len(fake_files)} fake clip(s).\n")
    print(f"  {'Label':<6}  {'P(fake)':<8}  File")
    print(f"  {'-'*6}  {'-'*8}  {'-'*50}")

    y_true, y_score = [], []

    for label_int, label_str, files in (
        (0, "REAL", real_files),
        (1, "FAKE", fake_files),
    ):
        for path in files:
            score = synthetic_probability(path)
            y_true.append(label_int)
            y_score.append(score)
            print(f"  {label_str:<6}  {score:.4f}    {path.name}")

    if len(np.unique(y_true)) < 2:
        print("\nNeed both real and fake clips to compute AUC / EER.")
        return

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = _eer(fpr, tpr)

    print(f"\nAUC : {auc:.4f}")
    print(f"EER : {eer:.4f}  ({eer * 100:.1f}%)")


if __name__ == "__main__":
    main()
