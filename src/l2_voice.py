"""
L2 Voice: deepfake voice detection via a fine-tuned wav2vec2 classifier.

synthetic_probability(audio_path) -> float in [0, 1]
    Returns mean P(AI-generated) across 6-second windows. Avoids OOM on long calls.

window_scores(audio_path) -> list[float]
    Returns per-window P(fake) so callers can inspect variation within a call.

Model is loaded once and cached for the lifetime of the process.
Audio is resampled to 16 kHz mono before inference.
"""

import sys
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from transformers import pipeline

_TARGET_SR = 16_000
_PRIMARY_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
_FALLBACK_MODEL = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

# Label keywords that signal the "fake / synthetic" class
_FAKE_KW = {"fake", "synthetic", "spoof", "deepfake", "artificial", "generated", "ai"}
# Label keywords that signal the "real / genuine" class (P_fake = 1 - score)
_REAL_KW = {"real", "genuine", "natural", "human", "bonafide", "bona"}


@lru_cache(maxsize=1)
def _get_pipe():
    """Load the classifier once; prefer primary, fall back to secondary model."""
    for model_id in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        try:
            pipe = pipeline("audio-classification", model=model_id)
            print(f"[l2_voice] Loaded: {model_id}", file=sys.stderr)
            return pipe, model_id
        except Exception as exc:
            print(f"[l2_voice] Could not load {model_id}: {exc}", file=sys.stderr)
    raise RuntimeError(
        "Neither deepfake voice model could be loaded. "
        "Check your network connection or HuggingFace model availability."
    )


def _fake_score(predictions: list[dict]) -> float:
    """Extract P(fake) from a [{label, score}] list with unknown label names."""
    # Direct fake-keyword match
    for p in predictions:
        if any(kw in p["label"].lower() for kw in _FAKE_KW):
            return float(p["score"])
    # Invert a real-keyword match
    for p in predictions:
        if any(kw in p["label"].lower() for kw in _REAL_KW):
            return 1.0 - float(p["score"])
    # Neither label is recognisable — treat highest-scoring label as proxy
    return float(max(predictions, key=lambda x: x["score"])["score"])


_WINDOW_SAMPLES = 6 * _TARGET_SR   # 6 s × 16 000 Hz = 96 000 samples
_MIN_SAMPLES = 1 * _TARGET_SR      # skip windows shorter than ~1 s


def window_scores(audio_path: str | Path) -> list[float]:
    """Return per-window P(fake) for every valid 6-second chunk of the file."""
    waveform, _ = librosa.load(audio_path, sr=_TARGET_SR, mono=True)
    pipe, _ = _get_pipe()

    scores = []
    for start in range(0, len(waveform), _WINDOW_SAMPLES):
        chunk = waveform[start : start + _WINDOW_SAMPLES]
        if len(chunk) < _MIN_SAMPLES:
            continue
        preds = pipe({"array": chunk, "sampling_rate": _TARGET_SR}, top_k=None)
        scores.append(_fake_score(preds))

    return scores


def synthetic_probability(audio_path: str | Path) -> float:
    """Return mean P(AI-generated) across 6-second windows. Returns 0.0 if no valid windows."""
    scores = window_scores(audio_path)
    if not scores:
        print(f"[l2_voice] Warning: no valid windows in {audio_path}; returning 0.0", file=sys.stderr)
        return 0.0
    return float(np.mean(scores))
