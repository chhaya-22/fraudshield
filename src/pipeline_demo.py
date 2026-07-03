"""
Full multimodal pipeline demo: audio file → verdict.

    python src/pipeline_demo.py <audio_file>

Steps:
    L1  ASR          audio → transcript   (Groq Whisper)
    L3  Classify     transcript → stage + red flags   (Groq LLM)
    L2  Voice check  audio → P(AI-generated)   (wav2vec2)
    L4  Fuse         (text verdict + voice score) → final verdict
"""

import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from l1_asr import transcribe
from l3_classify import classify
from l2_voice import synthetic_probability
from l4_fusion import fuse


def _divider(char: str = "─", width: int = 64) -> None:
    print(char * width)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pipeline_demo.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    _divider("=")
    print(f"  FraudShield — multimodal pipeline")
    print(f"  Input: {audio_path.name}")
    _divider("=")

    # ── L1: Transcribe ─────────────────────────────────────────────────────────
    print("\n[L1] Transcribing audio (Groq Whisper)...")
    t0 = time.time()
    transcript = transcribe(audio_path)
    print(f"     Done in {time.time()-t0:.1f}s  ({len(transcript.split())} words)")
    _divider()
    print(transcript[:500] + ("…" if len(transcript) > 500 else ""))
    _divider()

    # ── L3: Classify transcript ────────────────────────────────────────────────
    print("\n[L3] Classifying transcript (LLM)...")
    t0 = time.time()
    classification = classify(transcript)
    print(f"     Done in {time.time()-t0:.1f}s")

    # ── L2: Voice deepfake score ───────────────────────────────────────────────
    print("\n[L2] Running voice deepfake detector (wav2vec2)...")
    t0 = time.time()
    synth_prob = synthetic_probability(audio_path)
    print(f"     Done in {time.time()-t0:.1f}s")

    # ── L4: Fuse ───────────────────────────────────────────────────────────────
    result = fuse(classification, synthetic_prob=synth_prob)

    # ── Output ─────────────────────────────────────────────────────────────────
    verdict      = result["verdict"]
    stage        = classification["detected_stage"]
    flags        = classification["matched_red_flags"]
    reason       = result["reason"]
    voice_score  = result["synthetic_prob"]

    _divider("=")
    print(f"\n  VERDICT        {verdict}")
    print(f"  Stage          {stage} / 5")
    print(f"  Red flags      {', '.join(flags) if flags else '—'}")
    print(f"  P(AI voice)    {voice_score:.3f}")
    print(f"  Reason         {reason}")
    print()
    _divider("=")


if __name__ == "__main__":
    main()
