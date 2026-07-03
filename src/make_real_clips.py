"""
Slice the 4 original scam call recordings into 15-second segments for
the real-voice eval set, saving up to 4 per source file.

Output: data/audio/eval/real/real_<slug>_01.mp3 … real_<slug>_04.mp3
"""

import re
import sys
from pathlib import Path

import librosa
import soundfile as sf

_AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"
_OUT_DIR = _AUDIO_DIR / "eval" / "real"

_SEG_SECONDS = 15
_MIN_SECONDS = 8
_MAX_SEGS_PER_FILE = 4
_SKIP = {"test_fake.mp3"}


def _slug(stem: str, fallback: str) -> str:
    """ASCII-safe short identifier derived from a filename stem."""
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower())[:18].strip("_")
    return s if len(s) >= 3 else fallback


def slice_file(src: Path, slug: str) -> list[Path]:
    """Load *src* at native SR, slice into segments, save to _OUT_DIR."""
    print(f"\n[{slug}]  loading {src.name[:55]} ...")
    waveform, sr = librosa.load(src, sr=None, mono=True)

    seg_samples = _SEG_SECONDS * sr
    min_samples = _MIN_SECONDS * sr
    total = len(waveform)
    duration = total / sr

    n_possible = int(total // seg_samples)
    n_segs = min(n_possible, _MAX_SEGS_PER_FILE)
    print(f"  {duration:.0f}s  {sr} Hz  →  {n_possible} possible segments, using {n_segs}")

    saved = []
    for i in range(n_segs):
        start = i * seg_samples
        chunk = waveform[start : start + seg_samples]
        if len(chunk) < min_samples:
            print(f"  seg {i+1}: too short ({len(chunk)/sr:.1f}s), skipping")
            continue
        out_path = _OUT_DIR / f"real_{slug}_{i+1:02d}.mp3"
        sf.write(str(out_path), chunk, sr, format="MP3")
        kb = out_path.stat().st_size // 1024
        print(f"  seg {i+1}: {len(chunk)/sr:.1f}s  →  {out_path.name}  ({kb} KB)")
        saved.append(out_path)

    return saved


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        f for f in _AUDIO_DIR.glob("*.mp3")
        if f.name not in _SKIP
    )
    if not sources:
        print(f"No source .mp3 files found in {_AUDIO_DIR}")
        return

    print(f"Found {len(sources)} source file(s).  Output → {_OUT_DIR}\n")

    all_saved: list[Path] = []
    for idx, src in enumerate(sources):
        slug = _slug(src.stem, f"src{idx+1:02d}")
        all_saved.extend(slice_file(src, slug))

    print(f"\nDone.  {len(all_saved)} real clip(s) saved to {_OUT_DIR}")
    for p in all_saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
