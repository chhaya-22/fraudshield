"""
Batch-transcribe every .mp3 in data/audio/ and save transcripts to
data/audio/transcripts/<stem>.txt.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

import groq as _groq_module
from l1_asr import transcribe

_AUDIO_DIR = _SRC.parent / "data" / "audio"
_OUT_DIR = _AUDIO_DIR / "transcripts"


def main() -> None:
    _OUT_DIR.mkdir(exist_ok=True)

    mp3_files = sorted(_AUDIO_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {_AUDIO_DIR}")
        return

    print(f"Found {len(mp3_files)} file(s). Saving transcripts to {_OUT_DIR}\n")

    ok = failed = skipped = 0
    for audio_path in mp3_files:
        print(f"Transcribing: {audio_path.name}")
        out_path = _OUT_DIR / (audio_path.stem + ".txt")
        try:
            text = transcribe(audio_path)
            out_path.write_text(text, encoding="utf-8")
            words = len(text.split())
            print(f"  OK  -> {out_path.name}  ({words} words)")
            ok += 1
        except ValueError as exc:
            print(f"  SKIP: {exc}")
            skipped += 1
        except _groq_module.APIStatusError as exc:
            if exc.status_code == 413 or "too large" in str(exc).lower():
                print(
                    f"  SKIP: Groq rejected the file as too large (HTTP {exc.status_code}). "
                    "Split into segments under 25 MB."
                )
                skipped += 1
            else:
                print(f"  FAIL: Groq API error {exc.status_code}: {exc.message}")
                failed += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
            failed += 1

    print(f"\nDone. OK={ok}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
