"""
L1 ASR: transcribe an audio file via Groq Whisper and save the result.

CLI usage:
    python src/l1_asr.py <audio_file>

Saves a .txt transcript next to the audio file and prints it to stdout.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import groq as _groq_module
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_GROQ_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB

# Map file extensions to MIME types accepted by Groq Whisper
_MIME = {
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def transcribe(audio_path: Path) -> str:
    """Return the Whisper transcript for *audio_path*.

    Raises ValueError for files that exceed Groq's 25 MB limit.
    Raises groq.APIStatusError / groq.BadRequestError for API-level failures.
    """
    audio_path = Path(audio_path)
    size = audio_path.stat().st_size
    if size > _GROQ_SIZE_LIMIT:
        mb = size / 1024 / 1024
        raise ValueError(
            f"File is {mb:.1f} MB — Groq Whisper rejects files over 25 MB. "
            "Split the audio first (e.g. with ffmpeg -f segment -segment_time 600)."
        )

    mime = _MIME.get(audio_path.suffix.lower(), "audio/mpeg")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    with open(audio_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(audio_path.name, fh, mime),
            model="whisper-large-v3",
        )
    return resp.text


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python l1_asr.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    try:
        text = transcribe(audio_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except _groq_module.APIStatusError as exc:
        if exc.status_code == 413 or "too large" in str(exc).lower():
            print(
                f"Error: Groq rejected the file as too large (HTTP {exc.status_code}). "
                "Split the audio into segments under 25 MB and try again."
            )
        else:
            print(f"Groq API error {exc.status_code}: {exc.message}")
        sys.exit(1)

    out_path = audio_path.with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved: {out_path}\n")
    print(text)


if __name__ == "__main__":
    main()
