import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import call_llm

_DATA_DIR = Path(__file__).parent.parent / "data"

with open(_DATA_DIR / "scam_taxonomy.json") as _f:
    _TAXONOMY = json.load(_f)

_STAGES_TEXT = "\n".join(
    f"Stage {s['id']} — {s['name']}: {s['intent']}\n  Indicators: {'; '.join(s['indicators'])}"
    for s in _TAXONOMY["stages"]
)

_FLAGS_TEXT = "\n".join(
    f"{rf['id']} ({rf['weight']}) — {rf['label']}: {rf['description']}"
    for rf in _TAXONOMY["red_flags"]
)

_PROMPT_TEMPLATE = """\
You are a digital-arrest scam detection classifier.

## Scam Stage Taxonomy
{stages}

## Red Flags
{flags}

## Transcript
{transcript}

Analyse the transcript and return ONLY a JSON object with exactly these keys:
- "detected_stage": integer 0-5, the highest scam stage reached (use 0 if no scam signals)
- "matched_red_flags": list of matched red-flag IDs from the list above (e.g. ["RF01", "RF03"])
- "reason": one sentence explaining the classification

Return ONLY valid JSON, no markdown, no extra text."""


def classify(transcript: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(
        stages=_STAGES_TEXT,
        flags=_FLAGS_TEXT,
        transcript=transcript,
    )
    raw = call_llm(prompt, json_mode=True)
    return json.loads(raw)
