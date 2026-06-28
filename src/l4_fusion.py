import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"

with open(_DATA_DIR / "scam_taxonomy.json") as _f:
    _TAXONOMY = json.load(_f)

_IRREVERSIBLE = set(_TAXONOMY["irreversible_action_triggers"])  # RF06, RF07, RF08
_FLAG_WEIGHTS = {rf["id"]: rf["weight"] for rf in _TAXONOMY["red_flags"]}


def fuse(classification: dict) -> dict:
    """Apply scoring_guidance rules and return verdict + reason."""
    stage: int = classification["detected_stage"]
    flags: set = set(classification["matched_red_flags"])
    reason: str = classification["reason"]

    has_irreversible = bool(flags & _IRREVERSIBLE)
    high_critical_count = sum(
        1 for f in flags if _FLAG_WEIGHTS.get(f) in ("high", "critical")
    )

    # HIGH-RISK: stage >= 3 + any irreversible trigger, OR 3+ high/critical flags
    if (stage >= 3 and has_irreversible) or high_critical_count >= 3:
        return {"verdict": "HIGH-RISK", "reason": reason}

    # CAUTION: stages 1-2 with at least one high flag and no irreversible trigger yet
    if 1 <= stage <= 2 and high_critical_count >= 1 and not has_irreversible:
        return {"verdict": "CAUTION", "reason": reason}

    # ABSTAIN: only medium flags, single weak signal, or nothing
    return {"verdict": "ABSTAIN", "reason": reason}
