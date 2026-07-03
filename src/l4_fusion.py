import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"

with open(_DATA_DIR / "scam_taxonomy.json") as _f:
    _TAXONOMY = json.load(_f)

_IRREVERSIBLE = set(_TAXONOMY["irreversible_action_triggers"])  # RF06, RF07, RF08
_FLAG_WEIGHTS = {rf["id"]: rf["weight"] for rf in _TAXONOMY["red_flags"]}


def fuse(classification: dict, synthetic_prob: float | None = None) -> dict:
    """Apply scoring_guidance rules and return verdict + reason.

    Text-only path (synthetic_prob=None):
        Uses stage and red-flag counts from the taxonomy scoring_guidance.

    Voice amplifier (synthetic_prob provided):
        Applied *after* the text verdict is determined; voice alone can never
        produce HIGH-RISK or CAUTION when there are zero text red flags.

        >= 0.85  "strong AI"  — upgrade CAUTION → HIGH-RISK; append AI-voice note.
                               ABSTAIN with no flags: note only, no upgrade.
        0.5–0.85 "uncertain"  — keep tier; append uncertainty note.
                               ABSTAIN + ≥1 red flag → raise to CAUTION.
        < 0.5    "human"      — no tier change; append human note on HIGH/CAUTION.

        The 0.85 threshold is set conservatively: wav2vec2 deepfake detectors
        commonly report AUC > 0.95 on studio TTS but degrade on real call audio,
        so a high-confidence signal (≥ 0.85) is required before upgrading a tier.
    """
    stage: int = classification["detected_stage"]
    flags: set = set(classification["matched_red_flags"])
    reason: str = classification["reason"]

    has_irreversible = bool(flags & _IRREVERSIBLE)
    high_critical_count = sum(
        1 for f in flags if _FLAG_WEIGHTS.get(f) in ("high", "critical")
    )

    # ── Text-only verdict ──────────────────────────────────────────────────────
    # HIGH-RISK: stage >= 3 + any irreversible trigger, OR 3+ high/critical flags
    if (stage >= 3 and has_irreversible) or high_critical_count >= 3:
        verdict = "HIGH-RISK"
    # CAUTION: stages 1-2 with at least one high flag and no irreversible trigger yet
    elif 1 <= stage <= 2 and high_critical_count >= 1 and not has_irreversible:
        verdict = "CAUTION"
    # ABSTAIN: only medium flags, single weak signal, or nothing
    else:
        verdict = "ABSTAIN"

    # ── Voice amplifier ────────────────────────────────────────────────────────
    if synthetic_prob is not None:
        has_text_signal = len(flags) > 0

        if synthetic_prob >= 0.85:
            if verdict == "CAUTION":
                verdict = "HIGH-RISK"
                reason += " Caller's voice shows strong signs of AI generation."
            elif verdict == "HIGH-RISK":
                reason += " Caller's voice shows strong signs of AI generation."
            else:  # ABSTAIN
                if has_text_signal:
                    verdict = "CAUTION"
                    reason += " Caller's voice shows strong signs of AI generation."
                else:
                    reason += " Voice appears AI-generated but no scam content detected."

        elif 0.5 <= synthetic_prob < 0.85:
            reason += " Voice authenticity is uncertain."
            if verdict == "ABSTAIN" and has_text_signal:
                verdict = "CAUTION"

        else:  # < 0.5
            if verdict in ("HIGH-RISK", "CAUTION"):
                reason += " Voice appears human."

    return {"verdict": verdict, "reason": reason, "synthetic_prob": synthetic_prob}
