"""
Keyword-only baseline classifier for digital-arrest scam detection.
No LLM — used as a comparison baseline against the L3+L4 pipeline.
Returns the same verdict shape as l4_fusion.fuse so evaluate.py can
treat both classifiers uniformly.
"""

import re

# Keywords grouped by red-flag category (case-insensitive matching).
# Any single match → HIGH-RISK; no match → ABSTAIN.
_KEYWORDS: dict[str, list[str]] = {
    "RF01_authority": [
        "cbi", "enforcement directorate", r"\bed\b", "customs", "trai",
        "cyber cell", "narcotics", "income tax", "rbi", "supreme court",
        "inspector", "officer", "badge number",
    ],
    "RF02_crime": [
        "money laundering", "drug trafficking", "narcotics", r"\bmdma\b",
        "terror financing", "illegal", "seized", "contraband",
    ],
    "RF03_arrest": [
        "arrest", "warrant", r"\bfir\b", "non-bailable", "prosecution",
        "account freeze", "deactivat",
    ],
    "RF04_isolation": [
        "tell no one", "do not tell", "inform anyone", "obstruction",
        "confidential", "national security", "co-accused",
    ],
    "RF05_digital_custody": [
        "digital arrest", "digital custody", "stay on the call",
        "do not disconnect", "keep your camera", "camera on",
    ],
    "RF06_remote_access": [
        "anydesk", "teamviewer", "quicksupport", "remote access",
        "nine-digit code", "9-digit code",
    ],
    "RF07_credentials": [
        r"\botp\b", r"\bcvv\b", "upi pin", "net-banking", "netbanking",
        "card number", "share.*otp", "read out the code",
    ],
    "RF08_transfer": [
        "transfer.*account", "safe account", "verification account",
        "rbi.*account", "escrow", "break.*fd", "fixed deposit",
        "transfer your balance", "transfer the remaining",
    ],
    "RF09_refund_framing": [
        "refundable", "will be refunded", "after.*investigation",
        "clear your name",
    ],
    "RF11_urgency": [
        "within.*hour", "right now", "immediately", "expires",
        "two hours", "midnight",
    ],
}

# Flatten to (category_label, compiled_pattern) pairs
_COMPILED: list[tuple[str, re.Pattern]] = [
    (cat, re.compile(kw, re.IGNORECASE))
    for cat, keywords in _KEYWORDS.items()
    for kw in keywords
]


def classify(transcript: str) -> dict:
    """Return verdict dict compatible with l4_fusion.fuse output."""
    matched: list[str] = []
    for cat, pattern in _COMPILED:
        if pattern.search(transcript):
            label = pattern.pattern
            entry = f"{cat}:{label}"
            if entry not in matched:
                matched.append(entry)

    if matched:
        reason = f"Matched {len(matched)} keyword signal(s): {', '.join(m.split(':')[1] for m in matched[:5])}"
        return {"verdict": "HIGH-RISK", "matched_keywords": matched, "reason": reason}

    return {"verdict": "ABSTAIN", "matched_keywords": [], "reason": "No keyword signals detected."}
