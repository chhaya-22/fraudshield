"""
FraudShield API server.

    python src/server.py

Endpoints:
    GET  /health          liveness check
    POST /scan-text       {transcript} → verdict, stage, red_flags, reason
    GET  /                serves static/fraudshield_app.html (and other static assets)
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from l3_classify import classify
from l4_fusion import fuse

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="FraudShield", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = _SRC.parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────────────

class ScanTextRequest(BaseModel):
    transcript: str


class ScanTextResponse(BaseModel):
    verdict: str
    stage: int
    red_flags: list[str]
    reason: str

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/scan-text", response_model=ScanTextResponse, tags=["scan"])
def scan_text(body: ScanTextRequest):
    if not body.transcript.strip():
        raise HTTPException(status_code=422, detail="transcript must not be empty")
    classification = classify(body.transcript)
    result = fuse(classification)
    return ScanTextResponse(
        verdict=result["verdict"],
        stage=classification["detected_stage"],
        red_flags=classification["matched_red_flags"],
        reason=result["reason"],
    )

# Static files must be mounted LAST — a "/" mount acts as a catch-all and
# would swallow POST /scan-text before the route handler above could match it.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
