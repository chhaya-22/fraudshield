# FraudShield — Catch the digital-arrest scam before the money moves

**ET AI Hackathon 2026 · Problem 6 — AI for Digital Public Safety**

A multimodal AI system that listens to a suspicious call, reads the scammer's **script** and the caller's **voice** together, and warns the victim — in their own language — *before* any money is transferred.

---

## The problem

Digital-arrest scams cost India **₹19,000+ crore in 2025**. A fraudster impersonating CBI/ED/police traps a victim — often elderly — on a video call, accuses them of a crime, isolates them ("stay on video, tell no one"), and pressures them to transfer their savings "for verification."

The warning signs are all present in the call. Nobody is listening in time.

---

## What this does

| Layer | What it does | Model |
|---|---|---|
| **L1 — Transcription** | Call audio → text (Hindi & English) | Whisper `large-v3` (Groq) |
| **L3 — Script understanding** | Classifies scam **stage (0–5)** and **12 red flags** against a digital-arrest taxonomy | Llama 3.3 70B + retrieval |
| **L2 — Voice authenticity** | P(AI-generated voice) — real vs synthetic speech | fine-tuned wav2vec2 |
| **L4 — Fusion** | Combines both signals into `HIGH-RISK` / `CAUTION` / `ABSTAIN` | rule-based fusion |
| **Intervention** | Bilingual warning + auto-drafted 1930/NCRB complaint | — |

**Key design principle:** neither signal can raise a high-risk alarm alone. That is what keeps false alarms at zero.

**Why stages matter:** because the system tracks the scam's psychological arc, it reaches HIGH-RISK during *fear & isolation* — **before** the money is demanded. That lead time is what saves the victim.

---

## Results

Evaluated on **12 real call recordings** (public sources), adversarial text cases, and a 31-clip voice set.

| Metric | Result |
|---|---|
| Real scam calls flagged | **7 / 7** (6 HIGH-RISK, 1 CAUTION) |
| Real genuine calls correctly cleared | **5 / 5** |
| **False positives on real benign audio** | **0** |
| F1 on adversarial text cases | **1.00** (vs **0.60** keyword baseline) |
| Keyword baseline false-positive rate | 75% of legitimate calls wrongly flagged |
| Voice deepfake detection | **AUC 1.00 / EER 0%** (31 clips) |

The system also **correctly declined to fire** on a real *non*-digital-arrest (investment) scam — evidence of specificity, not blanket alarming.

> **Honest note on scope:** real recordings are sourced from publicly available, anonymized uploads; provenance of public recordings cannot be independently certified. Stage prediction shows some run-to-run variance; the binary verdict is stable.

---

## Methodology — why we don't train from scratch

No large labelled corpus of Indian digital-arrest calls exists. Training from scratch would underperform a strong LLM guided by domain knowledge, so:

1. **Retrieval-augmented classification** — a pretrained LLM reasons against the digital-arrest taxonomy we engineered. The domain knowledge lives in the taxonomy, not in fragile weights.
2. **A fine-tuned model where it fits** — the wav2vec2 voice detector *is* a fine-tuned model (AUC 1.0). We use training where it's the right tool, prompting where it isn't.
3. **Rigorous evaluation** — baselines, adversarial cases, and real-audio tests on both scam and genuine calls.

---

## Deployment & privacy

WhatsApp calls are end-to-end encrypted. We do **not** claim to intercept them.

FraudShield is a **consumer app that runs on the victim's own device**, where the call audio is already decrypted to be heard. Analysis happens locally with the user's consent — call content need not leave the phone. Real-time on-device capture is the deployment roadmap; this prototype proves the detection intelligence.

---

## Running it

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. add your key to .env
GROQ_API_KEY=your_key_here

# 3. start the server
python src/server.py

# 4. open the app
http://localhost:8000/fraudshield_app.html
```

Then open the **Scan** tab, paste any call transcript, and get a live verdict from the real model.

### Other entry points

```bash
python src/pipeline_demo.py <audio.mp3>   # full multimodal pipeline on an audio file
python src/evaluate.py                    # text evaluation (easy set)
python src/evaluate.py data/eval_transcripts_hard.json   # adversarial set
python src/evaluate.py --baseline data/eval_transcripts_hard.json  # keyword baseline
python src/evaluate_voice.py              # voice detector AUC / EER
python src/classify_real.py               # classify all real transcripts
```

---

## Repository layout

```
src/
  l1_asr.py            Whisper transcription
  l2_voice.py          wav2vec2 deepfake detection (windowed)
  l3_classify.py       LLM + taxonomy classification
  l4_fusion.py         multimodal decision fusion
  server.py            FastAPI backend + static hosting
  evaluate.py          text evaluation + keyword baseline comparison
  evaluate_voice.py    voice AUC / EER
  pipeline_demo.py     end-to-end audio → verdict
data/
  scam_taxonomy.json       6 scam stages + 12 weighted red flags
  eval_transcripts.json    labelled evaluation cases
  eval_transcripts_hard.json  adversarial cases
static/
  fraudshield_app.html     the consumer app (live, wired to backend)
```

---

## Scope

**Built & validated:** digital-arrest scam detection · citizen fraud shield app · AI-voice detection · multi-source fusion

**Roadmap:** counterfeit-currency CV · fraud-network graph intelligence · geospatial crime mapping · telecom integration & full 12-language coverage

---

## Ethics

All evaluation audio comes from publicly available uploads and is used for research evaluation only; transcripts are stored rather than redistributed audio. Synthetic voice samples were generated in-house for controlled evaluation.
