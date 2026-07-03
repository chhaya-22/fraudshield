"""
Generate 15 synthetic scam audio clips for deepfake voice evaluation.
Saves fake_01.mp3 … fake_15.mp3 into data/audio/eval/fake/.
Voices and scripts rotate independently to maximise variety.
"""

import asyncio
import sys
from itertools import cycle
from pathlib import Path

from edge_tts import Communicate

_OUT_DIR = Path(__file__).parent.parent / "data" / "audio" / "eval" / "fake"
_N_CLIPS = 15
_CONCURRENCY = 5  # max simultaneous TTS requests

_VOICES = [
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",
    "hi-IN-MadhurNeural",
    "hi-IN-SwaraNeural",
]

# 6 scripts — mix of English and Hindi, 3-4 sentences each
_SCRIPTS = [
    # 1 — English: CBI / parcel pretext, digital arrest
    (
        "A parcel sent in your name has been seized at Mumbai airport. "
        "It contained five hundred grams of narcotics and forged passports. "
        "This is Inspector Rajan from CBI Cyber Crime division, badge number 3812. "
        "A non-bailable warrant is being prepared in your name right now."
    ),
    # 2 — Hindi: TRAI number deactivation threat
    (
        "Namaskar, main TRAI ki taraf se bol raha hoon. "
        "Aapka mobile number ek fraud case mein paya gaya hai aur do ghante mein band ho jayega. "
        "Aapko abhi cyber crime officer se baat karni hogi. "
        "Agar aapne phone kaata toh aapke khilaf giraftari ka warrant jari ho jayega."
    ),
    # 3 — English: money laundering, isolation and fund transfer
    (
        "You are now under digital arrest. Do not disconnect this video call for any reason. "
        "Your bank account has been linked to a money laundering network worth forty crore rupees. "
        "Do not contact your family or lawyer — this is a national security matter. "
        "To prove your funds are clean, transfer your balance to an RBI verification account immediately."
    ),
    # 4 — Hindi: Enforcement Directorate, OTP demand
    (
        "Aapka bank account ek suspicious transaction ke karan freeze kar diya gaya hai. "
        "Main Enforcement Directorate se Deputy Director Sharma bol raha hoon. "
        "Apna account unfreeze karne ke liye aapko abhi OTP share karna hoga. "
        "Yeh ek government verification process hai, isliye bilkul ghabrana mat."
    ),
    # 5 — English: Customs Department, digital custody
    (
        "This is the Customs Department of India. A package from Dubai registered in your Aadhaar has been intercepted. "
        "It contained illegal foreign currency and expired passports. "
        "You are required to stay on this video call — you are now under our digital custody. "
        "If you disconnect or inform anyone, you will be charged with obstruction of a federal investigation."
    ),
    # 6 — Mixed Hindi-English: fake Supreme Court notice, escrow transfer
    (
        "Hello, main Supreme Court ke cyber division se bol raha hoon. "
        "Aapke naam pe ek money laundering case register hua hai, FIR number 0472 of 2026. "
        "You must transfer your savings to a government-verified escrow account within two hours. "
        "Yeh amount fully refundable hai jab investigation complete ho jayegi."
    ),
]


async def _generate(text: str, voice: str, out_path: Path, sem: asyncio.Semaphore) -> None:
    async with sem:
        comm = Communicate(text, voice)
        await comm.save(str(out_path))


async def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    voice_iter = cycle(_VOICES)
    script_iter = cycle(_SCRIPTS)

    plan: list[tuple[int, str, str, Path]] = []
    for i in range(1, _N_CLIPS + 1):
        script = next(script_iter)
        voice = next(voice_iter)
        plan.append((i, script, voice, _OUT_DIR / f"fake_{i:02d}.mp3"))

    print(f"Generating {_N_CLIPS} clips  ->  {_OUT_DIR}\n")
    col = max(len(v) for v in _VOICES)
    for idx, script, voice, path in plan:
        preview = script[:55].replace("\n", " ") + "…"
        print(f"  {path.name}  {voice:<{col}}  \"{preview}\"")

    print()
    sem = asyncio.Semaphore(_CONCURRENCY)
    await asyncio.gather(
        *[_generate(script, voice, path, sem) for _, script, voice, path in plan]
    )

    print("Done.\n")
    total_kb = 0
    for idx, _, _, path in plan:
        kb = path.stat().st_size // 1024
        total_kb += kb
        print(f"  {path.name}  {kb:>4} KB")
    print(f"\n  Total: {total_kb} KB  ({total_kb // 1024} MB)")


if __name__ == "__main__":
    asyncio.run(main())
