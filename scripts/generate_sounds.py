"""Generate placeholder audio feedback assets (success.wav / error.wav).

Writes two small beep tones to assets/sounds/. Run once from project root:
    .venv/bin/python scripts/generate_sounds.py
"""

import math
import os
import struct
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets/sounds")
SAMPLE_RATE = 22050
DURATION = 0.35
AMPLITUDE = 0.6


def _tone(freqs, duration=DURATION):
    frames = []
    total = int(SAMPLE_RATE * duration)
    for i in range(total):
        t = i / SAMPLE_RATE
        seg = len(freqs) - 1
        idx = min(int(t / (duration / (seg + 1))), seg)
        f = freqs[idx]
        value = AMPLITUDE * math.sin(2 * math.pi * f * t)
        frames.append(struct.pack("<h", int(value * 32767)))
    return b"".join(frames)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with wave.open(os.path.join(OUT_DIR, "success.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(_tone([880.0, 1320.0]))
    with wave.open(os.path.join(OUT_DIR, "error.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(_tone([220.0]))
    print("Wrote assets/sounds/success.wav, assets/sounds/error.wav")


if __name__ == "__main__":
    main()