"""Narrate every scene (and Shorts hook lines) with Kokoro-82M, keeping word timestamps.

usage: python -m pipeline.voice work/plan.json work/audio
writes sNNN.wav / hookN.wav (24 kHz mono) + words.json {name: {dur, words: [[word, start, end], ...]}}
Mastering (EQ, compression, pitch) happens once on the whole mix in render.py.
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SR = 24000


def main(plan_path, out):
    P = json.load(open(plan_path))
    S, V = P["settings"], P["video"]
    os.makedirs(out, exist_ok=True)
    voice = str(V.get("voice") or S["voice"]).replace(" ", "")
    speed = float(V.get("voice_speed") or S["voice_speed"])
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")  # voice ids start with a=US, b=UK
    # "Uatu -> Wah-too" in the Pronounce sheet becomes misaki's inline phoneme markup [Uatu](/wˈɑtu/)
    fixes = [(re.compile(rf"\b{re.escape(w)}\b"), f"[{w}](/{pipe.g2p(say)[0]}/)") for w, say in P["pronounce"].items()]
    lines = [(f"s{s['n']:03d}", s["narration"]) for s in P["scenes"]]
    lines += [(f"hook{h['n']}", h["hook_line"]) for h in P["shorts"] if h["hook_line"]]
    meta = {}
    for name, text in lines:
        if not text:
            continue
        for rx, rep in fixes:
            text = rx.sub(rep, text)
        chunks, words, t = [], [], 0.0
        for r in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            a = r.audio.numpy()
            for tok in r.tokens or []:
                if not re.search(r"\w", tok.text):
                    if words and tok.text.strip():  # glue punctuation onto the previous word for subtitle breaks
                        words[-1][0] += tok.text.strip()
                elif tok.start_ts is not None and tok.end_ts is not None:
                    words.append([tok.text, round(t + tok.start_ts, 3), round(t + tok.end_ts, 3)])
            chunks.append(a)
            t += len(a) / SR
        sf.write(f"{out}/{name}.wav", np.concatenate(chunks), SR)
        meta[name] = dict(dur=round(t, 3), words=words)
        print(f"{name} {t:.1f}s {len(words)} words", flush=True)
    json.dump(meta, open(f"{out}/words.json", "w"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
