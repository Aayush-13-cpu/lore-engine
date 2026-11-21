"""Narrate every scene (and Shorts hook lines), keeping word timestamps.

tts=chatterbox (default): Chatterbox-Turbo (MIT) clones assets/voice/<voice_ref>. Your own 10-20 s recording is the
most human option; without one, Kokoro reads a reference that Chatterbox then performs naturally. faster-whisper
aligns the script to the audio, and a line where the model dropped or garbled words is regenerated.
tts=kokoro: Kokoro-82M with its own word timestamps (faster, flatter delivery).

usage: python -m pipeline.voice work/plan.json work/audio
writes sNNN.wav / hookN.wav (24 kHz mono) + words.json {name: {dur, words: [[word, start, end], ...]}}
"""
import difflib
import hashlib
import json
import os
import re
import sys
import zlib

import numpy as np
import soundfile as sf

SR = 24000
REF_TEXT = ("Long before the first star was born, there were stories older than time. Some were written in fire. "
            "Some were carved into the bones of dead worlds. Tonight we open one of them, and we follow it to the very end.")


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def align(words, heard, dur):
    """Script words + whisper (word, start, end) -> script words with times; unheard words share the gap they sit in."""
    sm = difflib.SequenceMatcher(None, [norm(w) for w in words], [norm(h[0]) for h in heard], autojunk=False)
    t = [None] * len(words)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            t[a + k] = heard[b + k][1:]
    out, i = [], 0
    while i < len(words):
        if t[i]:
            out.append([words[i], round(t[i][0], 3), round(t[i][1], 3)])
            i += 1
            continue
        j = i
        while j < len(words) and not t[j]:
            j += 1
        lo = out[-1][2] if out else 0.0
        hi = max(t[j][0] if j < len(words) else dur, lo)
        total, x = sum(len(w) + 1 for w in words[i:j]), lo
        for w in words[i:j]:
            d = (hi - lo) * (len(w) + 1) / total
            out.append([w, round(x, 3), round(x + d, 3)])
            x += d
        i = j
    return out, sm.ratio()


def trim(a, thr=0.01):
    loud = np.flatnonzero(np.abs(a) > thr)
    return a[max(loud[0] - 1200, 0):loud[-1] + 2400] if len(loud) else a  # keep 50 ms before, 100 ms after


def kokoro_engine(P):
    from kokoro import KPipeline
    S, V = P["settings"], P["video"]
    voice = str(V.get("voice") or S["voice"]).replace(" ", "")
    speed = float(V.get("voice_speed") or S["voice_speed"])
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")  # voice ids start with a=US, b=UK
    # "Uatu -> Wah-too" becomes misaki's inline phoneme markup [Uatu](/wˈɑtu/); subtitles keep the real spelling
    fixes = [(re.compile(rf"\b{re.escape(w)}\b", re.I), pipe.g2p(say)[0]) for w, say in P["pronounce"].items()]

    def say(text):
        for rx, ph in fixes:
            text = rx.sub(lambda m, ph=ph: f"[{m.group(0)}](/{ph}/)", text)
        chunks, words, t = [], [], 0.0
        for r in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            for tok in r.tokens or []:
                if not re.search(r"\w", tok.text):
                    if words and tok.text.strip():  # glue punctuation onto the previous word for subtitle breaks
                        words[-1][0] += tok.text.strip()
                elif tok.start_ts is not None and tok.end_ts is not None:
                    words.append([tok.text, round(t + tok.start_ts, 3), round(t + tok.end_ts, 3)])
            chunks.append(r.audio.numpy())
            t += len(chunks[-1]) / SR
        return np.concatenate(chunks), words
    return say


def chatterbox_engine(P, out):
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from faster_whisper import WhisperModel
    ref = os.path.join("assets", "voice", str(P["settings"].get("voice_ref") or "").strip())
    if not os.path.isfile(ref):
        ref = os.path.join(out, "_reference.wav")
        try:
            sf.write(ref, kokoro_engine(P)(REF_TEXT)[0], SR)
        except Exception as e:  # Kokoro only supplies the reference timbre here; Chatterbox's own voice still works
            print(f"WARNING: Kokoro reference failed ({e!r}); using Chatterbox's built-in voice", flush=True)
            ref = None
    tts = ChatterboxTurboTTS.from_pretrained(device="cpu")
    assert tts.sr == SR, f"Chatterbox sample rate {tts.sr} != {SR}"
    if ref:
        tts.prepare_conditionals(ref)
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    import librosa  # installed with chatterbox; whisper gets arrays because its file reader breaks on newer PyAV
    respell = [(re.compile(rf"\b{re.escape(w)}\b", re.I), say) for w, say in P["pronounce"].items()]

    def say(text):
        spoken = text
        for rx, rep in respell:
            spoken = rx.sub(rep, spoken)
        best = None
        for attempt in range(3):  # a take that skips or mangles words gets redone
            torch.manual_seed(zlib.crc32(text.encode()) + attempt)
            a = trim(tts.generate(spoken).squeeze(0).numpy().astype(np.float32))
            segs, _ = whisper.transcribe(librosa.resample(a, orig_sr=SR, target_sr=16000), language="en",
                                         word_timestamps=True)
            heard = [(w.word, w.start, w.end) for seg in segs for w in seg.words]
            words, ratio = align(text.split(), heard, len(a) / SR)
            if not best or ratio > best[2]:
                best = (a, words, ratio)
            if ratio >= 0.8:
                break
        print(f"  match {best[2]:.2f} after {attempt + 1} take(s)", flush=True)
        return best[0], best[1]
    return say


def main(plan_path, out):
    P = json.load(open(plan_path))
    S = P["settings"]
    os.makedirs(out, exist_ok=True)
    engine = str(S.get("tts") or "chatterbox").strip().lower()
    ref = os.path.join("assets", "voice", str(S.get("voice_ref") or "").strip())
    voice_id = [engine, hashlib.sha1(open(ref, "rb").read()).hexdigest() if os.path.isfile(ref) else
                str(P["video"].get("voice") or S["voice"]), str(S.get("voice_speed")), P["pronounce"]]
    cache = os.environ.get("VOICE_CACHE", "")  # earlier renders' narration (actions/cache): unchanged lines are reused
    if cache:
        os.makedirs(cache, exist_ok=True)
    lines = [(f"s{s['n']:03d}", s["narration"]) for s in P["scenes"]]
    lines += [(f"hook{h['n']}", h["hook_line"]) for h in P["shorts"] if h["hook_line"]]
    say, meta = None, {}
    for name, text in lines:
        if not text:
            continue
        hit = os.path.join(cache, hashlib.sha1(json.dumps(voice_id + [text]).encode()).hexdigest()[:16]) if cache else ""
        if hit and os.path.exists(hit + ".json"):
            audio, words = sf.read(hit + ".wav", dtype="float32")[0], json.load(open(hit + ".json"))
        else:
            say = say or (chatterbox_engine(P, out) if engine == "chatterbox" else kokoro_engine(P))  # load models lazily
            audio, words = say(text)
            if hit:
                sf.write(hit + ".wav", audio, SR)
                json.dump(words, open(hit + ".json", "w"))
        sf.write(f"{out}/{name}.wav", audio, SR)
        meta[name] = dict(dur=round(len(audio) / SR, 3), words=words)
        print(f"{name} {meta[name]['dur']:.1f}s {len(words)} words", flush=True)
    json.dump(meta, open(f"{out}/words.json", "w"))


def _selfcheck():
    got, _ = align(["Hello", "Uatu", "world."], [("hello", 0.0, 0.4), ("wah", 0.5, 0.7), ("too", 0.7, 0.9),
                                               ("world", 1.0, 1.4)], 2.0)
    assert got == [["Hello", 0.0, 0.4], ["Uatu", 0.4, 1.0], ["world.", 1.0, 1.4]], got  # respelled name fills its gap


if __name__ == "__main__":
    _selfcheck()
    main(*sys.argv[1:3])
