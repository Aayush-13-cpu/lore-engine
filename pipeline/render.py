"""Long-form render: comic canvases -> camera-motion clips -> ffmpeg (xfade transitions, chromakey green screen,
grade, grain, watermark, optional burned subs) + mastered voice, ducked music and synthesized SFX.

usage: [DRAFT=1] python -m pipeline.render work/plan.json work/images work/audio out/VID001
writes video.mp4, thumbnail.jpg, captions.srt, description.txt, metadata.json
"""
import glob
import json
import math
import os
import re
import subprocess
import sys
import zlib
from multiprocessing import Pool

import cv2
import numpy as np
import soundfile as sf
from PIL import Image

from pipeline import comic
from pipeline.sheet import TRANSITIONS

SR = 48000
K = 1.25  # canvas is 25% larger than the frame: room for camera moves without upscaling
DRAFT = os.environ.get("DRAFT") == "1"
XFADE = {  # sheet name -> (ffmpeg xfade transition, seconds, whoosh)
    "cut": ("fade", 0.0, False), "fade": ("fade", 0.5, False), "dissolve": ("dissolve", 0.6, False),
    "black": ("fadeblack", 0.9, False), "flash": ("fadewhite", 0.4, True), "whip": ("slideleft", 0.3, True),
    "glitch": ("pixelize", 0.35, True), "zoom": ("zoomin", 0.45, True), "circle": ("circleopen", 0.6, True),
    "wipe": ("wipeleft", 0.45, True), "slide": ("slideup", 0.4, True), "radial": ("radial", 0.6, True),
}
assert set(XFADE) | {"auto"} == TRANSITIONS
AUTO_T = ["fade", "cut", "whip", "cut", "dissolve", "cut", "zoom", "cut"]  # mostly cuts: flashy-every-scene reads amateur
AUTO_M = ["in", "right", "out", "left", "in", "up", "out", "down"]
PRESETS = {  # keyframes (t 0..1, zoom, focus x, focus y)
    "in": [(0, 1.0, .5, .5), (1, 1.16, .5, .46)], "out": [(0, 1.16, .5, .46), (1, 1.0, .5, .5)],
    "left": [(0, 1.1, .62, .5), (1, 1.1, .38, .5)], "right": [(0, 1.1, .38, .5), (1, 1.1, .62, .5)],
    "up": [(0, 1.12, .5, .62), (1, 1.12, .5, .38)], "down": [(0, 1.12, .5, .38), (1, 1.12, .5, .62)],
    "static": [(0, 1.03, .5, .5), (1, 1.05, .5, .5)], "shake": [(0, 1.12, .5, .5), (1, 1.16, .5, .5)],
}
CARDS = {"title", "sting", "quote", "endscreen"}
ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX".split()
C = None  # per-process render context


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])} ...\n{r.stderr[-3000:]}")


class Ctx:
    def __init__(self, plan, imgdir, work):
        self.P, self.imgdir, self.work = plan, imgdir, work
        S = self.S = plan["settings"]
        self.W, self.H, self.fps = int(float(S["width"])), int(float(S["height"])), int(float(S["fps"]))
        self.CW, self.CH = round(self.W * K), round(self.H * K)
        self.k = float(S["print_pass"])
        self.style = plan["style"]
        self.accent = self.style.get("accent") or "#E62429"
        self.prompts = {j["name"]: j["prompt"] for j in plan["images"] + plan["refs"]}

    def image(self, name):
        p = os.path.join(self.imgdir, name)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
        if not DRAFT:
            raise RuntimeError(f"missing image {p}: did the images job fail?")
        return comic.placeholder(self.prompts.get(name, name), 1344, 768, zlib.crc32(name.encode()))


def init(*args):
    global C
    C = Ctx(*args)


def spread(text, dur):
    """Fallback word timing (proportional to word length) when TTS gave no timestamps."""
    words = text.split()
    if not words or not dur:
        return []
    total, t, out = sum(len(w) + 1 for w in words), 0.0, []
    for w in words:
        d = dur * (len(w) + 1) / total
        out.append([w, round(t, 3), round(t + d, 3)])
        t += d
    return out


def voice_key(s):
    return s.get("voice_key") or f"s{s['n']:03d}"  # Shorts hook lines carry their own audio name


def timeline(P, meta, fps, pad=None):
    pad = float(P["settings"]["scene_pad"]) if pad is None else pad
    out, start, chapter, titles = [], 0, None, 0
    for i, s in enumerate(P["scenes"]):
        tr = s["transition"]
        if tr == "auto":
            tr = "cut" if i == 0 else "black" if s["chapter"] != chapter or s["template"] in ("title", "sting") \
                else AUTO_T[i % len(AUTO_T)]
        Tf = 0 if i == 0 else max(1, round(XFADE[tr][1] * fps))
        m = meta.get(voice_key(s))
        speech = m["dur"] if m else 0.0
        body = (speech + pad + s["hold"]) if speech else (s["hold"] or 2.5)
        F = max(round((Tf / 2 / fps + body) * fps), Tf + 1)
        titles += s["template"] == "title"
        out.append(dict(i=i, s=s, tr=tr, Tf=Tf, F=F, start=start, voice=(start + Tf / 2) / fps, speech=speech,
                        words=(m or {}).get("words") or spread(s["narration"], speech), chapter_no=max(titles, 1)))
        start += F
        chapter = s["chapter"]
    for a, b in zip(out, out[1:]):
        a["out"] = b["Tf"]
    out[-1]["out"] = 0
    return out


# ---------------- pictures ----------------

def prep(img, x, y, t_in, pop=False):
    """RGBA overlay -> (float BGRA cropped to its visible box, x, y, appear time, pop)."""
    box = img.getchannel("A").getbbox()
    if not box:
        return None
    img = img.crop(box)
    a = np.asarray(img, np.float32)
    return np.dstack([a[..., 2::-1], a[..., 3:] / 255]), x + box[0], y + box[1], t_in, pop


def blend(frame, ov, tau):
    img, x, y, t_in, pop = ov
    a = min((tau - t_in) / 0.22, 1.0)
    if a <= 0:
        return
    if pop and a < 1:  # SFX lettering slams in from 135%
        sc = 1 + 0.35 * (1 - a)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, None, fx=sc, fy=sc)
        x, y = int(x - (img.shape[1] - w0) / 2), int(y - (img.shape[0] - h0) / 2)
    elif not pop:
        y = int(y + 18 * (1 - a))
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + img.shape[1], W), min(y + img.shape[0], H)
    if x1 <= x0 or y1 <= y0:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    al = src[..., 3:] * a
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (reg * (1 - al) + src[..., :3] * al).astype(np.uint8)


def interp(keys, t):
    for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
        if t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            return [p + (q - p) * u for p, q in zip(a, b)]
    return list(keys[-1][1:])


def motion_keys(s, i, L, lead):
    m = s["motion"]
    if m == "auto":
        m = {"title": "in", "sting": "in", "quote": "in", "dossier": "in", "endscreen": "static"}.get(
            s["template"], AUTO_M[i % len(AUTO_M)])
    if m == "punch":
        return [(0, 1.0, .5, .5), (min((lead + .3) / L, .5), 1.17, .5, .47), (1, 1.22, .5, .46)]
    return PRESETS[m]


def shaker(s, lead, seed):
    hand, impact = s["motion"] == "shake", s["sfx"] == "impact"
    if not (hand or impact):
        return None
    ph = np.random.default_rng(seed).random(4) * 6.283

    def f(tau):
        dx = dy = 0.0
        if hand:
            dx += 5 * math.sin(8.2 * tau + ph[0]) + 2.5 * math.sin(23 * tau + ph[1])
            dy += 4 * math.sin(6.9 * tau + ph[2]) + 2 * math.sin(26 * tau + ph[3])
        if impact and tau >= lead:
            e = 26 * math.exp(-7 * (tau - lead))
            dx += e * math.sin(82 * (tau - lead))
            dy += e * 0.7 * math.cos(69 * (tau - lead))
        return dx, dy
    return f


def particles(kind, W, H, seed):
    rng = np.random.default_rng(seed)
    N = {"embers": 80, "dust": 60, "snow": 140, "rain": 220, "stars": 110}[kind]
    p0 = rng.random((N, 2)) * [W, H]
    v = {"embers": np.c_[rng.normal(0, 18, N), -rng.uniform(50, 140, N)], "dust": rng.normal(0, 12, (N, 2)),
         "snow": np.c_[rng.normal(0, 20, N), rng.uniform(40, 100, N)],
         "rain": np.c_[np.full(N, -180.0), rng.uniform(1300, 1700, N)], "stars": rng.normal(0, 2, (N, 2))}[kind]
    col = np.float32({"embers": (40, 140, 255), "dust": (150, 160, 170), "snow": (235, 235, 235),
                      "rain": (120, 120, 130), "stars": (255, 255, 255)}[kind])
    r = rng.uniform(1.2, 3.4, N)
    ph = rng.random(N) * 6.283

    def draw(frame, tau):
        layer = np.zeros_like(frame)
        pos = (p0 + v * tau) % [W, H]
        bright = 0.55 + 0.45 * np.sin(tau * 5 + ph)
        for (x, y), rr, b in zip(pos.astype(int), r, bright):
            if kind == "rain":
                cv2.line(layer, (x, y), (x + 4, y - 34), col.tolist(), 1, cv2.LINE_AA)
                continue
            if kind == "embers":
                cv2.circle(layer, (x, y), int(rr * 3), (col * b * .25).tolist(), -1, cv2.LINE_AA)
            cv2.circle(layer, (x, y), max(1, round(rr)), (col * b).tolist(), -1, cv2.LINE_AA)
        cv2.add(frame, layer, dst=frame)  # additive = glow, never darkens the art
    return draw


def glitch(frame, rng):
    sh = int(rng.integers(8, 22))
    frame[:, :, 2] = np.roll(frame[:, :, 2], sh, axis=1)
    frame[:, :, 0] = np.roll(frame[:, :, 0], -sh, axis=1)
    for _ in range(3):
        y, h = int(rng.integers(0, frame.shape[0] - 60)), int(rng.integers(10, 60))
        frame[y:y + h] = np.roll(frame[y:y + h], int(rng.integers(-80, 80)), axis=1)


def build(s, tl, W, H, scale=1.0, captions=True):
    """Scene -> (canvas BGR, overlays, camera keys, panel reveal fn)."""
    cw, ch = round(C.CW * scale), round(C.CH * scale)
    tpl, k, acc, st = s["template"], C.k, C.accent, C.style
    imgs = [C.image(x) for x in s["images"]]
    lead = tl["Tf"] / 2 / C.fps
    L = (tl["F"] + tl["out"]) / C.fps
    over, keys, reveal = [], None, None
    if tpl == "full" or tpl.startswith("gs:"):
        canvas = comic.print_pass(comic.cover(imgs[0], cw, ch), k)
    elif tpl == "panels":
        canvas, tiles, centres = comic.panels(imgs, cw, ch, k)
        seg = max(tl["speech"], 1.0) / len(tiles)
        times = [lead + j * seg for j in range(len(tiles))]
        cur, states = canvas.copy(), [np.asarray(canvas)[..., ::-1].copy()]
        for tile, xy in tiles:
            cur.paste(tile, xy)
            states.append(np.asarray(cur)[..., ::-1].copy())

        def reveal(tau):
            j = sum(tau >= t for t in times)
            a = min((tau - times[j - 1]) / 0.25, 1.0) if j else 1.0
            return states[j] if a >= 1 else cv2.addWeighted(states[j], a, states[j - 1], 1 - a, 0)
        keys = [(0, 1.0, .5, .5)] + [(min((t + .6 * seg) / L, .97), 1.2, cx, cy) for t, (cx, cy) in zip(times, centres)]
        keys = sorted(keys) + [(1, 1.02, .5, .5)]
    elif tpl == "split":
        canvas = comic.split(imgs[0], imgs[-1], (s["caption"].split("|") + ["", ""])[:2], acc, cw, ch, k)
    elif tpl == "dossier":
        cid = s["characters"][0]
        canvas = comic.dossier(C.image(f"ref_{cid}.png"), C.P["characters"][cid], acc, C.P["video"].get("universe", ""), cw, ch, k)
    else:
        canvas = comic.darken(imgs[0], cw, ch) if imgs else comic.print_pass(Image.new("RGB", (cw, ch), (16, 16, 20)), 1.0)
        if tpl == "title":
            kw = dict(title=s["caption"] or s["chapter"], kicker=f"CHAPTER {ROMAN[min(tl['chapter_no'], 20) - 1]}")
        elif tpl == "sting":
            kw = dict(title=str(C.S["channel_name"]), kicker=str(C.S.get("channel_tagline", "")))
        elif tpl == "quote":
            parts = re.split(r"\s+[—–-]{1,2}\s+", s["caption"], maxsplit=1)  # "quote — who"
            kw = dict(text=parts[0].strip('"“” '), attribution=parts[1] if len(parts) > 1 else "")
        else:
            kw = dict(title=s["caption"])
        if W >= H:  # card text is laid out for landscape
            over.append(prep(comic.card_text(tpl, W, H, acc, **kw), 0, 0, lead + 0.1))
    if captions and s["caption"] and tpl not in CARDS | {"split"}:
        over.append(prep(comic.caption_box(s["caption"], st.get("caption_bg") or "#F6E27A", st.get("caption_fg") or "#111111"),
                         60, 50, lead + 0.3))
    if s["sfx_text"]:
        im = comic.sfx_word(s["sfx_text"].upper(), st.get("sfx_color") or "#FFE14D")
        over.append(prep(im, W - im.width - 70 if W >= H else (W - im.width) // 2, int(H * .52 - im.height / 2), lead, pop=True))
    if keys is None:
        keys = motion_keys(s, tl["i"], L, lead)
    return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR), [o for o in over if o], keys, reveal


def frames(canvas, keys, n, W, H, overlays=(), s=None, lead=0.0, reveal=None, seed=0):
    fps = C.fps
    ch_, cw_ = canvas.shape[:2]
    base = max(W / cw_, H / ch_)
    fx = s["fx"] if s else ""
    parts = particles(fx, W, H, seed) if fx in ("embers", "dust", "snow", "rain", "stars") else None
    shake = shaker(s, lead, seed) if s else None
    rng = np.random.default_rng(seed)
    for f in range(n):
        tau, t = f / fps, f / max(n - 1, 1)
        src = reveal(tau) if reveal else canvas
        z, fx_, fy_ = interp(keys, t)
        sc = base * z
        vw, vh = W / sc, H / sc
        cx = min(max(fx_ * cw_, vw / 2), cw_ - vw / 2)
        cy = min(max(fy_ * ch_, vh / 2), ch_ - vh / 2)
        dx, dy = shake(tau) if shake else (0.0, 0.0)
        M = np.float32([[sc, 0, W / 2 - sc * cx + dx], [0, sc, H / 2 - sc * cy + dy]])
        frame = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        for ov in overlays:
            blend(frame, ov, tau)
        if parts:
            parts(frame, tau)
        if fx == "flash" and lead <= tau < lead + 0.3:
            a = 1 - (tau - lead) / 0.3
            frame = cv2.addWeighted(frame, 1 - a, np.full_like(frame, 255), a, 0)
        if fx == "glitch" and lead <= tau < lead + 0.45 and f % 2 == 0:
            glitch(frame, rng)
        yield frame


def starfield(n, W, H, seed):
    rng = np.random.default_rng(seed)
    N = 700
    xy, z = rng.uniform(-1, 1, (N, 2)), rng.uniform(0.05, 1, N)
    f_ = W * 0.35
    for f in range(n):
        frame = np.full((H, W, 3), (10, 6, 8), np.uint8)
        zp = z.copy()
        z -= 0.4 / C.fps * (1 + 1.5 * f / n)  # accelerating warp
        dead = z <= 0.02
        z[dead], zp[dead] = 1.0, 1.0
        xy[dead] = rng.uniform(-1, 1, (int(dead.sum()), 2))
        p1 = (xy / z[:, None] * f_ + [W / 2, H / 2]).astype(int)
        p0 = (xy / zp[:, None] * f_ + [W / 2, H / 2]).astype(int)
        for a, b, zz in zip(p0, p1, z):
            v = int(min(255, 60 + 220 * (1 - zz)))
            cv2.line(frame, tuple(a), tuple(b), (v, v, v), 2 if zz < .3 else 1, cv2.LINE_AA)
        yield frame


def encode(frame_iter, path, W, H):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(C.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for fr in frame_iter:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")


def greenscreen(clip, kind, n, out):
    src = next(iter(sorted(glob.glob(f"assets/greenscreen/{kind}.*"))), None)
    if src is None:
        src = os.path.join(C.work, f"gs_{kind}.png")
        tmp = f"{src}.{os.getpid()}.png"
        comic.gs_frame(kind, C.W, C.H, C.accent).save(tmp)
        os.replace(tmp, src)
    still = src.lower().endswith((".png", ".jpg", ".jpeg"))
    if still:
        first = np.asarray(Image.open(src).convert("RGB").resize((C.W, C.H)))
    else:
        probe = os.path.join(C.work, f"gs_{kind}_{os.getpid()}.png")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-frames:v", "1", "-vf", f"scale={C.W}:{C.H}", probe])
        first = np.asarray(Image.open(probe).convert("RGB"))
    g = first.astype(int)
    ys, xs = np.nonzero((g[..., 1] > 180) & (g[..., 0] < 110) & (g[..., 2] < 110))
    if not len(xs):
        raise RuntimeError(f"green-screen asset {src} has no pure-green (#00FF00) area")
    x0, y0 = xs.min() // 2 * 2, ys.min() // 2 * 2
    bw, bh = (xs.max() + 2 - x0) // 2 * 2, (ys.max() + 2 - y0) // 2 * 2
    loop = ["-loop", "1", "-framerate", str(C.fps)] if still else ["-stream_loop", "-1"]
    fc = (f"[0:v]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},pad={C.W}:{C.H}:{x0}:{y0}:black[base];"
          f"[1:v]scale={C.W}:{C.H},fps={C.fps},format=yuva444p,chromakey=0x00FF00:0.16:0.06[key];"
          "[base][key]overlay=format=auto,format=yuv420p[v]")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, *loop, "-i", src, "-filter_complex", fc, "-map", "[v]",
         "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", out])


def render_scene(job):
    tl, path = job
    s = tl["s"]
    n = tl["F"] + tl["out"]
    seed = zlib.crc32(f"{C.P['video']['video_id']}{s['n']}".encode())
    if s["broll"] == "stars":
        encode(starfield(n, C.W, C.H, seed), path, C.W, C.H)
    elif s["broll"]:
        src = os.path.join("assets", "broll", s["broll"])
        if not os.path.exists(src):
            raise RuntimeError(f"scene {s['n']}: b-roll file {src} not found")
        run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", src, "-an", "-vf",
             f"scale={C.W}:{C.H}:force_original_aspect_ratio=increase,crop={C.W}:{C.H},fps={C.fps},format=yuv420p",
             "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", path])
    else:
        canvas, over, keys, reveal = build(s, tl, C.W, C.H)
        lead = tl["Tf"] / 2 / C.fps
        dst = path + ".pre.mp4" if s["template"].startswith("gs:") else path
        encode(frames(canvas, keys, n, C.W, C.H, over, s, lead, reveal, seed), dst, C.W, C.H)
        if dst != path:
            greenscreen(dst, s["template"][3:], n, path)
    return s["n"]


# ---------------- sound ----------------

def sfx_bank(seed=3):
    rng = np.random.default_rng(seed)

    def band(x, lo, hi):
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1 / SR)
        X[(f < lo) | (f > hi)] = 0
        return np.fft.irfft(X, len(x))

    def norm(x, peak):
        return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)

    n = int(.7 * SR); t = np.linspace(0, 1, n); x = rng.normal(0, 1, n)
    whoosh = norm((band(x, 150, 900) * (1 - t) + band(x, 900, 6000) * t * .7) * np.sin(np.pi * t ** .8) ** 2, .35)
    n = int(1.8 * SR); t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(35 + 55 * np.exp(-t * 6)) / SR) * np.exp(-t * 3.2)
    impact = norm(boom + band(rng.normal(0, 1, n), 200, 5000) * np.exp(-t * 40) * .5
                  + band(rng.normal(0, 1, n), 20, 160) * np.exp(-t * 2.2) * .6, .8)
    n = int(2.0 * SR); t = np.linspace(0, 1, n)
    riser = norm(band(rng.normal(0, 1, n), 2000, 9000) * t ** 2.5 * .6
                 + np.sin(2 * np.pi * np.cumsum(200 + 900 * t ** 2) / SR) * t ** 2 * .4, .4)
    n = int(3.0 * SR); t = np.linspace(0, 1, n)
    rumble = norm(band(rng.normal(0, 1, n), 20, 120) * np.sin(np.pi * t), .7)
    n = int(.35 * SR)
    tone = np.repeat(rng.uniform(200, 2000, 12), n // 12 + 1)[:n]
    glitchs = norm(np.sign(np.sin(2 * np.pi * tone * np.arange(n) / SR)) * (rng.random(n) > .3), .25)
    return dict(whoosh=whoosh, impact=impact, riser=riser, rumble=rumble, glitch=glitchs)


def place(dst, src, t, gain=1.0):
    i0 = int(t * SR)
    a, b = max(i0, 0), min(i0 + len(src), len(dst))
    if b > a:
        dst[a:b] += src[a - i0:b - i0] * gain


def tracks(tls, audiodir, total, fps):
    """Voice (24 kHz, as Kokoro made it) and SFX (48 kHz) timelines."""
    voice = np.zeros(int(total * 24000) + 1, np.float32)
    sfx = np.zeros(int(total * SR) + 1, np.float32)
    bank = sfx_bank()
    for tl in tls:
        s = tl["s"]
        if tl["speech"]:
            a, _ = sf.read(f"{audiodir}/{voice_key(s)}.wav", dtype="float32")
            i0 = int(tl["voice"] * 24000)
            voice[i0:i0 + len(a)] += a[:len(voice) - i0]
        if tl["Tf"] and XFADE[tl["tr"]][2]:
            place(sfx, bank["whoosh"], tl["start"] / fps - 0.2, 0.6)
        if s["sfx"]:
            place(sfx, bank[s["sfx"]], tl["voice"] - (len(bank["riser"]) / SR if s["sfx"] == "riser" else 0))
    return voice, sfx


def music_input(P, total, work):
    mood = str(P["video"].get("music_mood", "")).strip()
    ext = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
    files = [f for f in sorted(glob.glob(f"assets/music/{mood}/*") if mood else []) + sorted(glob.glob("assets/music/*"))
             if f.lower().endswith(ext)]
    if files:
        return ["-stream_loop", "-1", "-i", files[zlib.crc32(P["video"]["video_id"].encode()) % len(files)]]
    # shortcut: synthesized minor drone so a render never ships silent; drop real tracks in assets/music/<mood>/
    n = int(total * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(zlib.crc32(P["video"]["video_id"].encode()))
    root = 55 * 2 ** (int(rng.integers(0, 5)) / 12)
    y = np.zeros(n, np.float32)
    for mult, amp in ((1, .5), (1.5, .25), (2, .3), (2.4, .12), (3, .08)):
        for det in (-.15, .15):
            y += amp * np.sin(2 * np.pi * (root * mult + det) * t + rng.random() * 6.283)
    y *= .6 + .4 * np.sin(2 * np.pi * t / 23 + 1)
    path = os.path.join(work, "drone.wav")
    sf.write(path, y / (np.abs(y).max() + 1e-9) * .5, SR)
    return ["-i", path]


def mix(voice, sfx, total, out, P, work):
    S = P["settings"]
    vp, sp = os.path.join(work, "voice.wav"), os.path.join(work, "sfx.wav")
    sf.write(vp, voice, 24000)
    sf.write(sp, sfx, SR)
    pitch = float(S["voice_pitch"])
    rb = f"rubberband=pitch={2 ** (pitch / 12):.5f}:formant=preserved," if pitch else ""
    st = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    fc = (f"[0:a]aresample=48000,highpass=f=70,{rb}equalizer=f=140:t=q:w=1:g=2,equalizer=f=3200:t=q:w=1.4:g=2.5,"
          f"deesser=i=0.35,acompressor=threshold=0.08:ratio=3:attack=8:release=140:makeup=2,{st},asplit=2[v][key];"
          f"[2:a]{st},volume={float(S['music_db'])}dB,afade=t=in:d=2,afade=t=out:st={max(total - 4, 0):.2f}:d=4[m];"
          "[m][key]sidechaincompress=threshold=0.04:ratio=8:attack=30:release=500[duck];"
          f"[1:a]{st}[s];[v][duck][s]amix=inputs=3:normalize=0:duration=first,"
          f"loudnorm=I={float(S['lufs'])}:TP=-1.5:LRA=11,aresample=48000[a]")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", vp, "-i", sp, *music_input(P, total, work),
         "-filter_complex", fc, "-map", "[a]", "-t", f"{total:.3f}", "-ar", "48000", out])


# ---------------- text outputs ----------------

def stamp(t, srt=True):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"
    return f"{int(h)}:{int(m):02d}:{int(s):02d}" if h else f"{int(m)}:{int(s):02d}"


def cues(tls, max_words=7, max_chars=40):
    out = []
    for tl in tls:
        cur = []
        for w, a, b in tl["words"]:
            cur.append((w, tl["voice"] + a, tl["voice"] + b))
            text = " ".join(x[0] for x in cur)
            if len(cur) >= max_words or len(text) >= max_chars or w[-1] in ".?!;:,":
                out.append((cur[0][1], cur[-1][2], text))
                cur = []
        if cur:
            out.append((cur[0][1], cur[-1][2], " ".join(x[0] for x in cur)))
    return out


def ass_header(W, H, size, margin_v, align=2, fontname="Comic Neue", outline=3):
    return (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding\n"
            f"Style: Default,{fontname},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline},2,{align},60,60,{margin_v},1\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def ass_time(t):
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def chapters(tls, fps):
    out = []
    for tl in tls:
        name = tl["s"]["chapter"]
        if name and (not out or out[-1][1] != name):
            out.append((0.0 if not out else tl["start"] / fps, name))
    ok = len(out) >= 3 and all(b[0] - a[0] >= 10 for a, b in zip(out, out[1:]))  # YouTube's chapter rules
    return out if ok else []


def main(plan_path, imgdir, audiodir, outdir):
    P = json.load(open(plan_path))
    meta = json.load(open(os.path.join(audiodir, "words.json")))
    work = os.path.join(outdir, "_work")
    os.makedirs(work, exist_ok=True)
    init(P, imgdir, work)
    V, S, fps = P["video"], P["settings"], C.fps
    tls = timeline(P, meta, fps)
    clips = [os.path.join(work, f"clip{tl['i']:03d}.mp4") for tl in tls]
    with Pool(max(1, (os.cpu_count() or 2) // 2), initializer=init, initargs=(P, imgdir, work)) as pool:
        for done, n in enumerate(pool.imap_unordered(render_scene, zip(tls, clips)), 1):
            print(f"scene {n} rendered ({done}/{len(tls)})", flush=True)
    total = sum(tl["F"] for tl in tls) / fps
    voice, sfx = tracks(tls, audiodir, total, fps)
    mixed = os.path.join(work, "mix.wav")
    mix(voice, sfx, total, mixed, P, work)

    cs = cues(tls)
    with open(os.path.join(outdir, "captions.srt"), "w") as f:
        f.writelines(f"{i}\n{stamp(a)} --> {stamp(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cs, 1))
    burn = str(S["burn_subs"]).lower() in ("yes", "true", "1")
    if burn:
        subs = os.path.join(work, "subs.ass")
        with open(subs, "w") as f:
            f.write(ass_header(C.W, C.H, 54, 70) + "".join(
                f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Default,,0,0,0,,{t}\n" for a, b, t in cs))

    args = ["ffmpeg", "-y", "-loglevel", "error", "-stats"]
    for c in clips:
        args += ["-i", c]
    nv = len(clips)
    wm = os.path.join(work, "wm.png")
    comic.watermark(str(S["channel_name"]).upper()).save(wm)
    args += ["-i", wm, "-i", mixed]
    fc = [f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={fps},format=yuv420p[v{i}]" for i in range(nv)]
    prev, off = "[v0]", 0
    for i in range(1, nv):
        off += tls[i - 1]["F"]
        fc.append(f"{prev}[v{i}]xfade=transition={XFADE[tls[i]['tr']][0]}:duration={tls[i]['Tf'] / fps:.5f}"
                  f":offset={off / fps:.5f}[x{i}]")
        prev = f"[x{i}]"
    fc.append(f"{prev}{C.style.get('grade') or 'eq=contrast=1.04:saturation=1.08'},noise=c0s=5:allf=t,vignette=angle=PI/5[g]")
    last = "[g]"
    if str(S["watermark"]).lower() in ("yes", "true", "1"):
        fc.append(f"[g][{nv}:v]overlay=W-w-36:H-h-28[w]")
        last = "[w]"
    if burn:
        fc.append(f"{last}ass={subs}:fontsdir=assets/fonts[s]")
        last = "[s]"
    video = os.path.join(outdir, "video.mp4")
    run(args + ["-filter_complex", ";".join(fc), "-map", last, "-map", f"{nv + 1}:a", "-t", f"{total:.3f}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", video])

    thumb = "thumb.png" if any(j["name"] == "thumb.png" for j in P["images"]) else \
        next(s["images"][0] for s in P["scenes"] if s["images"])
    comic.thumbnail(C.image(thumb), V.get("thumbnail_text") or V["title"], C.accent, V.get("universe", ""), C.k).save(
        os.path.join(outdir, "thumbnail.jpg"), quality=92)
    chaps = chapters(tls, fps)
    desc = "\n".join([
        V.get("description", ""), "",
        *(["CHAPTERS", *[f"{stamp(t, False)} {name}" for t, name in chaps], ""] if chaps else []),
        *(["SOURCES", V["sources"], ""] if V.get("sources") else []),
        V.get("music_credit", ""), V.get("hashtags", "")])
    with open(os.path.join(outdir, "description.txt"), "w") as f:
        f.write(desc.strip() + "\n")
    json.dump(dict(video_id=V["video_id"], title=V["title"], description=desc.strip(),
                   tags=[t.strip() for t in V.get("tags", "").split(",") if t.strip()], publish_date=V.get("publish_date", ""),
                   duration=round(total, 2), chapters=[[stamp(t, False), n] for t, n in chaps]),
              open(os.path.join(outdir, "metadata.json"), "w"), indent=1)
    print(f"done: {video} ({total / 60:.1f} min)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
