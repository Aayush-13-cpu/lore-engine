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
import urllib.request
import zlib
from multiprocessing import Pool

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from pipeline import comic, sources
from pipeline.sheet import TRANSITIONS

SR = 48000
K = 1.25  # canvas is 25% larger than the frame: room for camera moves without upscaling
DRAFT = os.environ.get("DRAFT") == "1"
XFADE = {  # sheet name -> (ffmpeg xfade transition, seconds); transitions are silent by design
    "cut": ("fade", 0.0), "fade": ("fade", 0.5), "dissolve": ("dissolve", 0.6), "black": ("fadeblack", 0.9),
    "flash": ("fadewhite", 0.4), "whip": ("slideleft", 0.3), "glitch": ("pixelize", 0.35), "zoom": ("zoomin", 0.45),
    "circle": ("circleopen", 0.6), "wipe": ("wipeleft", 0.45), "slide": ("slideup", 0.4), "radial": ("radial", 0.6),
}
DEPTH_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
DEPTH = None  # per-process ONNX session, loaded on first use
PARALLAX = {"full", "page", "issue"}  # templates whose picture gets depth layers (gs:* too)
PAUSE = {"?": 1.5, "!": 1.4, ".": 1.2, ";": 0.8, ":": 0.8, ",": 0.45}  # breath after a line, by how it ends
assert set(XFADE) | {"auto"} == TRANSITIONS
AUTO_T = ["cut", "fade", "cut", "dissolve"]  # cuts and soft fades only: whips and zooms every scene read amateur
AUTO_M = ["in", "right", "out", "left", "in", "up", "out", "down"]
INTENT = {"threat": "in", "revelation": "in", "artifact": "in", "power": "in", "tragedy": "out", "cosmic": "right",
          "multiverse": "left", "origin": "up", "history": "right", "tech": "in"}  # beat -> the move that means it
FLIP = {"in": "out", "out": "in", "left": "right", "right": "left", "up": "down", "down": "up"}
FOCUS = {"artifact", "revelation", "tragedy"}  # beats that get a rack focus onto the subject
AUTO_FX = {"cosmic": "stars", "multiverse": "stars", "threat": "embers", "tragedy": "fog", "power": "rays",
           "revelation": "rays", "history": "film", "tech": "hud", "origin": "dust"}  # beat -> its one effect
MOOD = {  # beat -> (B, G, R gain, saturation, black lift): gentle, on top of the style's grade
    "cosmic": ((1.06, 1.0, .95), 1.05, 0), "multiverse": ((1.06, 1.0, .95), 1.05, 0),
    "threat": ((.95, .97, 1.07), 1.0, 0), "tragedy": ((1.03, 1.0, .98), .72, 5), "origin": ((.94, 1.0, 1.06), 1.0, 4),
    "history": ((.86, .97, 1.06), .7, 6), "tech": ((1.05, 1.04, .94), .95, 0), "power": ((1.0, 1.0, 1.0), 1.1, 0),
}
PRESETS = {  # keyframes (t 0..1, zoom, focus x, focus y): slow and small, like a rostrum camera
    "in": [(0, 1.0, .5, .5), (1, 1.06, .5, .48)], "out": [(0, 1.06, .5, .48), (1, 1.0, .5, .5)],
    "left": [(0, 1.06, .54, .5), (1, 1.06, .46, .5)], "right": [(0, 1.06, .46, .5), (1, 1.06, .54, .5)],
    "up": [(0, 1.06, .5, .54), (1, 1.06, .5, .46)], "down": [(0, 1.06, .5, .46), (1, 1.06, .5, .54)],
    "static": [(0, 1.02, .5, .5), (1, 1.03, .5, .5)], "shake": [(0, 1.08, .5, .5), (1, 1.1, .5, .5)],
}
CARDS = {"title", "sting", "quote", "endscreen"}
ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX".split()
C = None  # per-process render context


def fetch(url, path):
    """Download once. Honest user agent (Wikimedia & co. reject fake browsers); errors survive the worker pool."""
    if not os.path.exists(path):
        ua = f"lore-engine/1.0 (+https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'lore-engine')})"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=120) as r:
                data = r.read()
        except Exception as e:
            raise RuntimeError(f"could not download {url}: {e} (put the file in assets/ or use another link)") from None
        with open(f"{path}.{os.getpid()}", "wb") as f:
            f.write(data)
        os.replace(f"{path}.{os.getpid()}", path)
    return path


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
        self.pages = {}

    def page(self, spec):
        """Real comic page from assets/pages/ or a web link, plus its panel boxes in reading order."""
        src = spec.split("#")[0].split("@")[0]
        if src not in self.pages:
            path = os.path.join(self.imgdir, self.P.get("pages", {}).get(src, "-"))  # Real-ESRGAN copy from the runners
            if not os.path.exists(path):  # drafts skip the image runners: use the original page
                path = sources.page_image(src, self.P.get("sources", {}), self.work)
            img = Image.open(path).convert("RGB")
            small = img.copy()
            small.thumbnail((1024, 1024))  # gutter peaks for to_gutters(); full-size masks would eat RAM per worker
            self.pages[src] = (img, panel_boxes(img), gutter_nets(np.asarray(small.convert("L"))))
        return self.pages[src]

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


def gutter_nets(g):
    """The gutters + margins: one connected network of not-black paper (white or yellowed newsprint, stray colour).
    Only the black plate counts: colour plates printed off-register smear into gutters on old comics. Returned
    thickened by t px across rows / across columns (a slightly skewed gutter still crosses a cell), and as is."""
    h, w = g.shape
    k = max(3, round(w * 0.003)) | 1  # mends 1-2 px breaks in a scanned border line; gutters are far wider
    ink = cv2.morphologyEx((g < 110).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    _, lab, st, _ = cv2.connectedComponentsWithStats(1 - ink, connectivity=4)
    net = np.isin(lab, [i for i in range(1, len(st)) if st[i, 2] >= 0.5 * w and st[i, 3] >= 0.5 * h]).astype(np.uint8)
    t = max(3, round(0.012 * max(h, w))) | 1
    return cv2.dilate(net, np.ones((t, 1), np.uint8)) > 0, cv2.dilate(net, np.ones((1, t), np.uint8)) > 0, net > 0


def to_gutters(rect, nets, pw, ph):
    """Slide a hand-framed box's edges onto the nearest gutters, for pages whose panels could not be separated
    (a balloon crossing a gutter, staggered layouts): a broken gutter still peaks far above the panels around it."""
    rows_net, cols_net = nets[:2]
    h, w = rows_net.shape
    sx, sy = w / pw, h / ph

    def nearest(prof, v, lo, reach):  # centre of the strongest gutter peak within reach of v (nearest if tied)
        a, b = max(0, v - reach), min(len(prof), v + reach + 1)
        if b <= a or prof[a:b].max() < lo:
            return v
        top = np.nonzero(prof[a:b] >= prof[a:b].max() - 0.05)[0] + a
        i = j = int(top[np.argmin(abs(top - v))])
        while i > a and prof[i - 1] >= prof[i] - 0.05 and prof[i - 1] >= lo:
            i -= 1
        while j + 1 < b and prof[j + 1] >= prof[j] - 0.05 and prof[j + 1] >= lo:
            j += 1
        return (i + j) // 2

    x0, y0, x1, y1 = (round(rect[0] * sx), round(rect[1] * sy), round(rect[2] * sx), round(rect[3] * sy))
    rows = rows_net[:, x0:max(x1, x0 + 1)].mean(axis=1)
    y0 = nearest(rows, y0, 0.35, round(0.08 * h)) if y0 > 0.02 * h else y0  # the page's own edge needs no snapping
    y1 = nearest(rows, y1, 0.35, round(0.08 * h)) if y1 < 0.98 * h else y1
    cols = cols_net[y0:max(y1, y0 + 1)].mean(axis=0)
    x0 = nearest(cols, x0, 0.5, round(0.12 * w)) if x0 > 0.02 * w else x0
    x1 = nearest(cols, x1, 0.5, round(0.12 * w)) if x1 < 0.98 * w else x1
    return x0 / sx, y0 / sy, x1 / sx, y1 / sy


def panel_boxes(img):
    """Panels in western reading order. The gutters + margins form one connected network of not-black paper
    (white or yellowed newsprint, stray colour); the page is cut recursively (rows, then columns) along bands that network covers.
    The paper-coloured inside of a caption or balloon is enclosed by its border, so it is never taken for a gutter,
    and a lenient coverage threshold copes with slightly skewed scans. Each cell is then tightened to its art."""
    g = np.asarray(img.convert("L"))
    h, w = g.shape
    nets = gutter_nets(g)
    t = max(3, round(0.012 * max(h, w))) | 1  # skew tolerance: a slanted gutter only needs to pass within t/2 px
    net = nets[2]
    out = []

    def cut(x0, y0, x1, y1, axis, flipped):  # axis 0: split on horizontal gutters (rows), 1: vertical (columns)
        prof = nets[axis][y0:y1, x0:x1].mean(axis=1 - axis) >= 0.9  # gutter = crosses the whole cell
        spans, i, size = [], 0, (h, w)[axis]
        while i < len(prof):
            j = i
            while j < len(prof) and not prof[j]:
                j += 1
            if j - i >= 0.03 * size:  # ignore slivers: specks, page numbers
                spans.append((i, j))
            i = j + 1
        lim = (y1 - y0, x1 - x0)[axis]
        spans = [(max(i - t // 2, 0), min(j + t // 2, lim)) for i, j in spans]  # undo the thickening
        boxes = [(x0, y0 + i, x1, y0 + j) if axis == 0 else (x0 + i, y0, x0 + j, y1) for i, j in spans]
        if len(boxes) > 1 or (boxes and not flipped):
            for b in boxes:
                cut(*b, 1 - axis, len(boxes) == 1)
        elif boxes:
            bx0, by0, bx1, by1 = boxes[0]
            ys, xs = np.nonzero(~net[by0:by1, bx0:bx1])  # tighten the cell to its art (keeps a skewed panel whole)
            if len(xs) and (bx1 - bx0) * (by1 - by0) > 0.02 * w * h:
                out.append((bx0 + int(xs.min()), by0 + int(ys.min()), bx0 + int(xs.max()) + 1, by0 + int(ys.max()) + 1))

    cut(0, 0, w, h, 0, False)
    dark = g < 110
    for n, (x0, y0, x1, y1) in enumerate(out):  # a borderless caption printed on the paper just above its panel
        roof = max([o[3] for o in out if o[3] <= y0 and o[0] < x1 and o[2] > x0], default=0)
        top, gap = y0, 0
        for y in range(y0 - 1, max(roof, y0 - round(0.08 * h)), -1):
            if dark[y, x0:x1].sum() >= 3:
                top, gap = y, 0
            elif (gap := gap + 1) > 0.012 * h:  # a text line gap is ~0.5%; anything wider ends the caption
                break
        if top < y0:
            out[n] = (x0, max(top - round(0.003 * h), roof), x1, y1)
    return out or [(0, 0, w, h)]


def snap(rect, boxes, edges=lambda r: r):
    """A hand-framed page box -> whole panels, so a crop never slices one:
    - a zoom well inside one big panel (a splash page, a close-up) stays, clipped to that panel;
    - when less than half the box is detected art (the detector missed a title or text block), the box stays,
      grown over every panel it cuts into;
    - otherwise the panels at least 60% inside the box, or else the one it overlaps most."""
    x0, y0, x1, y1 = rect
    area = lambda b: max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    union = lambda bs: (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
    clip = lambda b: (max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1))
    for b in boxes:
        if area(clip(b)) >= 0.6 * area(rect) and area(rect) <= 0.6 * area(b):  # margins make up the rest
            x0, y0, x1, y1 = rect = edges(rect)  # a page the detector could not split: at least land on gutters
            return clip(b)
    if sum(area(clip(b)) for b in boxes) < 0.5 * area(rect):
        x0, y0, x1, y1 = rect = edges(rect)
        return union([rect] + [b for b in boxes if area(clip(b)) >= 0.3 * area(b)])
    share = {b: area(clip(b)) / area(b) for b in boxes}
    best = max(share.values())
    return union([b for b in boxes if share[b] >= 0.6] or [b for b in boxes if share[b] >= best - 0.05])


def desk(w, h):
    """Dark warm tabletop under a soft lamp (drawn small, scaled up: it is only a backdrop)."""
    sw, sh = max(w // 4, 8), max(h // 4, 8)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    lum = 40 - 42 * np.hypot((xx - sw / 2) / sw, (yy - sh / 2) / sh) + np.random.default_rng(3).normal(0, 2.5, (sh, sw))
    lum = np.clip(lum, 6, 60)[..., None] * np.float32([1.0, 0.92, 0.84])
    return Image.fromarray(lum.astype(np.uint8)).resize((w, h), Image.BILINEAR)


def page_focus(spec, boxes, pw, ph, nets=None):
    """'#3' = third panel, '@x0,y0,x1,y1' = manual box as page fractions, nothing = whole page."""
    if "@" in spec:
        x0, y0, x1, y1 = map(float, spec.split("@")[1].split(","))
        return snap((x0 * pw, y0 * ph, x1 * pw, y1 * ph), boxes, (lambda r: to_gutters(r, nets, pw, ph)) if nets else (lambda r: r))
    if "#" in spec:
        n = int(spec.split("#")[1])
        if not 1 <= n <= len(boxes):
            raise RuntimeError(f"page {spec}: only {len(boxes)} panels detected; frame it by hand with @x0,y0,x1,y1")
        return boxes[n - 1]
    return 0, 0, pw, ph


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


def bars(style):
    return str(style.get("letterbox", "")).strip().lower() in ("yes", "true", "1")


def voice_key(s):
    return s.get("voice_key") or f"s{s['n']:03d}"  # Shorts hook lines carry their own audio name


def page_file(s):
    return s.get("page", "").split("#")[0].split("@")[0]


def timeline(P, meta, fps, pad=None):
    pad = float(P["settings"]["scene_pad"]) if pad is None else pad
    sc = P["scenes"]
    out, start, chapter, titles = [], 0, None, 0
    for i, s in enumerate(sc):
        prev = sc[i - 1] if i else None
        tr = s["transition"]
        if tr == "auto":
            same_page = prev and page_file(s) and page_file(s) == page_file(prev)  # camera slides on, no cut visible
            new_page = prev and page_file(s) and page_file(prev) and not same_page
            tr = "cut" if i == 0 or same_page else "black" if s["chapter"] != chapter or s["template"] in ("title", "sting") \
                else "whip" if new_page else AUTO_T[i % len(AUTO_T)]  # whip between pages reads as a page turn
        Tf = 0 if i == 0 else max(1, round(XFADE[tr][1] * fps))
        m = meta.get(voice_key(s))
        speech = m["dur"] if m else 0.0
        text = s["narration"].rstrip()
        breath = 0.5 if text.endswith(("...", "…")) else PAUSE.get(text[-1:], 1.0)
        if i + 1 < len(sc) and sc[i + 1]["chapter"] != s["chapter"]:
            breath += 1.5  # let a chapter land before the next one starts
        body = (speech + pad * breath + s["hold"]) if speech else (s["hold"] or 2.5)
        F = max(round((Tf / 2 / fps + body) * fps), Tf + 1)
        titles += s["template"] == "title"
        out.append(dict(i=i, s=s, prev=prev, tr=tr, Tf=Tf, F=F, start=start, voice=(start + Tf / 2) / fps, speech=speech,
                        words=(m or {}).get("words") or spread(s["narration"], speech), chapter_no=max(titles, 1)))
        start += F
        chapter = s["chapter"]
    for a, b in zip(out, out[1:]):
        a["out"] = b["Tf"]
    out[-1]["out"] = 0
    if str(P["settings"].get("auto_sfx", "auto")).lower() != "off":
        letter(out, fps)
    if str(P["settings"].get("auto_text", "auto")).lower() != "off":
        texts(out, P, fps, float(P["settings"].get("text_gap", 15)))
    mood(out, P["settings"])
    return out


def mood(tls, S):
    """Each AI line's main beat sets its grade (s["mood"]) and, at most every fx_gap s, its one effect, unless the
    sheet chose an effect itself."""
    last = -1e9
    for tl in tls:
        s = tl["s"]
        beats = [b for b in s.get("beats") or [] if b]
        if s["template"] != "full" or s.get("page") or s["broll"] or not beats:
            continue
        beat = max(beats, key=beats.count)
        s = dict(s, mood=beat if str(S.get("mood_grade", "auto")).lower() != "off" else "")
        if not s["fx"] and AUTO_FX.get(beat) and str(S.get("auto_fx", "auto")).lower() != "off" \
                and tl["voice"] - last >= float(S.get("fx_gap", 12)):
            s["fx"], last = AUTO_FX[beat], tl["voice"]
        tl["s"] = s


IMPACT = {"snap": "SNAP!", "crash": "CRASH!", "slam": "SLAM!", "smash": "SMASH!", "explod": "BOOM!", "shatter": "KRASH!",
          "punch": "POW!", "roar": "ROAR!"}  # physical sounds only: "Devourer" or "Big Bang" are names, not impacts


def letter(tls, fps, gap=10.0):
    """tl["hit"] = (lettering, s into the clip): the sheet's sfx_text on its own word, else an impact verb in the
    line (at most one per `gap` s). Lettering, shake and the impact sound all land on that word."""
    last = -gap
    for tl in tls:
        s = tl["s"]
        if s["template"] != "full" or s.get("page"):  # never letter over a real comic page
            continue
        own = re.sub(r"[^a-z]", "", s["sfx_text"].lower())[:3]
        for w, a, _ in tl["words"]:
            k = re.sub(r"[^a-z]", "", w.lower())
            auto = next((v for stem, v in IMPACT.items() if k.startswith(stem)), "")
            if (own and (k.startswith(own) or auto)) or (not own and auto and tl["voice"] + a - last >= gap):
                tl["hit"] = (s["sfx_text"] or auto, tl["Tf"] / 2 / fps + a)
                last = tl["voice"] + a
                break


STOP = {"he", "she", "it", "they", "we", "i", "you", "him", "her", "them", "his", "its", "their", "was", "is", "were",
        "this", "that"}  # "He tricked some." or "Keeping it was." mean nothing on screen without the line around them


def name_at(words, name):
    """Index of the first word that is part of a character's name (any capitalised part: Surfer, Thanos, Death)."""
    parts = {re.sub(r"[^A-Za-z]", "", x) for x in str(name).split() if x[:1].isupper() and len(x) > 2}
    return next((i for i, w in enumerate(words) if re.sub(r"[^A-Za-z]", "", w[0]) in parts), None)


def level(ch):
    """A character's threat_level 1-10 from the sheet, 0 = none given."""
    try:
        return min(max(int(float(ch.get("threat_level") or 0)), 0), 10)
    except (TypeError, ValueError):
        return 0


def texts(tls, P, fps, gap=15.0):
    """tl["texts"] = [(kind, text, sub, in, out)] in s into the clip: a name plate on a character's first mention in the
    video, else the narration's own short punch sentences ("Time. Space. Reality.") in big type; one beat per gap s."""
    seen, met, pairs, last = set(), set(), set(), -gap
    for tl in tls:
        s, w = tl["s"], tl["words"]
        if s["template"] != "full" or s.get("page") or s["broll"] or not w:
            continue
        named = sorted((k, c) for c, ch in P["characters"].items() for k in [name_at(w, ch.get("name", ""))] if k is not None)
        lead, L, out = tl["Tf"] / 2 / fps, (tl["F"] + tl["out"]) / fps, []
        for c, ch in P["characters"].items():
            k = name_at(w, ch.get("name", ""))
            fa = str(ch.get("first_appearance", "") or "").strip()
            sub = " · ".join(x for x in (str(ch.get("alias", "") or "").strip(), fa and f"First appearance: {fa}") if x)
            if c in seen or k is None or not sub:
                continue
            if not out and tl["voice"] + w[k][1] - last >= gap:  # too soon after another beat: wait for the next mention
                out.append(("name", str(ch["name"]).strip(), sub, lead + w[k][1], min(lead + w[k][1] + 3.5, L)))
                last = tl["voice"] + w[k][1]
                seen.add(c)
        both = [(k, c) for k, c in named if c in met and level(P["characters"][c])]
        if not out and len(both) >= 2 and (both[0][1], both[1][1]) not in pairs \
                and tl["voice"] + w[both[1][0]][1] - last >= gap:  # two known characters measured against each other
            (_, a), (k, b) = both[:2]
            pairs |= {(a, b), (b, a)}
            ca, cb = P["characters"][a], P["characters"][b]
            out.append(("versus", f"{ca['name']}|{cb['name']}", f"{level(ca)}|{level(cb)}", lead + w[k][1],
                        min(lead + w[k][1] + 4.0, L)))
            last = tl["voice"] + w[k][1]
        met |= {c for _, c in named}
        yr = next((i for i, (x, _, _) in enumerate(w) if re.fullmatch(r"(19|20)\d\d[.,;:!?]?", x)), None)
        if not out and yr is not None and tl["voice"] + w[yr][1] - last >= gap:  # a year the narration gives
            out.append(("year", w[yr][0].strip(".,;:!?"), "", lead + w[yr][1], min(lead + w[yr][1] + 3.0, L)))
            last = tl["voice"] + w[yr][1]
        if not out and not tl.get("hit") and not s["caption"] and tl["voice"] - last >= gap:
            sents, cur = [], []
            for i, (x, _, _) in enumerate(w):
                cur.append(i)
                if x[-1:] in ".!?":
                    sents, cur = sents + [cur], []
            punchy = [len(q) <= 4 and not {re.sub(r"[^a-z]", "", w[i][0].lower()) for i in q} & STOP for q in sents]
            run = []
            for q, ok in zip(sents, punchy):  # the first run of punch sentences in the line is one beat
                if ok:
                    run.append(q)
                elif run:
                    break
            for q, nxt in zip(run, run[1:] + [None]):
                end = lead + w[nxt[0]][1] if nxt else min(lead + w[q[-1]][2] + 0.8, L)
                out.append(("punch", " ".join(w[i][0] for i in q), "", lead + w[q[0]][1], end))
            if run:
                last = tl["voice"] + w[run[0][0]][1]
        if out:
            tl["texts"] = out


# ---------------- pictures ----------------

def prep(img, x, y, t_in, pop=False, t_out=None):
    """RGBA overlay -> (float BGRA cropped to its visible box, x, y, appear time, pop, leave time or None)."""
    box = img.getchannel("A").getbbox()
    if not box:
        return None
    img = img.crop(box)
    a = np.asarray(img, np.float32)
    return np.dstack([a[..., 2::-1], a[..., 3:] / 255]), x + box[0], y + box[1], t_in, pop, t_out


def blend(frame, ov, tau):
    img, x, y, t_in, pop, t_out = ov
    a = min((tau - t_in) / (0.35 if pop in ("left", "right") else 0.22), 1.0)
    if t_out is not None:
        a = min(a, (t_out - tau) / 0.2)
    if a <= 0:
        return
    if pop in ("left", "right"):  # a green-screen character slides in from that side, easing out
        x = int(x + (-1 if pop == "left" else 1) * frame.shape[1] * .14 * (1 - a) ** 3)
    elif pop and a < 1:  # SFX lettering slams in from 135%
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
        return [(0, 1.0, .5, .5), (min((lead + .3) / L, .5), 1.08, .5, .48), (1, 1.1, .5, .47)]
    return PRESETS[m]


def moves(tl):
    """Camera move per picture of an AI line: the beat's meaning (push in on a threat, pull out on a loss, pan across
    scale), else the rotation; never the same move twice in a row. [] = the template/sheet decides (motion_keys)."""
    s = tl["s"]
    if s["template"] != "full" or s.get("page") or s["broll"] or s["motion"] != "auto":
        return []
    beats, out = s.get("beats") or [""], []
    for p in range(max(1, len(s["images"]))):
        m = INTENT.get(beats[min(p, len(beats) - 1)]) or AUTO_M[(tl["i"] + p) % len(AUTO_M)]
        out.append(FLIP.get(m, m) if out and m == out[-1] else m)
    return out


def scene_cuts(tl):
    """Cut times (s into the clip) between a line's pictures: on the word where each picture's part of the sentence
    starts (the sheet's splits), else an even share; [] = one picture (cards, pages, b-roll, single shots)."""
    s = tl["s"]
    m = len(s["images"]) if s["template"] == "full" and not s.get("page") and not s["broll"] else 1
    if m < 2:
        return []
    L, lead, w, at = (tl["F"] + tl["out"]) / C.fps, tl["Tf"] / 2 / C.fps, tl["words"], s.get("splits") or []
    if len(at) == m - 1 and len(w) == len(s["narration"].split()):
        return [lead + (w[i - 1][2] + w[i][1]) / 2 for i in at]
    return [L * k / m for k in range(1, m)]


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


def ambient(kind, W, H, seed):
    """Procedural effect layers, drawn on the picture under the text: light rays, drifting fog, old film, HUD scan."""
    rng = np.random.default_rng(seed)
    h4, w4 = max(H // 4, 1), max(W // 4, 1)
    if kind == "rays":
        yy, xx = np.mgrid[0:h4, 0:w4].astype(np.float32)
        sx, sy = w4 * rng.uniform(.25, .75), -h4 * .3
        ang, fall = np.arctan2(yy - sy, xx - sx), np.exp(-np.hypot(xx - sx, yy - sy) / h4 * .9)

        def draw(frame, tau):
            m = (.5 + .5 * np.sin(ang * 23 + tau * .35)) ** 6 * (.5 + .5 * np.sin(ang * 9 - tau * .2)) * fall * 120
            m = cv2.resize(m, (W, H))
            cv2.add(frame, np.dstack([m * .75, m * .9, m]).astype(np.uint8), dst=frame)  # warm, additive
    elif kind == "fog":
        a, b = (cv2.normalize(cv2.GaussianBlur(rng.random((h4, w4 * 2)).astype(np.float32), (0, 0), s), None, 0, 1,
                              cv2.NORM_MINMAX) for s in (18, 30))

        def draw(frame, tau):
            m = (np.roll(a, -int(tau * 6), 1)[:, :w4] * .6 + np.roll(b, int(tau * 3), 1)[:, :w4] * .4) ** 2 * .35
            m = cv2.resize(m, (W, H))[..., None]
            frame[:] = (frame * (1 - m) + np.float32([150, 150, 155]) * m).astype(np.uint8)
    elif kind == "film":
        def draw(frame, tau):
            r = np.random.default_rng(seed + int(tau * 24))  # changes 24 times a second, like a projector
            cv2.convertScaleAbs(frame, frame, 1 + r.uniform(-.03, .03))
            for _ in range(int(r.integers(0, 3))):
                x = int(r.integers(0, W))
                cv2.line(frame, (x, 0), (x + int(r.integers(-20, 20)), H), (205, 205, 205), 1, cv2.LINE_AA)
            for _ in range(int(r.integers(0, 6))):
                cv2.circle(frame, (int(r.integers(0, W)), int(r.integers(0, H))), int(r.integers(1, 4)), (25, 25, 25), -1)
    else:  # hud
        def draw(frame, tau):
            layer = np.zeros_like(frame)
            y = int(tau * .25 % 1 * H)
            cv2.rectangle(layer, (0, y), (W, y + 3), (120, 110, 0), -1)
            layer[::4] += 5
            mx, my, k = round(W * .05), round(H * .05), 60
            for cx, cy, dx, dy in ((mx, my, 1, 1), (W - mx, my, -1, 1), (mx, H - my, 1, -1), (W - mx, H - my, -1, -1)):
                cv2.line(layer, (cx, cy), (cx + dx * k, cy), (140, 130, 0), 2)
                cv2.line(layer, (cx, cy), (cx, cy + dy * k), (140, 130, 0), 2)
            cv2.add(frame, layer, dst=frame)
    return draw


def grader(beat):
    """The beat's gentle grade as a function on BGR frames, or None."""
    if beat not in MOOD:
        return None
    gains, sat, lift = MOOD[beat]
    x = np.arange(256, dtype=np.float32)
    lut = np.stack([np.clip(lift + x * g * (255 - lift) / 255, 0, 255) for g in gains], 1).astype(np.uint8)[:, None]

    def f(frame):
        out = cv2.LUT(frame, lut)
        if sat != 1:
            out = cv2.addWeighted(out, sat, cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 1 - sat, 0)
        return out
    return f


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
    vis = 0.76 if bars(st) and W > H else 1.0  # share of the frame height the letterbox bars leave visible
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
    elif tpl == "issue":  # the real cover on the desk, details beside it: caption "FANTASTIC FOUR #48|MARCH 1966|LEE & KIRBY"
        cover = C.page(s["page"])[0]
        canvas = desk(cw, ch)
        cover = cover.resize((round(ch * .8 * vis * cover.width / cover.height), round(ch * .8 * vis)), Image.LANCZOS)
        x, y = round(cw * .12), round(ch * (1 - .8 * vis) / 2)
        shadow = Image.new("L", canvas.size, 0)
        shadow.paste(170, (x + 14, y + 18, x + cover.width + 14, y + cover.height + 18))
        canvas.paste((0, 0, 0), mask=shadow.filter(ImageFilter.GaussianBlur(14)))
        canvas.paste(cover, (x, y))
        parts = (s["caption"].split("|") + ["", "", ""])[:3]
        if W >= H:
            over.append(prep(comic.card_text("issue", W, H, acc, title=parts[0], kicker=parts[1], text=parts[2]), 0, 0, lead + 0.3))
        keys = [(0, 1.0, .5, .5), (1, 1.07, .44, .5)]
    elif tpl == "page":
        img, boxes, nets = C.page(s["page"])
        pw, ph = img.size
        cw = max(pw * 1.06, ph * 1.06 * W / (H * vis))  # board in the frame's aspect, whole page clear of the bars
        ch = cw * H / W
        ox, oy = (cw - pw) / 2, (ch - ph) / 2
        canvas = desk(round(cw), round(ch))  # the page lies on a lamp-lit desk with a soft shadow, like a real book
        shadow = Image.new("L", canvas.size, 0)
        shadow.paste(175, (round(ox + 12), round(oy + 16), round(ox + pw + 12), round(oy + ph + 16)))
        canvas.paste((0, 0, 0), mask=shadow.filter(ImageFilter.GaussianBlur(max(8, pw // 60))))
        canvas.paste(img, (round(ox), round(oy)))

        def frame_on(rect, push=1.0):  # camera key that fits a page rectangle with a little air around it
            x0, y0, x1, y1 = rect
            vw = max((x1 - x0) * 1.08, (y1 - y0) * 1.08 * W / (H * vis))  # fits between the letterbox bars
            return max(cw / vw, 1.0) * push, (ox + (x0 + x1) / 2) / cw, (oy + (y0 + y1) / 2) / ch

        prev = tl.get("prev") or {}
        same = prev.get("template") == "page" and page_file(prev) == page_file(s)
        start = frame_on(page_focus(prev["page"], boxes, pw, ph, nets)) if same else frame_on((0, 0, pw, ph))
        target = page_focus(s["page"], boxes, pw, ph, nets)
        if target != (0, 0, pw, ph):  # spotlight the panel being explained: dim the page, outline the panel
            box = tuple(round(v) for v in (ox + target[0], oy + target[1], ox + target[2], oy + target[3]))
            lit = canvas.crop(box)
            canvas = ImageEnhance.Brightness(canvas).enhance(0.42)
            canvas.paste(lit, box[:2])
            ImageDraw.Draw(canvas).rectangle(box, outline=comic.rgb(acc), width=5)
        keys = [(0, *start), (min(0.9 / L, 0.4), *frame_on(target)), (1, *frame_on(target, 1.04))]
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
    if captions and s["caption"] and tpl not in CARDS | {"split", "issue"}:
        over.append(prep(comic.caption_box(s["caption"], st.get("caption_bg") or "#F6E27A", st.get("caption_fg") or "#111111"),
                         60, 160 if bars(st) else 50, lead + 0.3))  # clear of the letterbox bar
    plates = [x for x in tl.get("texts", []) if x[0] == "name"] if W >= H else []
    if s.get("callout") and W >= H:  # green screen: the named character slides in on their name
        k = name_at(tl["words"], C.P["characters"].get(s["callout"], {}).get("name", ""))
        at = lead + (tl["words"][k][1] if k is not None else 0.5)
        fig = cutout(f"cut_{s['callout']}.png", canvas, round(H * .8 * vis))
        if fig is not None:
            left = s["n"] % 2 == 1 and not plates  # the name plate sits bottom-left
            fig = fig if left else fig.transpose(Image.FLIP_LEFT_RIGHT)  # alternate sides face both ways
            over.append(prep(fig, round(W * .03) if left else round(W * .97) - fig.width,
                             round(H * (1 + vis) / 2) - round(fig.height * .94), at, "left" if left else "right"))
    for kind, text, sub, a, b in tl.get("texts", []) if W >= H else []:
        if kind == "name":
            im = comic.name_plate(text, sub, acc)
            x0 = round(W * .05)  # inside the 5% title-safe margin
            over.append(prep(im, x0, round(H * .76) - im.height, a, t_out=b))
            ch = next((c for c in C.P["characters"].values() if str(c.get("name", "")).strip() == text), {})
            for i, im in enumerate(comic.threat_bar(level(ch), acc) if level(ch) else []):  # frame, then cell by cell
                over.append(prep(im, x0, round(H * .76) + 14, a + .4 + .07 * i, t_out=b))
        elif kind == "versus":
            (na, nb), (la, lb) = text.split("|"), sub.split("|")
            im = comic.versus(na, nb, int(la), int(lb), acc)
            over.append(prep(im, (W - im.width) // 2, round(H * .9) - im.height, a, t_out=b))
        elif kind == "year":
            im = comic.year_tag(text, acc)
            over.append(prep(im, round(W * .95) - im.width, round(H * .08), a, t_out=b))
        else:
            im = comic.punch_text(text, maxw=W - 200)
            over.append(prep(im, (W - im.width) // 2, round(H * .1), a, pop=True, t_out=b))
    text, at = tl.get("hit") or (s["sfx_text"], lead)
    if text:
        im = comic.sfx_word(text.upper(), st.get("sfx_color") or "#FFE14D")
        over.append(prep(im, W - im.width - 70 if W >= H else (W - im.width) // 2, int(H * .52 - im.height / 2), at, pop=True))
    if keys is None:
        keys = motion_keys(s, tl["i"], L, lead)
    return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR), [o for o in over if o], keys, reveal


def cutout(name, scene, height):
    """Green-screen character -> RGBA figure, lightly colour-matched to the scene, with an ink outline and a contact
    shadow so it sits in the art; None when the key fails (a draft placeholder, or a background that isn't green)."""
    if not os.path.exists(os.path.join(C.imgdir, name)):
        return None
    rgb = np.asarray(Image.open(os.path.join(C.imgdir, name)).convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] > 40) & (hsv[..., 0] < 85) & (hsv[..., 1] > 90) & (hsv[..., 2] > 70)).astype(np.uint8)
    _, lab = cv2.connectedComponents(green)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge[edge > 0])  # only green joined to the border: a green cape inside the ink line stays...
    bg |= (hsv[..., 0] > 50) & (hsv[..., 0] < 70) & (hsv[..., 1] > 200) & (hsv[..., 2] > 200)  # ...but pure chroma
    # green trapped between an arm and the body is background too
    alpha = cv2.erode((~bg).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    if not .02 <= alpha.mean() / 255 <= .95:
        return None
    near = cv2.dilate(bg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)  # despill the rim only
    rgb = rgb.copy()
    rgb[..., 1] = np.where(near, np.minimum(rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2])), rgb[..., 1])
    f = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(cv2.resize(np.asarray(scene.convert("RGB")), (256, 144)), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    m = alpha > 128
    fm, fs, rm, rs = f[m].mean(0), f[m].std(0) + 1e-6, r.mean(0), r.std(0) + 1e-6
    f += np.float32([.35, .2, .2]) * ((f - fm) / fs * rs + rm - f)  # a light pull towards the scene's light and palette
    rgb = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    pad = 24
    rgb, alpha = (np.pad(x, ((pad, pad), (pad, pad)) + ((0, 0),) * (x.ndim - 2)) for x in (rgb, alpha))
    a = cv2.GaussianBlur(alpha, (3, 3), 0).astype(np.float32)[..., None] / 255
    ink = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)[..., None] / 255
    shadow = np.roll(cv2.GaussianBlur(ink[..., 0], (0, 0), 7), (14, 10), (0, 1))[..., None] * .35
    out_a = a + ink * (1 - a)
    out_a = out_a + shadow * (1 - out_a)
    col = (rgb * a + np.float32([12, 12, 14]) * ink * (1 - a)) / np.maximum(a + ink * (1 - a), 1e-6)
    fig = Image.fromarray(np.dstack([col, out_a * 255]).astype(np.uint8), "RGBA")
    fig = fig.crop(fig.getchannel("A").getbbox())  # the figure itself sets the height, not the empty green around it
    return fig.resize((round(fig.width * height / fig.height), height), Image.LANCZOS)


def depth_of(canvas):
    """Near=1 / far=0 depth at 1/4 canvas size from Depth-Anything-V2-Small (Apache-2.0, ONNX on CPU)."""
    global DEPTH
    if DEPTH is None:
        import onnxruntime
        DEPTH = onnxruntime.InferenceSession(fetch(DEPTH_URL, os.path.join(C.work, "depth.onnx")),
                                             providers=["CPUExecutionProvider"])
    x = cv2.resize(canvas, (518, 518), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    x = ((x - np.float32([.485, .456, .406])) / np.float32([.229, .224, .225])).transpose(2, 0, 1)[None]
    d = DEPTH.run(None, {DEPTH.get_inputs()[0].name: x.astype(np.float32)})[0].squeeze().astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    h, w = canvas.shape[:2]
    return cv2.GaussianBlur(cv2.resize(d, (max(w // 4, 1), max(h // 4, 1))), (0, 0), 2)


def frames(canvas, keys, n, W, H, overlays=(), s=None, lead=0.0, reveal=None, seed=0, depth=None, t0=0.0, focus=False):
    """t0 = where this shot starts inside its scene: captions, SFX and shake keep scene time across cuts.
    focus: rack focus from the background onto the nearest subject over the first 40% of the shot (needs depth)."""
    fps = C.fps
    ch_, cw_ = canvas.shape[:2]
    base = max(W / cw_, H / ch_)
    if depth is not None:  # 2.5D parallax: near layers drift against far ones, opposite to the camera's pan
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        amp = float(C.S.get("parallax", 14)) * (-1 if interp(keys, 1)[1] > interp(keys, 0)[1] else 1)
    fx = s["fx"] if s else ""
    parts = particles(fx, W, H, seed) if fx in ("embers", "dust", "snow", "rain", "stars") else \
        ambient(fx, W, H, seed) if fx in ("rays", "fog", "film", "hud") else None
    grade = grader(s.get("mood")) if s else None
    shake = shaker(s, lead, seed) if s else None
    rng = np.random.default_rng(seed)
    mips = [canvas]  # half-size copies: big zoom-outs (whole comic pages) sample these instead of shimmering
    while mips[-1].shape[1] > 2.2 * W:
        mips.append(cv2.pyrDown(mips[-1]))
    for f in range(n):
        tau, t = t0 + f / fps, f / max(n - 1, 1)
        z, fx_, fy_ = interp(keys, t)
        sc = base * z
        vw, vh = W / sc, H / sc
        cx = min(max(fx_ * cw_, vw / 2), cw_ - vw / 2)
        cy = min(max(fy_ * ch_, vh / 2), ch_ - vh / 2)
        lvl = 0
        while not reveal and lvl + 1 < len(mips) and sc * 2 ** lvl < 0.55:
            lvl += 1
        src, k = (reveal(tau), 1.0) if reveal else (mips[lvl], 2.0 ** lvl)
        dx, dy = shake(tau) if shake else (0.0, 0.0)
        M = np.float32([[sc * k, 0, W / 2 - sc * cx + dx], [0, sc * k, H / 2 - sc * cy + dy]])
        frame = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if depth is not None:
            Md = np.float32([[sc * 4, 0, W / 2 - sc * cx + dx], [0, sc * 4, H / 2 - sc * cy + dy]])
            near = cv2.warpAffine(depth, Md, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101) - 0.5
            shift = amp * (2 * t - 1)
            frame = cv2.remap(frame, gx + shift * near, gy + 0.3 * shift * near, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT101)
            if focus:
                f0 = -0.2 + 0.55 * min(t / 0.4, 1.0) ** 2 * (3 - 2 * min(t / 0.4, 1.0))  # eased: far -> subject
                m = np.clip((np.abs(near - f0) - 0.12) * 2.5, 0, 0.8)[..., None]
                frame = (frame * (1 - m) + cv2.GaussianBlur(frame, (0, 0), 5) * m).astype(np.uint8)
        if grade:  # the picture only: text and lettering keep their colours
            frame = grade(frame)
        under = fx in ("rays", "fog", "film", "hud")
        if parts and under:
            parts(frame, tau)
        for ov in overlays:
            blend(frame, ov, tau)
        if parts and not under:
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
        lead = tl["Tf"] / 2 / C.fps
        shots = [(s, tl, n, lead)]
        if s.get("page") and s["template"] == "full" and n >= 2 * C.fps:
            # both: the real panel first (the evidence), then a hard cut to the dramatized AI shot
            n1 = min(max(round(n * 0.45), C.fps), n - C.fps)
            evidence = dict(s, template="page", caption="", sfx_text="", fx="", sfx="")
            shots = [(evidence, dict(tl, F=n1, out=0), n1, lead),
                     (dict(s, sfx=""), dict(tl, Tf=0, F=n - n1, out=0, prev=None), n - n1, 0.0)]

        cuts = scene_cuts(tl)

        def shot_frames():
            for s2, tl2, count, ld in shots:
                pics = s2["images"] if cuts and len(shots) == 1 else s2["images"][:1]
                sets = [build(dict(s2, images=[p]), tl2, C.W, C.H) for p in pics] or [build(s2, tl2, C.W, C.H)]
                over, reveal = sets[0][1], sets[0][3]
                deep = float(C.S.get("parallax", 14)) and (s2["template"] in PARALLAX or s2["template"].startswith("gs:"))
                mv = moves(tl) if len(shots) == 1 else []
                keys = [PRESETS[mv[p]] if mv else sets[0][2] if p == 0 else PRESETS[AUTO_M[(tl["i"] + p) % len(AUTO_M)]]
                        for p in range(len(sets))]
                beats = s2.get("beats") or [""]
                rack = str(C.S.get("rack_focus", "auto")).lower() != "off"
                edges = [0] + [round(t * C.fps) for t in cuts[:len(sets) - 1]] + [count]
                if tl2.get("hit"):  # shake (and a sheet flash) on the lettered word, not at the line start
                    s2, ld = dict(s2, sfx=s2["sfx"] or "impact"), tl2["hit"][1]
                for p, (a, b) in enumerate(zip(edges, edges[1:])):
                    yield from frames(sets[p][0], keys[p], b - a, C.W, C.H, over, s2, ld, reveal, seed,
                                      depth_of(sets[p][0]) if deep else None, a / C.fps,
                                      rack and beats[min(p, len(beats) - 1)] in FOCUS)
        dst = path + ".pre.mp4" if s["template"].startswith("gs:") else path
        encode(shot_frames(), dst, C.W, C.H)
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
    """Voice (24 kHz, as the TTS made it) and SFX (48 kHz) timelines. Transitions stay silent."""
    voice = np.zeros(int(total * 24000) + 1, np.float32)
    sfx = np.zeros(int(total * SR) + 1, np.float32)
    bank = sfx_bank()
    for tl in tls:
        s = tl["s"]
        if tl["speech"]:
            a, _ = sf.read(f"{audiodir}/{voice_key(s)}.wav", dtype="float32")
            i0 = int(tl["voice"] * 24000)
            voice[i0:i0 + len(a)] += a[:len(voice) - i0]
        if s["sfx"]:
            at = tl["start"] / fps + tl["hit"][1] if tl.get("hit") and s["sfx"] == "impact" else tl["voice"]
            place(sfx, bank[s["sfx"]], at - (len(bank["riser"]) / SR if s["sfx"] == "riser" else 0))
    return voice, sfx


def music_input(P, work):
    """Your BGM: Videos.music_url (keeps licensed tracks out of the public repo) or assets/music/<mood>/. None = no bed."""
    url = str(P["video"].get("music_url", "")).strip()
    if url:
        return fetch(url, os.path.join(work, f"music_{zlib.crc32(url.encode())}"))
    mood = str(P["video"].get("music_mood", "")).strip()
    ext = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
    files = [f for f in sorted(glob.glob(f"assets/music/{mood}/*") if mood else []) + sorted(glob.glob("assets/music/*"))
             if f.lower().endswith(ext)]
    if not files:
        print("WARNING: no background music (set Videos.music_url or add files to assets/music/)", flush=True)
        return None
    return files[zlib.crc32(P["video"]["video_id"].encode()) % len(files)]


def music_bed(path, total):
    """Inputs + filter that repeat a short track with 6 s crossfades so the loop point is never a hard cut."""
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True, check=True).stdout)
    n = max(1, math.ceil((total - 6) / max(dur - 6, 1)))
    chain, label = "", "[2:a]"
    for k in range(1, n):
        chain += f"{label}[{2 + k}:a]acrossfade=d=6:c1=tri:c2=tri[mx{k}];"
        label = f"[mx{k}]"
    return [arg for _ in range(n) for arg in ("-i", path)], chain, label


def mix(voice, sfx, total, out, P, work):
    S = P["settings"]
    vp, sp = os.path.join(work, "voice.wav"), os.path.join(work, "sfx.wav")
    sf.write(vp, voice, 24000)
    sf.write(sp, sfx, SR)
    path = music_input(P, work)
    music, chain, bed = music_bed(path, total) if path else ([], "", "")
    st = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    # light touch keeps the narrator human: rumble cut, gentle 2:1 compression, soft de-ess, no pitch tricks
    fc = (f"[0:a]aresample=48000,highpass=f=60,acompressor=threshold=0.125:ratio=2:attack=15:release=200:makeup=1.5,"
          f"deesser=i=0.2,{st}" + (",asplit=2[v][key];" if music else "[v];") + f"[1:a]{st}[s];")
    if music:
        fc += (f"{chain}{bed}{st},volume={float(S['music_db'])}dB,afade=t=in:d=2,afade=t=out:st={max(total - 4, 0):.2f}:d=4[m];"
               "[m][key]sidechaincompress=threshold=0.05:ratio=6:attack=40:release=600[duck];"
               "[v][duck][s]amix=inputs=3:normalize=0:duration=first,")
    else:
        fc += "[v][s]amix=inputs=2:normalize=0:duration=first,"
    ln = f"loudnorm=I={float(S['lufs'])}:TP=-1.5:LRA=11"
    inputs = ["-i", vp, "-i", sp, *music]
    # two-pass: measure the whole mix, then correct it linearly, so every video lands on the exact target
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-filter_complex", fc + ln + ":print_format=json[a]",
                            "-map", "[a]", "-t", f"{total:.3f}", "-f", "null", "-"], capture_output=True, text=True)
    m = json.loads(probe.stderr[probe.stderr.rindex("{"):probe.stderr.rindex("}") + 1]) if probe.returncode == 0 else {}
    if m and float(m["input_i"]) > -70:  # silence (a draft with no voice) can't be measured: fall back to one pass
        ln += (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
               f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-filter_complex", fc + ln + ",aresample=48000[a]", "-map", "[a]", "-t", f"{total:.3f}", "-ar", "48000", out])


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


def my_clips(todo, shard):
    """'K/N': this runner's share of the clips, longest first onto the least-loaded runner (same split on every runner)."""
    k, n = map(int, shard.split("/"))
    load, mine = [0] * n, []
    for tl, c in sorted(todo, key=lambda x: (-x[0]["F"], x[0]["i"])):
        j = load.index(min(load))
        load[j] += tl["F"]
        if j == k:
            mine.append((tl, c))
    return mine


def main(plan_path, imgdir, audiodir, outdir, shard=""):
    P = json.load(open(plan_path))
    meta = json.load(open(os.path.join(audiodir, "words.json")))
    work = os.path.join(outdir, "_work")
    os.makedirs(work, exist_ok=True)
    init(P, imgdir, work)
    V, S, fps = P["video"], P["settings"], C.fps
    tls = timeline(P, meta, fps)
    clips = [os.path.join(work, f"clip{tl['i']:03d}.mp4") for tl in tls]
    todo = [(tl, c) for tl, c in zip(tls, clips) if not os.path.exists(c)]  # clips from parallel runners are kept
    if shard:
        todo = my_clips(todo, shard)
    for tl, _ in todo:  # fetch + analyse each comic page once here, not in every worker
        if tl["s"].get("page"):
            C.page(tl["s"]["page"])
    if float(S.get("parallax", 14)) and todo:  # download the depth model once, before workers race for it
        fetch(DEPTH_URL, os.path.join(work, "depth.onnx"))
    with Pool(max(1, (os.cpu_count() or 2) // 2), initializer=init, initargs=(P, imgdir, work)) as pool:
        for done, n in enumerate(pool.imap_unordered(render_scene, todo), 1):
            print(f"scene {n} rendered ({done}/{len(todo)})", flush=True)
    if shard:
        return print(f"clip shard {shard}: {len(todo)} clips")
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
        args += ["-threads", "1", "-i", c]  # one decoder per scene: keep each small or RAM grows with video length
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
    fc.append(f"{prev}{C.style.get('grade') or 'eq=contrast=1.04:saturation=1.08'},vignette=angle=PI/5"
              + (",drawbox=y=0:w=iw:h=ih*0.12:t=fill:c=black,drawbox=y=ih*0.88:w=iw:h=ih*0.12:t=fill:c=black" if bars(C.style) else "")
              + "[g]")  # letterbox bars = 2.39:1 cinema frame for movie-style videos
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
        next((s["images"][0] for s in P["scenes"] if s["images"]), None)
    if thumb:
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


def _selfcheck():
    page = Image.new("RGB", (1000, 1500), "white")  # wide top panel, tall left, two stacked right
    for x0, y0, x1, y1 in [(40, 40, 960, 500), (40, 540, 480, 1460), (520, 540, 960, 980), (520, 1020, 960, 1460)]:
        page.paste((60, 60, 60), (x0, y0, x1, y1))
    got = [b[:2] for b in panel_boxes(page)]
    assert got == [(40, 40), (40, 540), (520, 540), (520, 1020)], got
    B = panel_boxes(page)
    assert snap((0, 0, 1000, 555), B)[1::2] == (40, 500)  # a tier box past a gutter -> just the top panel
    assert snap((0, 0, 1000, 1500), [(0, 0, 100, 100)]) == (0, 0, 1000, 1500)  # detector missed the art: keep box
    assert snap((0, 0, 1000, 700), [(0, 0, 100, 100), (0, 600, 300, 900)]) == (0, 0, 1000, 900)  # never slice one
    assert snap((100, 100, 300, 300), [(0, 0, 1000, 1000)]) == (100, 100, 300, 300)  # a zoom inside a splash
    assert snap((0, 450, 1000, 1000), B) == (520, 540, 960, 980)  # a tall panel only half inside stays out
    rn = np.zeros((1000, 1000), bool)  # to_gutters: row gutters at 300 and 700, a weaker band just above 700
    rn[295:305] = rn[690:710] = True
    rn[650:690, :400] = True
    cn = np.zeros((1000, 1000), bool)
    cn[700:, 490:510] = True
    got = to_gutters((0, 320, 1000, 1000), (rn, cn), 1000, 1000)
    assert abs(got[1] - 300) <= 2, got
    got = to_gutters((560, 730, 1000, 1000), (rn, cn), 1000, 1000)
    assert abs(got[0] - 500) <= 2 and abs(got[1] - 700) <= 2, got  # the strongest peak, not the weak band
    global C
    C, old = type("C", (), dict(fps=30))(), C
    w5 = [(x, i * .5, i * .5 + .4) for i, x in enumerate("With one snap of his fingers, Thanos erased half".split())]
    s5 = dict(template="full", images=["a", "b"], splits=[6], narration=" ".join(x for x, *_ in w5), broll="")
    assert scene_cuts(dict(s=s5, F=150, out=0, Tf=0, words=w5)) == [(2.9 + 3.0) / 2]  # the cut lands on "Thanos"
    assert scene_cuts(dict(s=dict(s5, splits=[]), F=150, out=0, Tf=0, words=w5)) == [2.5]  # no splits: even share
    assert scene_cuts(dict(s=dict(s5, images=["a"]), F=150, out=0, Tf=0, words=w5)) == []
    m5 = dict(s5, images=["a", "b", "c"], beats=["threat", "threat", ""], motion="auto", page="")
    assert moves(dict(s=m5, i=1)) == ["in", "out", "left"]  # threat pushes in, a 2nd threat pulls out, plain rotates
    assert moves(dict(s=dict(m5, motion="static"), i=1)) == []  # the sheet's own motion wins
    C = old
    P = dict(characters=dict(THANOS=dict(name="Thanos", alias="The Mad Titan", first_appearance="Iron Man #55 (1973)")))
    mk = lambda text, v: dict(s=dict(template="full", caption="", broll="", narration=text), Tf=0, F=300, out=0, voice=v,
                              words=[(x, i * .5, i * .5 + .4) for i, x in enumerate(text.split())])
    tls = [mk("Then Thanos came for the stones.", 0), mk("Time. Space. Reality.", 5), mk("He tricked some. Power. Mind.", 30)]
    texts(tls, P, 30)
    assert tls[0]["texts"][0][:3] == ("name", "Thanos", "The Mad Titan · First appearance: Iron Man #55 (1973)"), tls[0]
    assert "texts" not in tls[1]  # within 15 s of the name plate: one beat at a time
    assert [x[1] for x in tls[2]["texts"]] == ["Power.", "Mind."] and tls[2]["texts"][0][4] == 1.5 + 0.5, tls[2]
    tls = [mk("Time.", 0), mk("Then Thanos came.", 3), mk("Thanos smiled at last.", 20)]
    texts(tls, P, 30)
    assert "texts" not in tls[1] and tls[2]["texts"][0][:2] == ("name", "Thanos")  # a blocked plate waits, not lost
    P["characters"]["THANOS"]["threat_level"] = "10"
    P["characters"]["DEATH"] = dict(name="Mistress Death", alias="", threat_level=9)
    tls = [mk("Thanos met Death.", 0), mk("Then Death faced Thanos at last.", 20), mk("It was 1991, and nobody knew.", 40)]
    texts(tls, P, 30)
    assert tls[1]["texts"][0][:3] == ("versus", "Mistress Death|Thanos", "9|10"), tls[1]
    assert tls[2]["texts"][0][:2] == ("year", "1991") and level({"threat_level": "x"}) == 0
    S = dict(fx_gap=12)
    sc = lambda beats, fx="": dict(template="full", page="", broll="", fx=fx, beats=beats)
    tls = [dict(s=sc(["", "cosmic", "cosmic"]), voice=0), dict(s=sc(["threat"]), voice=5),
           dict(s=sc(["tragedy"], "snow"), voice=20), dict(s=sc([""]), voice=40)]
    mood(tls, S)
    assert [tl["s"]["fx"] for tl in tls] == ["stars", "", "snow", ""] and tls[1]["s"]["mood"] == "threat"  # 12 s apart
    fr = np.full((8, 8, 3), 128, np.uint8)
    assert grader("tragedy")(fr).std() < 3 and grader("") is None
    for k in ("rays", "fog", "film", "hud"):
        g = np.full((72, 128, 3), 60, np.uint8)
        ambient(k, 128, 72, 1)(g, 1.0)  # draws in place without errors
    tl = lambda text, v: dict(s=dict(template="full", sfx_text=text), Tf=6, voice=v, words=[("He", 0, .2), ("snapped", .3, .7)])
    tls = [tl("", 0), tl("", 4), tl("", 12), tl("POW", 30)]
    letter(tls, 30)
    assert tls[0]["hit"] == ("SNAP!", 0.1 + .3) and "hit" not in tls[1] and tls[2]["hit"][0] == "SNAP!", tls  # 1 per 10 s
    assert tls[3]["hit"] == ("POW", 0.4)  # the sheet's own lettering wins and lands on the impact word


if __name__ == "__main__":
    _selfcheck()
    main(*sys.argv[1:6])
