"""Comic-book graphics with Pillow: print pass, lettering, page layouts, cards, green-screen frames, thumbnails."""
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")
FONT_FILES = {"title": "Bangers-Regular.ttf", "bold": "Anton-Regular.ttf", "letter": "ComicNeue-Bold.ttf"}
GREEN = (0, 255, 0)
INK = (12, 12, 14)


def font(kind, size):
    return ImageFont.truetype(os.path.join(FONTS, FONT_FILES[kind]), int(size))


def rgb(h):
    h = str(h).lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def wrap(text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if f.getlength(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def cover(img, w, h):
    return ImageOps.fit(img.convert("RGB"), (int(w), int(h)), Image.LANCZOS)


def darken(img, w, h, k=0.45, blur=6):
    return ImageEnhance.Brightness(cover(img, w, h).filter(ImageFilter.GaussianBlur(blur))).enhance(k)


def print_pass(img, k=0.5, cell=10.0):
    """Unify every render into one printed-comic look: halftone ink in the shadows, warm paper, crisp edges."""
    if k <= 0:
        return img
    a = np.asarray(img.convert("RGB"), np.float32) / 255
    h, w = a.shape[:2]
    lum = a @ np.float32([0.299, 0.587, 0.114])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + yy) / (cell * 1.4142), (xx - yy) / (cell * 1.4142)  # 45-degree screen like real print
    dist = np.hypot(u - np.round(u), v - np.round(v))
    dots = dist < 0.62 * np.sqrt(np.clip(0.72 - lum, 0, 1) / 0.72)
    a *= 1 - (0.3 * k) * dots[..., None]
    a = a * (1 - 0.22 * k) + a * np.float32([1.0, 0.972, 0.925]) * 0.22 * k
    a += np.random.default_rng(7).normal(0, 0.016 * k, (h, w, 1)).astype(np.float32)
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(70 * k), threshold=2))


def placeholder(text, w, h, seed=0):
    """Draft-mode stand-in so timing and script can be reviewed before paying for image generation."""
    rng = np.random.default_rng(seed)
    c1, c2 = rng.integers(20, 90, 3), rng.integers(90, 200, 3)
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    a = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).astype(np.uint8)
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    f = font("letter", h / 24)
    d.text((w * 0.05, h * 0.06), "DRAFT", font=font("title", h / 8), fill=(255, 255, 255))
    for i, line in enumerate(wrap(text[:420], f, w * 0.88)):
        d.text((w * 0.05, h * 0.25 + i * f.size * 1.2), line, font=f, fill=(240, 240, 240))
    return img


# ---------- lettering overlays (RGBA, placed in screen space) ----------

def caption_box(text, bg="#F6E27A", fg="#111111", maxw=820):
    f = font("letter", 54)
    lines = wrap(text.upper(), f, maxw - 48)
    width = int(max(f.getlength(line) for line in lines)) + 48  # hug the lettering like a real caption box
    lh = int(f.size * 1.2)
    h = lh * len(lines) + 36
    im = Image.new("RGBA", (width + 10, h + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, width + 9, h + 9], fill=(0, 0, 0, 150))
    d.rectangle([0, 0, width, h], fill=rgb(bg) + (255,), outline=INK, width=4)
    for i, line in enumerate(lines):
        d.text((24, 18 + i * lh), line, font=f, fill=rgb(fg))
    return im


def sfx_word(text, color="#FFE14D", size=200):
    f = font("title", size)
    w, h = int(f.getlength(text)) + 80, int(size * 1.5)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((48, 28), text, font=f, fill=(0, 0, 0, 200), stroke_width=12, stroke_fill=(0, 0, 0, 200))
    d.text((36, 16), text, font=f, fill=rgb(color), stroke_width=10, stroke_fill=INK)
    return im.rotate(8, expand=True, resample=Image.BICUBIC)


def name_plate(name, sub, accent="#E62429"):
    """Lower-third tag on a character's first mention: name in comic type over alias and first appearance."""
    f1, f2 = font("title", 92), font("letter", 36)
    name = re.sub(r"^THE ", "", name.upper())
    w = int(max(f1.getlength(name), f2.getlength(sub))) + 70
    im = Image.new("RGBA", (w + 10, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, w + 9, 169], fill=(0, 0, 0, 150))
    d.rectangle([0, 0, w, 159], fill=INK + (235,), outline=rgb(accent), width=3)
    d.rectangle([0, 0, 14, 159], fill=rgb(accent) + (255,))
    d.text((38, 4), name, font=f1, fill=(255, 255, 255))
    d.text((40, 108), sub, font=f2, fill=rgb("#F6E27A"))
    return im


def punch_text(text, color="#FFFFFF", size=130, maxw=1700):
    """The narration's short punch sentence ("Every star.") in big inked type, shrunk to fit the frame."""
    text = text.upper().rstrip(".")  # "SPACE" reads as lettering; ! and ? stay
    while size > 60 and font("title", size).getlength(text) > maxw - 60:
        size = int(size * .9)
    f = font("title", size)
    im = Image.new("RGBA", (int(f.getlength(text)) + 60, int(size * 1.35)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((38, 22), text, font=f, fill=(0, 0, 0, 170), stroke_width=10, stroke_fill=(0, 0, 0, 170))
    d.text((28, 12), text, font=f, fill=rgb(color), stroke_width=8, stroke_fill=INK)
    return im


def watermark(text):
    f = font("bold", 34)
    im = Image.new("RGBA", (int(f.getlength(text)) + 8, 52), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4, 2), text, font=f, fill=(255, 255, 255, 120))
    return im


def text_layer(w, h, draw_fn):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(im))
    return im


# ---------- canvases (RGB, larger than the frame so the camera can move) ----------

def panels(imgs, cw, ch, k):
    """Comic page: returns paper base, panel tiles [(img, (x, y))] and panel centres (0..1) for the camera."""
    m, g, n = 40, 30, len(imgs)
    if n == 2:
        rects = [(m, m, cw * .58 - g / 2, ch - m), (cw * .58 + g / 2, m, cw - m, ch - m)]
    elif n == 3:
        rects = [(m, m, cw - m, ch * .52 - g / 2), (m, ch * .52 + g / 2, cw * .45 - g / 2, ch - m),
                 (cw * .45 + g / 2, ch * .52 + g / 2, cw - m, ch - m)]
    else:
        imgs = imgs[:4]
        rects = [(m, m, cw / 2 - g / 2, ch / 2 - g / 2), (cw / 2 + g / 2, m, cw - m, ch / 2 - g / 2),
                 (m, ch / 2 + g / 2, cw / 2 - g / 2, ch - m), (cw / 2 + g / 2, ch / 2 + g / 2, cw - m, ch - m)][:len(imgs)]
    base = print_pass(Image.new("RGB", (cw, ch), (238, 229, 206)), 1.0)
    tiles, centres = [], []
    for img, (x0, y0, x1, y1) in zip(imgs, rects):
        tile = ImageOps.expand(print_pass(cover(img, x1 - x0 - 12, y1 - y0 - 12), k), border=6, fill=INK)
        tiles.append((tile, (int(x0), int(y0))))
        centres.append(((x0 + x1) / 2 / cw, (y0 + y1) / 2 / ch))
    return base, tiles, centres


def split(a, b, labels, accent, cw, ch, k):
    c = Image.new("RGB", (cw, ch), INK)
    half = cw // 2
    c.paste(print_pass(cover(a, half - 10, ch), k), (0, 0))
    c.paste(print_pass(cover(b, half - 10, ch), k), (half + 10, 0))
    d = ImageDraw.Draw(c)
    d.ellipse([half - 90, ch / 2 - 90, half + 90, ch / 2 + 90], fill=rgb(accent), outline=INK, width=8)
    d.text((half, ch / 2), "VS", font=font("title", 120), fill="white", anchor="mm", stroke_width=6, stroke_fill=INK)
    f = font("bold", 64)
    for x, lab in zip((half / 2, half * 1.5), labels):
        if lab:
            d.text((x, ch * 0.88), lab.upper(), font=f, fill="white", anchor="mm", stroke_width=6, stroke_fill=INK)
    return c


def dossier(ref, char, accent, universe, cw, ch, k):
    c = Image.new("RGB", (cw, ch), (14, 16, 22))
    d = ImageDraw.Draw(c)
    for x in range(0, cw, 60):
        d.line([(x, 0), (x, ch)], fill=(24, 27, 36), width=1)
    for y in range(0, ch, 60):
        d.line([(0, y), (cw, y)], fill=(24, 27, 36), width=1)
    d.polygon([(0, 0), (cw * .36, 0), (cw * .26, ch), (0, ch)], fill=tuple(int(v * .35) for v in rgb(accent)))
    photo = ImageOps.expand(print_pass(cover(ref, ch * .58, ch * .74), k), border=16, fill=(235, 232, 222))
    photo = photo.convert("RGBA").rotate(3, expand=True, resample=Image.BICUBIC)
    shadow = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    shadow.putalpha(photo.getchannel("A").point(lambda v: v * 150 // 255))
    px, py = int(cw * .07), int(ch * .12)
    c.paste(shadow, (px + 18, py + 18), shadow)
    c.paste(photo, (px, py), photo)
    stamp = text_layer(560, 160, lambda dd: (dd.rectangle([4, 4, 556, 156], outline=(200, 30, 30, 230), width=8),
                                             dd.text((280, 82), "CLASSIFIED", font=font("bold", 96), fill=(200, 30, 30, 230), anchor="mm")))
    stamp = stamp.rotate(-12, expand=True, resample=Image.BICUBIC)
    c.paste(stamp, (int(cw * .12), int(ch * .70)), stamp)
    x0, y = cw * .47, ch * .12
    d.text((x0, y), f"CHARACTER FILE  //  {universe.upper()}", font=font("bold", 44), fill=rgb(accent))
    d.text((x0, y + 60), str(char.get("name", "")).upper(), font=font("title", 150), fill="white", stroke_width=4, stroke_fill=INK)
    y += 250
    if char.get("alias"):
        d.text((x0, y - 30), str(char["alias"]), font=font("letter", 46), fill=(200, 200, 205))
        y += 50
    for label, key in (("FIRST APPEARANCE", "first_appearance"), ("CREATED BY", "creators"), ("POWERS", "powers")):
        val = str(char.get(key, "")).strip()
        if not val:
            continue
        d.text((x0, y), label, font=font("bold", 34), fill=rgb(accent))
        y += 48
        for line in wrap(val, font("letter", 44), cw * .48)[:3]:
            d.text((x0, y), line, font=font("letter", 44), fill="white")
            y += 54
        y += 22
    level = max(0, min(10, int(float(char.get("threat_level") or 0))))
    d.text((x0, y), "THREAT LEVEL", font=font("bold", 34), fill=rgb(accent))
    for i in range(10):
        bx = x0 + i * 70
        d.rectangle([bx, y + 52, bx + 58, y + 86], fill=rgb(accent) if i < level else None, outline=(110, 110, 120), width=3)
    return c


def card_text(kind, w, h, accent, title="", kicker="", text="", attribution=""):
    """Screen-space text for title / sting / quote / endscreen cards (fades in over the moving backdrop)."""
    def draw(d):
        if kind == "title":
            d.text((w / 2, h * .36), kicker.upper(), font=font("bold", 64), fill=rgb(accent), anchor="mm")
            lines = wrap(title.upper(), font("title", 190), w * .86)
            for i, line in enumerate(lines):
                d.text((w / 2, h * .52 + i * 180), line, font=font("title", 190), fill="white", anchor="mm",
                       stroke_width=7, stroke_fill=INK)
            yb = h * .52 + len(lines) * 180 - 40
            d.rectangle([w / 2 - 160, yb, w / 2 + 160, yb + 12], fill=rgb(accent))
        elif kind == "sting":
            f = font("title", 260)
            d.text((w / 2 + 14, h / 2 + 14), title.upper(), font=f, fill=rgb(accent), anchor="mm")
            d.text((w / 2, h / 2), title.upper(), font=f, fill="white", anchor="mm", stroke_width=8, stroke_fill=INK)
            if kicker:
                d.text((w / 2, h * .70), kicker.upper(), font=font("bold", 52), fill=(230, 230, 230), anchor="mm")
        elif kind == "quote":
            f = font("letter", 76)
            lines = wrap(text, f, w * .72)
            y0 = h / 2 - len(lines) * 92 / 2
            d.text((w * .11, y0 - 60), "“", font=font("title", 300), fill=rgb(accent), anchor="mm")
            for i, line in enumerate(lines):
                d.text((w / 2, y0 + i * 92), line, font=f, fill="white", anchor="mm", stroke_width=3, stroke_fill=INK)
            if attribution:
                d.text((w / 2, y0 + len(lines) * 92 + 50), "— " + attribution.upper(), font=font("bold", 48),
                       fill=rgb(accent), anchor="mm")
        elif kind == "issue":  # right of the cover: title, date, creators
            x0, f = w * .56, font("title", 120)
            lines = wrap(title.upper(), f, w * .4)
            for i, line in enumerate(lines):
                d.text((x0, h * .3 + i * 120), line, font=f, fill="white", stroke_width=5, stroke_fill=INK)
            y = h * .3 + len(lines) * 120 + 20
            d.rectangle([x0, y, x0 + 220, y + 10], fill=rgb(accent))
            d.text((x0, y + 40), kicker.upper(), font=font("bold", 60), fill=rgb(accent))
            for i, line in enumerate(wrap(text, font("letter", 48), w * .4)):
                d.text((x0, y + 130 + i * 58), line, font=font("letter", 48), fill=(230, 230, 230))
        elif kind == "endscreen":  # guide boxes sit where YouTube end-screen elements go
            d.text((w / 2, h * .16), title.upper() or "WATCH NEXT", font=font("title", 130), fill="white", anchor="mm",
                   stroke_width=6, stroke_fill=INK)
            d.rounded_rectangle([w * .47, h * .30, w * .92, h * .30 + w * .45 * 9 / 16], 18, outline=rgb(accent), width=8)
            r = w * .085
            cx, cy = w * .24, h * .53
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgb(accent), width=8)
            d.text((cx, cy + r + 50), "SUBSCRIBE", font=font("bold", 56), fill="white", anchor="mm")
    return text_layer(w, h, draw)


def gs_frame(kind, w, h, accent="#E62429"):
    """Built-in green-screen frames; the pure-green hole is keyed out by ffmpeg chromakey."""
    im = Image.new("RGB", (w, h), (10, 10, 14))
    d = ImageDraw.Draw(im)
    if kind == "tv":
        for y in range(h):
            v = int(14 + 26 * (1 - abs(y / h - .5) * 2))
            d.line([(0, y), (w, y)], fill=(v, v - 4, v - 8))
        d.rounded_rectangle([w * .14, h * .08, w * .86, h * .92], 60, fill=(58, 44, 34), outline=INK, width=10)
        d.rounded_rectangle([w * .19, h * .15, w * .70, h * .85], 80, fill=GREEN)
        for i in range(3):
            cy = h * (.25 + i * .18)
            d.ellipse([w * .75, cy, w * .81, cy + w * .06], fill=(30, 30, 30), outline=(150, 150, 150), width=5)
    elif kind == "monitor":
        d.rectangle([0, 0, w, h], fill=(18, 20, 26))
        d.rounded_rectangle([w * .09, h * .06, w * .91, h * .82], 24, fill=(8, 8, 10), outline=(70, 72, 80), width=6)
        d.rectangle([w * .11, h * .09, w * .89, h * .79], fill=GREEN)
        d.polygon([(w * .45, h * .82), (w * .55, h * .82), (w * .59, h * .95), (w * .41, h * .95)], fill=(50, 52, 58))
    elif kind == "page":
        d.rectangle([0, 0, w, h], fill=(238, 229, 206))
        d.rectangle([40, 40, w * .66, h - 40], fill=GREEN, outline=INK, width=8)
        for y0, y1 in ((40, h / 2 - 15), (h / 2 + 15, h - 40)):
            d.rectangle([w * .66 + 30, y0, w - 40, y1], fill=tuple(int(v * .8) for v in rgb(accent)), outline=INK, width=8)
        im = print_pass(im, 1.0)
        d = ImageDraw.Draw(im)
        d.rectangle([46, 46, w * .66 - 6, h - 46], fill=GREEN)  # re-punch: print noise must not leak into the key
    elif kind == "file":
        d.rectangle([0, 0, w, h], fill=(36, 28, 22))
        d.rounded_rectangle([w * .06, h * .07, w * .94, h * .95], 16, fill=(214, 182, 120))
        d.rectangle([w * .10, h * .14, w * .58, h * .86], fill=(250, 250, 245))
        d.rectangle([w * .12, h * .17, w * .56, h * .66], fill=GREEN)
        f = font("bold", 46)
        d.text((w * .62, h * .16), "SUBJECT FILE", font=f, fill=(90, 30, 20))
        for i in range(7):
            y = h * (.28 + i * .075)
            d.rectangle([w * .62, y, w * (.88 - (i % 3) * .07), y + 26], fill=(30, 30, 30) if i % 2 else (90, 80, 70))
    else:
        raise ValueError(f"unknown green-screen frame {kind!r}: use tv, monitor, page, file or put {kind}.png/.mp4 in assets/greenscreen/")
    return im


def thumbnail(img, text, accent, badge, k):
    W, H = 1280, 720
    im = ImageEnhance.Color(ImageEnhance.Contrast(cover(img, W, H)).enhance(1.18)).enhance(1.2)
    im = print_pass(im, k * .5, cell=7)
    shade = np.linspace(0.85, 0.0, W, dtype=np.float32)[None, :, None]  # darken the text side
    a = np.asarray(im, np.float32) * (1 - shade * 0.75)
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    words = text.upper().split()
    lines = [" ".join(words[:max(1, len(words) // 2)]), " ".join(words[max(1, len(words) // 2):])] if len(words) > 2 else [text.upper()]
    for i, line in enumerate(l for l in lines if l):
        size = 170
        while font("title", size).getlength(line) > W * .6 and size > 70:
            size -= 6
        d.text((48, 120 + i * 190), line, font=font("title", size), fill=(255, 228, 0) if i == 0 else "white",
               stroke_width=10, stroke_fill=INK)
    if badge:
        f = font("bold", 40)
        bw = f.getlength(badge.upper()) + 36
        d.rectangle([40, 36, 40 + bw, 96], fill=rgb(accent))
        d.text((58, 40), badge.upper(), font=f, fill="white")
    return im
