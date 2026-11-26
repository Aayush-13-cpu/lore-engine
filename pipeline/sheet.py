"""Read content/lore_content.xlsx and print the render plan (JSON) for one video.

usage: python -m pipeline.sheet VID001 [SCENES like 1-8] > work/plan.json
"""
import hashlib
import json
import os
import re
import sys
import zlib

from openpyxl import load_workbook

from pipeline.sources import ISSUE, locate

SHEET = "content/lore_content.xlsx"
TEMPLATES = {"full", "panels", "split", "dossier", "title", "sting", "quote", "endscreen", "page", "issue"}
PAGE = re.compile(r"^(\S+?)(#\d+|@[\d.]+,[\d.]+,[\d.]+,[\d.]+)?$")  # file-or-url [#panel | @x0,y0,x1,y1]
NEEDS_IMAGE = {"full", "panels", "split"}  # other templates use an image only as backdrop
TRANSITIONS = {"auto", "cut", "fade", "dissolve", "black", "flash", "whip", "glitch", "zoom", "circle", "wipe", "slide", "radial"}
MOTIONS = {"auto", "in", "out", "left", "right", "up", "down", "static", "shake", "punch"}
FX = {"", "none", "embers", "dust", "snow", "rain", "stars", "flash", "glitch"}
SFX = {"", "none", "whoosh", "impact", "riser", "rumble", "glitch"}
GRADE_OK = re.compile(r"^[\w=:.,\-/ ]*$")  # ffmpeg filter args only: no graph syntax like ; [ ]
DEFAULTS = dict(
    channel_name="LORE VAULT", fps=30, width=1920, height=1080,
    image_width=1024, image_height=576, image_steps=4, upscale="yes",
    tts="chatterbox", voice_ref="", voice="am_michael", voice_speed=1.0,
    scene_pad=0.3, shot_cuts="auto", auto_sfx="auto", auto_text="auto", shot_seconds=3.5, image_budget=140, cutouts=6, callout_gap=20,
    text_gap=15, music_db=-21, lufs=-14, burn_subs="no", print_pass=0.5, parallax=14,
    shorts_count=1, shorts_min=25, shorts_max=58, watermark="yes",
    shorts_cta="Want to know how it ends? The full story is on my channel.",
)


def table(ws):
    rows = ws.iter_rows(values_only=True)
    head = [str(h or "").strip() for h in next(rows)]
    out = []
    for r in rows:
        d = {h: ("" if v is None else v) for h, v in zip(head, r) if h}
        if any(str(v).strip() for v in d.values()):
            out.append(d)
    return out


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def ids(v):
    return [x.strip().upper() for x in str(v).split(",") if x.strip()]


COMPOSE = ("Composition: one clear focal subject placed on a rule-of-thirds line, its silhouette standing out against "
           "the background, with distinct foreground, midground and background layers for depth; a single full-bleed "
           "widescreen illustration that fills the frame edge to edge.")
LIGHT = ("Lighting: a strong key light from one side, deep spotted-black shadows and a bright rim light that separates the "
         "subject from the background, high contrast and a dramatic, cinematic mood.")


WPS = 2.6  # narration words per second: sizes subscenes before the voice exists
SHOTS = ["A tight close-up", "A wide establishing shot", "A low-angle hero shot", "An over-the-shoulder shot",
         "A detail insert shot"]  # each subscene picture is a different real shot of the moment, never a crop
CLAUSE = re.compile(r"^(and|but|then|when|as|while|until|so|because|who|which|where|before|after|or|now|yet)$", re.I)


def split_words(words, m):
    """Word indexes where pictures 2..m of a line start: near equal shares, on punctuation or a clause word,
    every piece at least 4 words."""
    cands = [(i, words[i - 1][-1:] in ",;:.!?—-" or bool(CLAUSE.match(words[i].strip(".,;:!?\"'").lower())))
             for i in range(4, len(words) - 3)]
    cuts = []
    for k in range(1, m):
        ok = [c for c in cands if c[0] >= (cuts[-1] + 4 if cuts else 4)]
        if not ok:
            break
        cuts.append(min(ok, key=lambda c: abs(c[0] - len(words) * k / m) - 2.5 * c[1])[0])
    return cuts


def name_re(ch):
    nm = str(ch.get("name", "")).strip()
    return re.compile(rf"\b{re.escape(nm)}\b") if nm else None


def scene_prompt(style, visual, cast):
    """Director's-note prose, most important first: the moment, who is in it and how they look, composition,
    lighting, then the style and the hard exclusions (FLUX.2 has no negative prompt)."""
    refs = " ".join(f"Image {i + 1} is the official look of {c['name']}: copy that exact costume, colors and face."
                    for i, c in enumerate(cast[:2]))
    looks = " ".join(f"{c['name']} drawn exactly on-model: {c['look']}." for c in cast)
    return " ".join(x for x in (f"{visual.rstrip('.')}.", looks, refs, COMPOSE, LIGHT, style["prompt_prefix"],
                                style["prompt_suffix"]) if x)


def cutout_prompt(style, ch):
    """One character alone on chroma green, keyed out by the renderer and slid into scenes on their name."""
    return " ".join([
        f"{ch['name']}, full figure from head to feet, in a strong dynamic three-quarter pose, the whole body inside the "
        f"frame with empty space around it. {ch['name']} drawn exactly on-model: {ch['look']}.",
        f"Image 1 is the official look of {ch['name']}: copy that exact costume, colors and face.",
        "The background is one flat, even, pure chroma-key green (#00FF00) with no shadows, gradients, floor or scenery; "
        "the figure is lit from the front-left with a bright rim light and outlined by a bold black ink line.",
        style["prompt_prefix"], style["prompt_suffix"]])


def build(vid, subset="", path=SHEET):
    wb = load_workbook(path, data_only=True)
    T = {ws.title: table(ws) for ws in wb.worksheets}
    errors = []
    S = dict(DEFAULTS, **{str(r["key"]).strip(): r["value"] for r in T["Settings"] if str(r.get("value", "")).strip()})
    video = next((v for v in T["Videos"] if str(v["video_id"]).strip() == vid), None)
    if not video:
        sys.exit(f"video_id {vid!r} not found in the Videos sheet")
    style = next((s for s in T["Styles"] if s["style_id"] == video["style_id"]), None)
    if not style:
        sys.exit(f"style_id {video['style_id']!r} (video {vid}) not found in the Styles sheet")
    if not GRADE_OK.match(str(style.get("grade", ""))):
        errors.append(f"style {style['style_id']}: grade must be plain ffmpeg filters like eq=contrast=1.05:saturation=1.1")
    chars = {str(c["char_id"]).strip().upper(): c for c in T["Characters"]}
    srcs = {str(x["source_id"]).strip(): str(x["location"]).strip()
            for x in T.get("Sources", []) if str(x.get("source_id", "")).strip()}

    def source_problem(ref):  # why a page reference can't be found, or None
        im = ISSUE.match(ref)
        if ref in srcs:  # a Sources row holding a single page
            ref = srcs[ref]
        if im:
            if im.group(1) not in srcs:
                return f"source {im.group(1)} is not in the Sources sheet"
            loc = srcs[im.group(1)]
            if not loc.startswith(("http://", "https://")) and not locate(loc):
                return f"source {im.group(1)}: {loc} not found in private/ (is ASSETS_REPO set up?) or assets/sources/"
        elif not ref.startswith(("http://", "https://")) and not locate(ref):
            return f"page {ref} not found in private/pages/ or assets/pages/ (or use a web link / Sources id)"
        return None
    rows = sorted((s for s in T["Scenes"] if str(s["video_id"]).strip() == vid), key=lambda s: num(s["scene"]))
    if subset:
        a, _, b = subset.partition("-")
        rows = [s for s in rows if int(a) <= num(s["scene"]) <= int(b or a)]
    if not rows:
        sys.exit(f"no scenes for {vid} {subset}")

    scenes, images, cast_ids, pages = [], [], [], {}
    have = set()  # scene numbers that produced images
    last_img = None
    for r in rows:
        n = int(num(r["scene"]))
        tpl = str(r.get("template") or "full").strip().lower()
        if tpl not in TEMPLATES and not tpl.startswith("gs:"):
            errors.append(f"scene {n}: unknown template {tpl!r}")
        cast = ids(r.get("characters", ""))
        for c in cast:
            if c not in chars:
                errors.append(f"scene {n}: character {c!r} not in Characters sheet")
        cast_rows = [chars[c] for c in cast if c in chars]
        visual = str(r.get("visual", "")).strip()
        s = dict(
            n=n, chapter=str(r.get("chapter", "")).strip(), narration=str(r.get("narration", "")).strip(),
            template=tpl, caption=str(r.get("caption", "")).strip(), sfx_text=str(r.get("sfx_text", "")).strip(),
            motion=str(r.get("motion") or "auto").strip().lower(), transition=str(r.get("transition") or "auto").strip().lower(),
            fx=str(r.get("fx") or "").strip().lower(), sfx=str(r.get("sfx") or "").strip().lower(),
            hold=num(r.get("hold")), short_score=num(r.get("short_score")), characters=cast, images=[], broll="",
        )
        for col, allowed in (("motion", MOTIONS), ("transition", TRANSITIONS), ("fx", FX), ("sfx", SFX)):
            if s[col] not in allowed:
                errors.append(f"scene {n}: {col} {s[col]!r} not one of {', '.join(sorted(a for a in allowed if a))}")
            if s[col] == "none":
                s[col] = ""
        page = str(r.get("page", "")).strip()
        if visual.lower().startswith("page:"):  # older form: the visual itself is the page
            page, visual = visual[5:].strip(), ""
        if page:  # a real comic page, framed panel by panel (alone, or before this row's AI shot)
            m = PAGE.match(page)
            if not m:
                errors.append(f"scene {n}: page must look like ff48_p05.jpg#2 or https://...jpg@0.1,0.2,0.6,0.5")
            elif source_problem(m.group(1)):
                errors.append(f"scene {n}: {source_problem(m.group(1))}")
            else:
                pages.setdefault(m.group(1), f"page_{zlib.crc32(m.group(1).encode()):08x}.png")
            if tpl not in ("full", "page", "issue"):
                errors.append(f"scene {n}: a comic page plays alone or before a 'full' AI shot, not with template {tpl}")
            s["page"] = page
        if page and not visual:
            s["template"] = "issue" if tpl == "issue" else "page"
        elif tpl in ("page", "issue"):
            errors.append(f"scene {n}: template page needs a page (column 'page' or visual page:<file or link>#panel)")
        elif visual.lower().startswith("broll:"):
            s["broll"] = visual[6:].strip()
        elif visual:
            for k, v in enumerate(p.strip() for p in visual.split("||")):
                if v.startswith("="):  # "=12" reuses scene 12's first image: no extra generation
                    ref = int(num(v[1:]))
                    if ref not in have:
                        errors.append(f"scene {n}: '{v}' points at scene {ref}, which has no image (or is outside this render)")
                    s["images"].append(f"s{ref:03d}_0.png")
                else:
                    name = f"s{n:03d}_{k}.png"
                    # seed from the shot text, not the row number: inserting a scene must not re-roll later art
                    images.append(dict(name=name, prompt=scene_prompt(style, v, cast_rows), seed=seed(vid, v),
                                       refs=[f"ref_{c}.png" for c in cast[:2]], visual=v, cast=cast[:2]))  # raw: Flow
                    s["images"].append(name)
                    cast_ids += cast[:2]
                    if k == 0:
                        have.add(n)
        elif last_img and (tpl in NEEDS_IMAGE or tpl.startswith("gs:") or tpl in ("title", "quote", "endscreen")):
            s["images"].append(last_img)  # blank visual = keep the previous picture on screen
        elif tpl in NEEDS_IMAGE or tpl.startswith("gs:"):
            errors.append(f"scene {n}: template {tpl} needs a visual and there is no earlier image to reuse")
        if len(s["images"]) > 1 and tpl not in ("panels", "split", "full"):  # full: subscenes, cut on the words
            errors.append(f"scene {n}: '||' gives several images but template {tpl} shows only one (use full, panels or split)")
        if tpl == "dossier":
            if visual:
                errors.append(f"scene {n}: dossier uses the character's reference image; leave visual blank")
            if not cast:
                errors.append(f"scene {n}: dossier needs a character in the characters column")
            cast_ids += cast[:1]
        if s["images"]:
            last_img = s["images"][0]
        scenes.append(s)

    jobs = {j["name"]: j for j in images}
    # subscenes: a long AI line gets 2-4 new pictures, one per part of the sentence, cut on the word it starts on.
    # Each costs ~36 min of runner time, so image_budget caps them and the longest lines get theirs first.
    if str(S.get("shot_cuts", "auto")).lower() != "off":
        lines = [s for s in scenes if s["template"] == "full" and not s.get("page") and len(s["images"]) == 1
                 and s["images"][0] in jobs]
        want = {s["n"]: min(4, round(len(s["narration"].split()) / WPS / num(S["shot_seconds"], 3.5))) for s in lines}
        spare = int(num(S["image_budget"])) - sum(not j.get("src") for j in images) - int(num(S.get("cutouts"), 0))
        give = dict.fromkeys(want, 1)
        for r in range(2, 5):  # round-robin: every long line gets its 2nd picture before any gets a 3rd
            for s in sorted(lines, key=lambda s: -len(s["narration"].split())):
                if spare > 0 and want[s["n"]] >= r:
                    give[s["n"]], spare = r, spare - 1
        for s in lines:
            words, j = s["narration"].split(), jobs[s["images"][0]]
            at = split_words(words, give[s["n"]])
            for k, (a, b) in enumerate(zip(at, at[1:] + [len(words)]), 1):
                part = words[a:b][1:] if CLAUSE.match(words[a].strip(".,;:!?\"'")) else words[a:b]
                part = " ".join(part).strip(" ,;:—-").rstrip(".!?")
                named = [c for c, ch in chars.items() if name_re(ch) and name_re(ch).search(part)]
                cast = (named or j["cast"])[:2]  # "he" in this part is still the line's character
                v = f"{SHOTS[(s['n'] + k) % len(SHOTS)]}: {part}. Same scene, setting and colours as: {j['visual']}"
                images.append(dict(name=f"s{s['n']:03d}_{k}.png", prompt=scene_prompt(style, v, [chars[c] for c in cast]),
                                   seed=seed(vid, v), refs=[f"ref_{c}.png" for c in cast], visual=v, cast=cast))
                s["images"].append(f"s{s['n']:03d}_{k}.png")
                cast_ids += cast
            s["splits"] = at[:len(s["images"]) - 1]
        jobs = {j["name"]: j for j in images}
    # green screen: a character named in a line whose pictures don't show them slides in on their name,
    # at most once every callout_gap seconds of narration
    cut_ids, t, last = [], 0.0, -1e9
    for s in scenes:
        shown = set(s["characters"]) | {c for i in s["images"] for c in jobs.get(i, {}).get("cast", [])}
        if s["template"] == "full" and not s.get("page") and s["images"] and t - last >= num(S.get("callout_gap"), 20):
            hits = sorted((m.start(), c) for c, ch in chars.items() if c not in shown and name_re(ch)
                          for m in [name_re(ch).search(s["narration"])] if m)
            c = next((c for _, c in hits if c in cut_ids or len(cut_ids) < int(num(S.get("cutouts"), 0))), None)
            if c:
                s["callout"], last = c, t
                cut_ids += [] if c in cut_ids else [c]
        t += len(s["narration"].split()) / WPS
    for c in cut_ids:
        images.append(dict(name=f"cut_{c}.png", prompt=cutout_prompt(style, chars[c]), seed=seed(vid, "cut", c),
                           refs=[f"ref_{c}.png"], visual="", cast=[c], cutout=True, size=[640, 1024], upscale=False))
        cast_ids.append(c)

    for src, name in pages.items():  # each real page gets fetched + Real-ESRGAN upscaled on an image runner
        images.append(dict(name=name, src=src, prompt="", seed=0, refs=[]))

    if str(video.get("thumbnail_prompt", "")).strip():
        tids = [c for c in ids(video.get("thumbnail_characters", "")) if c in chars]
        images.append(dict(name="thumb.png", prompt=scene_prompt(style, video["thumbnail_prompt"], [chars[c] for c in tids]),
                           seed=seed(vid, "thumb"), refs=[f"ref_{c}.png" for c in tids[:2]],
                           visual=str(video["thumbnail_prompt"]).strip(), cast=tids[:2]))
        cast_ids += tids[:2]

    refs = []
    for c in dict.fromkeys(cast_ids):
        ch = chars[c]
        crop = str(ch.get("ref_page", "")).strip()  # e.g. FF48:3@0.1,0.2,0.5,0.9 : the face cut from a real panel
        mc = PAGE.match(crop) if crop else None
        if crop and (not mc or not (mc.group(2) or "").startswith("@") or source_problem(mc.group(1))):
            errors.append(f"character {c}: ref_page must be <page>@x0,y0,x1,y1 on an existing page "
                          f"({source_problem(mc.group(1)) if mc else 'bad format'})")
        refs.append(dict(
            name=f"ref_{c}.png", asset=f"assets/characters/{c}.png", seed=int(num(ch.get("seed"), seed(c))), refs=[], char=c,
            crop=crop,
            prompt=f"{style['prompt_prefix']} Character reference sheet of {ch['name']}: {ch['look']}. Single full figure, "
                   f"three-quarter view, standing, plain light grey studio background, even lighting. {style['prompt_suffix']}"))

    shorts = [dict(n=int(num(x["short_no"], i + 1)), start=int(num(x["from_scene"])), end=int(num(x["to_scene"])),
                   hook_text=str(x.get("hook_text", "")).strip(), hook_line=str(x.get("hook_line", "")).strip(),
                   title=str(x.get("title", "")).strip())
              for i, x in enumerate(r for r in T.get("Shorts", []) if str(r["video_id"]).strip() == vid)]
    if errors:
        sys.exit("Sheet problems:\n  " + "\n  ".join(errors))
    return dict(
        video={k: str(v).strip() for k, v in video.items()}, settings=S, style=style,
        characters={c: chars[c] for c in dict.fromkeys(cast_ids + [c for c, ch in chars.items() if name_re(ch) and any(
            name_re(ch).search(s["narration"]) for s in scenes)])},  # named ones too: name plates, Flow's giveaway list
        scenes=scenes, images=images, refs=refs, shorts=shorts,
        pronounce={str(p["word"]).strip(): str(p["say_as"]).strip() for p in T.get("Pronounce", []) if p.get("word")},
        pages=pages, sources=srcs,
    )


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""), indent=1, default=str))
