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
    scene_pad=0.3, music_db=-21, lufs=-14, burn_subs="no", print_pass=0.5, parallax=14,
    shorts_count=3, shorts_min=25, shorts_max=58, watermark="yes",
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


def scene_prompt(style, visual, cast):
    refs = " ".join(f"Image {i + 1} is the official look of {c['name']}: copy that exact costume, colors and face."
                    for i, c in enumerate(cast[:2]))
    looks = " ".join(f"{c['name']} drawn exactly on-model: {c['look']}." for c in cast)
    frame = "One single full-bleed illustration filling the frame edge to edge: no panel borders, no gutters, no inset panels."
    return " ".join(x for x in (style["prompt_prefix"], refs, f"{visual}.", looks, frame, style["prompt_suffix"]) if x)


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
                                       refs=[f"ref_{c}.png" for c in cast[:2]]))
                    s["images"].append(name)
                    cast_ids += cast[:2]
                    if k == 0:
                        have.add(n)
        elif last_img and (tpl in NEEDS_IMAGE or tpl.startswith("gs:") or tpl in ("title", "quote", "endscreen")):
            s["images"].append(last_img)  # blank visual = keep the previous picture on screen
        elif tpl in NEEDS_IMAGE or tpl.startswith("gs:"):
            errors.append(f"scene {n}: template {tpl} needs a visual and there is no earlier image to reuse")
        if len(s["images"]) > 1 and tpl not in ("panels", "split"):
            errors.append(f"scene {n}: '||' gives several images but template {tpl} shows only one (use panels or split)")
        if tpl == "dossier":
            if visual:
                errors.append(f"scene {n}: dossier uses the character's reference image; leave visual blank")
            if not cast:
                errors.append(f"scene {n}: dossier needs a character in the characters column")
            cast_ids += cast[:1]
        if s["images"]:
            last_img = s["images"][0]
        scenes.append(s)

    for src, name in pages.items():  # each real page gets fetched + Real-ESRGAN upscaled on an image runner
        images.append(dict(name=name, src=src, prompt="", seed=0, refs=[]))

    if str(video.get("thumbnail_prompt", "")).strip():
        tids = [c for c in ids(video.get("thumbnail_characters", "")) if c in chars]
        images.append(dict(name="thumb.png", prompt=scene_prompt(style, video["thumbnail_prompt"], [chars[c] for c in tids]),
                           seed=seed(vid, "thumb"), refs=[f"ref_{c}.png" for c in tids[:2]]))
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
            name=f"ref_{c}.png", asset=f"assets/characters/{c}.png", seed=int(num(ch.get("seed"), seed(c))), refs=[],
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
        characters={c: chars[c] for c in dict.fromkeys(cast_ids)},
        scenes=scenes, images=images, refs=refs, shorts=shorts,
        pronounce={str(p["word"]).strip(): str(p["say_as"]).strip() for p in T.get("Pronounce", []) if p.get("word")},
        pages=pages, sources=srcs,
    )


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""), indent=1, default=str))
