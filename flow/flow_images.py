"""Videos whose comics we cannot use (Marvel, DC, The Boys, Invincible): draw the scene images with Google Flow,
driven through your own Chrome (Flow has no API). Public-domain videos keep using their real comic pages.
usage: python flow/flow_images.py work/plan.json out/VID001/flow
- Opens real Chrome with a dedicated profile (~/.cache/lore-flow-profile). First run: sign in to Google yourself
  in that window; the session is remembered, and no password is ever stored by this script.
- For each image job in the plan: types the prompt into Flow, waits for the new image, saves it as <name>.png.
  Already-saved images are skipped, so you can stop and resume any time.
Then ./lore.sh flow uploads the folder as release 'flow-<VID>', and the render uses those images.

Selectors checked live on 2026-11-24 (project chat box + "Start generation"); if Flow changes its page, fix them in main().
"""
import base64
import io
import json
import os
import re
import shutil
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

FLOW_URL = "https://flow.google.com/"  # moved from labs.google/fx/tools/flow in 2026
PROFILE = os.path.expanduser("~/.cache/lore-flow-profile")
PROJECT_FILE = os.path.expanduser("~/.cache/lore-flow-project")  # the Flow project every image goes into
WAIT_S = 240  # measured ~45 s per image (Nano Banana, PRO account, 2026-11-24)


def new_images(page, seen):
    """Large images on the page that were not there before the prompt was sent."""
    out = []
    for img in page.locator("img").all():
        try:
            src, w = img.get_attribute("src") or "", img.evaluate("e => e.naturalWidth")
        except Exception:  # element vanished while Flow re-rendered: just look again next round
            continue
        if src and src not in seen and w and w >= 512:
            out.append(src)
    return out


def trim_frame(im):
    """Cut a flat paper margin or printed border Flow sometimes draws around the art (up to 6% per side),
    then scale back to the original size so every image keeps the same shape."""
    import numpy as np
    g = np.asarray(im.convert("L"), np.float32)
    h, w = g.shape

    def depth(lines):  # edge lines that are flat paper (aged texture spreads ~25 grey levels, art 100+) ...
        n, cap = 0, round(0.06 * len(lines))
        while n < cap and np.percentile(lines[n], 90) - np.percentile(lines[n], 10) <= 40:
            n += 1
        if n:  # ... plus the printed ink rule just inside the paper
            while n < cap and lines[n].mean() < 70:
                n += 1
        return n

    top, bottom = depth(g), depth(g[::-1])
    left, right = depth(g.T), depth(g.T[::-1])
    if not (top or bottom or left or right):
        return im
    return im.crop((left, top, w - right, h - bottom)).resize((w, h), Image.LANCZOS)


def save(page, src, path):
    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith("blob:"):  # blob URLs only live inside the page: read them from there
        data = bytes(page.evaluate("async u => Array.from(new Uint8Array(await (await fetch(u)).arrayBuffer()))", src))
    else:
        data = page.request.get(src).body()
    trim_frame(Image.open(io.BytesIO(data)).convert("RGB")).save(path)  # Flow serves WebP; the render wants PNG


RULES = ("Quality: correct anatomy (five fingers on each hand, natural limbs, no extra or missing limbs), no duplicated "
         "or merged figures, coherent background and perspective, crisp clean inks, no blur, no smudges, no artifacts. "
         "Strictly no text of any kind: no letters, words, numbers, speech balloons, captions, sound-effect lettering, "
         "signs with writing, panel borders, gutters, split screens, frames, watermarks, signatures or logos.")


def flow_prompt(P, j, attached):
    """A structured brief for Flow: comic-panel look, the shot, fixed character designs, hard quality rules."""
    style, chars = P["style"]["prompt_prefix"], P["characters"]
    if j.get("char"):
        c = chars[j["char"]]
        return (f"Create one 3:4 portrait character model sheet in this exact comic art style: {style} "
                f"Subject: {c['name']}, full body from head to feet, neutral three-quarter standing pose, plain flat "
                f"light-grey background, even studio lighting. Design, follow exactly: {c['look']}. {RULES}")
    cast = [chars[c] for c in j.get("cast", []) if c in chars]
    out = ["Create one 16:9 widescreen illustration drawn exactly like professional comic book art (inks, flat "
           "printed colours, halftone texture), shown full-bleed: the artwork fills the whole image to all four edges, "
           f"with no frame, border or paper margin around it. Art style: {style}",
           f"Scene: {j.get('visual') or j['prompt']}."]
    if cast:
        out.append("Characters, identical to their design in every image (same face, hair, build, costume and colours): "
                   + " ".join(f"{c['name']}: {c['look']}." for c in cast))
    if attached:
        names = " and ".join(chars[c]["name"] for c in attached)
        out.append(f"The attached reference image{'s show' if len(attached) > 1 else ' shows'} the exact design of "
                   f"{names}: match it precisely, including costume details and colours.")
    out.append("Composition: cinematic comic-panel staging, one clear focal point, strong readable silhouettes, "
               "dramatic lighting, believable scale.")
    return " ".join(out + [RULES])


def attach(page, path):
    """Put one image file into the prompt as an ingredient (uploaded once per project, then picked by name)."""
    page.get_by_role("button", name="Add ingredients to the prompt box").click()
    name = os.path.basename(path)
    page.get_by_role("textbox", name=re.compile("search assets", re.I)).fill(name)  # newer images push it out of the list
    time.sleep(2)
    opt = page.get_by_role("option", name=re.compile(r"^" + re.escape(name)))
    if not opt.count():
        with page.expect_file_chooser() as fc:
            page.get_by_role("button", name="Upload media").click()
        fc.value.set_files(path)
    for _ in range(60):  # the option reads "Uploading <name>" until the upload finishes
        if opt.count() and "Uploading" not in opt.first.inner_text():
            break
        time.sleep(1)
    chips = page.get_by_role("button", name="Ingredient", exact=True)
    before = chips.count()
    opt.first.click()
    time.sleep(1)
    add = page.get_by_role("button", name="Add to prompt")
    if add.count() and add.first.is_visible():  # from the full list it needs this; a search hit is added directly
        add.first.click()
        time.sleep(1)
    if chips.count() <= before:
        raise RuntimeError(f"{name} did not reach the prompt box")


def main(plan_path, out):
    P = json.load(open(plan_path))
    vid = P["video"]["video_id"]
    os.makedirs(out, exist_ok=True)
    for r in P["refs"]:  # an approved image, or the face cut from a real panel, is the reference as-is
        dst = os.path.join(out, r["name"])
        if os.path.exists(dst):
            continue
        if r.get("asset") and os.path.exists(r["asset"]):
            Image.open(r["asset"]).convert("RGB").save(dst)
        elif r.get("crop"):
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from pipeline import sources
            ref, box = r["crop"].split("@")
            im = Image.open(sources.page_image(ref, P.get("sources", {}), os.path.join(out, "pages"))).convert("RGB")
            x0, y0, x1, y1 = map(float, box.split(","))
            im.crop((round(x0 * im.width), round(y0 * im.height), round(x1 * im.width), round(y1 * im.height))).save(dst)
    jobs = [j for j in P["refs"] + P["images"]  # refs first: every later scene is drawn against them
            if not j.get("src") and not j.get("crop") and not os.path.exists(os.path.join(out, j["name"]))]
    print(f"{len(jobs)} images to draw in Flow")
    if not jobs:
        return
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=False, viewport=None,
                                                    args=["--start-maximized"],
                                                    # Playwright's default 'basic' store can't decrypt the login
                                                    # cookies normal Chrome saved via the keyring: you'd be signed out
                                                    ignore_default_args=["--password-store=basic"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if os.path.exists(PROJECT_FILE):
            page.goto(open(PROJECT_FILE).read().strip())
        else:
            page.goto(FLOW_URL)
            input("Sign in to Google in the Chrome window if asked, open (or create) a Flow project, "
                  "then press Enter here... ")
            open(PROJECT_FILE, "w").write(page.url)
        box = page.get_by_role("textbox", name="What do you want to create?")
        box.wait_for(timeout=60000)
        for i, j in enumerate(jobs, 1):
            while page.get_by_role("button", name="Ingredient", exact=True).count():  # chips stay after a generation
                page.get_by_role("button", name="Ingredient", exact=True).first.click()
                time.sleep(0.5)
            attached = []
            for c in j.get("cast", []):
                ref = os.path.join(out, f"ref_{c}.png")
                if os.path.exists(ref):
                    tagged = os.path.join(out, "upload", f"{vid}_ref_{c}.png")  # unique name in the project's media
                    os.makedirs(os.path.dirname(tagged), exist_ok=True)
                    shutil.copy(ref, tagged)
                    try:
                        attach(page, tagged)
                        attached.append(c)
                    except Exception as e:  # draw it anyway from the written design; say so
                        print(f"  {j['name']}: could not attach {c}'s reference ({e.__class__.__name__})", flush=True)
                        page.keyboard.press("Escape")
            seen = {img.get_attribute("src") for img in page.locator("img").all()}  # after attaching: refs are not results
            box.fill(flow_prompt(P, j, attached))
            page.get_by_role("button", name="Start generation").click()
            t0, got = time.time(), []
            while time.time() - t0 < WAIT_S and not got:
                time.sleep(5)
                got = new_images(page, seen)
            if not got:
                print(f"[{i}/{len(jobs)}] {j['name']}: no image after {WAIT_S}s (blocked prompt?), skipped", flush=True)
                continue
            dst = os.path.join(out, j["name"])
            save(page, got[0], dst)
            w, h = Image.open(dst).size
            print(f"[{i}/{len(jobs)}] {j['name']} saved {w}x{h} ({time.time() - t0:.0f}s)"
                  + ("  <- not 16:9: check it" if not j.get("char") and not 1.6 < w / h < 1.95 else ""), flush=True)
        ctx.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
