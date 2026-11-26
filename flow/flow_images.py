"""Draw a video's AI shots in Google Flow, driven through your own Chrome (Flow has no API).
usage: python flow/flow_images.py work/plan.json out/VID001/flow   (./lore.sh flow VID001 does this + the upload)
- Opens real Chrome with a dedicated profile (~/.cache/lore-flow-profile). First run: sign in to Google yourself
  in that window; the session is remembered, and no password is ever stored by this script.
- Mix method: characters marked drawn_by=flux in the Characters tab (Marvel, DC... which Flow refuses) are NOT sent
  to Flow: every shot that shows them is left for FLUX on the GitHub runners. Flow draws everything else.
- Batches: up to FLOW_BATCH (default 4) shots per message, grouped by the character references they need. Each
  shot carries a tag (S012_0) that Flow repeats at the start of the image's own prompt, so every result is matched
  to its scene exactly; anything untagged falls back to the order of the shots.
- Already-saved images are skipped, so you can stop and resume any time.
Then ./lore.sh flow uploads the folder as release 'flow-<VID>', and the render uses those images.
Selectors checked live on 2026-11-24; if Flow changes its page, fix them in main()/attach()/results().
"""
import base64
import hashlib
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


def results(page):
    """Flow's generated images in the chat, oldest first: (src, alt). alt starts with the image's own prompt."""
    out = []
    for b in page.get_by_role("button", name="Open image in editor").all():
        try:
            im = b.locator("img").first
            out.append((im.get_attribute("src") or "", im.get_attribute("alt") or ""))
        except Exception:  # element vanished while Flow re-rendered: just look again next round
            continue
    return [r for r in out if r[0]]


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


def fingerprint(j):
    """The shot a Flow picture was drawn for (pipeline/images.py flow_ok checks the same): a rewritten shot is redrawn."""
    return hashlib.sha1(str(j.get("visual") or j.get("prompt") or "").encode()).hexdigest()[:16]


def current(out, j):
    """A saved Flow picture still fits its shot; pictures from before fingerprints (no .prompt file) count as fitting."""
    png, fp = os.path.join(out, j["name"]), os.path.join(out, j["name"][:-4] + ".prompt")
    return os.path.exists(png) and (not os.path.exists(fp) or open(fp).read().strip() == fingerprint(j))


def save(page, src, path):
    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith("blob:"):  # blob URLs only live inside the page: read them from there
        data = bytes(page.evaluate("async u => Array.from(new Uint8Array(await (await fetch(u)).arrayBuffer()))", src))
    else:  # the chat shows a sized copy (=s512 / =s1600): =s0 asks for the original
        data = page.request.get(re.sub(r"=s\d+(-rw)?$", "=s0", src)).body()
    trim_frame(Image.open(io.BytesIO(data)).convert("RGB")).save(path)  # Flow serves WebP; the render wants PNG


RULES = ("Quality: correct anatomy (five fingers on each hand, natural limbs, no extra or missing limbs), no duplicated "
         "or merged figures, coherent background and perspective, crisp clean inks, no blur, no smudges, no artifacts. "
         "Strictly no text of any kind: no letters, words, numbers, speech balloons, captions, sound-effect lettering, "
         "signs with writing, panel borders, gutters, split screens, frames, watermarks, signatures or logos.")


def tag(j):
    """Short id Flow repeats at the start of each image's prompt: s012_0.png -> S012_0, ref_JANE.png -> REF_JANE."""
    return os.path.splitext(j["name"])[0].upper()


def batch_prompt(P, jobs, attached):
    """One message for several shots: shared art direction once, then one tagged line per shot."""
    style, chars = P["style"]["prompt_prefix"], P["characters"]
    n = len(jobs)
    head = [f"Generate {n} separate image{'s' if n > 1 else ''}, one for each tagged shot below, in this order. "
            "Draw every shot as its own complete image; never combine shots. Begin each image's own prompt with its "
            "tag exactly as written (for example S012_0:)."]
    if jobs[0].get("char"):
        head.append("Each image is a 3:4 portrait character model sheet: one character, full body from head to feet, "
                    "neutral three-quarter standing pose, plain flat light-grey background, even studio lighting. "
                    f"Art style: {style}")
        shots = [f"{tag(j)}: {chars[j['char']]['name']}. Design, follow exactly: {chars[j['char']]['look']}." for j in jobs]
    else:
        head.append("Each image is a 16:9 widescreen illustration drawn exactly like professional comic book art (inks, "
                    "flat printed colours, halftone texture), shown full-bleed: the artwork fills the whole image to all "
                    f"four edges, with no frame, border or paper margin. Art style: {style}")
        cast = [chars[c] for c in dict.fromkeys(c for j in jobs for c in j.get("cast", [])) if c in chars]
        if cast:
            head.append("Characters, identical to their design in every image (same face, hair, build, costume and "
                        "colours): " + " ".join(f"{c['name']}: {c['look']}." for c in cast))
        if attached:
            names = " and ".join(chars[c]["name"] for c in attached)
            head.append(f"The attached reference image{'s show' if len(attached) > 1 else ' shows'} the exact design of "
                        f"{names}: match it precisely, including costume details and colours.")
        head.append("Composition: cinematic comic-panel staging, one clear focal subject on a rule-of-thirds line with a "
                    "strong readable silhouette, distinct foreground, midground and background layers, believable scale. "
                    "Lighting: a strong key light from one side, deep shadows and a bright rim light, high contrast.")
        shots = [f"{tag(j)}: {j.get('visual') or j['prompt']}." for j in jobs]
    return " ".join(head + [RULES, "Shots:"]) + "\n" + "\n".join(shots)


def flux_only(P):
    """Character ids Flow must never be asked for, and the names that give them away in a cast-less shot."""
    ids = {c for c, ch in P["characters"].items() if str(ch.get("drawn_by", "")).strip().lower() == "flux"}
    names = set()
    for c in ids:
        ch = P["characters"][c]
        for n in (ch.get("name", ""), ch.get("alias", "")):
            names.update(w.strip(" ,.'") for w in re.split(r"[ /]", str(n)) if len(w.strip(" ,.'")) >= 5)
        names.add(c.title())
    return ids, {n.lower() for n in names}


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


ERROR = re.compile(r"something went wrong", re.I)
POLICY = re.compile(r"might violate our policies", re.I)


def generate(page, box, P, batch, attached):
    """Send one batch and collect its images -> ({tag: src}, why) with why in ok / error / policy / timeout.
    Flow's error and policy messages are noticed at once instead of waiting out the timeout."""
    seen = {s for s, _ in results(page)}
    errs, refusals = page.get_by_text(ERROR).count(), page.get_by_text(POLICY).count()
    box.fill(batch_prompt(P, batch, attached))
    page.get_by_role("button", name="Start generation").click()
    t0, todo, new, why = time.time(), {tag(j): j for j in batch}, [], "timeout"
    while time.time() - t0 < WAIT_S + 60 * len(batch):
        time.sleep(6)
        new = [(s, a) for s, a in results(page) if s not in seen]
        if len(new) >= len(batch):
            time.sleep(4)  # let the last thumbnails settle
            new, why = [(s, a) for s, a in results(page) if s not in seen], "ok"
            break
        if page.get_by_text(ERROR).count() > errs or page.get_by_text(POLICY).count() > refusals:
            time.sleep(8)  # the rest of the batch may still be landing
            new = [(s, a) for s, a in results(page) if s not in seen]
            why = "policy" if page.get_by_text(POLICY).count() > refusals else "error"
            break
    got = {}
    for s, a in new:  # exact match on the tag Flow repeated at the start of the image's prompt
        m = re.match(r"\s*([A-Z]+\d*_?[A-Z0-9_]*)\s*:", a.upper())
        if m and m.group(1) in todo and m.group(1) not in got:
            got[m.group(1)] = s
    if why == "ok":  # untagged results of a complete batch: in the order the shots were listed
        rest = [s for s, _ in new if s not in got.values()]
        for t in todo:
            if t not in got and rest:
                got[t] = rest.pop(0)
    return got, why


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
    no_flow, giveaway = flux_only(P)
    jobs, flux = [], []
    for j in P["refs"] + P["images"]:  # refs first: every later scene is drawn against them
        if j.get("src") or j.get("crop") or current(out, j):
            continue
        words = set(re.findall(r"[a-z]+", str(j.get("visual") or "").lower()))
        if j.get("cutout") or j.get("char") in no_flow or set(j.get("cast", [])) & no_flow or words & giveaway:
            flux.append(j["name"])  # left for FLUX on the runners (copyrighted character)
        else:
            jobs.append(j)
    print(f"{len(jobs)} images to draw in Flow; {len(flux)} character shots left for FLUX on the runners")
    if not jobs:
        return
    size = max(1, int(os.environ.get("FLOW_BATCH", "4")))
    groups = {}  # same references attached -> same batch
    for j in jobs:
        key = ("ref",) if j.get("char") else tuple(c for c in j.get("cast", []) if c not in no_flow)
        groups.setdefault(key, []).append(j)
    batches = [g[k:k + size] for key in sorted(groups, key=lambda k: k != ("ref",)) for g in [groups[key]]
               for k in range(0, len(g), size)]
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=False, viewport=None,
                                                    args=["--start-maximized"], chromium_sandbox=True,  # no warning bar
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
        done = 0
        for bi, batch in enumerate(batches, 1):
            while page.get_by_role("button", name="Ingredient", exact=True).count():  # chips stay after a generation
                page.get_by_role("button", name="Ingredient", exact=True).first.click()
                time.sleep(0.5)
            attached = []
            for c in dict.fromkeys(c for j in batch for c in j.get("cast", []) if c not in no_flow):
                ref = os.path.join(out, f"ref_{c}.png")
                if os.path.exists(ref):
                    tagged = os.path.join(out, "upload", f"{vid}_ref_{c}.png")  # unique name in the project's media
                    os.makedirs(os.path.dirname(tagged), exist_ok=True)
                    shutil.copy(ref, tagged)
                    try:
                        attach(page, tagged)
                        attached.append(c)
                    except Exception as e:  # draw it anyway from the written design; say so
                        print(f"  could not attach {c}'s reference ({e.__class__.__name__})", flush=True)
                        page.keyboard.press("Escape")
            t0 = time.time()
            got, why = generate(page, box, P, batch, attached)
            if why == "error":  # Flow's "Something went wrong": send the same batch once more
                print(f"  batch {bi}: Flow error, retrying", flush=True)
                more, why = generate(page, box, P, [j for j in batch if tag(j) not in got], attached)
                got.update(more)
            if len(batch) > 1:  # whatever is still missing goes alone: one bad shot must not sink the others
                for j in [j for j in batch if tag(j) not in got]:
                    more, why = generate(page, box, P, [j], attached)
                    got.update(more)
                    if tag(j) not in got:
                        print(f"  {j['name']}: {'refused by Flow (policy)' if why == 'policy' else 'Flow ' + why}", flush=True)
            for j in batch:
                done += 1
                if tag(j) not in got:
                    print(f"[{done}/{len(jobs)}] {j['name']}: no image, skipped (reword this shot)", flush=True)
                    continue
                dst = os.path.join(out, j["name"])
                save(page, got[tag(j)], dst)
                open(dst[:-4] + ".prompt", "w").write(fingerprint(j))
                w, h = Image.open(dst).size
                print(f"[{done}/{len(jobs)}] {j['name']} saved {w}x{h} (batch {bi}/{len(batches)}, "
                      f"{time.time() - t0:.0f}s)" + ("  <- not 16:9: check it" if not j.get("char") and not 1.6 < w / h < 1.95
                                                     else ""), flush=True)
        ctx.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
