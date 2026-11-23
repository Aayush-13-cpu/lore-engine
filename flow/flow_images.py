"""Marvel videos only: draw the scene images with Google Flow, driven through your own Chrome (Flow has no API).

usage: python flow/flow_images.py work/plan.json out/VID001/flow
- Opens real Chrome with a dedicated profile (~/.cache/lore-flow-profile). First run: sign in to Google yourself
  in that window; the session is remembered, and no password is ever stored by this script.
- For each image job in the plan: types the prompt into Flow, waits for the new image, saves it as <name>.png.
  Already-saved images are skipped, so you can stop and resume any time.
Then ./lore.sh flow uploads the folder as release 'flow-<VID>', and the render uses those images.

shortcut: page selectors are best guesses until the first live test with your account; adjust FLOW_* below then.
"""
import base64
import json
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

FLOW_URL = "https://labs.google/fx/tools/flow"
PROFILE = os.path.expanduser("~/.cache/lore-flow-profile")
FLOW_GENERATE = re.compile(r"generate|create|send|submit|arrow_forward", re.I)
WAIT_S = 240  # Flow can take a few minutes per generation


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


def save(page, src, path):
    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith("blob:"):  # blob URLs only live inside the page: read them from there
        data = bytes(page.evaluate("async u => Array.from(new Uint8Array(await (await fetch(u)).arrayBuffer()))", src))
    else:
        data = page.request.get(src).body()
    with open(path, "wb") as f:
        f.write(data)


def main(plan_path, out):
    P = json.load(open(plan_path))
    os.makedirs(out, exist_ok=True)
    jobs = [j for j in P["refs"] + P["images"] if not j.get("src") and not os.path.exists(os.path.join(out, j["name"]))]
    print(f"{len(jobs)} images to draw in Flow")
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=False, viewport=None,
                                                    args=["--start-maximized"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(FLOW_URL)
        input("Sign in to Google in the Chrome window if asked, open a Flow project, choose image generation, "
              "then press Enter here... ")
        for i, j in enumerate(jobs, 1):
            seen = {img.get_attribute("src") for img in page.locator("img").all()}
            page.get_by_role("textbox").last.fill(j["prompt"])
            page.get_by_role("button", name=FLOW_GENERATE).last.click()
            t0, got = time.time(), []
            while time.time() - t0 < WAIT_S and not got:
                time.sleep(5)
                got = new_images(page, seen)
            if not got:
                print(f"[{i}/{len(jobs)}] {j['name']}: no image after {WAIT_S}s (blocked prompt?), skipped", flush=True)
                continue
            save(page, got[0], os.path.join(out, j["name"]))
            print(f"[{i}/{len(jobs)}] {j['name']} saved ({time.time() - t0:.0f}s)", flush=True)
        ctx.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
