"""Storyboard a video's comic sources (runs on a GitHub runner, uses no AI): every page gets its panels numbered,
the lettering is read with Tesseract OCR, and storyboard.xlsx holds one ready-made Scenes row per panel
(page reference filled in, OCR text in notes) to paste into the sheet and narrate.

usage: python -m pipeline.storyboard VID001 out/VID001/storyboard
"""
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from PIL import Image, ImageDraw

from pipeline import comic, sources
from pipeline.render import panel_boxes
from pipeline.sheet import SHEET, table


def ocr(panel, tmp):
    """Read only the lettering: OCR each light balloon / caption box, keep real all-caps words (comic lettering)."""
    a = np.asarray(panel.convert("RGB"))
    light = (a.max(axis=2) > 185) & ((a.max(axis=2).astype(int) - a.min(axis=2)) < 90)  # white or yellowed paper
    light = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    h, w = light.shape
    found = []
    for x, y, bw, bh, area in cv2.connectedComponentsWithStats(light)[2][1:]:
        if not (0.004 * w * h < area < 0.5 * w * h and bw > 40 and bh > 20):
            continue
        g = cv2.cvtColor(a[y:y + bh, x:x + bw], cv2.COLOR_RGB2GRAY)
        cv2.imwrite(tmp, cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        r = subprocess.run(["tesseract", tmp, "stdout", "--psm", "6"], capture_output=True, text=True)
        words = [t for t in r.stdout.split()
                 if re.fullmatch(r"[A-Z][A-Z'’!?.,\-]*", t) and sum(c.isalpha() for c in t) >= 2 or t in ("A", "I")]
        if len(words) >= 2:
            found.append((y, x, " ".join(words)))
    return " / ".join(t for _, _, t in sorted(found))  # top-to-bottom, left-to-right reading order


def main(vid, out):
    wb = load_workbook(SHEET, data_only=True)
    rows_in = table(wb["Sources"]) if "Sources" in wb.sheetnames else []
    srcs = {str(x["source_id"]).strip(): str(x["location"]).strip() for x in rows_in if str(x.get("source_id", "")).strip()}
    mine = [x for x in rows_in if str(x.get("video_id", "")).strip() == vid]
    if not mine:
        sys.exit(f"no Sources rows with video_id {vid}: add the comics for this video to the Sources sheet")
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    head = [c.value for c in wb["Scenes"][1]]
    rows = []
    for x in mine:
        sid = str(x["source_id"]).strip()
        single = srcs[sid].split("?")[0].lower().endswith(sources.IMG)
        pages = [sources.page_image(sid, srcs, work)] if single else \
            sources.issue_pages(sources.issue_file(sid, srcs, work), work)
        for k, path in enumerate(pages, 1):
            img = Image.open(path).convert("RGB")
            if max(img.size) < 1800:  # OCR reads small lettering far better at 2x
                img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
            ref = sid if single else f"{sid}:{k}"
            board = img.copy()
            d = ImageDraw.Draw(board)
            for i, b in enumerate(panel_boxes(img), 1):
                d.rectangle(b, outline=(230, 30, 30), width=8)
                d.text((b[0] + 16, b[1] + 8), str(i), font=comic.font("title", 110), fill=(230, 30, 30),
                       stroke_width=5, stroke_fill="white")
                text = ocr(img.crop(b), os.path.join(work, "ocr.png"))
                rows.append(dict(video_id=vid, chapter=x.get("title", ""), page=f"{ref}#{i}",
                                 notes=f"OCR: {text}" if text else ""))
            board.thumbnail((1400, 1400))
            board.save(os.path.join(out, f"{ref.replace(':', '_p')}.jpg"), quality=85)
            print(f"{ref}: {len(rows)} panels so far", flush=True)
    sb = Workbook()
    ws = sb.active
    ws.title = "Scenes"
    ws.append(head)
    for r in rows:
        ws.append([r.get(h, "") for h in head])
    sb.save(os.path.join(out, "storyboard.xlsx"))
    print(f"storyboard: {len(rows)} panels -> {out}/storyboard.xlsx (+ one numbered image per page)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
