"""Quality checks on a finished render -> report.md. Exit 1 on a hard failure (the release is still published first).

usage: python -m pipeline.check work/plan.json work/images work/audio out/VID001
Hard: loudness off target by >1 LU, true peak over -1 dBTP, audio/video length mismatch, a black hole >1 s.
Warnings: AI shots over 3.5 s, frozen picture >6 s, blurry or near-duplicate AI images.
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter

import cv2

from pipeline import render as R


def shots(tls, fps):
    """(scene n, seconds) for every AI-picture shot, cuts included."""
    out = []
    for tl in tls:
        if tl["s"]["template"] == "full" and not tl["s"].get("page") and not tl["s"]["broll"] and tl["s"]["images"]:
            edges = [0.0] + R.scene_cuts(tl) + [tl["F"] / fps]
            out += [(tl["s"]["n"], b - a) for a, b in zip(edges, edges[1:]) if b > a]
    return out


def pacing(tls, fps):
    """Editing numbers: unique AI pictures, hard-cut share of all cuts, back-to-back identical camera moves."""
    ai = [tl for tl in tls if tl["s"]["template"] == "full" and not tl["s"].get("page") and not tl["s"]["broll"]]
    pics = {i for tl in ai for i in tl["s"]["images"]}
    sub = sum(len(R.scene_cuts(tl)) for tl in tls)  # cuts between a line's own pictures are always hard cuts
    hard = sum(tl["tr"] == "cut" for tl in tls[1:]) + sub
    seq = [m for tl in ai for m in R.moves(tl)]
    return dict(pics=len(pics), hard=hard / max(len(tls) - 1 + sub, 1), repeats=sum(a == b for a, b in zip(seq, seq[1:])))


def dhash(path):
    g = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def images(imgdir):
    """(blurry, near-duplicate pairs) among the AI scene images."""
    paths = sorted(glob.glob(os.path.join(imgdir, "s*_*.png")))
    blurry = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (1024, round(1024 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
        if cv2.Laplacian(g, cv2.CV_64F).var() < 60:
            blurry.append(os.path.basename(p))
    h = {os.path.basename(p): dhash(p) for p in paths}
    names = list(h)
    dups = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if (h[a] != h[b]).sum() <= 4]
    return blurry, dups


def probe(video):
    """Stream lengths, loudness, black and frozen stretches, in one decoding pass at low resolution."""
    lens = dict(l.split(",") for l in subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout.split())
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vf",
                          "scale=320:-2,blackdetect=d=1.0:pix_th=0.08,freezedetect=n=0.002:d=6",
                          "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = log[log.rfind("Summary:"):]
    return dict(v=float(lens["video"]), a=float(lens["audio"]),
                lufs=float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1)),
                peak=float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1)),
                black=[(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)],
                frozen=[float(x) for x in re.findall(r"freeze_duration: ([\d.]+)", log)])


def verdict(S, sh, m, stars=(), pace=None):
    """(hard failures, warnings) from the measurements. stars: (start, end) of starfield b-roll, dark on purpose."""
    hard, warn = [], []
    m = dict(m, black=[(a, b) for a, b in m["black"] if not any(x <= a and b <= y for x, y in stars)])
    if abs(m["lufs"] - float(S["lufs"])) > 1:
        hard.append(f"loudness {m['lufs']:.1f} LUFS, target {float(S['lufs']):.0f} (±1)")
    if m["peak"] > -1.0:
        hard.append(f"true peak {m['peak']:.1f} dBTP > -1")
    if abs(m["v"] - m["a"]) > 0.1:  # AAC ends on 21 ms frames: up to ~2 video frames of tail is normal, not drift
        hard.append(f"audio {m['a']:.2f}s vs video {m['v']:.2f}s")
    if m["black"]:
        hard.append("black for >1 s at " + ", ".join(f"{a:.1f}s" for a, _ in m["black"]))
    long_ = [(n, d) for n, d in sh if d > 6]
    if long_:
        warn.append(f"{len(long_)} AI picture(s) held over 6 s (raise image_budget, or split the row): "
                    + ", ".join(f"scene {n} {d:.1f}s" for n, d in long_[:8]))
    if m["frozen"]:
        warn.append(f"{len(m['frozen'])} frozen stretch(es) over 6 s")
    if pace:
        per_min = pace["pics"] / max(m["v"] / 60, 1e-9)
        if per_min < 8:
            warn.append(f"{per_min:.1f} AI pictures per minute (aim 10-14): raise image_budget or lower shot_seconds")
        if pace["hard"] < .7:
            warn.append(f"only {pace['hard']:.0%} hard cuts (aim 70-85%): set more transitions to cut")
        if pace["repeats"] > 3:
            warn.append(f"{pace['repeats']} back-to-back identical camera moves")
    return hard, warn


def main(plan_path, imgdir, audiodir, outdir):
    P = json.load(open(plan_path))
    R.init(P, imgdir, os.path.join(outdir, "_work"))
    fps = R.C.fps
    tls = R.timeline(P, json.load(open(os.path.join(audiodir, "words.json"))), fps)
    sh = shots(tls, fps)
    m = probe(os.path.join(outdir, "video.mp4"))
    stars = [(tl["start"] / fps - .5, (tl["start"] + tl["F"] + tl["out"]) / fps + .5) for tl in tls if tl["s"]["broll"] == "stars"]
    pace = pacing(tls, fps)
    hard, warn = verdict(P["settings"], sh, m, stars, pace)
    blurry, dups = images(imgdir)
    warn += [f"blurry AI image(s): {', '.join(blurry)}"] if blurry else []
    warn += [f"near-identical AI images: {', '.join(f'{a}={b}' for a, b in dups)}"] if dups else []
    avg = sum(d for _, d in sh) / max(len(sh), 1)
    hits = [f"scene {tl['s']['n']} {tl['hit'][0]}" for tl in tls if tl.get("hit")]
    calls = [f"scene {tl['s']['n']} {tl['s']['callout']}" for tl in tls if tl["s"].get("callout")]
    fxs = [f"scene {tl['s']['n']} {tl['s']['fx']}" for tl in tls if tl["s"]["fx"]]
    said = [f"scene {tl['s']['n']} {kind} {text}" for tl in tls for kind, text, *_ in tl.get("texts", [])]
    lost = sorted({tl["s"]["callout"] for tl in tls if tl["s"].get("callout")
                   and not os.path.exists(os.path.join(imgdir, f"cut_{tl['s']['callout']}.png"))})
    warn += [f"green-screen picture missing, callout skipped: {', '.join(f'cut_{c}.png' for c in lost)}"] if lost else []
    lines = [f"# Render report: {P['video']['video_id']}", "",
             f"**{'FAIL' if hard else 'PASS'}**, {m['v'] / 60:.1f} min", "",
             "| Check | Result |", "|---|---|",
             f"| Loudness | {m['lufs']:.1f} LUFS (target {float(P['settings']['lufs']):.0f}), peak {m['peak']:.1f} dBTP |",
             f"| Audio/video length | {m['a']:.2f} s / {m['v']:.2f} s |",
             f"| AI shots | {len(sh)} shots on {len({n for n, _ in sh})} pictures, average {avg:.1f} s, "
             f"longest {max((d for _, d in sh), default=0):.1f} s |",
             f"| Lettering on the word | {', '.join(hits) or 'none'} |",
             f"| Subscenes (lines with 2+ pictures) | {sum(len(tl['s']['images']) > 1 for tl in tls if tl['s']['template'] == 'full')} |",
             f"| Green-screen callouts | {', '.join(calls) or 'none'} |",
             f"| Text on screen | {', '.join(said) or 'none'} |",
             f"| Pictures / shots per minute | {pace['pics'] / (m['v'] / 60):.1f} unique AI pictures, "
             f"{len(sh) / (m['v'] / 60):.1f} AI shots (aim 10-14 / 18-24) |",
             f"| Hard cuts | {pace['hard']:.0%} of all cuts (aim 70-85%), {pace['repeats']} repeated camera moves |",
             f"| Beats | {', '.join(f'{b} {n}' for b, n in Counter(b or 'plain' for tl in tls for b in tl['s'].get('beats', [])).most_common())} |",
             f"| Effects, ambience, risers | {', '.join(fxs) or 'none'}; "
             f"{sum(bool(R.AMBIENCE.get(tl['s'].get('beat'))) for tl in tls)} beds, {len(R.reveals(tls))} risers |",
             f"| Black over 1 s | {len(m['black'])} |", f"| Frozen over 6 s | {len(m['frozen'])} |", ""]
    lines += ["## Failures", *[f"- {x}" for x in hard], ""] if hard else []
    lines += ["## Warnings", *[f"- {x}" for x in warn], ""] if warn else []
    open(os.path.join(outdir, "report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    return 1 if hard else 0


def _selfcheck():
    S = dict(lufs=-14)
    ok = dict(v=60.0, a=59.95, lufs=-14.2, peak=-1.6, black=[], frozen=[])
    assert verdict(S, [(1, 2.0)], ok) == ([], [])
    hard, warn = verdict(S, [(1, 6.2)], dict(ok, lufs=-16.5, a=58.0, black=[(3.0, 4.5)]))
    assert len(hard) == 3 and "scene 1 6.2s" in warn[0], (hard, warn)
    assert verdict(S, [], dict(ok, black=[(132.3, 143.0)]), [(131.2, 144.3)]) == ([], [])  # warp starfield is not a hole
    _, warn = verdict(S, [], ok, (), dict(pics=4, hard=.5, repeats=5))
    assert len(warn) == 3 and "per minute" in warn[0], warn  # 4 pictures in a minute, half fades, 5 repeats


if __name__ == "__main__":
    _selfcheck()
    sys.exit(main(*sys.argv[1:5]))
