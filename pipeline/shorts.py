"""Vertical Shorts (1080x1920) cut from a video's strongest moments, re-rendered from the source art.

usage: [DRAFT=1] python -m pipeline.shorts work/plan.json work/images work/audio out/VID001
Windows come from the Shorts sheet when filled, otherwise from the scoring below.
"""
import json
import os
import re
import subprocess
import sys
import zlib

import numpy as np

from pipeline import render as R

W, H = 1080, 1920
PAD = 0.12  # tighter gaps than long-form: Shorts reward pace
HOOKY = re.compile(r"\b(never|nobody|no one|secret\w*|truth|real|actually|only|first|last|before|kill\w*|die[sd]?|death|"
                   r"dead|destroy\w*|devour\w*|eat\w*|ate|god\w*|strongest|powerful|universe|why|how|what if|imagine|"
                   r"wasn't|didn't|isn't|can't|won't|every|billions?|millions?|end)\b", re.I)
WEAK_START = re.compile(r"^(and|but|so|which|that|then|this|these|it|its|he|she|they|his|her|their|him|them|there)\b", re.I)
SKIP = {"title", "sting", "quote", "endscreen", "dossier", "split", "issue"}  # landscape-only layouts: a vertical crop cuts their text


def score(s):
    if s["short_score"]:
        return s["short_score"]
    t = s["narration"]
    return min(10.0, 2 * t.count("?") + 1.2 * len(HOOKY.findall(t)) + 2 * (s["sfx"] in ("impact", "riser"))
               + (s["motion"] in ("punch", "shake")))


def pick(scenes, meta, count, lo, hi):
    """Best non-overlapping windows of whole scenes, lo..hi seconds, that open on a strong self-contained line."""
    dur = [meta[f"s{s['n']:03d}"]["dur"] + PAD if f"s{s['n']:03d}" in meta else 0 for s in scenes]
    cands = []
    for i, first in enumerate(scenes):
        total = 0.0
        for j in range(i, len(scenes)):
            if scenes[j]["template"] in SKIP or not dur[j]:
                break
            total += dur[j]
            if total > hi:
                break
            if total >= lo:
                seg = scenes[i:j + 1]
                v = (1.5 * score(first) + sum(map(score, seg)) / len(seg)
                     - 4 * bool(WEAK_START.match(first["narration"]))
                     + scenes[j]["narration"].rstrip().endswith(("?", "!", "...", "…")))
                cands.append((v, i, j))
    chosen = []
    for v, i, j in sorted(cands, reverse=True):
        if all(j < a or i > b for _, a, b in chosen):
            chosen.append((v, i, j))
        if len(chosen) == count:
            break
    return sorted((i, j) for _, i, j in chosen)


def word_groups(tls, max_words=3, max_chars=18):
    groups = []
    for tl in tls:
        cur = []
        for w, a, b in tl["words"]:
            cur.append((w.upper(), tl["voice"] + a, tl["voice"] + b))
            if len(cur) >= max_words or len(" ".join(x[0] for x in cur)) >= max_chars or w[-1] in ".?!,;:":
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    return groups


def ass(groups, hook, total, accent, cta_from):
    bgr = "".join(reversed(re.findall("..", accent.lstrip("#")))).upper()
    head = R.ass_header(W, H, 116, 0, 5, "Anton", 8).replace(
        "\n\n[Events]",
        f"\nStyle: Hook,Bangers,92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,2,0,3,18,0,8,70,70,190,1"
        f"\nStyle: Cta,Anton,66,&H0000E5FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,300,1"
        "\n\n[Events]")
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{R.ass_time(0)},{R.ass_time(total)},Hook,,0,0,0,,{hook.upper()}")
    ev.append(f"Dialogue: 1,{R.ass_time(cta_from)},{R.ass_time(total)},Cta,,0,0,0,,FULL STORY ON THE CHANNEL")
    for g, nxt in zip(groups, groups[1:] + [None]):
        for k, (w, a, b) in enumerate(g):
            end = g[k + 1][1] if k + 1 < len(g) else min(b + 0.35, nxt[0][1] if nxt else total)
            words = [f"{{\\c&H{bgr}&\\t(0,90,\\fscx100\\fscy100)\\fscx118\\fscy118}}{x}{{\\r}}" if i == k else x
                     for i, (x, _, _) in enumerate(g)]
            ev.append(f"Dialogue: 0,{R.ass_time(a)},{R.ass_time(max(end, a + 0.05))},Default,,0,0,0,,"
                      f"{{\\an5\\pos({W // 2},{int(H * .64)})}}{' '.join(words)}")
    return head + "\n".join(ev) + "\n"


def shots(tl, seed):
    """One scene -> 1-2 camera shots: a slow pan across the wide art, then a punched-in close-up."""
    n, rng = tl["F"], np.random.default_rng(seed)
    a, b = (.3, .55) if rng.random() < .5 else (.7, .45)
    fy = rng.uniform(.4, .5)
    pan = [(0, 1.0, a, .5), (1, 1.06, b, .5)]
    if n < 2.2 * R.C.fps:
        return [(n, pan)]
    focus = rng.uniform(.38, .62)
    return [(n // 2, pan), (n - n // 2, [(0, 1.32, focus, fy), (1, 1.4, focus + .03, fy - .02)])]


def main(plan_path, imgdir, audiodir, outdir):
    P = json.load(open(plan_path))
    meta = json.load(open(os.path.join(audiodir, "words.json")))
    S, V = P["settings"], P["video"]
    work = os.path.join(outdir, "_work", "shorts")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(outdir, "shorts"), exist_ok=True)
    R.init(P, imgdir, work)
    fps = R.C.fps
    sc = P["scenes"]
    pos = {s["n"]: i for i, s in enumerate(sc)}
    if P["shorts"]:
        wins = [(pos[h["start"]], pos[h["end"]], h) for h in P["shorts"] if h["start"] in pos and h["end"] in pos]
    else:
        wins = [(i, j, {}) for i, j in pick(sc, meta, int(float(S["shorts_count"])), float(S["shorts_min"]), float(S["shorts_max"]))]
    out = []
    for k, (i, j, h) in enumerate(wins, 1):
        scenes = [dict(s, transition="cut", hold=0.0) for s in sc[i:j + 1]]
        hook = h.get("hook_text") or scenes[0]["caption"] or V.get("thumbnail_text", "")
        if h.get("hook_line") and f"hook{h['n']}" in meta:  # spoken cold-open line before the clip
            scenes.insert(0, dict(scenes[0], voice_key=f"hook{h['n']}", narration=h["hook_line"], sfx="",  # no sound effects on cuts
                                  sfx_text="", caption=""))
        if "cta" in meta:  # funnel: stop on the cliffhanger, then send viewers to the long video for the ending
            scenes.append(dict(scenes[-1], voice_key="cta", narration=S["shorts_cta"], sfx="", sfx_text="", caption=""))
        tls = R.timeline(dict(P, scenes=scenes), meta, fps, pad=PAD)
        total = sum(tl["F"] for tl in tls) / fps
        voice, sfx = R.tracks(tls, audiodir, total, fps)
        mixed = os.path.join(work, f"mix{k}.wav")
        R.mix(voice, sfx, total, mixed, P, work)
        subs = os.path.join(work, f"short{k}.ass")
        with open(subs, "w") as f:
            f.write(ass(word_groups(tls), hook, total, R.C.accent, tls[-1]["voice"] if "cta" in meta else max(total - 1.8, 0)))
        dst = os.path.join(outdir, "shorts", f"short{k}.mp4")
        grade = R.C.style.get("grade") or "eq=contrast=1.04:saturation=1.08"
        fc = (f"[0:v]{grade},vignette=angle=PI/5,ass={subs}:fontsdir=assets/fonts[g];"
              f"[g][2:v]overlay=x='-w+w*t/{total:.3f}':y=H-h:shortest=1[v]")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                              "-r", str(fps), "-i", "-", "-i", mixed, "-f", "lavfi", "-i",
                              f"color=c=0x{R.C.accent.lstrip('#')}:s={W}x14:r={fps}", "-filter_complex", fc,
                              "-map", "[v]", "-map", "1:a", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium",
                              "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                              dst], stdin=subprocess.PIPE)
        for tl in tls:
            s = tl["s"]
            seed = zlib.crc32(f"{V['video_id']}short{k}{s['n']}".encode())
            canvas, over, keys0, reveal = R.build(s, tl, W, H, scale=1.45, captions=False)
            # comic pages keep their own camera path (it slides to each panel being explained)
            for n, keys in [(tl["F"], keys0)] if s["template"] in ("panels", "page") else shots(tl, seed):
                for fr in R.frames(canvas, keys, n, W, H, over, s, 0.0, reveal, seed):
                    p.stdin.write(fr.tobytes())
                over = []  # SFX lettering pops once per scene, not again on the punch-in
        p.stdin.close()
        if p.wait():
            sys.exit(f"ffmpeg failed on {dst}")
        first = re.split(r"(?<=[.!?])\s", scenes[0]["narration"])[0]
        title = h.get("title") or (hook.title() if hook else first)[:90]
        out.append(dict(file=f"shorts/short{k}.mp4", title=f"{title} #shorts", scenes=[s["n"] for s in sc[i:j + 1]],
                        seconds=round(total, 1),
                        description=f"Full story: {V['title']}\n(Link the full video as this Short's Related Video.)\n"
                                    f"{V.get('hashtags', '')} #shorts"))
        print(f"short {k}: scenes {sc[i]['n']}-{sc[j]['n']} {total:.1f}s", flush=True)
    json.dump(out, open(os.path.join(outdir, "shorts.json"), "w"), indent=1)


def _selfcheck():
    s = lambda n, text, tpl="full": dict(n=n, narration=text, template=tpl, short_score=0, sfx="", motion="auto")
    sc = [s(1, "He walked."), s(2, "Galactus was never born in our universe."), s(3, "He ate a billion worlds."),
          s(4, "", "title"), s(5, "Why did he spare Earth?")]
    meta = {f"s{i:03d}": dict(dur=d) for i, d in ((1, 10), (2, 12), (3, 14), (5, 30))}
    # "He walked." (weak start) is skipped; the title card breaks windows; scene 5 alone is a valid 30 s Short
    assert pick(sc, meta, 2, 25, 58) == [(1, 2), (4, 4)], pick(sc, meta, 2, 25, 58)


if __name__ == "__main__":
    _selfcheck()
    main(*sys.argv[1:5])
