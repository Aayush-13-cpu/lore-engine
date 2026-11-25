---
name: comic-video-pipeline
description: Make, render, debug or tune a Lore Engine comic video (sheet -> GitHub Actions -> release). Use when adding a video or scenes to content/lore_content.xlsx, running draft/render/flow, reading a render's report.md, fixing a failed render job, or changing shot cuts, lettering, camera, audio mix or image QA in pipeline/.
---

# Comic video pipeline (Lore Engine)

## Stages and where they run
| Stage | Runs on | Cost | Cache |
|---|---|---|---|
| plan (`pipeline/sheet.py`) | runner | seconds | - |
| refs, images (`pipeline/images.py`, FLUX.2-klein CPU) | 12-18 runners | **30-44 min per image** | `img-<VID>-*`, keyed on prompt+seed+refs+settings |
| Flow images (`./lore.sh flow`) | laptop, user's Google session | ~1 min per 4 | release `flow-<VID>` |
| voice (Chatterbox + faster-whisper) | runner | 15-40 min | `voice-<VID>-*`, per line |
| clips, shorts, render (`pipeline/render.py`, `shorts.py`) | runners | 2-20 min | - |
| check (`pipeline/check.py`) | render runner | ~2 min | writes `report.md` |

Rule of thumb: never add generated images to get motion. Re-frame what exists.

## How a line becomes shots
1. `render.timeline()` gives each Scenes row its frames from the narration length plus a breath pause.
2. `letter()` finds the impact word: the row's `sfx_text` lands on its own word, else snap, crash, slam, smash,
   explode, shatter, punch or roar gets SNAP!, CRASH!, POW!... (max one per 10 s, `full` AI shots only, never on pages).
   Lettering, shake, a sheet `flash` and an `impact` sound all fire on that word.
3. `scene_cuts()` cuts a long `full` line in the gaps between words (punctuation and clause words preferred),
   1.2-3.5 s per shot. `cut_plan()` frames shot 1 with the row's own move (the wide), then alternates a close-up
   (1.75×) on the subject and a medium (1.3×) on the group. Focus comes from the depth map, else ink density.
   The canvas is built at 1.4× so close-ups stay sharp. Captions and lettering keep scene time across cuts (`t0`).
4. Not cut: cards (title, sting, quote, endscreen), comic pages (camera walks panel to panel), b-roll, motion
   `static`/`shake`, and lines too short to cut.

Switches in Settings: `shot_cuts` and `auto_sfx` (`off` = the old look).

## Anti-slop checklist
- No text inside generated art: `images.py` OCRs the first take and redraws once (`REDRAW_BUDGET_MIN`, default 180).
- No stray panel borders (`has_gutter` redraw).
- Lookalike prompts for famous characters are forbidden in Flow. Use `drawn_by=flux` and name the character.
- Mostly hard cuts; transitions are silent; one lettering per 10 s at most.
- Loudness: two-pass loudnorm to Settings `lufs` (-14), true peak -1.5.

## Reading report.md (in the release and the run summary)
- **FAIL**: loudness off by >1 LU, true peak > -1 dBTP, audio and video lengths differ by >0.1 s, black >1 s.
  The release is still published; the run turns red after it.
- Warnings: AI shots over 3.5 s (a line with no pause: add a comma or split the row), frozen >6 s, blurry or
  duplicate AI images (redraw by changing the visual or seed).

## Common errors
| Symptom | Fix |
|---|---|
| `Sheet problems:` in plan | Fix the listed cells; `python -m pipeline.sheet VID` locally |
| an images shard hits its timeout | lower `image_width/height` or `upscale=no`, or re-run failed jobs (finished shards are cached) |
| `missing image ... did the images job fail?` | an images shard failed: re-run failed jobs |
| Flow "might violate our policies" | that shot shows a famous character: it falls back to FLUX automatically |
| cuts feel too busy | Settings `shot_cuts=off`, or `motion=static` on that row |
