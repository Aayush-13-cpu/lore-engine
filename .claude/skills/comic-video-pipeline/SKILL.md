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
1. `sheet.py` splits a long `full` line into 2-4 parts at punctuation or clause words (`shot_seconds`, about 3.5 s each)
   and gives every part after the first its own NEW picture: a different shot type (close-up, wide, low angle,
   over the shoulder, insert) of what that part of the sentence says, in the same setting. `image_budget` (140) caps
   the new pictures, longest lines first. Never crop or re-frame one picture into fake angles: it reads amateur.
2. `render.scene_cuts()` cuts each picture on the word its part starts (`splits` + word timings).
   Camera: one slow, small move per picture (zoom <= 1.06, pan <= 8%), transitions mostly cuts and soft fades.
3. `letter()`: the row's `sfx_text`, else snap/crash/slam/smash/explode/shatter/punch/roar, gets SNAP!, CRASH!, POW!
   on its word (max one per 10 s, AI shots only). Lettering, shake, flash and the impact sound land together.
4. `texts()` (Settings `auto_text`, `text_gap` 15 s between beats): a name plate (name, alias, first appearance from
   Characters) on a character's first mention, else the narration's short punch sentences ("Time. Space. Reality.")
   in big type, each on its word. Write short punch sentences in the script to get them.
5. Green screen: a character named in a line whose pictures don't show them slides in on their name (`cutouts`
   characters, `callout_gap` 20 s apart). One `cut_<ID>.png` per character on chroma green, keyed in `render.cutout()`.
6. Not split: cards, comic pages, b-roll, `||` rows (already several pictures), short lines.

Switches in Settings: `shot_cuts`, `auto_sfx`, `auto_text` (`off` = the old look).

## Anti-slop checklist
- No text inside generated art: `images.py` OCRs the first take and redraws once, only while the rest of the shard still fits in `SHARD_LIMIT_MIN` (default 300).
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
| too many cuts or text | raise `shot_seconds` / `text_gap`, or `shot_cuts=off` / `auto_text=off` |
| images shards near 330 min | lower `image_budget` (each picture ~36 min, 7 per runner on 20 runners) |
