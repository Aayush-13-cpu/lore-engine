# Lore Engine

Excel sheet → GitHub Actions → finished YouTube comic video, Shorts, thumbnail and captions, published as a GitHub
Release. All compute runs on free CPU runners; the laptop only edits the sheet and drives Google Flow.

## Commands
```
./lore.sh draft  VID002          # placeholder art + real voice, fast pacing check
./lore.sh render VID002 [1-12]   # full render (or a scene range) -> release VID002-r<run>
./lore.sh flow   VID002          # draw non-character shots in Google Flow on this laptop -> release flow-VID002
./lore.sh status | watch | get VID002 | releases
python -m pipeline.sheet VID002 > /dev/null          # validate the sheet (needs openpyxl)
PYTHONPATH=. python -c "from pipeline import render; render._selfcheck()"   # also check, shorts, voice
```
Local Python with the deps: `~/.cache/lore-engine-venv/bin/python`.

## Map
| Path | Role |
|---|---|
| `content/lore_content.xlsx` | Videos, Scenes (one row = one narration line), Characters (`drawn_by` flux/flow), Sources, Styles, Settings |
| `pipeline/sheet.py` | sheet → `plan.json`, validates cells, `DEFAULTS` for Settings |
| `pipeline/images.py` | FLUX.2-klein on CPU (30-44 min/image), image cache, redraw gates (panel border, text in the art) |
| `flow/flow_images.py` | Google Flow via Playwright, 4 per batch; famous characters stay with FLUX |
| `pipeline/voice.py` | Chatterbox narration per line + faster-whisper word timings |
| `pipeline/render.py` | timeline, shot cuts, word-timed lettering, camera, parallax, transitions, two-pass loudnorm |
| `pipeline/check.py` | `report.md`: loudness, A/V length, black frames, shot lengths, image QA |
| `pipeline/shorts.py`, `comic.py`, `sources.py`, `storyboard.py` | Shorts, comic look, page sources, page OCR |
| `.github/workflows/render.yml` | plan → refs → images ×12 → voice → clips → shorts → render → check → release |

## Rules
- Never pirate; never commit Marvel/DC scans (public repo). Public-domain pages come from Commons.
- Never prompt Flow with lookalikes of copyrighted characters: those rows are `drawn_by=flux`.
- No GPU, no paid APIs, no video generation: all motion comes from still images.
- New generated images are the expensive stage; add motion by re-framing pictures we already have.
- Every module keeps its `_selfcheck()` passing; Settings switches (`shot_cuts`, `auto_sfx`) set to `off` restore the old look.

Details: `.claude/skills/comic-video-pipeline/SKILL.md`, `docs/UPGRADE_PLAN.md`, `COMICS.md`.
