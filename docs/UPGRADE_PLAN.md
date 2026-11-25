# Motion-comic upgrade plan

Goal: stop holding one picture for a whole sentence, and make every video look human-edited. The upgrade keeps the
current flow (Excel sheet → GitHub Actions → release), the genre (comic lore and public-domain comics), and the
Flux and Flow image sources.

## 1. Repo map

| Part | File | What it does |
|---|---|---|
| Content | `content/lore_content.xlsx` | Videos, Scenes (one row = one narration line), Characters, Sources, Styles, Settings, Pronounce |
| CLI | `lore.sh` | `draft`, `render`, `flow`, `storyboard`, `status`, `watch`, `get`, `releases` |
| Plan | `pipeline/sheet.py` | Sheet → `plan.json` (scenes, image jobs, ref jobs, settings) and validates every cell |
| Images | `pipeline/images.py` | FLUX.2-klein-4B Q8 through stable-diffusion.cpp on CPU, Real-ESRGAN 4×, the panel-border quality gate, and an image cache keyed on prompt, seed, refs and model |
| Flow | `flow/flow_images.py` | Google Flow driven by Playwright on the laptop, batches of 4; output goes to release `flow-<VID>` |
| Comic pages | `pipeline/sources.py` | Commons DjVu, cbz or pdf pages from the private assets repo |
| Voice | `pipeline/voice.py` | Chatterbox (or Kokoro) per line, faster-whisper word timings aligned to the script, voice cache |
| Render | `pipeline/render.py` | Timeline, canvases, eased camera keys, depth parallax, overlays, particles, shake, xfade transitions, green screen, mix, captions, chapters |
| Comic look | `pipeline/comic.py` | Print pass (halftone), caption boxes, SFX lettering, cards, dossier, panels, green-screen frames, thumbnail |
| Shorts | `pipeline/shorts.py` | Picks hook moments and renders vertical Shorts with word captions |
| Storyboard | `pipeline/storyboard.py` | Panel detection and Tesseract OCR of real pages |
| CI | `.github/workflows/render.yml` | plan → refs ×N → images ×12-18 → voice → cache → clips ×≤8 → shorts → render → GitHub release |

## 2. How each input is obtained today

| Input | Method | Limits and failure modes |
|---|---|---|
| FLUX images | stable-diffusion.cpp `sd-cli` on the runner CPU, 1024×576, 4 steps, `-r` character refs, then Real-ESRGAN 4× (4096×2304) | **30-44 min per image** on 4 vCPU (measured: s031 2617 s, s045 1815 s). A stray panel border triggers a redraw with a new seed (up to 3 takes). No API, no key, no rate limit. |
| Flow images | Playwright with your signed-in Chrome profile on the laptop, agent panel, 4 per batch, about 1 min per batch | Refuses famous characters (policy), so those go to FLUX via `drawn_by=flux`. Sometimes shows "Something went wrong" (retry, then single shots). Needs a human-signed-in session; it can't run on Actions. |
| Narration | Chatterbox-Turbo (Kokoro fallback) on the runner CPU, one line at a time, cached per line | 16-37 min per video |
| Word timings | faster-whisper on the CPU, aligned to the script with difflib | Already part of the voice job |
| LLM text | None in the pipeline. Scripts are written into the sheet (by Claude in this repo). | The `LLM.json` adapter would have nothing to wrap. |
| Comic pages | Wikimedia Commons (public domain) or the private `lore-assets` repo | Commons needs a User-Agent; pages are upscaled with ESRGAN |

The repo is **public**, so runners are `ubuntu-24.04`/`26.04` with 4 vCPU and 16 GB.

## 3. Measured runtimes (final runs, 2026-11-24)

| Job | VID001 Galactus (77 scenes) | VID002 Infinity Gauntlet (56) | VID003 Dracula (124, mostly pages) |
|---|---|---|---|
| refs | ≤38 min | ≤19 min | <1 min |
| images (12 shards) | **≤227 min** | ≤93 min | ≤3 min |
| voice | 26 min | 17 min | 37 min |
| clips (≤8 shards) | ≤3 min | ≤3 min | ≤4 min |
| shorts | 4 min | 2 min | 1 min |
| render (assemble, mix) | 15 min | 13 min | 18 min |

**What the numbers mean:** image generation is the only expensive stage. Compositing is cheap. Any idea that adds
generated images (plates, masters, cut-outs) multiplies the slowest stage, while ideas that add shots from pictures
we already have cost almost nothing.

The current "static" problem: VID002 holds each AI picture for about 8 s with one camera move.

## 4. The prompt, mapped to this repo

| Prompt item | Status | Decision |
|---|---|---|
| Several shots per sentence | **Gap: the core of the upgrade** | Cut each long `full` scene into 2-4 shots **on the same image** (wide → medium → close, focus taken from the depth map). Cuts land on clause boundaries using the word timings we already have. No extra image generation. |
| Elements appear on their word | Partial: overlays fade or pop at the scene start | Time `sfx_text` (with shake and flash) to its word, and the caption to its word. |
| Eased camera, parallax, shake, particles, halftone, lettering | Already exists | Keep it. |
| Transitions, mostly cuts | Already exists (`AUTO_T`) | Add a `smash` cut (2-frame white flash) on impact lines. |
| QA: gibberish text in AI images | **Gap** | Add a Tesseract check to the existing redraw loop in `images.py` (beside `has_gutter`): a picture with readable or fake text is redrawn with a new seed, once. |
| QA: blur and duplicates | Gap, cheap | Laplacian blur and perceptual-hash checks; logged in the report, not redrawn (a redraw costs 30+ min). |
| Audio: ducking, compression, loudnorm | Already exists (sidechain duck, 1-pass loudnorm) | Make loudnorm two-pass so -14 LUFS / -1.5 dBTP is exact. Add an optional ambience bed per chapter if files exist in `assets/sfx/ambience/`. |
| Final checks and report | **Gap** | The render job writes `report.md` (shot lengths, loudness, A/V sync, black/frozen frames), attaches it to the release, and fails on hard errors. |
| Dry run | Already exists: `./lore.sh draft` (placeholder art, real voice) | Keep it. Add local self-checks for the new logic (the repo's `_selfcheck()` convention). |
| Project skill and CLAUDE.md | Gap | Phase 6. |

### Skipped, with reasons
- **Separate `motion_comic` pipeline, new workflow, `timeline.json`, pydantic:** you asked to keep the current flow and execution. `plan.json` and `render.timeline()` already are the timeline.
- **Location plates, masters and cut-outs per scene; rembg composites:** each needs extra FLUX images at 30-44 min each, and image shards already take up to 227 min (6 h limit).
- **edge-tts:** Chatterbox is already the narrator. edge-tts uses an unofficial Microsoft endpoint that can break or be blocked.
- **open_clip consistency:** it adds a 1 GB+ torch install to every image shard. The character refs already lock faces. Revisit only if drift shows up.
- **Local LLM planner:** the rules planner is enough, and scripts are written in the sheet.
- **24 fps:** compositing isn't the bottleneck, and 30 fps is already set in Settings.
- **pysubs2:** SRT and ASS are already written directly.
- **Remotion:** only if you say "enable Remotion".

## 5. Phases (each one commit on `feature/motion-comic`, tested before the next)

1. **Shot cuts.** `render.py`: split long `full` scenes into sub-shots at word boundaries (min 1.2 s, max 3.5 s, target about 2.3 s), never the same framing twice in a row, with focus from the depth map. Settings switch `shot_cuts = auto | off`; `off` reproduces today's output exactly (the legacy path). Self-check plus a local CPU render of one scene.
2. **Word-timed lettering.** SFX text, shake and flash land on the impact word; captions appear on their word; add a `smash` transition.
3. **Image QA.** Tesseract text check in the redraw loop; blur and duplicate scores in the report. Install `tesseract-ocr` in the refs and images jobs.
4. **Audio.** Two-pass loudnorm, plus an optional ambience bed from `assets/sfx/ambience/<tag>.*` (skipped when the folder is empty).
5. **Checks and report.** `report.md` in the release; the job fails on A/V desync over 1 frame, black frames, or loudness off by more than 1 LU.
6. **Test runs.** Real renders of VID002 and VID003 on the branch; skill in `.claude/skills/comic-video-pipeline/SKILL.md`, repo guide in `CLAUDE.md`.

Time budget: phases 1, 2, 4 and 5 add under 5 min to the clips and render jobs. Phase 3 adds a second or two per
image, plus one redraw (30+ min) only when text is detected.

## 6. Questions for you
1. **Approach:** upgrade in place behind a Settings switch (this plan) rather than a parallel pipeline?
2. **Music license:** the only track is `assets/music/Hidden Powers - usesuno.com.mp3`. Suno's free plan doesn't allow monetized use, and this repo is public. Was it made on a paid Suno plan? If not, replace it or link it through `Videos → music_url`.
3. **Ambience:** you asked earlier to remove transition sound effects. Do you want a quiet ambience bed under the narration (for example a crypt drone for Dracula, a space hum for cosmic stories)? If yes, I'll list exact CC0 files to download.
4. **Test video:** use VID002 scenes 1-12 (sheet-driven) instead of `stories/sample.txt`? The pipeline reads the sheet, not text files.
