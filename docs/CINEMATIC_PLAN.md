# Cinematic quality upgrade: audit and plan

The 18-section "cinematic lore documentary" prompt, checked against what Lore Engine already does. The flow stays the
same (sheet → GitHub Actions → release), as do the free CPU runners and the genre.

## 1. Limits that override the prompt

| Prompt asks | Decision | Why |
|---|---|---|
| §16 generative video for 5-15% of the timeline | **Skip** | Project rule: no GPU, no paid APIs, no video generation. All motion comes from stills. |
| §2 "reuse images through crops, close-ups, alternate compositions" | **Skip** | You watched VID002 and said the re-framed angles look unprofessional and too aggressive. Every new shot is a new picture. |
| §5 background inpainting behind cut-out characters | **Skip** | It needs a second diffusion pass per picture (+36 min each). Depth parallax and green-screen characters give the layers instead. |
| §12 leitmotifs, licensed foley packs | **Skip**, unless you supply files | There's one music track (your Suno one). Effects are synthesised or come from files you drop into `assets/sfx/`. |
| §2 10-14 unique images per minute | **Fits the budget** | Each FLUX picture takes ~36 min. 20 runners × 7 pictures = 140 per video: 14/min for a 10-min video, about 5 h of runner time. Flow draws the shots without famous characters in minutes on your laptop. |

## 2. Audit

Status: **Done**, **Stage A** (written, not yet merged), **Partial**, **Missing**.

| § | Capability | Status | What exists / what's missing |
|---|---|---|---|
| 1 | Sentence → visual beats, on narration timestamps | Stage A | Long lines are split into 2-4 parts, each with its own new picture, cut on the word the part starts (whisper word timings). It splits on punctuation and clause words; it doesn't yet look for a **change of subject** (new character, place, object). Stage B adds that. |
| 2 | Density targets (long and short form) | Stage A | `shot_seconds` 3.5, `image_budget` 140. The report doesn't yet show images/min or shots/min (Stage H). Shorts reuse the long-form pictures. |
| 3 | Shot choice by meaning | Partial | Stage A rotates close-up, wide, low angle, over-the-shoulder, insert. **Missing:** choosing by meaning (scale → extreme wide, threat → low angle, emotion → close-up, artifact → insert, loneliness → high-angle wide, reveal → silhouette). |
| 4 | Contextual B-roll | Partial | `broll:` rows (starfield, files in `assets/broll/`). Stage A subscenes are on-topic by construction, since each is drawn from its own words. **Missing:** artifact and location inserts chosen by meaning (Stage B). No stock filler, by design. |
| 5 | Layered compositing | Partial | Depth Anything V2 parallax on every AI picture; green-screen characters with key, despill, colour match, ink outline and shadow; particle layers. |
| 6 | Camera by narrative intent | Partial | Stage A: slow, small moves only (zoom ≤ 1.06), each picture alternating its move. **Missing:** intent (push in on a threat, pull out on loneliness, pan on scale, still on tragedy), rack focus and depth of field from the depth map (Stage C). |
| 7 | Green screen and transparent overlays | Partial | Chroma-key characters (Stage A: spaced by time, bigger, alternating sides); `gs:` frame templates; particles (embers, dust, snow, rain, stars), flash, glitch. **Missing:** a categorised effects library chosen by meaning (Stage E). |
| 8 | Comic storytelling | Done (mostly) | Real comic pages framed panel by panel, page then AI "match" shot, page-turn whip, issue cards, split panels, dossier cards, halftone print pass. Provenance: pages come only from Commons or your private repo. |
| 9 | Lore motion graphics | Partial | Dossier card, issue card, chapter cards, quote cards; Stage A adds name plates (name, alias, first appearance from the Characters tab). **Missing:** power-level bars, timelines for years in the narration, comparisons (Stage D). Everything comes from sheet columns, never invented. |
| 10 | Transitions, mostly cuts | Done | Stage A: auto transitions are only cuts and soft fades; fade to black on new chapters; page-turn whip only between comic pages. |
| 11 | Colour grading | Partial | One grade per style (Styles `grade`), vignette, grain, halftone. **Missing:** a grade per mood (cosmic horror, tragedy, heroic...) (Stage F). |
| 12 | Sound design | Partial | Sidechain ducking, two-pass loudnorm to -14 LUFS / -1.5 dBTP, impact, whoosh, riser, rumble and glitch, impact on the lettered word. **Missing:** ambience bed by mood, a riser and silence before revelations, J/L cuts at chapters (Stage G). |
| 13 | Narrative classification | Missing | Stage B: a rules classifier per beat (intro, power, origin, cosmic, revelation, tragedy, multiverse, comic history) from keywords and sheet data. It drives shot type, camera, grade, effects and sound. |
| 14 | Pacing, variety | Partial | Stage A gives a new picture every ~3.5 s and text beats at most every 15 s. **Missing:** a report on repeated compositions and camera moves (Stage H). |
| 15 | Character consistency | Done | Reference sheet per character (FLUX `-r`), looks restated in every prompt, `drawn_by=flux` for famous characters. Versions: add one Characters row per version (THANOS_1991). |
| 16 | Generative video | Skipped | See section 1. |
| 17 | Quality control | Partial | `report.md`: loudness, A/V sync, black, frozen, blur, duplicates, text in the art, missing cut-outs. **Missing:** images/min, shots/min, repeated moves, transition share, text in safe areas (Stage H). Fixing one scene: `./lore.sh render VID 12-14`; the cache redraws only what changed. |
| 18 | Modular, switchable | Done | Each feature has a Settings switch; `off` restores the old look. |

**Status 2026-11-26:** Stages A-H are built on `feature/cinematic` (commits cf7164f..c534121), with self-checks and
local renders. VID002 render run 45751953047 has Stage A only.

## 3. Stages (one commit each, self-checks plus a local render before the next)

All stages after A are compositing and audio only: **no new pictures and no extra image time**.

- **A. Subscenes, gentle camera, name plates, punch text, green screen** (written, waiting for your render check). Also: redraws only when they fit the 330-min runner limit, and 20 image runners.
- **B. Meaning.** A beat classifier (§13) and splits that prefer a change of subject (§1). Shot type by meaning (§3, §4: artifact and location inserts).
- **C. Camera by intent** (§6): push, pull, pan or hold chosen by the beat's meaning; rack focus and shallow depth of field from the depth map we already compute.
- **D. Lore graphics** (§9): power-level bars from `threat_level` and `powers`, a year timeline when the narration gives a date, a two-character comparison card. All animated in step with the narration.
- **E. Effects by meaning** (§7): procedural light rays, fog, energy aura on cut-outs, HUD scan, film damage, one at a time and only when the beat calls for it. Your own alpha PNG/WebM files in `assets/fx/` are used when present.
- **F. Mood grading** (§11): a gentle grade per beat type on top of the style grade.
- **G. Sound by meaning** (§12): an ambience bed by mood, a riser and a short music dip before revelations, sound bridges across chapters.
- **H. QC** (§17): images/min, shots/min, repeated moves and compositions, transition share and safe-area checks in `report.md`.
