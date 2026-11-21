# Lore Engine

Turns one Excel sheet into a finished YouTube lore video (Marvel, DC, The Boys, Invincible...), its Shorts,
thumbnail, captions, chapters and description. Everything renders on free GitHub Actions runners with
open-source models. No paid APIs, no AI assistant needed to run it, and almost no work for your own computer.

```
content/lore_content.xlsx ──► plan ──┬─► refs (character faces) ─► images ×16 runners (FLUX.2-klein) ─┐
                                     └─► voice (Chatterbox narrator + Whisper word timings) ─────────┴─► render
render: comic print pass ► camera moves ► ffmpeg transitions / green screen / grade ► mixed audio
        ► video.mp4 + shorts/*.mp4 + thumbnail.jpg + captions.srt + description.txt + metadata.json
```

## Make a video (all from the terminal, no AI needed)

```
./lore.sh sheet               # 1. write the video: a Videos row + its Scenes rows (VID001 is a full example)
./lore.sh draft  VID001       # 2. preview voice and pacing with placeholder art (~15-20 min)
./lore.sh render VID001       # 3. full video + Shorts + thumbnail (first time ~2 h, re-renders reuse images)
./lore.sh status              #    what is running and how far along
./lore.sh watch               #    follow the newest run, download when done
./lore.sh get VID001          #    download the newest finished result
```

Results land in `out/VID001/`. Upload `video.mp4`, `thumbnail.jpg` and `captions.srt`, and paste `description.txt`
(it already has the chapter timestamps). Upload `shorts/*.mp4` with titles from `shorts.json`, and set each Short's
*Related video* to the long video. That link is the funnel.

Your laptop only needs `git` and the GitHub CLI (`gh auth login`). All rendering happens on GitHub Actions.
No terminal? **GitHub → Actions → render → Run workflow**, then download the `VID001-final` artifact.

## Keeping characters consistent

The first render generates a reference image per character (`out/VID001/refs/ref_GALACTUS.png`) and feeds it to every
scene that character appears in. When you like a face, commit it as `assets/characters/GALACTUS.png`. From then on,
every video uses that exact design, which is how the channel gets a recognisable look.

## Make it yours (optional assets)

| Folder | What to drop in |
| --- | --- |
| `assets/music/<mood>/` | Your BGM, only if its license allows redistribution (this repo is public). Otherwise paste a direct download link in **Videos → music_url**. No music = voice + SFX only. |
| `assets/voice/` | A clean 10-20 s recording of the narrator (your own voice is best); set **Settings → voice_ref** to its file name. |
| `assets/pages/` | Real comic pages to explain panel by panel (or use web links, see below). |
| `assets/greenscreen/` | `name.png` or `name.mp4` with a pure #00FF00 area. Use it as template `gs:name`. Built in: `gs:tv`, `gs:monitor`, `gs:page`, `gs:file`. |
| `assets/broll/` | Clips used with `visual = broll:file.mp4`. `broll:stars` is a built-in warp starfield. |
| `assets/characters/` | Approved character reference images (see above). |

Settings (voice, loudness, print look, Shorts length, channel name) live in the **Settings** tab.

## Explaining real comic pages (sliding window)

Put `page:` in a scene's **visual**, using a file in `assets/pages/` or a web link:

| visual | What the camera does |
| --- | --- |
| `page:ff48_p05.jpg` | Shows the whole page |
| `page:ff48_p05.jpg#2` | Slides to panel 2 and spotlights it (panels are detected automatically) |
| `page:https://site/page.jpg@0.1,0.2,0.6,0.5` | Frames any box by hand (fractions of the page) |

Give each panel its own row and narration. Consecutive rows on the same page glide from panel to panel with no cut.
Run a draft first to check the framing. Public-domain comics (Digital Comic Museum, Comic Book Plus, Wikimedia
Commons) are free to use. For Marvel/DC pages, show them briefly while you comment on them, never use piracy
sites, and don't commit their scans to this public repo (use links).

## Models and licenses (all commercial-use friendly)

| Part | Tool | License |
| --- | --- | --- |
| Images | FLUX.2-klein-4B (GGUF Q8) + Qwen3-4B encoder via stable-diffusion.cpp | Apache-2.0 / MIT |
| Upscale | Real-ESRGAN x4plus anime | BSD-3 |
| Voice | Chatterbox-Turbo (narrator), faster-whisper (timing), Kokoro-82M (reference / fallback) | MIT / MIT / Apache-2.0 |
| Fonts | Bangers, Anton, Comic Neue | OFL |
| Video/audio | ffmpeg, OpenCV, Pillow | LGPL/GPL, Apache, HPND |

## Before you publish

- **Fact-check the script.** The sample Galactus script cites its issues in `sources`; check every claim. Accuracy is what earns a lore audience.
- **YouTube's inauthentic-content policy** demonetises mass-produced, template-only videos. Your human-written script and analysis are what keep this channel on the right side of it. Vary structure, add opinion, and never bulk-publish thin videos.
- **Characters are other companies' IP.** Commentary and analysis is the norm for lore channels, but don't use official logos, film stills or actors' likenesses in prompts.
- **AI disclosure:** stylised comic art usually doesn't need YouTube's "altered or synthetic" label. Tick it if a shot could pass as real footage.

## Troubleshooting

- *Sheet problems: ...*: the `plan` job lists every bad cell (unknown template, missing character, bad `=N` reference). Fix the cells and run again.
- *An image shard failed*: on the run page, use **Re-run failed jobs**. Finished shards are kept.
- *A word sounds wrong*: add it to the **Pronounce** tab (`Uatu` → `Wah-too`).
- *Too slow*: lower `image_width/height`, set `upscale` to `no`, or render a range like `1-20` first.
- *Re-rendering after a small change*: images are cached per video, so changing music, narration or one visual
  only redraws what changed.
