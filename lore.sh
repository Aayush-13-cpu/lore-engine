#!/usr/bin/env bash
# Lore Engine, driven entirely from the terminal. Needs only git + gh (logged in); no AI, no local rendering:
# this laptop edits the sheet, pushes, and downloads results. All rendering runs on GitHub Actions.
set -euo pipefail
cd "$(dirname "$0")"
WF=render.yml

usage() {
  cat <<'EOF'
Lore Engine - make a video from the terminal (rendering runs on GitHub, not this laptop)

  ./lore.sh sheet               open the content sheet to write a video
  ./lore.sh storyboard VID001   read the video's real comics: numbered panels + lettering + ready Scenes rows
  ./lore.sh flow VID001         non-public-domain videos (Marvel, DC, Boys, Invincible): draw images in Google Flow
  ./lore.sh draft  VID001       fast preview: real voice + placeholder art (~15-20 min)
  ./lore.sh render VID001       full video + Shorts + thumbnail (first time ~2 h; re-renders reuse cached images)
  ./lore.sh render VID001 1-8   only scenes 1-8
  ./lore.sh status              what is running and how far along it is
  ./lore.sh watch               follow the newest run and download it when it finishes
  ./lore.sh get VID001          download the newest finished result into out/VID001
  ./lore.sh releases            finished videos published on GitHub (video, Shorts, thumbnail, captions)

Ctrl-C while watching is safe: the render keeps going on GitHub.
EOF
}

need() {
  gh auth status >/dev/null 2>&1 || { echo "GitHub login missing: run  gh auth login -h github.com --with-token"; exit 1; }
}

latest() { # the run still queued/rendering, else the newest one
  gh run list --workflow "$WF" --limit 10 --json databaseId,status \
    -q '([.[] | select(.status != "completed")][0] // .[0]) | .databaseId'
}

status() {
  gh run list --workflow "$WF" --limit 5
  local id; id=$(latest)
  [ -n "$id" ] || return 0
  echo
  gh run view "$id" --json displayTitle,status,jobs -q \
    '"\(.displayTitle): \(.status) - \([.jobs[] | select(.status == "completed")] | length)/\(.jobs | length) jobs done",
     (.jobs[] | "  \(.name): \(.status) \(.conclusion // "")")'
}

get() { # $1 video id, $2 optional run id
  local run=${2:-}
  [ -n "$run" ] || run=$(gh run list --workflow "$WF" --status success --limit 30 --json databaseId,displayTitle \
    -q "[.[] | select(.displayTitle | startswith(\"render $1 \"))][0].databaseId // empty")
  [ -n "$run" ] || { echo "No finished render of $1 yet. Check: ./lore.sh status"; exit 1; }
  rm -rf "out/${1:?}"
  gh run download "$run" -n "$1-final" -D out
  echo "Done -> out/$1  (video.mp4, shorts/, thumbnail.jpg, captions.srt, description.txt, shorts.json)"
}

watch_run() { # $1 run id
  local vid; vid=$(gh run view "$1" --json displayTitle -q '.displayTitle' | awk '{print $2}')
  echo "Run $1 ($vid): $(gh run view "$1" --json url -q .url)"
  echo "Watching... (Ctrl-C is safe; resume later with ./lore.sh watch)"
  if gh run watch "$1" --interval 60 --exit-status >/dev/null; then
    get "$vid" "$1"
    echo "Watch / share it on GitHub: $(gh release list --limit 1 --json tagName -q '.[0].tagName' | xargs -r gh release view --json url -q .url)"
  else
    gh run view "$1" --log-failed | tail -60
    echo "Failed. Fix the cause, then run the same command again (finished images are reused)."
    exit 1
  fi
}

flow() { # videos whose comics we cannot use (Marvel, DC, ...): draw the images in Google Flow in your own Chrome, then publish them for the runners
  need
  local py="$HOME/.cache/lore-engine-venv/bin/python"
  [ -x "$py" ] || py=python3 # needs openpyxl + playwright: pip install openpyxl playwright pillow
  mkdir -p "out/${1:?}"
  "$py" -m pipeline.sheet "$1" > "out/$1/plan.json"
  # VS Code's snap terminal leaks its GTK/GIO paths, which crash Chrome (GLib settings schema error)
  env -u GSETTINGS_SCHEMA_DIR -u GTK_PATH -u GTK_EXE_PREFIX -u GIO_MODULE_DIR -u GDK_PIXBUF_MODULE_FILE \
    -u GDK_PIXBUF_MODULEDIR -u GTK_IM_MODULE_FILE -u LOCPATH \
    XDG_DATA_DIRS="${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-${XDG_DATA_DIRS:-/usr/share}}" \
    "$py" flow/flow_images.py "out/$1/plan.json" "out/$1/flow"
  gh release view "flow-$1" >/dev/null 2>&1 ||
    gh release create "flow-$1" --prerelease --title "Flow images for $1" --notes "Google Flow images used by renders of $1"
  gh release upload "flow-$1" "out/$1"/flow/*.png --clobber
  echo "Flow images published. Now render with: ./lore.sh render $1"
}

push_sheet() {
  need
  git add content assets
  git diff --cached --quiet || git commit -qm "content: $1"
  git push -q
}

storyboard() { # $1 video id: numbered panels + OCR lettering + Scenes rows for every page of its Sources
  push_sheet "$1"
  gh workflow run storyboard.yml -f video_id="$1"
  sleep 8
  local run; run=$(gh run list --workflow storyboard.yml --limit 1 --json databaseId -q '.[0].databaseId')
  echo "Storyboard run $run: $(gh run view "$run" --json url -q .url)"
  gh run watch "$run" --interval 30 --exit-status >/dev/null || { gh run view "$run" --log-failed | tail -40; exit 1; }
  rm -rf "out/${1:?}/storyboard"
  gh run download "$run" -n "$1-storyboard" -D "out/$1/storyboard"
  echo "Done -> out/$1/storyboard: one image per page with numbered panels + storyboard.xlsx (rows to paste into Scenes)"
}

start() { # $1 video id, $2 scenes, $3 draft true/false
  push_sheet "$1"
  gh workflow run "$WF" -f video_id="$1" -f scenes="$2" -f draft="$3"
  sleep 8 # shortcut: assumes the newest run is ours; fine for one person triggering renders
  watch_run "$(latest)"
}

case "${1:-help}" in
  render) start "${2:?usage: ./lore.sh render VID001 [1-8]}" "${3:-}" false ;;
  draft) start "${2:?usage: ./lore.sh draft VID001}" "" true ;;
  storyboard) storyboard "${2:?usage: ./lore.sh storyboard VID001}" ;;
  flow) flow "${2:?usage: ./lore.sh flow VID001}" ;;
  status) need; status ;;
  watch) need; watch_run "$(latest)" ;;
  get) need; get "${2:?usage: ./lore.sh get VID001}" ;;
  releases) need; gh release list --limit 10 && echo "https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases" ;;
  sheet) xdg-open content/lore_content.xlsx >/dev/null 2>&1 & ;;
  *) usage ;; # anything unknown only prints help: it never triggers a render
esac
