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

start() { # $1 video id, $2 scenes, $3 draft true/false
  need
  git add content assets
  git diff --cached --quiet || git commit -qm "content: $1"
  git push -q
  gh workflow run "$WF" -f video_id="$1" -f scenes="$2" -f draft="$3"
  sleep 8 # shortcut: assumes the newest run is ours; fine for one person triggering renders
  watch_run "$(latest)"
}

case "${1:-help}" in
  render) start "${2:?usage: ./lore.sh render VID001 [1-8]}" "${3:-}" false ;;
  draft) start "${2:?usage: ./lore.sh draft VID001}" "" true ;;
  status) need; status ;;
  watch) need; watch_run "$(latest)" ;;
  get) need; get "${2:?usage: ./lore.sh get VID001}" ;;
  releases) need; gh release list --limit 10 && echo "https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases" ;;
  sheet) xdg-open content/lore_content.xlsx >/dev/null 2>&1 & ;;
  *) usage ;; # anything unknown only prints help: it never triggers a render
esac
