#!/usr/bin/env bash
# Render a video on GitHub Actions. This machine only pushes the sheet and downloads the result.
#   ./lore.sh VID001          full render (video + Shorts + thumbnail + metadata)
#   ./lore.sh VID001 draft    placeholder art, real voice: check script and pacing in ~10 min
#   ./lore.sh VID001 1-8      only scenes 1-8 at full quality
set -euo pipefail
vid=${1:?usage: ./lore.sh VIDEO_ID [draft|FROM-TO]}
draft=false scenes=""
case "${2:-}" in draft) draft=true ;; "") ;; *) scenes=$2 ;; esac

git add content assets
git diff --cached --quiet || git commit -qm "content: $vid"
git push -q
gh workflow run render.yml -f video_id="$vid" -f scenes="$scenes" -f draft="$draft"
sleep 8 # shortcut: assumes the newest run is ours; fine for one person triggering renders
run=$(gh run list --workflow render.yml --limit 1 --json databaseId -q '.[0].databaseId')
echo "Run $run: $(gh run view "$run" --json url -q .url)"
echo "Ctrl-C is safe; the render keeps going. Download later with: gh run download $run -n $vid-final -D out"
gh run watch "$run" --interval 60 --exit-status >/dev/null || { gh run view "$run" --log-failed | tail -60; exit 1; }
rm -rf "out/$vid"
gh run download "$run" -n "$vid-final" -D out
echo "Done -> out/$vid"
