#!/usr/bin/env bash
# Render media/mission2.npz to PNG frames (resumable: finished frames are kept, half-written ones re-rendered),
# then compose the HUD video.   BLENDER=/path/to/blender OUT=/path/to/frames/ render/render_mission2.sh
set -e
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
mkdir -p "$OUT"
find "$OUT" -name 'f_*.png' -size 0 -delete          # placeholders left by an interrupted render
$B -b -P render/animate_mission.py -- --mission media/mission2.npz --res 960x540 --samples 6 --out "${OUT}f_"
python render/compose_video.py --frames "$OUT" --mission media/mission2.npz --out media/gyra_mk21_mission.mp4
