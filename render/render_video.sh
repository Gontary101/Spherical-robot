#!/usr/bin/env bash
# Render queue for the GYRA Mk2.1 video (resumable: every renderer skips frames already on disk).
#   render/render_video.sh <frames_dir>
set -u
OUT=${1:?frames dir}
cd "$(dirname "$0")/.."
mkdir -p "$OUT/robot" "$OUT/mission"
for sh in title assembly xray roll; do
  nice -n 19 blender -b -P render/robot_shots.py -- --shot $sh --out "$OUT/robot" --res 1280x720 --samples 10 --step 2 > "$OUT/render_$sh.log" 2>&1
done
nice -n 19 blender -b -P render/animate_showcase.py -- --mission media/showcase.npz --out "$OUT/mission" --res 1280x720 --samples 8 --step 2 > "$OUT/render_mission.log" 2>&1
touch "$OUT/ALL_DONE"
