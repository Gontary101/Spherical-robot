#!/usr/bin/env bash
# Render queue for the GYRA Mk2.1 video (resumable: every renderer skips frames already on disk).
#   render/render_video.sh <frames_dir>
# Quality is set by environment variables (defaults = the preview-grade settings used for the repo film):
#   RES=1280x720        Blender resolution (1920x1080, 2560x1440, 3840x2160)
#   STEP=2              2 renders every other frame (interpolated to 24 fps when composing), 1 renders all of them
#   SAMPLES_CAD=10      Cycles samples for the CAD shots (title, assembly, x-ray, roll)
#   SAMPLES_MISSION=8   Cycles samples for the field mission
#   HQ=0                1 adds final-quality light bounces and a tighter adaptive-sampling threshold
#   GYRA_DEVICE=        OPTIX / CUDA / HIP / METAL / ONEAPI to render on a GPU (empty = CPU)
#   SHOTS="title assembly xray roll mission"   subset to render
set -u
OUT=${1:?frames dir}
RES=${RES:-1280x720}
STEP=${STEP:-2}
SAMPLES_CAD=${SAMPLES_CAD:-10}
SAMPLES_MISSION=${SAMPLES_MISSION:-8}
SHOTS=${SHOTS:-"title assembly xray roll mission"}
HQFLAG=""
[ "${HQ:-0}" = "1" ] && HQFLAG="--hq"
cd "$(dirname "$0")/.."
mkdir -p "$OUT/robot" "$OUT/mission"
for sh in $SHOTS; do
  if [ "$sh" = mission ]; then
    nice -n 19 blender -b -P render/animate_showcase.py -- --mission media/showcase.npz --out "$OUT/mission" \
      --res "$RES" --samples "$SAMPLES_MISSION" --step "$STEP" $HQFLAG > "$OUT/render_mission.log" 2>&1
  else
    nice -n 19 blender -b -P render/robot_shots.py -- --shot "$sh" --out "$OUT/robot" \
      --res "$RES" --samples "$SAMPLES_CAD" --step "$STEP" $HQFLAG > "$OUT/render_$sh.log" 2>&1
  fi
done
touch "$OUT/ALL_DONE"
