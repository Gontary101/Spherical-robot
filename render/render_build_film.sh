#!/usr/bin/env bash
# Final render of the build film (resumable: frames on disk are skipped), then composition.
#   render/render_build_film.sh <frames_dir> [workers=2] [threads_per_worker=2]
# Environment: RES (1920x1080), SAMPLES (8). Needs: python render/film_data.py (once).
set -u
OUT=${1:?frames dir}
NW=${2:-2}
TH=${3:-2}
RES=${RES:-1920x1080}
SAMPLES=${SAMPLES:-8}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
pids=()
for ((k = 0; k < NW; k++)); do
  nice -n 10 blender -b -t "$TH" -P render/build_film.py -- --out "$OUT" --res "$RES" --samples "$SAMPLES" --worker "$k/$NW" \
    > "$OUT/worker_$k.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
# frames that no worker owned a JSON for (non-rendered in-betweens) are written by a JSON-only pass
blender -b -P render/build_film.py -- --out "$OUT" --res "$RES" --json-only > "$OUT/json.log" 2>&1
python render/compose_build_film.py --frames "$OUT" --out media/gyra_mk21_build_film.mp4 > "$OUT/compose.log" 2>&1
touch "$OUT/ALL_DONE"
