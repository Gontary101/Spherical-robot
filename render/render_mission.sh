#!/usr/bin/env bash
# Render the recorded mission: real-time except 20.2-45.8 s (every 2nd frame -> 2x playback, labelled in the HUD).
# Usage: BLENDER=/path/to/blender OUT=/path/to/frames/ render/render_mission.sh
set -e
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
A="$B -b -P render/animate_mission.py -- --res 960x540 --samples 6 --out ${OUT}f_"
$A --start 1 --end 486
$A --start 487 --end 1099 --step 2
$A --start 1100 --end 0
