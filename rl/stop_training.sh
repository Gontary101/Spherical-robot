#!/usr/bin/env bash
# Stop a training job: rl/stop_training.sh loco2  (pattern lives in this file, so it never matches the caller's shell)
pat="rl/train_${1:-}"
for p in $(pgrep -f "python.*${pat}"); do
  [ "$p" != "$$" ] && kill "$p" 2>/dev/null
done
sleep 2
pgrep -f "python.*${pat}" | wc -l
