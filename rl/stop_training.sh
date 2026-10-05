#!/usr/bin/env bash
# stop any running training job (kept in a file so the pattern never matches this shell)
pkill -f "python3 rl/train_${1:-}" ; sleep 2
