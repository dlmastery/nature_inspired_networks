#!/bin/bash
# Auto-checkpoint loop for the A4 / A1 campaign (Rule 20).
# Commits new artifacts every ~10 min so a crash loses ≤1 run.
# Stop by creating the file logs/.stop_a4_checkpoint
set -u
cd "$(dirname "$0")/.."
echo "[auto-checkpoint] started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while [ ! -f logs/.stop_a4_checkpoint ]; do
  sleep 600
  ts=$(date -u +%Y%m%d_%H%M%S)
  # Stage anything new in the experiment dirs + the rolling log
  git add experiments_modern_debug/ logs/a4_seed0.log logs/a4_seed1.log logs/a4_seed2.log 2>/dev/null || true
  if ! git diff --staged --quiet; then
    git commit -m "auto-checkpoint: A4 recipe-debug campaign progress $ts" 2>&1 | tail -3
    for i in 1 2 3; do
      if git push origin main 2>&1 | tail -3; then
        echo "[auto-checkpoint] push OK at $ts"
        break
      fi
      echo "[auto-checkpoint] push retry $i at $ts"
      sleep 30
    done
  else
    echo "[auto-checkpoint] nothing new at $ts"
  fi
done
echo "[auto-checkpoint] stopped at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
