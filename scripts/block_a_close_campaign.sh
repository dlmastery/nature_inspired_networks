#!/bin/bash
# Block A close campaign — sequential runs to close R5 BLOCKER 1.
#
# Sequence:
#   1. A4-v1 baseline seeds 1, 2  (~7 GPU-h, closes n=3 baseline)
#   2. 3 priors × seed 0 at iso-FLOPs (~10.5 GPU-h)
#   3. 3 priors × seed 1 at iso-FLOPs (~10.5 GPU-h)
#   4. 3 priors × seed 2 at iso-FLOPs (~10.5 GPU-h)
# Total: ~38 GPU-h ≈ 1.6 calendar days.
#
# Robustness:
#   * skip-if-exists per run → resumable after any crash
#   * per-run git commit + push → ≤1 run lost on any crash
#   * set -e → halts on first error so operator investigates
#
# Stop signal: create logs/.stop_campaign to halt cleanly between runs.

set -euo pipefail
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

PY=.venv/Scripts/python.exe
LOG="logs/block_a_close.log"
mkdir -p logs experiments_modern_debug experiments_iso_flops_v1

echo "[campaign] started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

run_if_missing() {
  local config=$1 tag=$2 seed=$3 root=$4
  local mj="$root/cifar100/${tag}_seed${seed}/metrics.json"

  if [ -f logs/.stop_campaign ]; then
    echo "[stop] stop flag detected; halting between runs" | tee -a "$LOG"
    exit 0
  fi

  if [ -f "$mj" ]; then
    local top1
    top1=$(python -c "import json; print(json.load(open(r'$mj'))['top1'])" 2>/dev/null || echo "?")
    echo "[skip] $tag seed=$seed already exists (top1=$top1)" | tee -a "$LOG"
    return 0
  fi

  echo "[run] starting $tag seed=$seed at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  "$PY" -m nature_inspired_networks.runner \
    --config "$config" \
    --tag "$tag" \
    --seed "$seed" \
    --root "$root" 2>&1 | tee -a "$LOG"
  echo "[run] finished $tag seed=$seed at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

  # Commit + push with retry
  git add "$root/" 2>/dev/null || true
  if ! git diff --staged --quiet; then
    local top1
    top1=$(python -c "import json; print(json.load(open(r'$mj'))['top1'])" 2>/dev/null || echo "?")
    git commit -m "campaign Block A: $tag seed=$seed top1=$top1" 2>&1 | tee -a "$LOG"
    for i in 1 2 3 4 5; do
      if git push origin main 2>&1 | tee -a "$LOG"; then
        echo "[push] OK ($tag seed=$seed)" | tee -a "$LOG"
        break
      fi
      echo "[push] retry $i" | tee -a "$LOG"
      sleep 30
    done
  fi
}

# ---- Phase 1: complete A4-v1 baseline n=3 (seed 0 already exists at 0.6747) ----
run_if_missing configs/cifar100_modern_200ep_he2019_debug.yaml \
  baseline_resnet20_he2019_debug 1 experiments_modern_debug
run_if_missing configs/cifar100_modern_200ep_he2019_debug.yaml \
  baseline_resnet20_he2019_debug 2 experiments_modern_debug

# ---- Phase 2: 3 priors × n=3 at iso-FLOPs (phi_budget_total=125000 → 43.24M FLOPs) ----
# Interleave seeds so a partial completion still has comparable n across priors.
for seed in 0 1 2; do
  run_if_missing configs/cifar100_modern_200ep_sg_only_phi_budget.yaml \
    sg_only_phi_budget_iso_flops_v1 "$seed" experiments_iso_flops_v1
  run_if_missing configs/cifar100_modern_200ep_pair_gm_pdw.yaml \
    pair_gm_pdw_iso_flops_v1 "$seed" experiments_iso_flops_v1
  run_if_missing configs/cifar100_modern_200ep_slot_act_sine.yaml \
    slot_act_sine_iso_flops_v1 "$seed" experiments_iso_flops_v1
done

echo "[campaign] Block A close complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[campaign] Next: paired Wilcoxon + bootstrap CI + Holm-Bonferroni vs A4-v1 baseline" | tee -a "$LOG"
