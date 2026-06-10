#!/bin/bash
# Wave-0 (Imagenette recipe validation) campaign script.
#
# Pre-registration: pre-registration/wave0_imagenette_recipe_validation.md
# SYNTHESIS_100.md Block B item B1.
#
# Sequence: 3 recipes x 5 seeds = 15 runs, ~20 min each = ~5 GPU-h total.
#
#   recipe_legacy             -- legacy 11-trick (no EMA)
#   recipe_modern_naive       -- legacy + EMA 0.9999 (full timm-style)
#   recipe_modern_cifar_tuned -- A4-v1 He-2019 / Playbook (the CIFAR winner)
#
# Robustness (mirrors scripts/block_a_close_campaign.sh):
#   * skip-if-exists per run -> resumable after any crash
#   * per-run git commit + push with retry -> <= 1 run lost on any crash
#   * set -euo pipefail -> halts on first error so operator investigates
#
# Stop signal: ``touch logs/.stop_wave0`` to halt cleanly between runs.

set -euo pipefail
cd "$(dirname "$0")/.."

# Rule 26 thread caps -- mandatory for any long-running sweep on Windows.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

PY=.venv/Scripts/python.exe
LOG="logs/wave0_imagenette.log"
ROOT="experiments_wave0"
mkdir -p logs "$ROOT"

echo "[campaign] Wave-0 Imagenette started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

run_if_missing() {
  local config=$1 tag=$2 seed=$3 root=$4
  local mj="$root/imagenette/${tag}_seed${seed}/metrics.json"

  if [ -f logs/.stop_wave0 ]; then
    echo "[stop] stop flag logs/.stop_wave0 detected; halting between runs" | tee -a "$LOG"
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

  # Commit + push with retry (Rule 11 + Rule 20).
  git add "$root/" 2>/dev/null || true
  if ! git diff --staged --quiet; then
    local top1
    top1=$(python -c "import json; print(json.load(open(r'$mj'))['top1'])" 2>/dev/null || echo "?")
    git commit -m "wave0 imagenette: $tag seed=$seed top1=$top1" 2>&1 | tee -a "$LOG"
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

# Per pre-reg: 3 recipes x 5 seeds (n=5 per recipe). Interleave seeds so
# a partial completion still has comparable n across recipes (a crash
# after 9 runs gives n=3 per recipe, not n=5/0/4).
for seed in 0 1 2 3 4; do
  run_if_missing configs/wave0_imagenette_legacy.yaml \
    wave0_imagenette_legacy "$seed" "$ROOT"
  run_if_missing configs/wave0_imagenette_modern_naive.yaml \
    wave0_imagenette_modern_naive "$seed" "$ROOT"
  run_if_missing configs/wave0_imagenette_modern_cifar_tuned.yaml \
    wave0_imagenette_modern_cifar_tuned "$seed" "$ROOT"
done

# Wave-0 decision gate per pre-registration:
# Choose the recipe with median top1 in [0.85, 0.95] band that closes
# Imagenette small-model band; this gate is operator-driven (the script
# prints the per-recipe median + IQR and the operator updates
# ``pre-registration/wave1_imagenette_iso_flops_pareto.md`` with the
# selected recipe BEFORE any Wave-1 seed launches).
echo "[wave0-gate] computing per-recipe median + IQR ..." | tee -a "$LOG"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
import statistics
from pathlib import Path

ROOT = Path("experiments_wave0/imagenette")
RECIPES = [
    "wave0_imagenette_legacy",
    "wave0_imagenette_modern_naive",
    "wave0_imagenette_modern_cifar_tuned",
]

print("\n[wave0-gate] per-recipe top1 medians + IQR (n=5)")
print("=" * 72)
for r in RECIPES:
    top1s = []
    for seed in range(5):
        mj = ROOT / f"{r}_seed{seed}" / "metrics.json"
        if mj.is_file():
            top1s.append(float(json.loads(mj.read_text())["top1"]))
    if not top1s:
        print(f"  {r:48s}  NO RUNS YET")
        continue
    med = statistics.median(top1s)
    if len(top1s) >= 2:
        q1, q3 = statistics.quantiles(top1s, n=4)[0], statistics.quantiles(top1s, n=4)[2]
        iqr = q3 - q1
    else:
        iqr = 0.0
    band = "PROMOTE" if med >= 0.90 else ("CONDITIONAL" if med >= 0.85 else "REFUSE")
    print(f"  {r:48s}  median={med:.4f}  IQR={iqr:.4f}  n={len(top1s)}  -> {band}")
print("=" * 72)
print("\n[wave0-gate] decision rule per pre-registration:")
print("  median >= 0.90  -> PROMOTE to Wave-1 (working modern recipe)")
print("  median in [0.85, 0.90)  -> CONDITIONAL (proceed with caveat)")
print("  median < 0.85  -> REFUSE (recipe debug -> wave0b_recipe_debug.md)")
PYEOF

echo "[campaign] Wave-0 Imagenette complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[campaign] Next: operator selects the promoted recipe and pre-registers wave1_imagenette_iso_flops_pareto.md" | tee -a "$LOG"
