#!/usr/bin/env bash
# Orchestrator: waits for the in-flight pre-fix combo ladder to complete,
# then cleans the 14 pre-fix run dirs (affected by fixers) and launches
# the post-fix campaign (31 runs total: 7 single-axis re-runs + 7 combo
# re-runs + 17 Tier-A ladders). Auto-checkpoint loop is the user's
# responsibility (launch separately).
set -uo pipefail
cd /c/Users/evija/sacgeometry

SENTINEL="experiments/cifar10/combo8_pb_gm_pd_pdw_plr_fe_sa_fp_seed0/metrics.json"
echo "[orchestrator] waiting for pre-fix combo ladder to finish (sentinel: $SENTINEL)"
while [ ! -f "$SENTINEL" ]; do sleep 60; done
echo "[orchestrator] sentinel detected; pre-fix combo ladder done"

# Commit + push whatever pre-fix combo results landed
git add experiments/cifar10/ 2>/dev/null
git -c user.name="dlmastery" -c user.email="eranti@gmail.com" \
  commit -m "Pre-fix combo ladder complete (7 rows, broken phi_budget + 1-step-saturating golden_momentum)" \
  >/dev/null 2>&1 || true
git push >/dev/null 2>&1 || true

# Wipe the 14 affected dirs so --skip-existing re-runs them on POST-FIX code.
# Tags affected (Fixer commits → modules touched):
#   Fixer-PhiScaling  → phi_budget widths changed                      sg_only_phi_budget + combo{2..8}*
#   Fixer-PhiScaling  → golden_bottleneck c_mid changed                sg_only_golden_bottleneck
#   Fixer-Priors      → chladni_modes_banded dedup for out_c > k^2     sg_only_cymatic_init
#   Fixer-InitFilter  → golden_spiral_init uses true phi-growth + 137.508°  sg_only_golden_spiral_init
#   Fixer-Opt         → GoldenRatioAdamW eps default 1e-8 (clean β-only test) sg_only_golden_adam
#   Fixer-Opt         → PhiDropout per-epoch curriculum                sg_only_phi_dropout + combo{3..8}*
#   Fixer-Opt         → GoldenMomentumScheduler non-saturating         sg_only_golden_momentum + combo{2..8}*
DIRS=(
  "experiments/cifar10/sg_only_phi_budget_seed0"
  "experiments/cifar10/sg_only_golden_bottleneck_seed0"
  "experiments/cifar10/sg_only_cymatic_init_seed0"
  "experiments/cifar10/sg_only_golden_spiral_init_seed0"
  "experiments/cifar10/sg_only_golden_adam_seed0"
  "experiments/cifar10/sg_only_phi_dropout_seed0"
  "experiments/cifar10/sg_only_golden_momentum_seed0"
  "experiments/cifar10/combo2_pb_gm_seed0"
  "experiments/cifar10/combo3_pb_gm_pd_seed0"
  "experiments/cifar10/combo4_pb_gm_pd_pdw_seed0"
  "experiments/cifar10/combo5_pb_gm_pd_pdw_plr_seed0"
  "experiments/cifar10/combo6_pb_gm_pd_pdw_plr_fe_seed0"
  "experiments/cifar10/combo7_pb_gm_pd_pdw_plr_fe_sa_seed0"
  "experiments/cifar10/combo8_pb_gm_pd_pdw_plr_fe_sa_fp_seed0"
)
echo "[orchestrator] removing ${#DIRS[@]} pre-fix run dirs"
for d in "${DIRS[@]}"; do rm -rf "$d"; done

# Launch unified post-fix sweep
TAGS=(
  # 7 single-axis re-runs
  sg_only_phi_budget sg_only_golden_bottleneck sg_only_cymatic_init
  sg_only_golden_spiral_init sg_only_golden_adam sg_only_phi_dropout
  sg_only_golden_momentum
  # 7 combo re-runs (post-fix architecture + schedules)
  combo2_pb_gm combo3_pb_gm_pd combo4_pb_gm_pd_pdw
  combo5_pb_gm_pd_pdw_plr combo6_pb_gm_pd_pdw_plr_fe
  combo7_pb_gm_pd_pdw_plr_fe_sa combo8_pb_gm_pd_pdw_plr_fe_sa_fp
  # 7 Tier-A LOO subtractive
  loo_no_gm loo_no_pd loo_no_pdw loo_no_plr loo_no_fe loo_no_sa loo_no_fp
  # 5 Tier-A PAIR interaction matrix
  pair_gm_pdw pair_gm_plr pair_pd_pdw pair_pd_plr pair_pdw_plr
  # 5 Tier-A SLOT ablation
  slot_act_sine slot_act_phi slot_init_spiral slot_init_phi slot_init_cymatic
)
echo "[orchestrator] launching post-fix sweep over ${#TAGS[@]} tags"
KMP_DUPLICATE_LIB_OK=TRUE /c/Users/evija/sacgeometry/.venv/Scripts/python.exe \
  -u scripts/run_sweep.py \
  --config configs/cifar10_quick.yaml --seeds 0 --root experiments --skip-existing \
  --only "${TAGS[@]}" 2>&1 | tee experiments/sweep_postfix_campaign.log
echo "[orchestrator] post-fix sweep finished"
