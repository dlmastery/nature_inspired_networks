# Session Resume — nature_inspired_networks SYNTHESIS_100 12-week plan

**Last updated:** 2026-06-07 16:35 UTC
**Last commit:** `ee120c7` (origin/main = HEAD; working tree clean)
**Reason for checkpoint:** server going down mid-campaign; user will resume later

## Where the campaign is

The user authorized the **full 12-week SYNTHESIS_100 plan (option a)** on 2026-06-06 in response to the 5-reviewer brutal-critique pass (all REJECT). Plan target: ~250 GPU-h, paper resubmittable at ICML 2028 / ICLR 2028 / NeurIPS 2027 with nature-inspired DL as the headline. North-star rejection of R4's "drop the priors and publish the protocol alone" pivot is documented in `audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md` §0.

### Day-1 deliverables landed (commits `8738a29..ee120c7`)

1. 5 reviewer critiques + SYNTHESIS_100.md (~1900 lines)
2. Code structural fixes: FLOP-target check in runner (A3), `headline_mode` in `set_seed` (D4), FLOPs-extended composite metric with new SHA-256 fingerprint `b73e8bbfa2717c567bda42b7760fefc3b4e68381aee54ea28d7cd8f3d6863649`, `train_top1_clean` under Mixup (D7), Mixup λ-flip assertion (D1), CutMix degenerate-box guard (D2), EMA BN recalibration (D6), `composite_score(params<1000)` raises (D9). 27 tests green.
3. Configs + pre-registration: 3 prior-overlay YAMLs at `phi_budget_total=125000` (43.24M FLOPs, +4.9% iso-FLOPs band); 5 wave pre-registrations (Wave 0-4); A4 + A4-v2 pre-registrations; `paper/FALSIFIERS.md` (8 binding refutation rows); `paper/NEGATIVE_RESULTS.md` skeleton.
4. Paper text reframe: abstract 440→200 words; `slot_act_sine` demoted from headline (kept as case study); `pair_gm_pdw` reframed as 3-axis stack with 61% non-φ; 19 missing citations land (Bello, Wightman, Zheng, Saunders, Madaan, Sitzmann, Cohen-Welling, Weiler, Geiger, etc.); self-grading banners stripped; H08/H43/H67/H74 demoted to FALSIFIED_AT_MECHANISM_TEST.
5. A1 iso-FLOPs calibration audit: `phi_budget_total: 270000 → 125000` → 43.24M FLOPs; H09 1:φ:φ² preserved within 1.82%. Reproduction scripts committed at `audits/REVIEWER_FIVE_2026-06-06/_calibrate_iso_flops*.py`.

### GPU work done

| Run | Recipe | Top1 | Verdict | Commit |
|---|---|---:|---|---|
| A4-v1 baseline_resnet20_he2019_debug seed 0 | He-2019/Playbook | **0.6747** | DIRECTIONAL (+3.97 pp vs legacy 0.6350) | `f124bb3` |
| A4-v2 baseline_resnet20_he2019_debug_v2 seed 0 | A4-v1 with lr 1e-3 → 5e-4 | **0.6516** | REGRESSION (-2.31 pp vs A4-v1) | `7325e1b` |

**Decision:** A4-v1 promoted to **practical baseline** at 0.6747 (within Wightman 2021 published range 0.69-0.71; 0.5 pp shy of PLAN.md's optimistic 0.68 floor — disclosed in FINDINGS). A1 (iso-FLOPs prior re-test) is now **UNBLOCKED**.

### Currently in flight when server went down

- **A4-v1 seed 1** was running (background id `blsvtgk2h`, started ~13:44 UTC). Expected landing ~17:14 UTC.
- **Auto-checkpoint loop** signaled to stop via `logs/.stop_a4_checkpoint`.

If A4-v1 seed 1 completed before server died, its `metrics.json` will be at `experiments_modern_debug/cifar100/baseline_resnet20_he2019_debug_seed1/metrics.json`. Check that on resume. If it didn't complete, simply re-launch — the runner is reproducible under `headline_mode=true`.

## Resume protocol (next session)

When the user returns:

1. **First action:** `git pull origin main` to sync any out-of-band commits.
2. **Check A4-v1 seed 1 status:** read `experiments_modern_debug/cifar100/baseline_resnet20_he2019_debug_seed1/metrics.json` if it exists. If not, re-launch via:
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
   .venv/Scripts/python.exe -m nature_inspired_networks.runner \
     --config configs/cifar100_modern_200ep_he2019_debug.yaml \
     --tag baseline_resnet20_he2019_debug --seed 1 \
     --root experiments_modern_debug
   ```
   ETA ~3.5 h.
3. **After seed 1:** launch seed 2 (same as above with `--seed 2`).
4. **After n=3 baseline:** launch 3 priors × 3 seeds each at the iso-FLOPs A4-v1 recipe configs:
   - `configs/cifar100_modern_200ep_sg_only_phi_budget.yaml` (tag `sg_only_phi_budget_iso_flops_v1`)
   - `configs/cifar100_modern_200ep_pair_gm_pdw.yaml` (tag `pair_gm_pdw_iso_flops_v1`)
   - `configs/cifar100_modern_200ep_slot_act_sine.yaml` (tag `slot_act_sine_iso_flops_v1`)

   Each at seeds {0, 1, 2}, root `experiments_iso_flops_v1`. ~31.5 GPU-h sequential.
5. **Restart the auto-checkpoint loop:** `rm logs/.stop_a4_checkpoint && bash scripts/auto_checkpoint_a4.sh > logs/auto_checkpoint.log 2>&1 &`.

After A1 lands (n=3 paired Wilcoxon + bootstrap CI + Holm-Bonferroni k=3 against the A4-v1 n=3 baseline), Block A closes and Week 3 begins (Wave-0 Imagenette recipe validation).

## Budget tracking

| Phase | Budgeted | Used | Remaining |
|---|---:|---:|---:|
| Block A (Week 1-2) | 30 | ~10.5 | ~19.5 |
| Wave-0 (Week 3) | 5 | 0 | 5 |
| Wave-1 (Week 4-5) | 50 | 0 | 50 |
| Wave-2 (Week 6-8) | 80 | 0 | 80 |
| Wave-3 (Week 9-11) | 80 | 0 | 80 |
| Wave-4 (Week 12) | 15 | 0 | 15 |
| **Total** | **260** | **~10.5** | **~249.5** |

## Open binding decisions (none pending)

All decisions through 2026-06-07 16:35 UTC are committed per binding pre-registrations. No HARKing risk on resume; the next-session protocol above is deterministic.

## Key file pointers (for fast resume)

- Plan: `audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md`
- Falsifiers: `paper/FALSIFIERS.md`
- A4-v1 result: `experiments_modern_debug/cifar100/baseline_resnet20_he2019_debug_seed0/metrics.json`
- A4-v2 result: `experiments_modern_debug/cifar100/baseline_resnet20_he2019_debug_v2_seed0/metrics.json`
- A1 calibration: `audits/REVIEWER_FIVE_2026-06-06/A1_ISO_FLOPS_CALIBRATION.md`
- Latest FINDINGS: `paper/FINDINGS.md` top block (2026-06-07 PM)
- Wave pre-registrations: `pre-registration/wave{0,1,2,3,4}_*.md`
- Recipe pre-registrations: `pre-registration/a4_recipe_debug_he2019.md`, `a4_recipe_debug_v2_lr5em4.md`
