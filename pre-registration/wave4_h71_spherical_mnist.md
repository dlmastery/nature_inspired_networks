# Wave-4 Pre-Registration — H71 IcosaRoPE3D on Spherical MNIST

**Date filed:** 2026-06-06
**Commit hash:** _to be filled at commit time and propagated to PAPER.md §5 before any seed launches_
**Status:** PRE-REGISTERED (no seeds run yet)
**Depends on:** none (Wave-4 can run in parallel with Wave-1/2/3)
**SYNTHESIS_100.md items:** B5 + B18 + E9
**Estimated wall-clock:** ~15 GPU-h on RTX 4090 Laptop, 1 calendar week

## Hypothesis under test

H71 IcosaRoPE3D — a rotary positional encoding (RoPE; Su 2021
arXiv:2104.09864) generalized to the SO(3) action via the icosahedral
group — lifts non-equivariant ViT-Tiny on **rotated-test Spherical MNIST**
by Δ ≥ +3 pp at n=5 seeds. The icosahedral symmetry is **load-bearing**
on this benchmark: Spherical MNIST embeds 2-D digits on the unit sphere,
the test set rotates each digit by U[SO(3)], and a non-equivariant
classifier loses ~10-20 pp top1 vs train-set top1 on the rotated test.
H71 is the project's sole NOVEL+TESTABLE sci-critic survivor across the
84-hypothesis substrate; this is its first empirical test.

SYNTHESIS_100.md §11 identifies Wave-4 as "the experiment that should
run FIRST, not last" because (i) it is the lowest-cost (15 GPU-h)
nature-inspired claim in the substrate and (ii) it is the headline
candidate for the abstract's third "winner" slot if it lifts.

## Recipe (exact)

- **Architectures (3):**
  - `vit_tiny_p12_nonequiv` — ViT-Tiny patch=12, embed=192, 12 layers,
    heads=3 (Dosovitskiy 2021). Standard 1-D RoPE on the patch sequence.
    **The non-equivariant baseline.**
  - `vit_tiny_p12_h71_icosarope3d` — same backbone with H71 IcosaRoPE3D
    replacing the standard 1-D RoPE. H71 uses the icosahedral group
    rotations (60 elements) to construct the rotary basis; the prior
    claim is that this approximates SO(3) equivariance via the
    icosahedral covering at order-60 resolution.
  - `vit_tiny_p12_e3nn_baseline` — same backbone with a published
    e3nn-based steerable attention block (Geiger 2022 arXiv:2207.09453)
    as the equivariance literature anchor. Iso-FLOPs to the other two
    arms within ±5%. This is the upper bound on what equivariance
    "should" deliver.
- **Dataset:** Spherical MNIST 60×60 (Cohen 2018 arXiv:1801.10130 standard
  e3nn benchmark). Train set unrotated, test set rotated by U[SO(3)] per
  the standard protocol. The rotated test set is where the symmetry
  becomes load-bearing.
- **Augmentation:** none (rotation is the controlled test condition;
  augmenting train with rotations would defeat the equivariance test).
- **Optimizer:** AdamW, lr=5.0e-4, wd=5.0e-4.
- **Scheduler:** cosine, warmup 5 ep, 100 ep total.
- **Seeds:** 0, 1, 2, 3, 4 (n=5).
- **Epochs:** 100.
- **Hardware:** laptop RTX 4090 16 GB. bf16 AMP, batch 256, num_workers 0,
  OMP/MKL caps per CLAUDE.md Rule 26. Headline-mode seeding per D4.

## Decision rule (pre-committed)

The pre-registered Δ-prediction is **Δ_H71 ≥ +3 pp at rotated Spherical
MNIST test top1, paired Wilcoxon at n=5** (SYNTHESIS_100 B18).

- **IF** H71 lifts non-equivariant ViT-Tiny by ≥ +3 pp median AND paired
  Wilcoxon p_one ≤ 0.05 at n=5 AND BCa 95% CI on Δmean excludes +1 pp
  (a more demanding floor than excluding 0) **THEN** H71 is PROMOTED to
  the abstract's third winner slot; PAPER.md §5 reframes around H71 as
  the strongest nature-inspired claim in the paper.
- **IF** H71 lifts by [+1, +3) pp median at p_one ≤ 0.05 **THEN** H71 is
  CONDITIONAL — reported as "directional lift, smaller than the
  pre-registered Δ"; the prior is retained in the paper but does not
  carry the headline.
- **IF** H71 lifts by < +1 pp median OR fails Wilcoxon **THEN** H71 is
  FALSIFIED on Spherical MNIST. PAPER.md §5 honestly reports the
  negative result; the abstract's third slot pivots to the hex AID
  result (B6) or the H22 toroidal tiled-CIFAR result (B7) if either
  succeeds. If neither succeeds, the paper's nature-inspired claim
  collapses to a 2-winner abstract.
- **IF** the e3nn upper-bound arm does NOT lift baseline by ≥ +5 pp
  **THEN** the benchmark is not actually testing what we think it is
  (Spherical MNIST 60×60 may have insufficient resolution or
  insufficient rotation in the test distribution); WAVE-4 is REFUSED
  pending benchmark re-calibration.

## Statistical test

- **Test:** paired Wilcoxon signed-rank, paired permutation. At n=5 the
  exact achievable one-sided minimum is p ≈ 0.031 (all 5 positive).
- **α:** 0.05 two-sided (one-sided used here because Δ-prediction is
  pre-registered directional, ≥ +3 pp).
- **Multiple-comparison correction:** k=1 (H71 is the single
  pre-registered Wave-4 prior). The e3nn upper-bound arm is a sanity
  check, not a competing hypothesis.
- **Auxiliary:** 10⁴-resample BCa bootstrap 95% CI on Δmean — REPORTED
  WITH UNDER-COVERAGE DISCLOSURE per SYNTHESIS_100 C6 (n=5 bootstrap
  under-covers by ~30%).
- **Required n for stated power:** at Δ=3 pp, σ=0.6 pp on Spherical MNIST
  rotated test (estimated from e3nn-benchmark literature; pre-registered),
  80% power at α=0.05 one-sided requires n≥3. n=5 is comfortably above
  this floor.

## Expected wall-clock

- 100 ep at ViT-Tiny on Spherical MNIST 60×60 batch 256 ≈ ~1 GPU-h per seed.
- 3 arms × 5 seeds × 1 GPU-h = 15 GPU-h.

## Pre-registered analysis plan

1. Compute per-(arm, seed) top1 on the rotated test set at epoch 100.
   **Headline-mode seeded.** Also report unrotated-test top1 as a sanity
   metric (the prior should not hurt the unrotated test).
2. Verify iso-FLOPs band ([0.95, 1.05] × baseline FLOPs) for all 3 arms.
3. Compute equivariance error per arm: `‖block(g·x) − g·block(x)‖ / ‖x‖`
   averaged over 100 sampled g ∈ SO(3) and 1000 x. Report per
   SYNTHESIS_100 E9. H71 expected to land between 0 (e3nn) and ~1
   (non-equivariant ViT-Tiny).
4. Compute paired Wilcoxon p_one for (H71 vs non-equiv baseline) at n=5.
5. Compute BCa 95% bootstrap CI on Δmean with under-coverage disclosure.
6. Apply the decision rule.
7. Cross-family auditor pass (GPT-5 + Gemini-3-Pro) on the H71 mechanism
   — independent verdict logged in `audits/CROSS_FAMILY_HONEST_REAUDIT.md`.

## What would falsify this

1. H71 lifts the non-equivariant baseline by < +1 pp median (or loses)
   — IcosaRoPE3D is refuted on Spherical MNIST.
2. The equivariance-error test shows H71's error ≈ the non-equivariant
   baseline's error — the icosahedral-rotary basis does not deliver
   approximate SO(3) equivariance; the prior is malformed.
3. The e3nn upper-bound arm fails to lift baseline by ≥ +5 pp — the
   benchmark is not testing equivariance; Wave-4 refused.
4. H71's iso-FLOPs band fails (e.g., the icosahedral rotary basis
   doubles attention compute) — comparison malformed; H71 re-engineered
   under iso-FLOPs before re-test.
5. Cross-family auditor (GPT-5 / Gemini-3-Pro) flags BROKEN /
   NUMEROLOGY on the IcosaRoPE3D mechanism — H71 demoted from NOVEL
   sci-critic survivor pending audit reconciliation.

## Cross-references

- SYNTHESIS_100.md Block B items
  [B5](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md),
  [B18](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md),
  Block E item [E9](../audits/REVIEWER_FIVE_2026-06-06/SYNTHESIS_100.md)
- FALSIFIERS.md row F5
- CLAUDE.md Rules 28, 35, 36
- Su 2021 arXiv:2104.09864 RoPE
- Cohen 2018 arXiv:1801.10130 Spherical CNNs (Spherical MNIST benchmark)
- Cohen & Welling 2016 arXiv:1602.07576 "G-Equivariant CNNs"
- Geiger 2022 arXiv:2207.09453 e3nn
- Dosovitskiy 2021 arXiv:2010.11929 ViT
