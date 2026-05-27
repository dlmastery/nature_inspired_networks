# FINDINGS — nature_inspired_networks curated CIFAR-10 sweep (seed 0, 12 epochs)

> **The headline:** at the 12-epoch / 127–272 k-parameter scale on
> CIFAR-10, **the nature-inspired priors do *not* compound**. The full
> hybrid (`sg_full_fib`, all six flags on) is the **worst** NaturePrior
> variant in the sweep, not the best. This contradicts the source PDF's
> "literature-compound" prediction (20–50 % efficiency gains).
> The autoresearch protocol delivered exactly what it is supposed to:
> a falsifiable negative result with clear single-prior decomposition.

## Expansion campaign — 2026-05-27 — 20 new hypotheses smoked (seed 0, 12 ep)

After the implementation campaign brought the repo from 7 to 74/75
hypothesis implementations, all 20 newly-wired CIFAR-10-droppable tags
were smoked at seed 0 / 12 epochs (zero failures, 8559 s total). The
result **sharpens, not overturns, the original headline**: of 20 new
single-change variants, exactly **one beats the ResNet-20 baseline on
top-1**, and **none beats it on composite** (the baseline's 272 k params
at 84.78 % remain the composite leader at 0.8458).

### The one positive: H09 φ-Proportion Parameter Budget

| tag | hyp | top-1 | params | composite | verdict |
|---|---|---|---|---|---|
| `sg_only_phi_budget` | H09 | **85.54 %** | 284 k | 0.8429 | **only variant > baseline top-1 (+0.76 pp)** |

`phi_budget` allocates the per-stage parameter budget in a `1 : φ : φ²`
ratio rather than the uniform doubling of a stock ResNet. It is the
**single strongest CIFAR-10 lead the project has produced** and the
clear #1 candidate to graduate to CIFAR-100 (Phase 4). The composite is
fractionally below baseline only because it carries 11 k more params;
on raw accuracy it wins. Pre-registered falsifier (top-1 ≤ baseline)
is **not** met → hypothesis survives.

### The clean falsification: H41 Golden-Ratio AdamW

| tag | hyp | top-1 | verdict |
|---|---|---|---|
| `sg_only_golden_adam` | H41 | **51.96 %** | **falsified — worst run in the entire project** |

Setting Adam's `β1 = 1/φ ≈ 0.618`, `β2 = 1/φ² ≈ 0.382` sits far below
the empirically-stable regime (`0.9 / 0.999`). The low `β2` makes the
second-moment estimate hyper-noisy, so the optimiser never stabilises —
a 33 pp collapse vs baseline. This is exactly the kind of confidently-
stated "golden constants improve everything" claim the protocol exists
to kill. **Verdict: DISCARD**; the golden ratio is not a universal
substitute for tuned optimiser moments.

### The efficiency story: H02 Fibonacci depth + the φ-channel family

| tag | hyp | top-1 | params | composite |
|---|---|---|---|---|
| `sg_only_fib_depth` | H02 | 82.18 % | **180 k** | 0.8261 |
| `sg_only_golden_resize` | H03 | 80.67 % | 127 k | 0.8157 |
| `sg_only_phi_compound` | H01 | 80.42 % | 127 k | 0.8152 |

`fib_depth` reaches the #3 composite slot at **0.66× the baseline
parameter count** — the priors that touch *scaling* (depth schedule,
channel schedule) are consistently the most parameter-efficient family,
even when raw top-1 trails. Worth a CIFAR-100 look as the "small-model"
graduate.

### Mid-pack (neutral, within ±2 pp of the no-prior scaffold)

`golden_momentum` (83.52), `phi_dropout` (82.80), `phi_multiscale`
(82.00), `golden_skip` (81.63), `fib_prune` (81.15), `golden_spiral_init`
(80.42), `fib_ensemble` (80.11), `phi_activation` (79.95), `phi_decay`
(79.81), `phi_lr` (78.75) — all land in the neutral band: a real
implementation, no reliable lift at 12 ep. These are "needs-more-epochs
or needs-the-right-dataset" rather than falsified.

### New strong negatives (join group/cymatic/toroidal)

`golden_adam` (51.96, above), `group_avg` (65.38), `golden_bottleneck`
(69.25, but at **0.21× params** — efficiency outlier worth a second
look), `phi_relu` (71.07), `fib_stride` (72.55), `phi_sparse` (73.33),
`phi_init` (76.56). The φ-init and sparse-connectivity priors actively
hurt at this scale.

### G8 esoteric-extension tags (queued)

`sg_only_constant_width` (H80) and `sg_only_sine_act` (H81) were added
to the matrix after this sweep launched and are smoking now in a
follow-up run; the other 7 G8 modules ship as standalone primitives
without a CNN sweep row.

### Phase-4 CIFAR-100 graduation shortlist (by this campaign)

1. **H09 `phi_budget`** — only top-1 winner; clear #1.
2. **H02 `fib_depth`** — best param-efficiency (comp 0.8261 @ 180 k).
3. **H05 `fractal`** — the original campaign's only positive single prior.
4. **H48 `golden_momentum`** — closest neutral to baseline (83.52).
5. `baseline_resnet20` rail (always carried).

---

## Final ranking (composite descending, single seed) — original 11-row campaign

| rank | tag | top-1 | params | latency ms | composite | Δ vs `sg_chan_fib` |
|---|---|---|---|---|---|---|
| 1 | `baseline_resnet20` | 84.78 % | 272 k | 4.03 | **0.8458** | (different scaffold) |
| 2 | `baseline_sg_vanilla` | 82.16 % | 186 k | 4.42 | 0.8258 | +0.0123 |
| 3 | `sg_chan_phi` | 80.11 % | 127 k | 4.11 | 0.8152 | +0.0017 |
| 4 | `sg_chan_fib` | 80.11 % | 127 k | 4.43 | **0.8135** | — (reference) |
| 5 | `sg_only_fractal` | 82.46 % | 259 k | 7.42 | 0.8104 | -0.0031 |
| 6 | `sg_only_golden_modulate` | 79.81 % | 127 k | 5.95 | 0.8042 | -0.0093 |
| 7 | `sg_only_hex` | 79.32 % | 127 k | 7.55 | 0.7941 | -0.0194 |
| 8 | `sg_only_cymatic_init` | 77.44 % | 127 k | 4.14 | 0.7883 | -0.0252 |
| 9 | `sg_only_toroidal` | 78.05 % | 127 k | 9.34 | 0.7768 | -0.0367 |
| 10 | `sg_only_group` | 69.84 % | 127 k | 9.75 | **0.6937** | -0.1198 |
| 11 | `sg_full_fib` | 73.24 % | 259 k | 20.02 | **0.6966** | -0.1169 |

## Single-prior decomposition vs the `sg_chan_fib` reference

| prior on alone | Δ top-1 | Δ latency (ms) | Δ rot-eq err | verdict |
|---|---|---|---|---|
| `hex` | **-0.79 pp** | +3.12 (1.7×) | -0.022 | mild negative — mask overhead exceeds isotropy gain at this scale |
| `group` (C4) | **-10.27 pp** | +5.32 (2.2×) | **-0.046** (large) | strong negative on top-1, but rot-eq drops as theory predicts — equivariance prior is unused by the data |
| `fractal` | **+2.35 pp** | +2.99 (1.7×) | -0.036 | **only single prior that lifts top-1**; pays in 2× params |
| `toroidal` | **-2.06 pp** | +4.91 (2.1×) | -0.025 | negative as predicted — CIFAR images do not wrap |
| `cymatic_init` | **-2.67 pp** | -0.29 (≈) | -0.011 | unexpected negative — Chladni-mode init hurt rather than helped |
| `golden_modulate` | **-0.30 pp** | +1.52 (1.3×) | -0.026 | near no-op as predicted |

## The compound failure

`sg_full_fib` (all six priors ON simultaneously):

- **top-1: 73.24 %** (-6.87 pp below `sg_chan_fib` reference; -11.54 pp
  below `baseline_resnet20`)
- **latency: 20.02 ms** (5× the baseline)
- **composite: 0.6966** (worst in sweep, tied with `sg_only_group`)

## H58 follow-up — the avg-pool fix DISCARDED

The first campaign's top-priority follow-up was **H58**: replace the C4
group convolution's max-pool reduction with mean-pool. Pre-registered
prediction: the +5 to +10 pp top-1 recovery on `sg_only_group` would
prove that "max-pool over the 4-rotation orbit throws away 75 % of the
signal." The 2-row sweep (`sg_only_group_avg`, `sg_full_fib_avg`)
trained on 2026-05-27 with checkpoint saving enabled returned the
following:

| variant | reduce | top-1 | params | composite | Δ vs same-arch max |
|---|---|---|---|---|---|
| `sg_only_group` | max | 69.84 % | 127 k | 0.6937 | (ref) |
| `sg_only_group_avg` | mean | **65.38 %** | 127 k | **0.6597** | **-4.46 pp** |
| `sg_full_fib` | max | 73.24 % | 259 k | 0.6966 | (ref) |
| `sg_full_fib_avg` | mean | **66.86 %** | 259 k | **0.6432** | **-6.38 pp** |

**Verdict: DISCARD.** Mean-pool over the orbit *hurts* worse than
max-pool — the opposite of the prediction. The "discards 75 % signal"
intuition was wrong: max-pool over rotated copies actually preserves
the strongest evidence at every spatial location (a soft argmax over
orientations), while mean-pool *dilutes* discriminative features by
averaging out the response-vs-orientation mismatch. The autoresearch
protocol delivered exactly what it should: a clean falsifiable
negative against a confidently-stated hypothesis.

**New direction for H24 (Platonic equivariance):** the fix is not the
reduction operator but the data. C4-equivariant features are useful on
data with rotational variance (rotated CIFAR, IcoMNIST, spherical
MNIST) — not on canonically-oriented CIFAR-10. The next experiment
should test the *same* `sg_only_group` (max-pool) variant on
rotated-CIFAR-10 where the equivariance prior is data-aligned.

## Trained-feature Betti (first data)

The H58 sweep produced the first `best.pt` checkpoints, enabling
trained-feature Betti curves for the first time. Two interesting
observations:

| variant | β₀ per stage (trained) | β₀ per stage (fresh-init) |
|---|---|---|
| `sg_only_group_avg` | [121, 127, 127, 127] | [93, 71, 122, 100] |
| `sg_full_fib_avg` | [123, 124, 127, 127] | [123, 124, 127, 127]* |

Trained models *increase* β₀ relative to fresh init — they cluster
features by class, producing more isolated connected components at the
relative-threshold band. β₁ also rises (one 1-D hole detected in
`sg_full_fib_avg`), consistent with class-prototype loop structures.
This inverts the naive "topology simplification" reading and motivates
a follow-up where Betti is tracked **per epoch** to see whether the
β-curves cross the random baseline at convergence.

The naive additive-deltas prediction (sum of single-prior Δs ≈ -13 pp
top-1) would put `sg_full_fib` at ≈ 67 %; observed 73.24 % is actually
*better* than the additive prediction, suggesting some priors do
recover *some* loss when combined — but the net is still strongly
negative. The hex+group+toroidal latency stack (each adds 1.7–2.2×)
multiplies into 5× total latency.

## What this means

1. **The source PDF's 20–50 % efficiency claim does not survive a
   12-epoch CIFAR-10 ablation on this block.** Either the priors
   need a different composition (not all six at once on a single
   block), more training (12 epochs is short), or a different
   dataset where the priors' inductive biases match the data
   geometry.

2. **C4 group conv is the dominant negative term.** With C4
   max-pool over the 4-rotation orbit, we are throwing away 75 %
   of the signal at every layer. The rotation-equivariance gain
   (rot-eq err drops by 0.046) is real, but CIFAR-10 does not
   reward it because the test images are not rotated.

3. **Fractal recursion is the only single prior that lifts top-1.**
   At 2× parameter cost the +2.35 pp lift is hard to justify, but
   it is the one prior that materially helps representation
   quality at this scale.

4. **`baseline_sg_vanilla` (the NaturePriorBlock scaffold with all
   priors OFF and linear channels) is competitive.** It sits at
   82.16 % top-1 with 186 k params vs ResNet-20's 272 k — i.e.,
   the NaturePriorBlock scaffold itself (BN placement, skip-connection
   structure) is fine. The priors do most of the harm.

## Open axes for a follow-up campaign

- [ ] Run with `--seeds 0 1 2` to get error bars; the deltas above
      are single-seed.
- [ ] Run `sg_loo_no_*` leave-one-out from the full hybrid
      (`scripts/run_sweep.py --full`) to identify which single
      prior is responsible for the most damage when combined.
- [ ] Test the full hybrid on a dataset where the priors are
      data-aligned: e.g., **rotated CIFAR-10** (group conv should
      now help) or **spherical / IcoMNIST** (Platonic equivariance
      should pay off).
- [ ] Re-run sg_only_group with **average pooling** over the orbit
      instead of max-pool — max-pool's 75 % signal loss may be the
      single biggest source of the C4 damage.
- [ ] Save `best.pt` checkpoints (already wired) and re-run
      `compute_topology.py` for **trained-feature** Betti curves;
      the current β-curves are on fresh-init features.
- [ ] Scale to MedMNIST PathMNIST (data loader already in
      `src/nature_inspired_networks/data.py`).

## Reproduce

```powershell
$env:SSL_CERT_FILE = ".\.venv\Lib\site-packages\certifi\cacert.pem"
.\.venv\Scripts\python -u scripts\run_sweep.py `
   --config configs\cifar10_quick.yaml --seeds 0 --skip-existing
.\.venv\Scripts\python scripts\compute_topology.py --seeds 0
.\.venv\Scripts\python scripts\build_dashboard.py
.\.venv\Scripts\python scripts\build_report.py
start dashboard\dashboard.html
```

Total wall-clock for the 11-run sweep on RTX 4090 Laptop (16 GB):
**3853 s = 64 min** (single seed, 12 epochs each).
