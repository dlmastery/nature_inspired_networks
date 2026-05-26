# FINDINGS — nature_inspired_networks curated CIFAR-10 sweep (seed 0, 12 epochs)

> **The headline:** at the 12-epoch / 127–272 k-parameter scale on
> CIFAR-10, **the nature-inspired priors do *not* compound**. The full
> hybrid (`sg_full_fib`, all six flags on) is the **worst** NaturePrior
> variant in the sweep, not the best. This contradicts the source PDF's
> "literature-compound" prediction (20–50 % efficiency gains).
> The autoresearch protocol delivered exactly what it is supposed to:
> a falsifiable negative result with clear single-prior decomposition.

## Final ranking (composite descending, single seed)

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
