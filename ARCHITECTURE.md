# ARCHITECTURE

> Module-level diagram, shape tables, and forward-pass flow for the
> NaturePriorBlock and the NaturePriorNet stack it builds.

## Module diagram

```
                  ┌─────────────────────────────────────────┐
                  │             NaturePriorNet                │
                  └──────────────────┬──────────────────────┘
                                     │
   x ∈ ℝ^{B×3×32×32}                 │
        │                            │
        ▼                            │
   ┌──────────┐                      │
   │   stem   │  Conv2d 3×3 → BN → ReLU  (3 → widths[0])
   └────┬─────┘
        │
        ▼  ──── stage 1 (3 blocks, stride 1) ──── widths[0] = 16
   ┌──────────┐
   │ SGBlock  │  c_in = widths[0], c_out = widths[0]
   │ SGBlock  │
   │ SGBlock  │
   └────┬─────┘
        │
        ▼  ──── stage 2 (3 blocks, stride 2 then 1) ──── widths[1]
   ┌──────────┐
   │ SGBlock  │  c_in = widths[0], c_out = widths[1]   (stride 2)
   │ SGBlock  │
   │ SGBlock  │
   └────┬─────┘
        │
        ▼  ──── stage 3 (3 blocks, stride 2 then 1) ──── widths[2]
   ┌──────────┐
   │ SGBlock  │  c_in = widths[1], c_out = widths[2]   (stride 2)
   │ SGBlock  │
   │ SGBlock  │
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ AvgPool  │  → (B, widths[2])
   └────┬─────┘
        ▼
   ┌──────────┐
   │ Linear   │  → (B, num_classes)
   └──────────┘
```

`widths` is chosen by `priors.fibonacci_channels(c0=16, n_stages=3, mode=cfg.channel_mode)`:

| mode | widths (c0=16, n=3) | total approx params |
|---|---|---|
| `fib` | 16, 24, 40 | 127 k (no fractal) / 259 k (full) |
| `phi` | 16, 24, 40 | 127 k / 259 k (φ²≈2.6, similar to fib) |
| `linear` | 16, 32, 48 | 159 k / 311 k |

## NaturePriorBlock — forward pass

```
            x  ∈ ℝ^{B×c_in×H×W}                    skip(x) ∈ ℝ^{B×c_out×H'×W'}
            │
            ▼
   ┌─────────────────────────┐
   │      _GenericConv       │  ← chosen by flags
   │  (conv1, stride=stride) │
   └────────────┬────────────┘
                │
                ▼   ReLU
   ┌─────────────────────────┐
   │  _FractalPath  if flags.fractal else _GenericConv
   │     (conv2, stride=1)
   │
   │   if fractal:
   │      a = GenericConv(c_out)
   │      b = GenericConv(c_out) → recurse depth-1
   │      out = 0.5*(a + b)
   │   else:
   │      out = GenericConv(c_out)
   └────────────┬────────────┘
                │
                ▼
   ┌─────────────────────────┐
   │ y · gate(phases, α)     │   if flags.golden_modulate
   │  gate = ½cos(phases+α)+½│   per-channel multiplicative gate
   └────────────┬────────────┘
                │
                ▼
            y + skip(x) ─► ReLU  ──► out ∈ ℝ^{B×c_out×H'×W'}
```

`_GenericConv` is one of:

| flag set | implementation |
|---|---|
| `group=True` | `GroupConv2d`(in, out, k=3, stride, padding=1, group="c4") |
| `group=False, hex=True` | `HexConv2d`(in, out, k=3, stride, padding=1, toroidal=flags.toroidal) |
| `group=False, hex=False` | plain `nn.Conv2d` (+ circular pad if `toroidal`) |

In all three cases, the conv is followed by `BatchNorm2d`. If
`flags.cymatic_init`, the conv's weight is overwritten in place by
`cymatic_init_` (Chladni-mode + He variance scaling) — unless the conv
is `GroupConv2d`, where the orbit-aware init is left as future work.

## Shape table for CIFAR-10 (32×32, fib widths)

| stage | block | shape after (B=128) |
|---|---|---|
| stem | conv 3→16 | (128, 16, 32, 32) |
| 1 | SGBlock(16→16, s=1) | (128, 16, 32, 32) |
| 1 | SGBlock(16→16) ×2 | (128, 16, 32, 32) |
| 2 | SGBlock(16→24, s=2) | (128, 24, 16, 16) |
| 2 | SGBlock(24→24) ×2 | (128, 24, 16, 16) |
| 3 | SGBlock(24→40, s=2) | (128, 40, 8, 8) |
| 3 | SGBlock(40→40) ×2 | (128, 40, 8, 8) |
| head | avgpool + fc(40 → 10) | (128, 10) |

## nature-inspired priors — code map

| prior | symbol | file | line-of-truth |
|---|---|---|---|
| Fibonacci channels | `fibonacci_channels()` | `priors.py` | rounds to multiples of 8 |
| φ-scaling | mode `'phi'` in same | `priors.py` | geometric c·φ^k |
| Hex kernel mask | `hex_kernel_mask(k)` | `priors.py` | 3×3 = 7 taps; 5×5 = 19 taps |
| Chladni init | `cymatic_init_(conv)` | `priors.py` | sin(mπx)·sin(nπy) modes + He scale |
| Toroidal pad | `toroidal_pad(x, pad)` | `priors.py` | `F.pad(..., mode='circular')` |
| C4 group conv | `GroupConv2d` | `priors.py` | weight rotated 4× + max-pool orbit |
| Hex conv | `HexConv2d` | `priors.py` | masked conv with optional toroidal pad |
| Golden-angle gate | inside `NaturePriorBlock` | `blocks.py` | `cos(2π·k/φ + α)` per channel |

## Module families — `src/nature_inspired_networks/` (80 modules)

The shared package now exposes **80 modules** (single import surface,
Rule 14), exercised by **78 `tests/test_*.py` files (~780+ unit tests,
all green)** and backing **84 hypothesis docs across groups g1–g8**.
Modules are grouped below by the 8 thematic families; only the key
module per family is tabled with a one-line shape/role note (the full
list is the directory itself).

| family | theme | key modules | shape / role (one-liner) |
|---|---|---|---|
| **g1** | scaling & growth | `priors.fibonacci_channels`, `phi_budget`, `phi_compound`, `phi_multiscale`, `fib_depth` | choose stage widths/depths on φ/Fibonacci rules; budget-matched channel allocation (H09 best) |
| **g2** | layer / channel / neuron | `phi_activation`, `phi_relu`, `sine_act`, `constant_width`, `fib_ensemble` | per-channel/per-neuron transforms; activation + width policies; (B,C,H,W)→(B,C,H,W) |
| **g3** | topologies & graphs | `toroidal_pad`, `HexConv2d`, `GroupConv2d`, `hex_kernel_mask` | lattice/graph-structured conv: hex 7-tap mask, C4 orbit pool, circular (toroidal) padding |
| **g4** | kernels / attention / filters | `cymatic_init_`, `golden_modulate`, `golden_skip`, `fibottention` | kernel init + channel gating + Fib-dilated attention; (B,C,H,W)→gated (B,C,H,W) |
| **g5** | optimization / init / reg / NAS | `phi_lr`, `golden_momentum`, `golden_adam`, `phi_dropout`, `phi_decay`, `phi_init`, `fib_prune` | LR/momentum/β schedules, φ-weight init, φ-dropout, Fib structured pruning (training-time, no shape change) |
| **g6** | topological bridging | `compute_betti`, betti-collapse hooks | β₀/β₁ persistence on trained features; diagnostic, not in forward path |
| **g7** | cross-paradigm hybrids | `NaturePriorBlock`, `NaturePriorNet`, `full_fib` composer | compose multiple priors into one residual block / ResNet-shaped stack |
| **g8** | esoteric extensions | `golden_spiral_init`, `golden_resize`, `metatron`/`vesica` routing, `harmonic_cymatic` | golden-spiral PE/init, Metatron-graph routing, harmonic-cymatic SwiGLU FFN variants |

Every module is a drop-in flag/composer for `NaturePriorBlock` /
`NaturePriorNet`; no module duplicates code from another (Rule 14). The
mapping module↔hypothesis is in `IDEA_TABLE.md`.

## What the runner writes

Per experiment (`experiments/<dataset>/<tag>_seed<S>/`):

- `metrics.json` — `RunMetrics` dataclass (tag, top1/5, params, FLOPs,
  latency, rot-eq-err, composite, fingerprint, …)
- `history.json` — per-epoch list (train_loss, train_top1, test_top1, lr)

Global (`experiments/`):

- `experiment_log.jsonl` — one row per run, append-only
- `betti.json` — β₀/β₁ curves (produced by `compute_topology.py`)
- `plot_*.png` — generated by `dashboard.py`

Dashboard (`dashboard/`):

- `dashboard.html` — self-contained, sortable/filterable
- `plot_pareto.png`, `plot_ablation.png`, `plot_curves.png`,
  `plot_betti.png`

## Why is the block this shape?

The block follows the **`Sketch (Input → Core → Output)`** in the source
PDF (Sec. *The block design sketch*):

> Input → φ-scaled Fib channel expansion + hexagonal/platonic group conv
> (Flower-of-Life overlapping paths).
> Core → Fractal recursive sub-blocks (self-similar at φ ratios) with
> toroidal closure and cymatic wavelet init (simulated interference for
> "resonant" features).
> Output → Metatron-projected residuals + golden-angle positional/rotary
> modulation.

— mapped onto a residual-network-shaped block so it is drop-in
compatible with existing CIFAR backbones and ablation rows can be
created by flipping individual flags.
