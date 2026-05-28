"""Run the CIFAR-10 ablation matrix.

Each row in the matrix is (model, channel_mode, flags-dict, tag). Seeds are
swept inside the runner. Outputs go under experiments/cifar10/<tag>_seed<S>/.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.runner import run_one  # noqa: E402


def build_matrix(curated: bool = True) -> list[dict]:
    """Return list of {tag, overrides} dicts.

    `curated=True` returns a ~11-row matrix designed to fit in ~60 min on
    an RTX 4090 Laptop at 15 epochs/run; `curated=False` returns the full
    ~20-row sweep (baseline, single-prior, channel, full, leave-one-out).
    """
    base_flags = dict(hex=False, group=False, fractal=False, toroidal=False,
                      cymatic_init=False, golden_modulate=False)
    full = {k: True for k in base_flags}
    rows: list[dict] = []

    # Baselines (always)
    rows.append(dict(tag="baseline_resnet20",
                     overrides=dict(model="resnet20")))
    rows.append(dict(tag="baseline_sg_vanilla",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="linear",
                                    flags=base_flags.copy())))

    # Channel scaling ablation (priors all off; only channel schedule changes)
    for mode in ("fib", "phi"):
        rows.append(dict(tag=f"sg_chan_{mode}",
                         overrides=dict(model="NaturePrior",
                                        channel_mode=mode,
                                        flags=base_flags.copy())))

    # Single-prior on (each nature-inspired prior switched on alone, fib channels)
    for k in ("hex", "group", "fractal", "toroidal", "cymatic_init",
              "golden_modulate"):
        f = base_flags.copy(); f[k] = True
        rows.append(dict(tag=f"sg_only_{k}",
                         overrides=dict(model="NaturePrior",
                                        channel_mode="fib", flags=f)))

    # H58: avg-pool group reduction (fix for the dominant -10pp finding).
    # group_reduce lives INSIDE the flags dict so make_flags picks it up.
    f = base_flags.copy(); f["group"] = True; f["group_reduce"] = "mean"
    rows.append(dict(tag="sg_only_group_avg",
                     overrides=dict(model="NaturePrior", channel_mode="fib",
                                    flags=f)))

    # Full NaturePrior with each channel mode
    rows.append(dict(tag="sg_full_fib",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib", flags=full.copy())))
    # H58 + full hybrid: full priors but with avg-pool group reduction
    full_avg = full.copy(); full_avg["group_reduce"] = "mean"
    rows.append(dict(tag="sg_full_fib_avg",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib", flags=full_avg)))
    if not curated:
        rows.append(dict(tag="sg_full_phi",
                         overrides=dict(model="NaturePrior",
                                        channel_mode="phi", flags=full.copy())))
        # Leave-one-out from full (only in full sweep)
        for k in ("hex", "group", "fractal", "toroidal", "cymatic_init",
                  "golden_modulate"):
            f = full.copy(); f[k] = False
            rows.append(dict(tag=f"sg_loo_no_{k}",
                             overrides=dict(model="NaturePrior",
                                            channel_mode="fib", flags=f)))

    # ---------------------------------------------------------------
    # Code-Agent-2 — H06, H07, H09, H17.pure rows.
    # Each row is one config change vs. the baseline_resnet20 line
    # (Rule 1). Models live in src/nature_inspired_networks/phi_scaling.py
    # and are routed via the `phi_model` override key; the runner-side
    # wiring is intentionally additive (a follow-up extends build_model
    # to dispatch on phi_model when set). The rows are listed here as
    # the documented sweep deltas; future wiring is purely additive.
    # ---------------------------------------------------------------
    # H06 — Golden Ratio Bottleneck: c -> c/phi -> c bottleneck stack.
    rows.append(dict(tag="sg_only_golden_bottleneck",
                     overrides=dict(model="golden_bottleneck",
                                    phi_model="golden_bottleneck",
                                    phi_inverted=False)))
    # H07 — phi-Modulated Multi-Scale: feature pyramid with widths * phi^k.
    rows.append(dict(tag="sg_only_phi_multiscale",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    phi_fpn=True,
                                    phi_fpn_c0=16,
                                    phi_fpn_levels=4)))
    # H09 — Golden Proportion Parameter Budget: per-stage params in 1:phi:phi^2.
    rows.append(dict(tag="sg_only_phi_budget",
                     overrides=dict(model="phi_budget",
                                    phi_model="phi_budget",
                                    phi_budget_total=270_000,
                                    phi_budget_n_stages=3,
                                    phi_budget_mode="phi")))
    # H17.pure — Golden Skip Connections: residual scaled by learnable
    # alpha init=1/phi. Distinct from sg_only_golden_modulate (channel gate).
    rows.append(dict(tag="sg_only_golden_skip",
                     overrides=dict(model="golden_skip",
                                    phi_model="golden_skip",
                                    phi_skip_init=None,
                                    phi_skip_trainable=True)))

    # ---------------------------------------------------------------
    # Code-Agent-4 — H31, H39, H42, H44 rows.
    # Each row is one config change vs. baseline_sg_vanilla (Rule 1).
    # Primitives live in src/nature_inspired_networks/{inits,activations,
    # phi_decay}.py; the runner-side wiring is intentionally additive
    # (init / activation / regulariser overrides are picked up by the
    # runner via the documented override keys: phi_init, phi_activation,
    # golden_spiral_init, phi_decay_wd).
    # ---------------------------------------------------------------
    # H31 — Golden Spiral Kernel: 5x5 conv init from discretised golden
    # spiral, He-variance-preserving. Vanilla baseline + init flip only.
    rows.append(dict(tag="sg_only_golden_spiral_init",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    golden_spiral_init=True,
                                    golden_spiral_kernel=5)))
    # H39 — Harmonic φ-Activation: PhiGELU (x·σ(φx)) replaces ReLU.
    rows.append(dict(tag="sg_only_phi_activation",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    phi_activation=True)))
    # H42 — φ-Weight Initialization: He's √2 swapped for √φ. Init only.
    rows.append(dict(tag="sg_only_phi_init",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    phi_init=True)))
    # H44 — Golden Regularization: per-layer wd = base_wd / φ^k.
    rows.append(dict(tag="sg_only_phi_decay",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    phi_decay_wd=True,
                                    phi_decay_base=1e-2)))

    # ---------------------------------------------------------------
    # Code-Agent-5 — H41, H43, H47, H48 rows.
    # Each row is one config change vs. baseline_resnet20 (Rule 1).
    # Primitives live in src/nature_inspired_networks/{optimizers,
    # pruning,regularizers,schedulers}.py; runner-side wiring is
    # intentionally additive — the override keys (`optimizer`,
    # `prune_schedule`, `dropout`, `momentum_schedule`) are picked up
    # by a follow-up runner patch.
    # ---------------------------------------------------------------
    # H41 — Golden Ratio Optimizer: AdamW with β1=1/φ, β2=1/φ².
    rows.append(dict(tag="sg_only_golden_adam",
                     overrides=dict(model="resnet20",
                                    optimizer="golden_adam")))
    # H43 — Fibonacci Pruning: iterative magnitude prune at Fib epochs.
    rows.append(dict(tag="sg_only_fib_prune",
                     overrides=dict(model="resnet20",
                                    prune_schedule="fibonacci",
                                    prune_length=5)))
    # H47 — φ-Dropout: cyclical Fibonacci-ratio dropout rates.
    rows.append(dict(tag="sg_only_phi_dropout",
                     overrides=dict(model="resnet20",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib",
                                    dropout_length=5)))
    # H48 — Golden Momentum Scheduler: β1 *= 1/φ per epoch, floor 1/φ².
    rows.append(dict(tag="sg_only_golden_momentum",
                     overrides=dict(model="resnet20",
                                    momentum_schedule="golden")))

    # ---------------------------------------------------------------
    # Code-Agent-3 — H13, H18, H19, H20 (G2 layer/channel/neuron).
    # Each row toggles exactly one Rule-1 atomic change vs.
    # baseline_sg_vanilla. The new model variants
    # (``natureprior_phi_sparse``, ``natureprior_fib_stride``,
    # ``natureprior_phi_relu``) are registered with build_model at
    # import time by the corresponding src/ modules
    # (``sparse.py``, ``stride.py``, ``phi_threshold.py``) so the
    # runner resolves the names without an explicit edit to
    # models.build_model.
    # ---------------------------------------------------------------
    # H13 — Golden Neuron Connectivity: classifier head swapped for
    # PhiSparseLinear at density 1/phi (~0.618). Backbone priors all
    # off; the sparse-head effect is isolated.
    rows.append(dict(tag="sg_only_phi_sparse",
                     overrides=dict(model="natureprior_phi_sparse",
                                    channel_mode="fib",
                                    flags=base_flags.copy())))
    # H18 — Fibonacci Stage Transition: per-stage downsampling schedule
    # (1, 2, 3) instead of (1, 2, 2). AdaptiveAvgPool guarantees a
    # final 1x1 spatial regardless of cascade.
    rows.append(dict(tag="sg_only_fib_stride",
                     overrides=dict(model="natureprior_fib_stride",
                                    channel_mode="fib",
                                    flags=base_flags.copy())))
    # H19 — phi-Neuron Activation Threshold: stem ReLU replaced with
    # PhiReLU (per-channel learnable threshold init = 1/phi).
    rows.append(dict(tag="sg_only_phi_relu",
                     overrides=dict(model="natureprior_phi_relu",
                                    channel_mode="fib",
                                    flags=base_flags.copy())))
    # H20 — Fibonacci Ensemble: inference-time checkpoint averaging
    # with Fib weights (FibEnsemble / FibEMA wrappers in
    # src/nature_inspired_networks/ensemble.py). Training is unchanged;
    # the ``fib_ensemble`` override key documents the intent so a
    # follow-up runner patch can wire the post-hoc averaging.
    rows.append(dict(tag="sg_only_fib_ensemble",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    fib_ensemble=dict(enabled=True, K=8))))

    # ---------------------------------------------------------------
    # Code-Agent-1 — H01, H02, H03, H10 rows (G1 scaling-growth).
    # Each row toggles exactly one Rule-1 atomic change vs.
    # baseline_sg_vanilla. Primitives live in
    # src/nature_inspired_networks/{scaling,multi_scale,schedulers}.py
    # and are picked up by build_model + Trainer via documented
    # override keys (channel_mode, blocks_mode, input_resolution,
    # scheduler).
    # ---------------------------------------------------------------
    # H01 — phi-Compound Scaling: channel_mode='phi_compound' (single
    # change vs. baseline_sg_vanilla which uses 'linear'). All priors off.
    rows.append(dict(tag="sg_only_phi_compound",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="phi_compound",
                                    flags=base_flags.copy())))
    # H02 — Fibonacci Depth Progression: blocks_mode='fib' (single
    # change vs. baseline; 3-stage backbone → [2, 3, 5] block schedule).
    rows.append(dict(tag="sg_only_fib_depth",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    blocks_mode="fib",
                                    fib_start=2,
                                    flags=base_flags.copy())))
    # H03 — Golden Spiral Resolution Scaling: input_resolution=45
    # (the phi^1 multiple of the 28-base spec; closest /1-aligned value
    # to CIFAR's 32 that still inflates by phi). All priors off so the
    # resize wrapper is the only delta.
    rows.append(dict(tag="sg_only_golden_resize",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    input_resolution=45,
                                    flags=base_flags.copy())))
    # H10 — phi-Decay LR Scheduler: scheduler='phi_decay' replaces
    # CosineAnnealingLR. All priors off so the LR schedule is the only
    # delta (Rule 1).
    rows.append(dict(tag="sg_only_phi_lr",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    scheduler="phi_decay",
                                    flags=base_flags.copy())))

    # ---------------------------------------------------------------
    # G8 esoteric-extension rows (bonus_hypothesis.md, neutral-recast).
    # Only the two CNN-droppable G8 hypotheses get smoke rows; the rest
    # (attention / latent / graph / memory modules) ship as standalone
    # primitives without a ResNet-20 sweep row, matching how the G2/G4/G7
    # attention modules ship. Both are post-build mutators (Rule 1 atomic),
    # wired in runner.post_build_mutators via documented override keys.
    # ---------------------------------------------------------------
    # H80 — Constant-Width (Reuleaux) Kernel: every square Conv2d (k>=3)
    # swapped for a weight-preserving ConstantWidthConv2d (near-isotropic
    # receptive field). All priors off so the kernel mask is the only delta.
    rows.append(dict(tag="sg_only_constant_width",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    constant_width_kernel=True)))
    # H81 — Sinusoidal (SIREN-style) Activation: every nn.ReLU replaced
    # with sin(omega*x), omega init 1.0 (near-identity start). All priors
    # off so the activation swap is the only delta (Rule 1).
    rows.append(dict(tag="sg_only_sine_act",
                     overrides=dict(model="NaturePrior",
                                    channel_mode="fib",
                                    flags=base_flags.copy(),
                                    sine_activation=True,
                                    omega_init=1.0)))

    # ---------------------------------------------------------------
    # COMBO LADDER — additive stack of N orthogonal (non-competing)
    # priors on top of phi_budget (the only verified single-prior
    # positive). Each row adds exactly one new orthogonal-axis prior:
    #   N=2: + golden_momentum   (momentum schedule, trainer callback)
    #   N=3: + phi_dropout       (regulariser, model-level inject)
    #   N=4: + phi_decay_wd      (per-layer wd, optimizer)
    #   N=5: + phi_lr            (LR scheduler)
    #   N=6: + fib_ensemble      (post-hoc inference averaging)
    #   N=7: + sine_act          (activation swap; ReLU -> sin(omega*x))
    #   N=8: + fib_prune         (Fib-epoch magnitude pruning)
    # Axes are mutually orthogonal (each touches a different layer of
    # the training stack); they CANNOT compete by construction. This
    # is the additivity test missing from the original sg_full_fib
    # ablation (which stacked 6 priors on the SAME _GenericConv
    # forward path and was catastrophic). Compute: ~8 runs * ~8 min on
    # the 4090 ~= 65 min, CIFAR-10 seed 0, 12-epoch smoke.
    # ---------------------------------------------------------------
    PB = dict(model="phi_budget", phi_model="phi_budget",
              phi_budget_total=270_000, phi_budget_n_stages=3,
              phi_budget_mode="phi")  # the verified-winner base

    rows.append(dict(tag="combo2_pb_gm",
                     overrides=dict(**PB, momentum_schedule="golden")))
    rows.append(dict(tag="combo3_pb_gm_pd",
                     overrides=dict(**PB, momentum_schedule="golden",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib", dropout_length=5)))
    rows.append(dict(tag="combo4_pb_gm_pd_pdw",
                     overrides=dict(**PB, momentum_schedule="golden",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib", dropout_length=5,
                                    phi_decay_wd=True, phi_decay_base=5e-4)))
    rows.append(dict(tag="combo5_pb_gm_pd_pdw_plr",
                     overrides=dict(**PB, momentum_schedule="golden",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib", dropout_length=5,
                                    phi_decay_wd=True, phi_decay_base=5e-4,
                                    scheduler="phi_decay")))
    rows.append(dict(tag="combo6_pb_gm_pd_pdw_plr_fe",
                     overrides=dict(**PB, momentum_schedule="golden",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib", dropout_length=5,
                                    phi_decay_wd=True, phi_decay_base=5e-4,
                                    scheduler="phi_decay",
                                    fib_ensemble=dict(enabled=True, K=8))))
    rows.append(dict(tag="combo7_pb_gm_pd_pdw_plr_fe_sa",
                     overrides=dict(**PB, momentum_schedule="golden",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib", dropout_length=5,
                                    phi_decay_wd=True, phi_decay_base=5e-4,
                                    scheduler="phi_decay",
                                    fib_ensemble=dict(enabled=True, K=8),
                                    sine_activation=True, omega_init=1.0)))
    rows.append(dict(tag="combo8_pb_gm_pd_pdw_plr_fe_sa_fp",
                     overrides=dict(**PB, momentum_schedule="golden",
                                    dropout="phi_dropout",
                                    dropout_cycle="fib", dropout_length=5,
                                    phi_decay_wd=True, phi_decay_base=5e-4,
                                    scheduler="phi_decay",
                                    fib_ensemble=dict(enabled=True, K=8),
                                    sine_activation=True, omega_init=1.0,
                                    prune_schedule="fibonacci", prune_length=5)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar10_quick.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="*", help="run only these tags",
                    default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--curated", action="store_true",
                    help="use the 11-row curated matrix (default)")
    ap.add_argument("--full", action="store_true",
                    help="use the full ~20-row sweep")
    ap.add_argument("--root", default="experiments")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    matrix = build_matrix(curated=not args.full)
    if args.only:
        matrix = [m for m in matrix if m["tag"] in args.only]

    print(f"[sweep] {len(matrix)} configs × {len(args.seeds)} seeds = "
          f"{len(matrix) * len(args.seeds)} runs")
    t0 = time.perf_counter()
    for row in matrix:
        for seed in args.seeds:
            tag = row["tag"]
            out_dir = Path(args.root) / base_cfg["dataset"] / f"{tag}_seed{seed}"
            if args.skip_existing and (out_dir / "metrics.json").exists():
                print(f"  [skip] {tag} seed={seed} (exists)")
                continue
            cfg = copy.deepcopy(base_cfg)
            cfg.update(row["overrides"])
            print(f"  [run]  {tag} seed={seed}")
            t1 = time.perf_counter()
            try:
                run_one(cfg, tag=tag, seed=seed, root=args.root)
                print(f"         {time.perf_counter() - t1:.1f}s")
            except Exception as exc:  # keep going on a single failure
                err_dir = out_dir; err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / "error.txt").write_text(repr(exc))
                print(f"         FAILED: {exc!r}")
    print(f"[sweep] total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
