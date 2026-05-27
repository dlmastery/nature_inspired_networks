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
