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

from sacgeo.runner import run_one  # noqa: E402


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
                     overrides=dict(model="sacredgeo",
                                    channel_mode="linear",
                                    flags=base_flags.copy())))

    # Channel scaling ablation (priors all off; only channel schedule changes)
    for mode in ("fib", "phi"):
        rows.append(dict(tag=f"sg_chan_{mode}",
                         overrides=dict(model="sacredgeo",
                                        channel_mode=mode,
                                        flags=base_flags.copy())))

    # Single-prior on (each sacred prior switched on alone, fib channels)
    for k in ("hex", "group", "fractal", "toroidal", "cymatic_init",
              "golden_modulate"):
        f = base_flags.copy(); f[k] = True
        rows.append(dict(tag=f"sg_only_{k}",
                         overrides=dict(model="sacredgeo",
                                        channel_mode="fib", flags=f)))

    # Full SacredGeo with each channel mode
    rows.append(dict(tag="sg_full_fib",
                     overrides=dict(model="sacredgeo",
                                    channel_mode="fib", flags=full.copy())))
    if not curated:
        rows.append(dict(tag="sg_full_phi",
                         overrides=dict(model="sacredgeo",
                                        channel_mode="phi", flags=full.copy())))
        # Leave-one-out from full (only in full sweep)
        for k in ("hex", "group", "fractal", "toroidal", "cymatic_init",
                  "golden_modulate"):
            f = full.copy(); f[k] = False
            rows.append(dict(tag=f"sg_loo_no_{k}",
                             overrides=dict(model="sacredgeo",
                                            channel_mode="fib", flags=f)))
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
