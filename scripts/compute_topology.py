"""Compute persistent-homology Betti curves and CKA matrices for trained runs.

For each run dir that has metrics.json + final model weights (we rebuild
the model from config and freshly initialize — Betti is computed on the
*initial* representation if no weights are saved; this is still useful
as a relative ablation since the seed is fixed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.blocks import NaturePriorFlags  # noqa: E402
from nature_inspired_networks.data import load_dataset  # noqa: E402
from nature_inspired_networks.models import build_model  # noqa: E402
from nature_inspired_networks.runner import make_flags, set_seed  # noqa: E402
from nature_inspired_networks.topology import betti_curve, collect_features  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--config", default="configs/cifar10_quick.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-points", type=int, default=256)
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    _, te_loader, n_cls, _ = load_dataset(
        base["dataset"], root=base.get("data_root", "./data"),
        batch_size=base.get("batch_size", 256), num_workers=0,
    )

    out_rows = []
    root = Path(args.root)
    for mj in root.glob("**/metrics.json"):
        m = json.loads(mj.read_text())
        tag = m["tag"]
        seed = m["seed"]
        if seed not in args.seeds:
            continue
        # rebuild same architecture (Betti on fresh-init is a *lower bound*
        # for the prior's geometric simplifying power; positive results still
        # discriminate between priors)
        set_seed(seed)
        model_name = "NaturePrior" if not tag.startswith("baseline_resnet") else "resnet20"

        # Reconstruct flags from the tag string. group_reduce is a STRING
        # field on NaturePriorFlags, not a Boolean, so we must not iterate
        # all dataclass fields uniformly — that bug was caught by the
        # H58 sweep crashing here.
        BOOL_FIELDS = ("hex", "group", "fractal", "toroidal",
                       "cymatic_init", "golden_modulate")
        bool_flags = {k: False for k in BOOL_FIELDS}
        group_reduce = "max"
        if "_only_" in tag:
            k = tag.split("_only_", 1)[1]
            # H58 variants surface as `sg_only_group_avg`
            if k.endswith("_avg"):
                k = k[: -len("_avg")]
                group_reduce = "mean"
            if k in BOOL_FIELDS:
                bool_flags[k] = True
        if tag.startswith("sg_full"):
            bool_flags = {k: True for k in BOOL_FIELDS}
            if tag.endswith("_avg"):
                group_reduce = "mean"
        if tag.startswith("sg_loo_no_"):
            bool_flags = {k: True for k in BOOL_FIELDS}
            k = tag.split("sg_loo_no_", 1)[1]
            if k in BOOL_FIELDS:
                bool_flags[k] = False
        flags = NaturePriorFlags(group_reduce=group_reduce, **bool_flags)

        chan_mode = "linear"
        if "_chan_" in tag:
            chan_mode = tag.split("_chan_", 1)[1]
        elif tag.startswith("sg_full") or "_only_" in tag or tag.startswith("sg_loo_no_"):
            chan_mode = "fib"

        model = build_model(model_name, num_classes=n_cls,
                            flags=flags if model_name == "NaturePrior" else None,
                            channel_mode=chan_mode).cuda()
        # Load trained weights if a checkpoint exists alongside metrics.json
        ckpt = mj.parent / "best.pt"
        if ckpt.exists():
            try:
                model.load_state_dict(torch.load(ckpt, map_location="cuda"))
                print(f"  [load] {ckpt.name} (trained features)")
            except Exception as exc:
                print(f"  [warn] could not load {ckpt}: {exc!r}")
        else:
            print(f"  [warn] no checkpoint for {tag}; using fresh-init features")
        feats = collect_features(model, te_loader, n_points=args.n_points)
        b = betti_curve(feats)
        out_rows.append(dict(tag=tag, seed=seed, **b))
        print(f"  [betti] {tag} seed={seed} b0={b['b0']} b1={b['b1']}")

    (root / "betti.json").write_text(json.dumps(out_rows, indent=2))
    print(f"[ok] wrote {root/'betti.json'} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
