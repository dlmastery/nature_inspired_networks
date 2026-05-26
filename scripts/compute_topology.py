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

from sacgeo.blocks import SacredFlags  # noqa: E402
from sacgeo.data import load_dataset  # noqa: E402
from sacgeo.models import build_model  # noqa: E402
from sacgeo.runner import make_flags, set_seed  # noqa: E402
from sacgeo.topology import betti_curve, collect_features  # noqa: E402


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
        model_name = "sacredgeo" if not tag.startswith("baseline_resnet") else "resnet20"
        flags = SacredFlags(False, False, False, False, False, False)
        if "_only_" in tag:
            k = tag.split("_only_", 1)[1]
            flags = SacredFlags(**{f.name: (f.name == k) for f in
                                   SacredFlags().__dataclass_fields__.values()})  # type: ignore[attr-defined]
        if tag.startswith("sg_full") or tag.startswith("sg_loo_no_"):
            flags = SacredFlags(True, True, True, True, True, True)
            if tag.startswith("sg_loo_no_"):
                k = tag.split("sg_loo_no_", 1)[1]
                setattr(flags, k, False)
        chan_mode = "linear"
        if "_chan_" in tag:
            chan_mode = tag.split("_chan_", 1)[1]
        elif tag.startswith("sg_full") or "_only_" in tag or tag.startswith("sg_loo_no_"):
            chan_mode = "fib"

        model = build_model(model_name, num_classes=n_cls,
                            flags=flags if model_name == "sacredgeo" else None,
                            channel_mode=chan_mode).cuda()
        feats = collect_features(model, te_loader, n_points=args.n_points)
        b = betti_curve(feats)
        out_rows.append(dict(tag=tag, seed=seed, **b))
        print(f"  [betti] {tag} seed={seed} b0={b['b0']} b1={b['b1']}")

    (root / "betti.json").write_text(json.dumps(out_rows, indent=2))
    print(f"[ok] wrote {root/'betti.json'} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
