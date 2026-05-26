"""Experiment runner — one ablation = one experiment.

CLI:
    python -m sacgeo.runner --config configs/cifar10_ablation.yaml --tag sg_full --seed 0

Outputs go under experiments/<dataset>/<tag>_seed<S>/.
Writes per-run metrics.json + history.json and appends to experiment_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .blocks import SacredFlags
from .data import load_dataset
from .eval import COMPOSITE_FINGERPRINT, COMPOSITE_FORMULA
from .models import build_model
from .train import TrainConfig, Trainer, evaluate_full, save_run


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_flags(d: dict) -> SacredFlags:
    return SacredFlags(
        hex=bool(d.get("hex", True)),
        group=bool(d.get("group", True)),
        fractal=bool(d.get("fractal", True)),
        toroidal=bool(d.get("toroidal", True)),
        cymatic_init=bool(d.get("cymatic_init", True)),
        golden_modulate=bool(d.get("golden_modulate", True)),
    )


def run_one(cfg: dict, tag: str, seed: int, root: str = "experiments") -> Path:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_name = cfg["dataset"]
    tr_loader, te_loader, n_cls, _ = load_dataset(
        ds_name, root=cfg.get("data_root", "./data"),
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 4),
    )

    model_name = cfg["model"]
    channel_mode = cfg.get("channel_mode", "fib")
    if model_name == "sacredgeo":
        flags = make_flags(cfg.get("flags", {}))
    else:
        flags = None
    model = build_model(model_name, num_classes=n_cls, flags=flags,
                        channel_mode=channel_mode)

    train_cfg = TrainConfig(
        epochs=cfg.get("epochs", 30),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 5e-4),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        target_top1=cfg.get("target_top1", 0.85),
        use_bf16=cfg.get("use_bf16", True),
    )
    tr = Trainer(model, tr_loader, te_loader, n_cls, train_cfg, device=device)
    fit_info = tr.fit()

    metrics = evaluate_full(model, te_loader, dataset=ds_name, tag=tag,
                            seed=seed, epochs=train_cfg.epochs,
                            fit_info=fit_info, device=device)
    out_dir = Path(root) / ds_name / f"{tag}_seed{seed}"
    save_run(str(out_dir), metrics, fit_info, model=model)

    # append to experiment_log.jsonl
    log_path = Path(root) / "experiment_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            **metrics.to_dict(),
            "model": model_name,
            "channel_mode": channel_mode,
            "flags": cfg.get("flags", {}) if flags else None,
            "composite_formula": COMPOSITE_FORMULA,
        }) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["seed"] = args.seed
    out = run_one(cfg, args.tag, args.seed, root=args.root)
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
