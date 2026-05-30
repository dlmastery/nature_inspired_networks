"""Experiment runner — one ablation = one experiment.

CLI:
    python -m nature_inspired_networks.runner --config configs/cifar10_ablation.yaml --tag sg_full --seed 0

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
import torch.nn as nn
import yaml

from .blocks import NaturePriorFlags
from .data import load_dataset
from .eval import COMPOSITE_FINGERPRINT, COMPOSITE_FORMULA
# Importing models triggers the H13 / H18 / H19 self-registration of new
# model names (natureprior_phi_sparse / natureprior_fib_stride /
# natureprior_phi_relu). build_model is re-bound by those modules on
# this module too — we re-read it via the models attribute at call
# time to pick up any later wrappers.
from . import models as _models  # noqa: F401 — triggers self-registration
from .models import build_model
from .train import TrainConfig, Trainer, evaluate_full, save_run


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_flags(d: dict) -> NaturePriorFlags:
    return NaturePriorFlags(
        hex=bool(d.get("hex", True)),
        group=bool(d.get("group", True)),
        fractal=bool(d.get("fractal", True)),
        toroidal=bool(d.get("toroidal", True)),
        cymatic_init=bool(d.get("cymatic_init", True)),
        golden_modulate=bool(d.get("golden_modulate", True)),
        group_reduce=str(d.get("group_reduce", "max")),
    )


# ---------------------------------------------------------------------------
# Override-keys that ``build_model`` forwards into the per-family
# factories. Pulled out as a constant so the runner stays in sync with
# the sweep matrix in ``scripts/run_sweep.py``.
# ---------------------------------------------------------------------------
_MODEL_BUILD_KW: tuple[str, ...] = (
    "phi_inverted",
    "phi_budget_total", "phi_budget_n_stages", "phi_budget_mode",
    "phi_budget_blocks_per_stage",
    "phi_skip_init", "phi_skip_trainable",
)


def post_build_mutators(model, cfg: dict):
    """Apply optional post-construction model mutators in a fixed order.

    Hypotheses wired here (Rule 1 atomic — each override is the only
    delta vs. the baseline):

    * H07 — ``phi_fpn=True``: wrap a NaturePriorNet in a phi-spaced FPN.
      Uses ``phi_fpn_c0`` (default 16) and ``phi_fpn_levels`` (default
      4); the wrapper exposes a list of pyramid features at training
      time but for classification we reduce the deepest pyramid level
      through the original head.
    * H31 — ``golden_spiral_init=True``: re-initialise every Conv2d
      whose kernel matches ``golden_spiral_kernel`` (default 5).
    * H39 — ``phi_activation=True``: swap every ``nn.ReLU`` for the
      H39 :class:`PhiGELU` activation.
    * H42 — ``phi_init=True``: re-initialise every Conv2d/Linear with
      :func:`inits.phi_weight_init_`.

    The function is idempotent on a fresh model — the four flags are
    independent. Returns the (possibly-wrapped) model so the caller can
    re-bind ``model = post_build_mutators(model, cfg)``.
    """
    # H39 — activation swap (do this BEFORE re-initialisation so the
    # new modules are also subject to phi_init if both flags are set).
    if bool(cfg.get("phi_activation", False)):
        from .activations import swap_relu_with_phigelu
        swap_relu_with_phigelu(model)

    # H81 (G8) — sinusoidal (SIREN-style) activation swap. Single-flag
    # ablation: replace every nn.ReLU with SinusoidalActivation(sin(omega*x)).
    if bool(cfg.get("sine_activation", False)):
        from .sinusoidal_activation import swap_relu_with_sine
        swap_relu_with_sine(model, omega_init=float(cfg.get("omega_init", 1.0)))

    # H80 (G8) — constant-width (Reuleaux) kernel swap. Replaces every
    # square Conv2d (kernel >= 3) with a weight-preserving ConstantWidthConv2d
    # so the receptive field is near-isotropic. 1x1 skip convs are untouched.
    if bool(cfg.get("constant_width_kernel", False)):
        from .constant_width_kernel import apply_constant_width
        apply_constant_width(model)

    # H31 — golden-spiral init. Applied per-Conv2d that matches the
    # kernel size; mismatched kernels keep their default He init.
    if bool(cfg.get("golden_spiral_init", False)):
        from .inits import apply_golden_spiral_init
        k = int(cfg.get("golden_spiral_kernel", 5))
        apply_golden_spiral_init(model, k=k)

    # H42 — phi-init across every Conv2d / Linear.
    if bool(cfg.get("phi_init", False)):
        from .inits import apply_phi_init
        apply_phi_init(model)

    # H47 — phi-dropout. Injects a single PhiDropout BEFORE the model's
    # final ``fc`` Linear so the regulariser is data-flow-correct. If
    # the model has no ``fc`` attribute the override is a silent no-op.
    if str(cfg.get("dropout", "")).lower() in ("phi_dropout", "phidropout", "phi"):
        from .regularizers import PhiDropout
        cycle = str(cfg.get("dropout_cycle", "fib"))
        length = int(cfg.get("dropout_length", 5))
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            old_fc = model.fc
            model.fc = nn.Sequential(
                PhiDropout(cycle=cycle, length=length), old_fc,
            )

    # H07 — phi-spaced FPN wrap. Done last so the wrapped backbone
    # carries any init / activation deltas above. The wrapper builds
    # its own lateral / smoothing convs at default (He) init.
    if bool(cfg.get("phi_fpn", False)):
        from .phi_scaling import PhiSpacedFPN
        c0 = int(cfg.get("phi_fpn_c0", 16))
        n_levels = int(cfg.get("phi_fpn_levels", 4))
        model = _wrap_with_phi_fpn(model, c0=c0, n_levels=n_levels)

    return model


def _wrap_with_phi_fpn(backbone, c0: int, n_levels: int):
    """Wrap a NaturePriorNet-shaped backbone with a PhiSpacedFPN head.

    The FPN consumes the per-stage feature maps emitted by
    ``stagewise_features`` and reduces the deepest pyramid level
    through the backbone's original classifier. ``n_levels`` is
    clamped to the available number of stages; surplus levels are
    silently dropped (the H07 row's documented behaviour).
    """
    from .phi_scaling import PhiSpacedFPN

    if not hasattr(backbone, "stagewise_features"):
        # No multi-stage feature emitter — return backbone unmodified.
        return backbone

    # Probe channel widths by running a 1-sample dummy. This is
    # deterministic at fresh init and avoids hard-coding the schedule.
    backbone.eval()
    with torch.no_grad():
        feats = backbone.stagewise_features(torch.zeros(1, 3, 32, 32))
    in_channels = [int(f.shape[1]) for f in feats]
    n_levels = max(1, min(n_levels, len(in_channels)))
    in_channels = in_channels[-n_levels:]  # use deepest n_levels stages
    fpn = PhiSpacedFPN(in_channels=in_channels, c0=c0, phi_widths=True)
    head_in = fpn.widths[-1]
    # Replace the backbone's classifier so the deepest pyramid level
    # routes through a fresh nn.Linear of the right shape.
    fc_out_features = backbone.fc.out_features  # type: ignore[union-attr]

    class _FpnWrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.fpn = fpn
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(head_in, fc_out_features)

        def forward(self, x):
            feats = self.backbone.stagewise_features(x)[-n_levels:]
            pyramid = self.fpn(feats)
            deepest = pyramid[-1]
            return self.fc(self.pool(deepest).flatten(1))

    return _FpnWrapped()


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
    if model_name == "NaturePrior":
        flags = make_flags(cfg.get("flags", {}))
    else:
        flags = None
    # H02 / H03 — optional new kwargs (defaults preserve legacy behaviour).
    blocks_mode = cfg.get("blocks_mode", "uniform")
    blocks_per_stage = int(cfg.get("blocks_per_stage", 3))
    fib_start = int(cfg.get("fib_start", 3))
    input_resolution = cfg.get("input_resolution", None)

    # Pluck out model-build-only override keys (Phase A) into a sidecar
    # dict so the legacy build_model paths receive them as **kwargs.
    build_kwargs = {k: cfg[k] for k in _MODEL_BUILD_KW if k in cfg}
    # Resolve build_model fresh from the models module so the H13/H18/
    # H19 self-registration wrappers (installed at import time) are
    # picked up regardless of import order.
    _build_model = _models.build_model
    model = _build_model(
        model_name, num_classes=n_cls, flags=flags,
        channel_mode=channel_mode,
        blocks_mode=blocks_mode,
        blocks_per_stage=blocks_per_stage,
        fib_start=fib_start,
        input_resolution=input_resolution,
        **build_kwargs,
    )

    # Phase B — H07 / H31 / H39 / H42 post-build mutators.
    model = post_build_mutators(model, cfg)

    train_cfg = TrainConfig(
        epochs=cfg.get("epochs", 30),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 5e-4),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        target_top1=cfg.get("target_top1", 0.85),
        use_bf16=cfg.get("use_bf16", True),
        scheduler=cfg.get("scheduler", "cosine"),       # H10
        phi_lr_floor=float(cfg.get("phi_lr_floor", 1e-6)),
        # Phase C — optimizer + per-layer weight-decay routing.
        optimizer=str(cfg.get("optimizer", "adamw")),
        phi_decay_wd=bool(cfg.get("phi_decay_wd", False)),
        phi_decay_base=float(cfg.get("phi_decay_base", 5e-4)),
        # Phase D — trainer callbacks.
        prune_schedule=str(cfg.get("prune_schedule", "")),
        prune_length=int(cfg.get("prune_length", 5)),
        momentum_schedule=str(cfg.get("momentum_schedule", "")),
        fib_ensemble=cfg.get("fib_ensemble", None),
        # Control 1 (reviewer-flagged) — pin β1 to a constant value,
        # bypassing the H48 schedule. None preserves legacy behaviour.
        const_beta1=cfg.get("const_beta1", None),
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
