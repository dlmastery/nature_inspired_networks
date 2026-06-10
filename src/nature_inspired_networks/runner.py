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
from .eval import COMPOSITE_FINGERPRINT, COMPOSITE_FORMULA, count_flops
# Importing models triggers the H13 / H18 / H19 self-registration of new
# model names (natureprior_phi_sparse / natureprior_fib_stride /
# natureprior_phi_relu). build_model is re-bound by those modules on
# this module too — we re-read it via the models attribute at call
# time to pick up any later wrappers.
from . import models as _models  # noqa: F401 — triggers self-registration
from .models import build_model
from .train import TrainConfig, Trainer, evaluate_full, save_run


class FLOPTargetError(RuntimeError):
    """Synthesis-100 A3 (2026-06-06): raised when ``flops_target`` is set
    in the run config and the measured model FLOPs lie outside the
    accepted band ``flops_target * (1 +/- flops_tolerance)``.

    The runner aborts BEFORE any GPU compute is spent on training so an
    operator who mis-spec'd a prior (e.g. the Phase-9i 80.8 MFLOP overrun
    against a 41.2 MFLOP baseline) gets immediate feedback instead of a
    full training run + post-hoc audit.
    """


def set_seed(seed: int, headline_mode: bool = False) -> None:
    """Seed every RNG library the training stack touches.

    Synthesis-100 D4 (2026-06-06): the legacy ``set_seed`` set only the
    three top-level RNGs and enabled ``cudnn.benchmark``. That is enough
    for *informal* reproducibility (median over 3 seeds is stable to
    ~0.4 pp) but NOT for headline / cert runs where two seeded runs must
    produce bit-identical outputs.

    Parameters
    ----------
    seed
        The integer seed for ``random`` / ``numpy`` / ``torch``.
    headline_mode
        When True, this additionally:
          * sets ``torch.use_deterministic_algorithms(True)``
          * sets ``torch.backends.cudnn.deterministic = True``
          * sets ``torch.backends.cudnn.benchmark = False``
          * sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (required by some
            deterministic CUDA kernels on Ampere / Ada).
        When False (default), legacy fast-mode is preserved
        byte-for-byte so existing screening sweeps stay reproducible
        with respect to their prior runs.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if headline_mode:
        # CUBLAS workspace must be set BEFORE the first CUDA workspace
        # is allocated. Setting it later is a no-op; setting it here
        # before model build is correct.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch versions don't accept warn_only; fall back.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Per-worker DataLoader seed function.

    Synthesis-100 D4 / D5 (2026-06-06): every DataLoader worker spawns
    with its own ``random`` / ``numpy`` RNG state that is otherwise
    derived non-deterministically from process startup time. The
    PyTorch-recommended fix is to seed every worker from
    ``torch.initial_seed()`` (which IS already deterministic per the
    main-process seed + DataLoader generator). RandAugment, RandomErasing,
    Mixup, CutMix all consume np.random / random, so without this hook
    augmented batches differ between bit-identical-otherwise runs.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    # Control 3b — RegNetX-200MF dispatch keys.
    "regnetx_param_budget",
    # Control 4 — ViT-Tiny / H71 IcosaRoPE3D dispatch keys.
    "vit_embed_dim", "vit_num_heads", "vit_head_dim",
    "vit_depth", "vit_patch_size", "vit_img_size",
    "vit_mlp_ratio", "vit_rope_kind", "vit_rope_base",
)


def _phi_budget_build_kwargs(cfg: dict) -> dict:
    """Return phi_budget-only build kwargs derived from cfg.

    Phase-9e Wave-1 H88 wiring fix: when ``model=phi_budget`` the
    ``toroidal`` top-level cfg key (previously honoured only under
    ``model=NaturePrior`` via ``flags.toroidal``) is forwarded to the
    phi_budget factory so the H22 boundary mechanism stacks with the
    H09 channel mechanism. Default behaviour preserves legacy phi_budget
    rows: when the key is absent the kwarg is omitted and the factory
    defaults to ``toroidal=False``.
    """
    out: dict = {}
    if "toroidal" in cfg:
        out["toroidal"] = bool(cfg["toroidal"])
    return out


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

    # Control 2 (reviewer-flagged) — generic activation swap.
    # ``slot_activation`` ∈ {"tanh", "softplus", "gelu", "swish", "silu"}
    # dispatches to swap_relu_with(model, factory). The "sine" / "phi"
    # ablations remain on the existing dedicated helpers above so the
    # H81 omega-init / H39 beta-init wiring is preserved byte-for-byte.
    slot_act = str(cfg.get("slot_activation", "") or "").lower()
    if slot_act:
        from .activations import SLOT_ACTIVATION_FACTORIES, swap_relu_with
        if slot_act in SLOT_ACTIVATION_FACTORIES:
            swap_relu_with(model, SLOT_ACTIVATION_FACTORIES[slot_act])
        elif slot_act in ("sine", "sin"):
            # Delegate to the existing sine helper so omega_init is
            # respected.
            from .sinusoidal_activation import swap_relu_with_sine
            swap_relu_with_sine(
                model, omega_init=float(cfg.get("omega_init", 1.0)),
            )
        elif slot_act in ("phi", "phi_gelu", "phigelu"):
            from .activations import swap_relu_with_phigelu
            swap_relu_with_phigelu(model)
        else:
            raise ValueError(
                f"unknown slot_activation {slot_act!r}; "
                f"expected one of {sorted(SLOT_ACTIVATION_FACTORIES)} "
                f"+ {'sine'!r} / {'phi'!r}"
            )

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


def _check_flops_target(model, cfg: dict, input_size: tuple[int, ...]) -> None:
    """Synthesis-100 A3 (2026-06-06): refuse to launch on a model whose
    measured FLOPs lie outside the ``flops_target`` band.

    The check is OPTIONAL — when ``flops_target`` is absent (default) the
    function is a no-op, preserving legacy launch behaviour. When set,
    we accept either a single number (target ± ``flops_tolerance`` *
    target, default 10%) or a (min, max) tuple/list.

    Prints a one-line PASS / FAIL summary to stdout before any GPU
    training cycles are consumed.
    """
    target = cfg.get("flops_target")
    if target is None:
        return
    tolerance = float(cfg.get("flops_tolerance", 0.10))
    flops = count_flops(model, input_size=input_size)
    import math as _math
    if not _math.isfinite(flops):
        # fvcore returned NaN -- common on group-conv / hex-conv models.
        # Print a clear note but do NOT block; this is a measurement gap,
        # not a model gap. Operator can use ``flops_target_strict=true``
        # to escalate (default false).
        msg = (
            f"[runner] measured FLOPs: NaN (fvcore unable to count) / "
            f"target {target} -> WARNING (check skipped)"
        )
        print(msg)
        if bool(cfg.get("flops_target_strict", False)):
            raise FLOPTargetError(
                "flops_target set with flops_target_strict=true but "
                "count_flops returned NaN; cannot verify model is "
                "within the FLOP band."
            )
        return
    flops_M = flops / 1e6
    if isinstance(target, (list, tuple)) and len(target) == 2:
        # Tuple/list: interpret as a (lo, hi) raw-FLOP band. Caller can
        # pass either raw-FLOPs (e.g. (4e7, 4.5e7)) or MFLOPs (e.g.
        # (40, 45)); we detect the unit by magnitude (>= 1e4 -> raw).
        t0, t1 = float(target[0]), float(target[1])
        if t0 >= 1e4:
            lo = t0 / 1e6
            hi = t1 / 1e6
        else:
            lo = t0
            hi = t1
        target_repr = f"[{lo:.2f}M, {hi:.2f}M]"
    else:
        # Scalar: raw-FLOPs if > 1e4, otherwise already MFLOPs. The
        # threshold of 1e4 means a config with ``flops_target: 41.2``
        # is read as 41.2 MFLOPs (the convergence-regime baseline) and
        # ``flops_target: 41200000`` is read as 41.2 MFLOPs (same
        # number written as raw count).
        t = float(target)
        target_M = t / 1e6 if t >= 1e4 else t
        lo = target_M * (1.0 - tolerance)
        hi = target_M * (1.0 + tolerance)
        target_repr = f"{target_M:.2f}M +/- {tolerance * 100:.0f}%"
    ok = lo <= flops_M <= hi
    status = "PASS" if ok else "FAIL"
    print(
        f"[runner] measured FLOPs: {flops_M:.2f}M / target {target_repr} "
        f"-> {status}"
    )
    if not ok:
        raise FLOPTargetError(
            f"model FLOPs {flops_M:.2f}M outside target band {target_repr}; "
            f"synthesis-100 A3 refuses to launch."
        )


def run_one(cfg: dict, tag: str, seed: int, root: str = "experiments") -> Path:
    # Synthesis-100 D4 (2026-06-06): headline_mode is opt-in via the YAML
    # ``headline_mode`` knob. Default False preserves the legacy
    # cudnn.benchmark=True fast-path for screening sweeps.
    headline_mode = bool(cfg.get("headline_mode", False))
    set_seed(seed, headline_mode=headline_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_name = cfg["dataset"]
    # Wave-0 (2026-06-08): image_size is forwarded to the dataset loader
    # for benchmarks where the native resolution is not 32x32. The
    # CIFAR loaders ignore the kwarg (image_size only matters for
    # Imagenette / Tiny-ImageNet / ImageNet-100). Default 160 matches
    # Imagenette v2-160's native short-edge resize.
    image_size = int(cfg.get("image_size", 160))
    tr_loader, te_loader, n_cls, _ = load_dataset(
        ds_name, root=cfg.get("data_root", "./data"),
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 4),
        # Modern-recipe DataLoader knobs (default 0 -> legacy pipeline).
        randaugment_n=int(cfg.get("randaugment_n", 0)),
        randaugment_m=int(cfg.get("randaugment_m", 14)),
        random_erasing_p=float(cfg.get("random_erasing_p", 0.0)),
        image_size=image_size,
        # Synthesis-100 D4: pass headline_mode + seed for deterministic
        # DataLoader workers.
        headline_mode=headline_mode, seed=seed,
    )
    # Wave-0 (2026-06-08): YAML can override the dataset-implied class
    # count (e.g. Imagenette is 10-class but the future ImageNet-100
    # task will be 100-class on the same loader stub). When present,
    # ``num_classes`` overrides the loader-reported count. The model
    # factory then receives the YAML-declared count, not the loader's.
    if "num_classes" in cfg:
        n_cls = int(cfg["num_classes"])

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
    # Phase-9e Wave-1 H88 wiring fix: forward ``toroidal`` from cfg to
    # the phi_budget factory (only — NaturePrior consumes toroidal via
    # ``flags.toroidal``, RegNet / ViT do not have a boundary axis).
    if model_name.lower() == "phi_budget":
        build_kwargs.update(_phi_budget_build_kwargs(cfg))
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

    # Synthesis-100 A3 (2026-06-06): refuse to launch if the model's
    # FLOPs are outside the YAML-declared band. Done BEFORE any training
    # cycle so an iso-FLOPs misconfiguration costs zero GPU time.
    #
    # Wave-0 (2026-06-08): for non-CIFAR datasets the input is not
    # 32x32. We derive the eval input side from the YAML ``image_size``
    # for any dataset whose native side differs from CIFAR's 32, so
    # the FLOPs measurement / latency reflect the actual evaluation
    # tensor shape.
    if str(ds_name).lower() in {"imagenette", "imagenette2", "imagenette2-160"}:
        eval_input_size = (1, 3, image_size, image_size)
    else:
        eval_input_size = (1, 3, 32, 32)
    _check_flops_target(model, cfg, input_size=eval_input_size)

    train_cfg = TrainConfig(
        epochs=cfg.get("epochs", 30),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 5e-4),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        warmup_epochs=int(cfg.get("warmup_epochs", 0)),
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
        # H51 — Topological Betti Loss (Phase-9e Wave-1 H88 wiring).
        # Default 0.0 preserves legacy training byte-for-byte; positive
        # values add a BettiLoss(features) surrogate to the per-step CE
        # loss. See ``train.Trainer._step`` for the integration point.
        betti_loss_weight=float(cfg.get("betti_loss_weight", 0.0)),
        betti_persistence_threshold=float(cfg.get("betti_persistence_threshold", 0.1)),
        betti_max_pts=int(cfg.get("betti_max_pts", 64)),
        # Modern-recipe knobs — Mixup, CutMix, EMA. Default 0 preserves
        # legacy training byte-for-byte. See train.Trainer._step / fit
        # for the wiring points; RandAugment + RandomErasing live on the
        # DataLoader pipeline (load_dataset above) and are not part of
        # TrainConfig.
        mixup_alpha=float(cfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(cfg.get("cutmix_alpha", 0.0)),
        mixup_cutmix_prob=float(cfg.get("mixup_cutmix_prob", 0.5)),
        ema_decay=float(cfg.get("ema_decay", 0.0)),
        # Synthesis-100 D6 / D7 (2026-06-06): BN recalibration after the
        # EMA shadow load + un-mixed clean train-set top-1 sampling. Both
        # default to ON for any run that activates mixing / EMA.
        recalibrate_bn_after_ema=bool(cfg.get("recalibrate_bn_after_ema", True)),
        recalibrate_bn_max_batches=int(cfg.get("recalibrate_bn_max_batches", 50)),
        train_top1_clean_samples=int(cfg.get("train_top1_clean_samples", 1024)),
    )
    tr = Trainer(model, tr_loader, te_loader, n_cls, train_cfg, device=device)
    fit_info = tr.fit()

    metrics = evaluate_full(model, te_loader, dataset=ds_name, tag=tag,
                            seed=seed, epochs=train_cfg.epochs,
                            fit_info=fit_info, input_size=eval_input_size,
                            device=device)
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
