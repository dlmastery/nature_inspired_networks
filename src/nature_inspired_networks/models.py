"""NaturePriorNet (CIFAR-scale) + a strict ResNet-20 baseline.

Both models share the same head/stem so flop and param counts are
directly comparable — the only difference is the block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NaturePriorFlags, NaturePriorBlock
from .multi_scale import GoldenSpiralResize
from .priors import fibonacci_channels
from .scaling import resolve_blocks_schedule


# ---------------------------------------------------------------------------
# Baseline ResNet-20 (CIFAR), He et al. 2015 — Section 4.2
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class ResNet20(nn.Module):
    """3 stages × 3 blocks → 18 conv layers + stem/head = 20 layers."""

    def __init__(self, num_classes: int = 10, widths: Sequence[int] = (16, 32, 64)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(widths[0], widths[0], n=3, stride=1)
        self.stage2 = self._make_stage(widths[0], widths[1], n=3, stride=2)
        self.stage3 = self._make_stage(widths[1], widths[2], n=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

    @staticmethod
    def _make_stage(c_in: int, c_out: int, n: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [_BasicBlock(c_in, c_out, stride=stride)]
        for _ in range(n - 1):
            layers.append(_BasicBlock(c_out, c_out, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def stagewise_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        x = self.stem(x); feats.append(x)
        x = self.stage1(x); feats.append(x)
        x = self.stage2(x); feats.append(x)
        x = self.stage3(x); feats.append(x)
        return feats


# ---------------------------------------------------------------------------
# NaturePriorNet — same depth/width schedule but blocks = NaturePriorBlock
# ---------------------------------------------------------------------------
@dataclass
class NaturePriorConfig:
    num_classes: int = 10
    # Channel schedule mode for the three stages
    channel_mode: str = "fib"     # 'fib' | 'phi' | 'phi_compound' | 'linear'
    base_channels: int = 16
    n_stages: int = 3
    blocks_per_stage: int = 3
    fractal_depth: int = 2
    flags: NaturePriorFlags = None       # type: ignore[assignment]
    # H02 — Fibonacci depth progression. blocks_mode='uniform' (default)
    # replicates the legacy scalar behaviour byte-for-byte; 'fib' selects
    # per-stage Fibonacci block counts via scaling.fibonacci_depths.
    blocks_mode: str = "uniform"     # 'uniform' | 'fib' | 'linear'
    fib_start: int = 3               # offset into (1,1,2,3,5,8,13,21,...)
    # H03 — Golden-spiral input resize. None preserves the legacy input
    # resolution (no resize); an int wraps the stem with GoldenSpiralResize.
    input_resolution: int | None = None


class NaturePriorNet(nn.Module):
    def __init__(self, cfg: NaturePriorConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or NaturePriorConfig()
        cfg.flags = cfg.flags or NaturePriorFlags()
        self.cfg = cfg

        widths = fibonacci_channels(
            cfg.base_channels, cfg.n_stages, mode=cfg.channel_mode
        )
        self.widths = widths

        # H02 — resolve per-stage block counts (uniform / fib / linear).
        self.block_counts = resolve_blocks_schedule(
            cfg.blocks_per_stage, cfg.n_stages,
            mode=cfg.blocks_mode, fib_start=cfg.fib_start,
        )

        # H03 — optional input-resize wrapper. None preserves legacy.
        if cfg.input_resolution is not None:
            self.resize = GoldenSpiralResize(int(cfg.input_resolution))
        else:
            self.resize = None

        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        c_in = widths[0]
        for i, c_out in enumerate(widths):
            stride = 1 if i == 0 else 2
            n_blocks = self.block_counts[i]
            blocks: list[nn.Module] = [
                NaturePriorBlock(c_in, c_out, stride=stride,
                               flags=cfg.flags, fractal_depth=cfg.fractal_depth)
            ]
            for _ in range(n_blocks - 1):
                blocks.append(NaturePriorBlock(c_out, c_out, stride=1,
                                             flags=cfg.flags,
                                             fractal_depth=cfg.fractal_depth))
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize is not None:
            x = self.resize(x)
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def stagewise_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        if self.resize is not None:
            x = self.resize(x)
        x = self.stem(x); feats.append(x)
        for s in self.stages:
            x = s(x); feats.append(x)
        return feats


def _regnetx_stock_baseline_widths() -> dict[str, tuple]:
    """Hard-coded RegNetX-200MF reference parameters from the original
    paper (Radosavovic et al. 2020 "Designing Network Design Spaces",
    arXiv:2003.13678 Table 9 / torchvision.models.regnet docstring).

    RegNetX-200MF was dropped from torchvision >= 0.21 (the smallest
    stock RegNetX is now 400MF). We provide the canonical 200MF init
    parameters here so the reviewer-flagged Control 3b row can build a
    stock 200MF (~2.7M params) AND a shrunk variant at iso-params with
    ``phi_budget`` (~270k params) via a binary search over ``w_0``.
    """
    # Reference: Designing Network Design Spaces paper Table 9.
    return {
        "regnet_x_200mf": dict(
            depth=13, w_0=24, w_a=36.44, w_m=2.49, group_width=8,
        ),
        "regnet_x_400mf": dict(
            depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16,
        ),
    }


def _build_regnet_from_params(init_params: dict, num_classes: int,
                              stem_width: int = 32) -> nn.Module:
    """Wrap ``BlockParams.from_init_params`` + ``RegNet(...)`` so a single
    callable produces a fresh CIFAR-shaped RegNet. ``stem_width`` is
    forwarded as RegNet's stem; default 32 matches torchvision."""
    from torchvision.models.regnet import BlockParams, RegNet
    bp = BlockParams.from_init_params(**init_params)
    return RegNet(bp, num_classes=num_classes, stem_width=stem_width)


def width_multiplier_search(target_params: int,
                            base_init: dict | None = None,
                            num_classes: int = 100,
                            tol_frac: float = 0.05,
                            max_iter: int = 32) -> tuple[dict, int, float]:
    """Binary-search the ``w_0`` width multiplier of a RegNetX-200MF
    base such that the realised total param count lands within
    ``tol_frac`` of ``target_params``.

    Returns
    -------
    tuple[dict, int, float]
        ``(init_params, realised_params, w0_scale)`` -- the resolved
        BlockParams kwargs (with both ``w_0`` and ``w_a`` scaled by
        ``w0_scale``), the realised param count of the resulting model,
        and the discovered ``w0_scale`` itself.

    Notes
    -----
    Both ``w_0`` and ``w_a`` are scaled because the RegNet width
    parameterisation widens both the starting width and the per-block
    growth slope; scaling only ``w_0`` would otherwise leave the deep
    stages largely unchanged. The search is monotone in ``w0_scale``
    (param count grows ~quadratically), so a bisection converges in
    log2(range/tol) iterations.
    """
    if base_init is None:
        base_init = _regnetx_stock_baseline_widths()["regnet_x_200mf"]
    if target_params <= 0:
        raise ValueError(f"target_params must be positive, got {target_params}")

    def _count(scale: float) -> tuple[int, dict]:
        init = dict(base_init)
        # RegNet's BlockParams.from_init_params asserts w_0 % 8 == 0;
        # quantise to the nearest multiple of 8 with a floor of 8.
        w0_raw = init["w_0"] * scale
        init["w_0"] = max(8, int(round(w0_raw / 8.0)) * 8)
        init["w_a"] = max(0.0, float(init["w_a"]) * scale)
        m = _build_regnet_from_params(init, num_classes=num_classes)
        n = sum(p.numel() for p in m.parameters())
        return n, init

    lo, hi = 0.05, 4.0
    # Verify the bracket actually straddles the target.
    n_lo, _ = _count(lo)
    n_hi, _ = _count(hi)
    if n_lo > target_params * (1 + tol_frac):
        # Even the smallest scale overshoots — return the lo bracket.
        n, init = _count(lo)
        return init, n, lo
    if n_hi < target_params * (1 - tol_frac):
        # Even the largest scale undershoots — return the hi bracket.
        n, init = _count(hi)
        return init, n, hi

    best: tuple[float, int, dict, float] | None = None  # (err, n, init, scale)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        n_mid, init_mid = _count(mid)
        err = abs(n_mid - target_params) / target_params
        if best is None or err < best[0]:
            best = (err, n_mid, init_mid, mid)
        if err < tol_frac:
            return init_mid, n_mid, mid
        if n_mid < target_params:
            lo = mid
        else:
            hi = mid
    # Best-effort return: tightest match found.
    assert best is not None
    return best[2], best[1], best[3]


def build_regnetx(name: str, num_classes: int = 100,
                  target_params: int | None = None) -> nn.Module:
    """Build a RegNetX model.

    Names:
      - ``"regnetx_200mf"`` -- stock RegNetX-200MF (~2.7M params, the
        canonical baseline from Radosavovic 2020 Table 9). Built via
        local BlockParams since torchvision >= 0.21 no longer exposes
        the 200MF helper directly.
      - ``"regnetx_200mf_shrunk"`` -- RegNetX-200MF shrunk via
        :func:`width_multiplier_search` to hit ``target_params``
        (default 270_000 to iso-compare with the H09 ``phi_budget``).

    Raises
    ------
    ImportError
        If torchvision is unavailable (the venv lacks the RegNet
        module). The error message points the user at the
        ``torchvision>=0.15`` install requirement (RegNetX has been
        present since torchvision 0.14).
    """
    try:
        from torchvision.models.regnet import (  # noqa: F401
            BlockParams,
            RegNet,
        )
    except ImportError as exc:  # pragma: no cover — env-only
        raise ImportError(
            "torchvision RegNetX not present in env; install "
            "torchvision >= 0.15 (Control 3b wiring; see "
            "controls/PLAN.md control3_baseline_tuned.yaml)"
        ) from exc

    n = name.lower()
    baseline = _regnetx_stock_baseline_widths()
    if n in ("regnetx_200mf", "regnet_x_200mf"):
        init = dict(baseline["regnet_x_200mf"])
        return _build_regnet_from_params(init, num_classes=num_classes)
    if n in ("regnetx_200mf_shrunk", "regnet_x_200mf_shrunk"):
        budget = target_params if target_params is not None else 270_000
        init, _, _ = width_multiplier_search(
            int(budget), base_init=baseline["regnet_x_200mf"],
            num_classes=num_classes,
        )
        return _build_regnet_from_params(init, num_classes=num_classes)
    raise ValueError(f"unknown RegNetX variant {name!r}")


def build_model(name: str, num_classes: int, flags: NaturePriorFlags | None = None,
                channel_mode: str = "fib",
                blocks_mode: str = "uniform",
                blocks_per_stage: int = 3,
                fib_start: int = 3,
                input_resolution: int | None = None,
                **kwargs) -> nn.Module:
    """Dispatch a model name to its constructor.

    ``**kwargs`` carries optional sweep-row override keys that target
    sibling factory modules (``phi_scaling.build_phi_model`` for the H06
    / H07 / H09 / H17.pure families). Unknown kwargs are ignored at the
    legacy paths so the runner can forward all sweep keys uniformly.
    """
    # Accept any casing; canonicalize once.
    n = name.lower()
    if n == "resnet20":
        return ResNet20(num_classes=num_classes)
    if n in ("natureprior", "nature_prior", "sacredgeo"):  # legacy alias for old configs
        cfg = NaturePriorConfig(num_classes=num_classes, channel_mode=channel_mode,
                                flags=flags,
                                blocks_mode=blocks_mode,
                                blocks_per_stage=blocks_per_stage,
                                fib_start=fib_start,
                                input_resolution=input_resolution)
        return NaturePriorNet(cfg)
    # Control 4 (reviewer-flagged) — ViT-Tiny smoke (H71 IcosaRoPE3D).
    if n in ("vit_tiny", "vit_tiny_icosa", "vit_tiny_rope1d"):
        from .vit_tiny import build_vit_tiny
        # The cfg keys are namespaced ``vit_*`` so they don't collide
        # with the existing phi_* family.
        rope_kind = str(kwargs.get("vit_rope_kind", ""))
        if not rope_kind:
            # Sensible default per the cfg row.
            rope_kind = "icosa3d" if n == "vit_tiny_icosa" else "rope1d"
        return build_vit_tiny(
            num_classes=num_classes,
            embed_dim=int(kwargs.get("vit_embed_dim", 198)),
            num_heads=int(kwargs.get("vit_num_heads", 6)),
            head_dim=int(kwargs.get("vit_head_dim", 33)),
            depth=int(kwargs.get("vit_depth", 12)),
            patch_size=int(kwargs.get("vit_patch_size", 4)),
            img_size=int(kwargs.get("vit_img_size", 32)),
            mlp_ratio=float(kwargs.get("vit_mlp_ratio", 4.0)),
            rope_kind=rope_kind,
            rope_base=float(kwargs.get("vit_rope_base", 10_000.0)),
        )
    # Control 3b (reviewer-flagged) — RegNetX-200MF stock + shrunk.
    if n in ("regnetx_200mf", "regnet_x_200mf",
             "regnetx_200mf_shrunk", "regnet_x_200mf_shrunk"):
        target = None
        if "regnetx_param_budget" in kwargs:
            target = int(kwargs["regnetx_param_budget"])
        return build_regnetx(n, num_classes=num_classes,
                             target_params=target)
    # H06 / H09 / H17.pure — phi-scaling family. Route via build_phi_model
    # (imported lazily to avoid a circular dep at module load time).
    if n in ("golden_bottleneck", "phi_budget", "golden_skip"):
        from .phi_scaling import build_phi_model
        # Translate the runner-side ``phi_*`` cfg keys into the kwarg
        # names that build_phi_model accepts.
        phi_kw: dict = {}
        if n == "golden_bottleneck":
            if "phi_inverted" in kwargs:
                phi_kw["inverted"] = bool(kwargs["phi_inverted"])
        elif n == "phi_budget":
            if "phi_budget_total" in kwargs:
                phi_kw["B_total"] = int(kwargs["phi_budget_total"])
            if "phi_budget_n_stages" in kwargs:
                phi_kw["n_stages"] = int(kwargs["phi_budget_n_stages"])
            if "phi_budget_mode" in kwargs:
                phi_kw["budget_mode"] = str(kwargs["phi_budget_mode"])
            if "phi_budget_blocks_per_stage" in kwargs:
                phi_kw["blocks_per_stage"] = int(kwargs["phi_budget_blocks_per_stage"])
        elif n == "golden_skip":
            if "phi_skip_init" in kwargs:
                init_val = kwargs["phi_skip_init"]
                phi_kw["init"] = None if init_val is None else float(init_val)
            if "phi_skip_trainable" in kwargs:
                phi_kw["trainable"] = bool(kwargs["phi_skip_trainable"])
        return build_phi_model(n, num_classes=num_classes, **phi_kw)
    raise ValueError(f"unknown model '{name}'")


# Ensure the H13 / H18 / H19 self-registering variants are imported so
# their ``build_model`` wrappers run at package import time. The imports
# are guarded — if one fails the runner still works on the legacy paths.
def _autoregister_variants() -> None:
    for mod in ("sparse", "stride", "phi_threshold"):
        try:
            __import__(f"nature_inspired_networks.{mod}")
        except Exception:  # noqa: BLE001 — diagnostic-only autoregister
            pass


_autoregister_variants()
