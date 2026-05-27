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
    fib_start: int = 4               # offset into (1,1,2,3,5,8,13,21,...)
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


def build_model(name: str, num_classes: int, flags: NaturePriorFlags | None = None,
                channel_mode: str = "fib",
                blocks_mode: str = "uniform",
                blocks_per_stage: int = 3,
                fib_start: int = 4,
                input_resolution: int | None = None) -> nn.Module:
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
    raise ValueError(f"unknown model '{name}'")
