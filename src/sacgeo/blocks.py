"""SacredGeoBlock — composable, ablation-friendly drop-in residual block.

Toggle priors with constructor flags. Output shape matches a standard
ResNet basic block so it can replace BasicBlock in CIFAR-ResNet stacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import (
    GroupConv2d,
    HexConv2d,
    cymatic_init_,
    golden_angle_phases,
    toroidal_pad,
)


@dataclass
class SacredFlags:
    """Boolean toggles for ablation studies."""

    hex: bool = True
    group: bool = True          # C4 group conv
    fractal: bool = True        # fractal-recursive sub-block
    toroidal: bool = True       # circular padding
    cymatic_init: bool = True   # Chladni-mode init
    golden_modulate: bool = True  # golden-angle channel rotary

    def tag(self) -> str:
        on = [k for k, v in self.__dict__.items() if v]
        return "+".join(on) if on else "vanilla"


class _GenericConv(nn.Module):
    """Conv unit chosen by flags. Used inside fractal paths."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int,
        flags: SacredFlags,
    ) -> None:
        super().__init__()
        if flags.group:
            self.conv = GroupConv2d(c_in, c_out, kernel_size=3, stride=stride,
                                    padding=1, group="c4", bias=False)
        elif flags.hex:
            self.conv = HexConv2d(c_in, c_out, kernel_size=3, stride=stride,
                                  padding=1, toroidal=flags.toroidal, bias=False)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if flags.cymatic_init:
            target = self.conv.conv if isinstance(self.conv, HexConv2d) else None
            if target is not None:
                cymatic_init_(target)
            elif isinstance(self.conv, nn.Conv2d):
                cymatic_init_(self.conv)
            # GroupConv2d weight already kaiming; cymatic init for it is
            # left to a future iteration (would need orbit-aware init).
        self.toroidal = flags.toroidal and isinstance(self.conv, nn.Conv2d)
        self.padding = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.toroidal:
            x = toroidal_pad(x, self.padding)
            y = F.conv2d(x, self.conv.weight, self.conv.bias,
                         stride=self.conv.stride, padding=0)
        else:
            y = self.conv(x)
        return self.bn(y)


class _FractalPath(nn.Module):
    """Self-similar recursive path (FractalNet 2017 style).

    depth=1 → single conv. depth=2 → conv(c→c) on one branch, conv-conv
    on the other, merged by mean. φ-scaled implicitly by channel
    schedule chosen at the model level.
    """

    def __init__(self, c_in: int, c_out: int, stride: int,
                 depth: int, flags: SacredFlags) -> None:
        super().__init__()
        self.depth = depth
        self.flags = flags
        if depth == 1:
            self.path = _GenericConv(c_in, c_out, stride, flags)
        else:
            # branch A: single conv at this depth (downsample if stride>1)
            self.a = _GenericConv(c_in, c_out, stride, flags)
            # branch B: conv → recursive sub-block at depth-1
            self.b1 = _GenericConv(c_in, c_out, stride, flags)
            self.b2 = _FractalPath(c_out, c_out, 1, depth - 1, flags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth == 1:
            return self.path(x)
        a = self.a(x)
        b = self.b2(self.b1(x))
        return 0.5 * (a + b)


class SacredGeoBlock(nn.Module):
    """Drop-in residual block with toggleable sacred-geometry priors.

    Architecture (residual form):
        x → conv1 (sacred priors) → ReLU
          → fractal-or-conv2 (sacred priors)
          → optional golden-angle channel modulation
          → + residual(x) → ReLU
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int = 1,
        flags: SacredFlags | None = None,
        fractal_depth: int = 2,
    ) -> None:
        super().__init__()
        flags = flags or SacredFlags()
        self.flags = flags
        self.conv1 = _GenericConv(c_in, c_out, stride, flags)
        if flags.fractal:
            self.conv2 = _FractalPath(c_out, c_out, 1, fractal_depth, flags)
        else:
            self.conv2 = _GenericConv(c_out, c_out, 1, flags)

        # residual skip
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.skip = nn.Identity()

        # golden-angle channel modulation (output stage)
        if flags.golden_modulate:
            phases = golden_angle_phases(c_out)
            # learnable amplitude → multiplicative gate by cos(phase + α·t)
            # where t is a learnable scalar; this is a cheap rotary proxy
            self.register_buffer("phases", phases)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.phases = None
            self.alpha = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        if self.phases is not None:
            gate = (torch.cos(self.phases + self.alpha) * 0.5 + 0.5)
            y = y * gate.view(1, -1, 1, 1)
        return F.relu(y + identity, inplace=True)
