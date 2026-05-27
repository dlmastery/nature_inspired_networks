"""H18 — Fibonacci Stage Transition (alternating-stride downsampling).

The standard ResNet-style backbone halves spatial resolution at every
stage (stride 2 across the board). H18 hypothesises that an alternating
Fibonacci-pair cascade ``(stride 2, 3, 2, ...)`` produces a finer
multi-scale feature pyramid because the per-stage downsampling ratio
oscillates around ``sqrt(6) ~= 2.449``, closer to ``phi^2 ~= 2.618``
than the uniform ``2``. The retinal -> LGN -> V1 -> V2 cascade has
roughly the same average compression rate (Markram 2015, LeCun 1998).

This module provides:

* :func:`fib_stride_schedule` — generate an alternating stride schedule.
* :class:`FibStrideNaturePriorNet` — a NaturePriorNet variant that
  applies a custom per-stage stride schedule. The stem is unchanged.
  ``AdaptiveAvgPool2d(1)`` is used at the head so the final spatial
  size is collapsed to ``1x1`` regardless of the cumulative downsample
  factor (32 -> ~3, 64 -> ~5).

Wire-in: imports register a ``"natureprior_fib_stride"`` model name
with :func:`build_model` so the sweep row ``sg_only_fib_stride``
resolves through the standard runner path. Rule 1 atomicity: only the
per-stage stride schedule changes; channel widths and block contents
are identical to the curated NaturePriorNet baseline.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default alternating Fibonacci-pair stride schedule for a 3-stage
# CIFAR-scale NaturePriorNet. Stage 0 is the first feature stage and
# does not downsample (matches ResNet-20 stage1 stride=1). Subsequent
# stages alternate between stride 2 and stride 3.
DEFAULT_FIB_STRIDES: tuple[int, ...] = (1, 2, 3)


def fib_stride_schedule(n_stages: int,
                        pair: tuple[int, int] = (2, 3),
                        first_stage_stride: int = 1) -> list[int]:
    """Return an alternating-stride schedule of length ``n_stages``.

    Stage 0 takes ``first_stage_stride`` (default 1 to match the
    ResNet-CIFAR convention of no downsampling in the first feature
    stage). Subsequent stages alternate between ``pair[0]`` and
    ``pair[1]``. For ``pair=(2, 3)`` and ``n_stages=3`` the result is
    ``[1, 2, 3]``; for ``n_stages=4`` it is ``[1, 2, 3, 2]``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if pair[0] < 1 or pair[1] < 1:
        raise ValueError(f"stride pair entries must be >= 1; got {pair}")
    out = [int(first_stage_stride)]
    for i in range(1, n_stages):
        out.append(int(pair[(i - 1) % 2]))
    return out


def _downsampled_size(h_in: int, stride: int, kernel: int) -> int:
    """Compute output spatial size of a stride-`stride` conv with
    same-padding-style behaviour. Mirrors the PyTorch formula for
    ``Conv2d(stride=stride, padding=kernel // 2)``.
    """
    padding = kernel // 2
    return (h_in + 2 * padding - kernel) // stride + 1


class FibStrideNaturePriorNet(nn.Module):
    """NaturePriorNet variant with an alternating-stride schedule.

    The first block of each stage takes ``stride=strides[i]``;
    subsequent blocks in the same stage take stride 1, matching
    :class:`NaturePriorBlock` semantics. A stride-3 block uses a
    standard 3x3 kernel with ``padding=1`` so the output spatial size
    follows ``floor((H + 2 - 3) / 3) + 1 = floor((H - 1) / 3) + 1``.

    For a 32x32 input and strides ``(1, 2, 3)``: 32 -> 32 -> 16 -> 6.
    For strides ``(1, 2, 5)``: 32 -> 32 -> 16 -> 4. In every case the
    final stage feeds :class:`~torch.nn.AdaptiveAvgPool2d(1)` so the
    classifier head sees a fixed-shape feature regardless of the
    downsampling cascade.
    """

    def __init__(self, num_classes: int = 10, channel_mode: str = "fib",
                 flags=None,
                 strides: Sequence[int] = DEFAULT_FIB_STRIDES) -> None:
        super().__init__()
        from .blocks import NaturePriorBlock, NaturePriorFlags
        from .priors import fibonacci_channels
        flags = flags or NaturePriorFlags()
        self.flags = flags
        self.strides = tuple(int(s) for s in strides)
        widths = fibonacci_channels(16, len(self.strides), mode=channel_mode)
        self.widths = widths

        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        c_in = widths[0]
        for i, c_out in enumerate(widths):
            stride = self.strides[i]
            blocks: list[nn.Module] = [
                NaturePriorBlock(c_in, c_out, stride=stride, flags=flags,
                                 fractal_depth=2)
            ]
            # 2 additional same-stride-1 blocks per stage (matches the
            # curated NaturePriorNet 3-blocks-per-stage convention).
            for _ in range(2):
                blocks.append(NaturePriorBlock(c_out, c_out, stride=1,
                                               flags=flags, fractal_depth=2))
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.ModuleList(stages)

        # AdaptiveAvgPool guarantees a 1x1 output even when the
        # cumulative stride product exceeds the input resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)

    def predicted_spatial_cascade(self, h_in: int) -> list[int]:
        """Trace the spatial-size cascade for an input of size ``h_in``.

        Returns ``[h_in, h_after_stage0, ..., h_after_stageN_minus_1]``.
        Each step uses kernel=3, padding=1, so the per-stage rule is
        ``floor((h + 2 - 3) / s) + 1 = floor((h - 1) / s) + 1``.
        """
        out = [int(h_in)]
        h = h_in
        for s in self.strides:
            h = _downsampled_size(h, stride=int(s), kernel=3)
            out.append(h)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# build_model registration
# ---------------------------------------------------------------------------
def _register_fib_stride_variant() -> None:
    from . import models as _models
    from . import runner as _runner

    original = _models.build_model
    if getattr(original, "_fib_stride_wrapped", False):
        return

    def build_model(name: str, num_classes: int, flags=None,
                    channel_mode: str = "fib", **kwargs):
        if name.lower() in {"natureprior_fib_stride", "fib_stride_natureprior"}:
            return FibStrideNaturePriorNet(
                num_classes=num_classes, channel_mode=channel_mode,
                flags=flags, strides=DEFAULT_FIB_STRIDES,
            )
        return original(name, num_classes, flags=flags,
                        channel_mode=channel_mode, **kwargs)

    build_model._fib_stride_wrapped = True  # type: ignore[attr-defined]
    build_model._original = original  # type: ignore[attr-defined]
    _models.build_model = build_model  # type: ignore[assignment]
    _runner.build_model = build_model  # type: ignore[assignment]


_register_fib_stride_variant()
