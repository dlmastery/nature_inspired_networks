"""phi-scaling primitives for hypotheses H06, H07, H09, H17.pure.

This module is a stand-alone primitive layer that augments the shared
nature-inspired prior set in :mod:`nature_inspired_networks.priors` and
:mod:`nature_inspired_networks.blocks`. It is deliberately additive --
it does not replace existing primitives -- so the legacy
`NaturePriorBlock` / `NaturePriorNet` behaviour is preserved
byte-for-byte.

Public surface
--------------
- ``GoldenBottleneck``      (H06)  c -> c/phi -> c bottleneck block
- ``PhiSpacedFPN``          (H07)  feature-pyramid with widths * phi**k
- ``phi_budget_widths``     (H09)  param-budget allocator (1:phi:phi**2:...)
- ``PhiBudgetNet``          (H09)  conv stack whose stage widths consume B
- ``GoldenSkipBlock``       (H17p) residual block with learnable skip scale init=1/phi
- ``build_phi_model``       dispatcher used by sweep rows; mirrors build_model

References (Citation Rigor):
    He, Zhang, Ren, Sun 2016 CVPR 'Deep Residual Learning for Image
    Recognition' (arXiv:1512.03385) -- bottleneck baseline (H06).
    Sandler, Howard, Zhu, Zhmoginov, Chen 2018 CVPR 'MobileNetV2:
    Inverted Residuals and Linear Bottlenecks' (arXiv:1801.04381) --
    inverted bottleneck baseline (H06).
    Lin, Dollar, Girshick, He, Hariharan, Belongie 2017 CVPR 'Feature
    Pyramid Networks for Object Detection' (arXiv:1612.03144) --
    power-of-two FPN baseline (H07).
    Tan, Le 2019 ICML 'EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks' (arXiv:1905.11946) -- compound
    scaling baseline (H09).
    Hayou, Ghosh, Doucet 2021 ICLR 'Stable ResNet' (arXiv:2010.12859)
    -- residual rescaling motivation (H17.pure).
    Bachlechner, Majumder, Mao, Cottrell, McAuley 2021 UAI 'ReZero is
    All You Need' (arXiv:2003.04887) -- learnable skip alpha (H17.pure).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


# ---------------------------------------------------------------------------
# H06 -- Golden Ratio Bottleneck
# ---------------------------------------------------------------------------
def _round8(x: int) -> int:
    """Round to the nearest multiple of 8 with a floor of 8.

    Mirrors the contract used by :func:`priors.fibonacci_channels` so that
    GoldenBottleneck widths align with the rest of the codebase. Kept as
    a private helper to avoid coupling -- there is no shared rounding API
    in the existing module surface.
    """
    return max(8, int(round(x / 8)) * 8)


class GoldenBottleneck(nn.Module):
    """phi:1:phi bottleneck conv block (H06).

    Architecture:

        x : (B, c_in, H, W)
          -> 1x1 conv reduce    (c_in   -> c_mid = round8(c_in / phi))
          -> ReLU
          -> 3x3 conv           (c_mid  -> c_mid, stride)
          -> ReLU
          -> 1x1 conv expand    (c_mid  -> c_out)
          -> BN
          -> (+ optional skip)
          -> ReLU

    The conventional ResNet-50 bottleneck uses a 4:1 ratio:
    c -> c/4 -> c. H06 replaces 4 with phi**2 = phi+1 ~= 2.618 so the
    middle channel is c/phi and the expansion is phi. With ``inverted=True``
    the block instead goes c -> c*phi -> c (MobileNetV2-style inverted
    residual, expansion ratio phi**2).

    Parameters
    ----------
    c_in, c_out : int
        Input/output channels. Default behaviour preserves c_out=c_in so
        the residual skip is identity-cheap.
    stride : int
        Spatial stride of the inner 3x3 conv (1 or 2).
    inverted : bool
        If True, c_mid = round8(c_in * phi) (expand-contract). Default
        False uses the contract-expand pattern from H06's hypothesis text.
    residual : bool
        If True (default), add a (possibly projected) skip from x.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int | None = None,
        stride: int = 1,
        inverted: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if c_out is None:
            c_out = c_in
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.inverted = inverted
        self.residual = residual
        if inverted:
            c_mid = _round8(int(round(c_in * PHI)))
        else:
            c_mid = _round8(int(round(c_in / PHI)))
        self.c_mid = c_mid
        self.reduce = nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.mid = nn.Conv2d(c_mid, c_mid, kernel_size=3,
                             stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.expand = nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)

        if residual:
            if stride != 1 or c_in != c_out:
                self.skip = nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(c_out),
                )
            else:
                self.skip = nn.Identity()
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.reduce(x)), inplace=True)
        y = F.relu(self.bn2(self.mid(y)), inplace=True)
        y = self.bn3(self.expand(y))
        if self.skip is not None:
            y = y + self.skip(x)
        return F.relu(y, inplace=True)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# H07 -- phi-Modulated Multi-Scale FPN
# ---------------------------------------------------------------------------
def phi_pyramid_widths(c0: int, n_levels: int) -> list[int]:
    """Return n_levels channel widths spaced by powers of phi.

    width_k = round8(c0 * phi**k), k = 0, 1, ..., n_levels - 1.

    Unlike ``fibonacci_channels(mode='phi')`` this returns a *pyramid*
    schedule (used for the lateral FPN width per scale level), not a
    *stage* schedule. Both can coexist.
    """
    if n_levels <= 0:
        raise ValueError("n_levels must be >= 1")
    return [_round8(int(round(c0 * (PHI ** k)))) for k in range(n_levels)]


class PhiSpacedFPN(nn.Module):
    """Feature-Pyramid Network with phi-spaced level widths (H07).

    Standard FPN (Lin et al 2017) widens all lateral outputs to a fixed
    width (e.g. 256 channels). H07 instead widens the k-th pyramid level
    to ``c0 * phi**k`` so high-resolution features have less capacity per
    channel than low-resolution features. This is the multi-scale flavour
    of phi-scaling -- complementary to the per-stage channel_mode='phi'
    that already exists.

    Input
    -----
    A list of per-stage feature maps ``[C2, C3, C4, ...]`` with strictly
    *decreasing* spatial resolution (channels-first, ``(B, C_k, H_k, W_k)``)
    -- the conventional bottom-up feed from a backbone.

    Output
    ------
    A list of per-level pyramid features ``[P2, P3, P4, ...]`` with the
    same length and the same spatial sizes as the input list, but
    channel widths now ``[c0, c0*phi, c0*phi**2, ...]``.

    The top-down pathway uses bilinear up-sampling so that strides
    between input scales need not be powers of two (which is the H07
    point -- phi-spaced strides are non-integer in the source resolution
    but we resample explicitly).
    """

    def __init__(self, in_channels: Sequence[int], c0: int = 16,
                 phi_widths: bool = True) -> None:
        super().__init__()
        n = len(in_channels)
        if phi_widths:
            self.widths = phi_pyramid_widths(c0, n)
        else:
            # Baseline: uniform widths (standard FPN)
            self.widths = [_round8(c0) for _ in range(n)]
        # lateral 1x1s to reproject each input to the per-level pyramid width
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels[k], self.widths[k], kernel_size=1, bias=False)
            for k in range(n)
        ])
        # smoothing 3x3s after top-down merge (per FPN)
        self.smooth = nn.ModuleList([
            nn.Conv2d(self.widths[k], self.widths[k], kernel_size=3,
                      padding=1, bias=False)
            for k in range(n)
        ])
        # 1x1 reprojections used to align top-down channel widths when
        # merging two adjacent levels (widths differ by phi).
        self.reproject = nn.ModuleList()
        for k in range(n - 1):
            # reproject self.widths[k+1] -> self.widths[k] for the merge at level k
            self.reproject.append(
                nn.Conv2d(self.widths[k + 1], self.widths[k],
                          kernel_size=1, bias=False)
            )

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        assert len(feats) == len(self.lateral), (
            f"PhiSpacedFPN expects {len(self.lateral)} feature levels, "
            f"got {len(feats)}"
        )
        # lateral 1x1
        laterals = [self.lateral[k](feats[k]) for k in range(len(feats))]
        # top-down merge starting from the deepest level
        n = len(laterals)
        merged: list[torch.Tensor | None] = [None] * n
        merged[n - 1] = laterals[n - 1]
        for k in range(n - 2, -1, -1):
            up = F.interpolate(
                merged[k + 1],
                size=laterals[k].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            up = self.reproject[k](up)
            merged[k] = laterals[k] + up
        outs = [self.smooth[k](merged[k]) for k in range(n)]  # type: ignore[arg-type]
        return outs


# ---------------------------------------------------------------------------
# H09 -- Golden Proportion Parameter Budget
# ---------------------------------------------------------------------------
def phi_budget_allocations(B_total: int, n_stages: int) -> list[int]:
    """Allocate B_total params across n_stages in ratio 1 : phi : phi**2 : ...

    Returns a list of integer param-budget per stage that sums to (at most)
    B_total. The closed-form share for stage k (0-indexed) is

        share_k = (phi - 1) * phi**k / (phi**n - 1)

    which sums to 1.0 across k = 0..n-1.
    """
    if B_total <= 0 or n_stages <= 0:
        raise ValueError("B_total and n_stages must be positive")
    denom = (PHI ** n_stages) - 1.0
    shares = [(PHI - 1.0) * (PHI ** k) / denom for k in range(n_stages)]
    raw = [B_total * s for s in shares]
    out = [int(round(r)) for r in raw]
    # repair rounding drift: trim/give last stage so total equals B_total
    drift = B_total - sum(out)
    out[-1] += drift
    return out


def phi_budget_widths(B_total: int, n_stages: int, c_in: int = 3,
                      kernel: int = 3, blocks_per_stage: int = 2) -> list[int]:
    """Derive per-stage channel widths so the network consumes ~B_total params.

    For a 2-block (each block = 2 convs of c_out x c_out x k x k plus a
    transition c_in_stage -> c_out at the stage boundary) approximation,
    the param count per stage is dominated by

        params_k ~= blocks_per_stage * 2 * c_k**2 * k**2

    Inverting: c_k = sqrt(params_k / (2 * blocks_per_stage * k**2)).
    The first stage's input width c_in is small (e.g. 3 channels in)
    so its true cost is closer to c_in * c_0 * k**2 + (blocks-1) * c_0**2
    -- we use the c_k**2 quadratic only because it dominates for k >= 1.
    Resulting widths are rounded to the nearest multiple of 8 (>=8).
    """
    alloc = phi_budget_allocations(B_total, n_stages)
    widths: list[int] = []
    denom_factor = 2.0 * float(blocks_per_stage) * float(kernel * kernel)
    for params_k in alloc:
        c_sq = max(64.0, params_k / denom_factor)
        c = int(round(c_sq ** 0.5))
        widths.append(_round8(c))
    return widths


class _ConvBlock(nn.Module):
    """Tiny double-conv block for PhiBudgetNet (BasicBlock-shaped)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride,
                               padding=1, bias=False)
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


class PhiBudgetNet(nn.Module):
    """CIFAR-scale conv backbone whose stage widths sum to ~B_total params (H09).

    Two ``budget_mode`` options:

    - ``"phi"`` -- per-stage params in ratio 1 : phi : phi**2 : ...
    - ``"uniform"`` -- per-stage params equal (B_total / n_stages each)

    Both modes use the same overall param budget B_total so the comparison
    is iso-parameter. The width derivation uses
    :func:`phi_budget_widths` (and an equivalent uniform allocator).
    """

    def __init__(self, num_classes: int = 10, B_total: int = 270_000,
                 n_stages: int = 3, blocks_per_stage: int = 2,
                 budget_mode: str = "phi") -> None:
        super().__init__()
        assert budget_mode in {"phi", "uniform"}, budget_mode
        self.budget_mode = budget_mode
        self.B_total = B_total
        self.n_stages = n_stages
        if budget_mode == "phi":
            widths = phi_budget_widths(B_total, n_stages,
                                       blocks_per_stage=blocks_per_stage)
        else:
            per = B_total // n_stages
            widths = []
            denom_factor = 2.0 * float(blocks_per_stage) * 9.0
            for _ in range(n_stages):
                c_sq = max(64.0, per / denom_factor)
                widths.append(_round8(int(round(c_sq ** 0.5))))
        self.widths = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        c_prev = widths[0]
        for i, c_out in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks: list[nn.Module] = [_ConvBlock(c_prev, c_out, stride=stride)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(_ConvBlock(c_out, c_out, stride=1))
            stages.append(nn.Sequential(*blocks))
            c_prev = c_out
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# H17.pure -- Golden Skip Connections (learnable scalar init=1/phi)
# ---------------------------------------------------------------------------
class GoldenSkipBlock(nn.Module):
    """Residual block with a learnable skip-path scalar initialised at 1/phi.

    This is the *pure* skip-scale variant of H17 -- it does NOT modify
    the conv path, and it does NOT involve golden-angle channel gating
    (which is the existing ``NaturePriorFlags.golden_modulate`` confound).

    Forward:
        y = conv2(relu(conv1(x))) + alpha * skip(x)

    where ``alpha`` is a single learnable scalar initialised to 1/phi
    (~ 0.618). The branch path retains unit weight 1.0 (in contrast to
    Hayou et al's "Stable ResNet" which scales BOTH paths). Setting
    ``trainable=False`` freezes alpha as a non-trainable buffer at
    initialisation value -- useful as a control.

    Parameters
    ----------
    c_in, c_out : int
    stride : int
    init : float
        Initial value of alpha. Defaults to 1/phi (~ 0.618). Pass 1.0 to
        recover the vanilla ResNet skip and ``phi`` to test the
        amplified variant.
    trainable : bool
        If True (default), alpha is learnable. If False, alpha is a
        frozen buffer.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1,
                 init: float | None = None, trainable: bool = True) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        if init is None:
            init = 1.0 / PHI
        self.init = float(init)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride,
                               padding=1, bias=False)
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
        if trainable:
            self.alpha = nn.Parameter(torch.tensor(self.init, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(self.init,
                                                      dtype=torch.float32))
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.alpha * self.skip(x), inplace=True)


class GoldenSkipResNet(nn.Module):
    """ResNet-20-shaped backbone where every BasicBlock is a GoldenSkipBlock.

    Matches the (3, 3, 3) block layout of ResNet-20 from
    :mod:`nature_inspired_networks.models` (16, 32, 64 widths) so the
    H17.pure hypothesis isolates exactly the skip-scaling change.
    """

    def __init__(self, num_classes: int = 10,
                 widths: Sequence[int] = (16, 32, 64),
                 blocks_per_stage: int = 3,
                 init: float | None = None,
                 trainable: bool = True) -> None:
        super().__init__()
        self.init = (1.0 / PHI) if init is None else float(init)
        self.trainable = trainable
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(widths[0], widths[0],
                                       blocks_per_stage, stride=1,
                                       init=self.init, trainable=trainable)
        self.stage2 = self._make_stage(widths[0], widths[1],
                                       blocks_per_stage, stride=2,
                                       init=self.init, trainable=trainable)
        self.stage3 = self._make_stage(widths[1], widths[2],
                                       blocks_per_stage, stride=2,
                                       init=self.init, trainable=trainable)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

    @staticmethod
    def _make_stage(c_in: int, c_out: int, n: int, stride: int,
                    init: float, trainable: bool) -> nn.Sequential:
        layers: list[nn.Module] = [
            GoldenSkipBlock(c_in, c_out, stride=stride,
                            init=init, trainable=trainable)
        ]
        for _ in range(n - 1):
            layers.append(GoldenSkipBlock(c_out, c_out, stride=1,
                                          init=init, trainable=trainable))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def alphas(self) -> list[float]:
        """Return the current alpha values across every GoldenSkipBlock."""
        out: list[float] = []
        for m in self.modules():
            if isinstance(m, GoldenSkipBlock):
                out.append(float(m.alpha.detach().item()))
        return out


# ---------------------------------------------------------------------------
# Bottleneck stack -- thin wrapper around GoldenBottleneck for H06 sweep row
# ---------------------------------------------------------------------------
class GoldenBottleneckNet(nn.Module):
    """CIFAR-scale backbone where every block is a GoldenBottleneck (H06).

    Mirrors the ResNet-20 stage layout (3 stages x 3 blocks @ 16-32-64)
    so H06's claim of "matches or beats at >=5% fewer params" can be
    measured directly against the ResNet-20 baseline already in the
    sweep.
    """

    def __init__(self, num_classes: int = 10,
                 widths: Sequence[int] = (16, 32, 64),
                 blocks_per_stage: int = 3,
                 inverted: bool = False) -> None:
        super().__init__()
        self.inverted = inverted
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        c_in = widths[0]
        for i, c_out in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks: list[nn.Module] = [
                GoldenBottleneck(c_in, c_out, stride=stride,
                                 inverted=inverted, residual=True)
            ]
            for _ in range(blocks_per_stage - 1):
                blocks.append(GoldenBottleneck(c_out, c_out, stride=1,
                                               inverted=inverted, residual=True))
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Dispatcher -- mirrors models.build_model so sweep rows can route to new
# variants WITHOUT modifying the runner's central build_model. Callers
# may import build_phi_model directly; sweep rows are documentation-only
# until the runner is extended in a follow-up.
# ---------------------------------------------------------------------------
def build_phi_model(name: str, num_classes: int = 10, **kw) -> nn.Module:
    n = name.lower()
    if n == "golden_bottleneck":
        return GoldenBottleneckNet(num_classes=num_classes,
                                   inverted=bool(kw.get("inverted", False)))
    if n == "phi_budget":
        return PhiBudgetNet(num_classes=num_classes,
                            B_total=int(kw.get("B_total", 270_000)),
                            n_stages=int(kw.get("n_stages", 3)),
                            blocks_per_stage=int(kw.get("blocks_per_stage", 2)),
                            budget_mode=str(kw.get("budget_mode", "phi")))
    if n == "golden_skip":
        return GoldenSkipResNet(
            num_classes=num_classes,
            init=kw.get("init"),
            trainable=bool(kw.get("trainable", True)),
        )
    raise ValueError(f"unknown phi-model '{name}'")
