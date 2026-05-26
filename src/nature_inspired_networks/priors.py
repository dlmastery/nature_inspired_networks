"""nature-inspired priors as small, composable PyTorch utilities.

Each function/class implements ONE nature-inspired prior so the NaturePriorBlock
can ablate them independently. References to the literature are inline.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PHI = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio ≈ 1.618


# ---------------------------------------------------------------------------
# φ / Fibonacci channel scaling (Fibonacci-Net 2025; EfficientNet compound)
# ---------------------------------------------------------------------------
def fibonacci_channels(c0: int, n_stages: int, mode: str = "fib") -> list[int]:
    """Return a channel schedule of length n_stages.

    mode='fib'   → c0, c0*F2/F1, c0*F3/F1, ...  (Fibonacci ratios, divisible-by-8)
    mode='phi'   → c0 * φ^k         (geometric golden growth)
    mode='linear'→ c0 * (k+1)       (control: arithmetic growth)
    """
    out: list[int] = []
    if mode == "fib":
        # Generate Fibonacci(>=1): 1, 2, 3, 5, 8, 13, 21, 34, 55
        fib = [1, 2]
        while len(fib) < n_stages + 2:
            fib.append(fib[-1] + fib[-2])
        base = fib[1]
        for k in range(n_stages):
            c = int(round(c0 * fib[k + 1] / base))
            out.append(_round8(c))
    elif mode == "phi":
        for k in range(n_stages):
            out.append(_round8(int(round(c0 * (PHI ** k)))))
    elif mode == "linear":
        for k in range(n_stages):
            out.append(_round8(c0 * (k + 1)))
    else:
        raise ValueError(f"unknown scaling mode '{mode}'")
    return out


def _round8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)


# ---------------------------------------------------------------------------
# Toroidal padding (Pittorino 2022, TopoCN)
# ---------------------------------------------------------------------------
def toroidal_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular padding on the last two spatial dims → torus topology."""
    if pad <= 0:
        return x
    return F.pad(x, (pad, pad, pad, pad), mode="circular")


# ---------------------------------------------------------------------------
# Hexagonal kernel mask (HexaConv 2018, HexagDLy 2019)
# ---------------------------------------------------------------------------
def hex_kernel_mask(k: int = 3) -> torch.Tensor:
    """3x3 hex-shaped mask on a square kernel: keep the 7 'honeycomb'
    positions, zero out the two corners. For k=3 the pattern is

        [ 1 1 0 ]
        [ 1 1 1 ]
        [ 0 1 1 ]

    which gives 6 neighbors + center (axial-coordinate hex on offset rows).
    For k=5 we extend to a hexagon of radius 2 (19 positions).
    """
    m = torch.ones(k, k)
    if k == 3:
        m[0, 2] = 0.0
        m[2, 0] = 0.0
    elif k == 5:
        # radius-2 hex mask (19 cells)
        for i in range(k):
            for j in range(k):
                # axial: q = j - 2, r = i - 2; require |q|+|r|+|q+r| <= 4
                q, r = j - 2, i - 2
                if abs(q) + abs(r) + abs(q + r) > 4:
                    m[i, j] = 0.0
    else:
        raise ValueError("hex_kernel_mask supports k in {3, 5}")
    return m


# ---------------------------------------------------------------------------
# Cymatic / Chladni wavelet init (cymatic-resonance hypothesis)
# ---------------------------------------------------------------------------
def chladni_modes(k: int, n_modes: int = 4) -> torch.Tensor:
    """Return `n_modes` orthogonal Chladni-plate basis patterns of size k×k.

    Each pattern = sin(m·π·x/(k+1)) · sin(n·π·y/(k+1)) for (m,n) drawn from
    low frequencies — these are the eigenmodes of the 2-D wave equation
    on a square plate (Chladni patterns).
    """
    xs = torch.linspace(0, math.pi, k + 2)[1:-1]
    ys = torch.linspace(0, math.pi, k + 2)[1:-1]
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    modes: list[torch.Tensor] = []
    pairs = []
    # use lowest-frequency mode pairs first (m+n small)
    for s in range(2, 2 + n_modes * 2):
        for m in range(1, s):
            n = s - m
            pairs.append((m, n))
            if len(pairs) >= n_modes:
                break
        if len(pairs) >= n_modes:
            break
    for m, n in pairs[:n_modes]:
        mode = torch.sin(m * X) * torch.sin(n * Y)
        mode = mode / (mode.abs().max() + 1e-8)
        modes.append(mode)
    return torch.stack(modes, dim=0)  # (n_modes, k, k)


def cymatic_init_(conv: nn.Conv2d, n_modes: int | None = None) -> None:
    """Initialize Conv2d weights from Chladni eigenmodes, broadcast across
    in/out channels with random phases. Fan-in scaling preserves variance.
    """
    out_c, in_c, kh, kw = conv.weight.shape
    assert kh == kw, "cymatic init expects square kernel"
    n = n_modes or min(8, max(2, kh * kh // 2))
    basis = chladni_modes(kh, n_modes=n)  # (n, k, k)
    g = torch.Generator().manual_seed(0xC1A171C)
    fan_in = in_c * kh * kw
    scale = math.sqrt(2.0 / fan_in)  # He-style
    with torch.no_grad():
        for o in range(out_c):
            for i in range(in_c):
                # random convex combination over modes + sign
                coef = torch.randn(n, generator=g)
                coef = coef / (coef.norm() + 1e-8)
                w = (basis * coef.view(n, 1, 1)).sum(0)
                conv.weight[o, i].copy_(w * scale)
        if conv.bias is not None:
            conv.bias.zero_()


# ---------------------------------------------------------------------------
# C4 / D4 group-equivariant conv (proxy for Platonic equivariance)
# ---------------------------------------------------------------------------
class GroupConv2d(nn.Module):
    """Cohen-Welling style group convolution over C4 (4 rotations) or D4
    (4 rotations + flips). Weight-shared across group, output channels =
    `out_channels`. Aggregation across the group orbit is max-pool, which
    is invariant; this is a *light* Platonic prior cheap enough for ablations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        group: str = "c4",
        bias: bool = False,
        mask: torch.Tensor | None = None,
        reduce: str = "max",
    ) -> None:
        super().__init__()
        assert group in {"c4", "d4"}
        assert reduce in {"max", "mean"}
        self.group = group
        self.n_orbits = 4 if group == "c4" else 8
        self.stride = stride
        self.padding = padding
        self.reduce = reduce
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer(
            "mask",
            mask if mask is not None else torch.ones_like(self.weight[0, 0]),
        )

    def _orbit(self) -> torch.Tensor:
        w = self.weight * self.mask  # (O, I, k, k)
        outs = [w]
        for r in (1, 2, 3):
            outs.append(torch.rot90(w, k=r, dims=(2, 3)))
        if self.group == "d4":
            wf = torch.flip(w, dims=(3,))
            outs.append(wf)
            for r in (1, 2, 3):
                outs.append(torch.rot90(wf, k=r, dims=(2, 3)))
        return torch.stack(outs, dim=0)  # (G, O, I, k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orbit = self._orbit()  # (G, O, I, k, k)
        G, O, I, k, _ = orbit.shape
        w = orbit.reshape(G * O, I, k, k)
        y = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        # Group orbit reduction. Max-pool (legacy) is the dominant negative in
        # the prior CIFAR sweep — it throws away 75% of the signal at C4. The
        # H58 fix is mean-pool, which preserves the orbit's full signal while
        # still giving rotation invariance after averaging.
        y = y.view(x.shape[0], G, O, y.shape[-2], y.shape[-1])
        if self.reduce == "mean":
            y = y.mean(dim=1)
        else:
            y = y.amax(dim=1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


# ---------------------------------------------------------------------------
# Toroidal-padded hex conv (drop-in Conv2d replacement)
# ---------------------------------------------------------------------------
class HexConv2d(nn.Module):
    """Hex-masked Conv2d with optional toroidal padding. The mask zeroes
    out the two corner taps so the receptive field is the 7-cell honeycomb
    (HexaConv 2018) emulated on a square lattice.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        toroidal: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.toroidal = toroidal
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias,
        )
        self.register_buffer("mask", hex_kernel_mask(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.toroidal:
            x = toroidal_pad(x, self.padding)
        else:
            x = F.pad(x, [self.padding] * 4, mode="constant", value=0.0)
        # Apply mask each forward — keeps gradients flowing to all weights
        w = self.conv.weight * self.mask
        y = F.conv2d(x, w, self.conv.bias, stride=self.stride, padding=0)
        return y


# ---------------------------------------------------------------------------
# Golden-angle positional / rotary modulation (output stage)
# ---------------------------------------------------------------------------
def golden_angle_phases(n: int) -> torch.Tensor:
    """Return n golden-angle phases in [0, 2π). Used for rotary-style
    modulation of channel groups (Metatron projection proxy).
    """
    golden_angle = 2 * math.pi * (1 - 1 / PHI)  # ≈ 2.3999632...
    return torch.arange(n, dtype=torch.float32) * golden_angle % (2 * math.pi)
