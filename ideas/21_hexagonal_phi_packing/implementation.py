"""H21 — Hexagonal phi-Packing — implementation module.

The shared infrastructure already exposes:

- ``hex_kernel_mask(k)`` -- the 7-cell honeycomb mask
- ``HexConv2d`` -- a hex-masked Conv2d with optional toroidal padding

What it does NOT yet expose is the **phi-radial weighting** that scales
the six peripheral taps by 1/phi relative to the centre tap. This is
the second sacred prior that the H21 hypothesis composes on top of
HexConv2d, and it is the missing ingredient in T1.3's negative
single-seed result (which used UNIFORM weights).

This module:

1. Builds the phi-radial scale buffer ``hex_phi_radial_mask(k)`` -- a
   k x k tensor of 1.0 at the centre, 1/phi at the six hex neighbours,
   0.0 at the corners. The product of this buffer with the standard
   hex mask gives the effective per-tap scale.

2. Provides ``PhiRadialHexConv2d`` -- a thin subclass of
   ``nn.Module`` that wraps an ``nn.Conv2d`` and applies the
   phi-radial-masked weight at every forward. The mask is registered
   as a buffer (non-trainable). The variance-preserving gain
   ``gain = 1.0 / sqrt(1 + 6 * (1/phi)**2)`` is documented but NOT
   applied to the init (we leave kaiming_normal_ untouched to match
   the H21 hypothesis doc).

DO NOT redefine ``hex_kernel_mask`` here; import it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI, hex_kernel_mask, toroidal_pad


def hex_phi_radial_mask(k: int = 3) -> torch.Tensor:
    """Return the combined hex-mask x phi-radial scale buffer.

    For k=3:
        [ 0       1/phi   0      ]
        [ 1/phi   1.0     1/phi  ]
        [ 0       1/phi   1/phi  ]

    Wait, that's a 6-neighbour + centre = 7-tap pattern but the
    `hex_kernel_mask` zeros the (0, 2) and (2, 0) corners only --
    NOT a symmetric hexagon. The resulting active cells are
    {(0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2)} -- the centre
    is at (1,1), and the other six taps are the "neighbours". We
    apply 1.0 to (1,1) and 1/phi everywhere else the hex mask is
    non-zero.
    """
    base = hex_kernel_mask(k)  # zeros at corners, ones elsewhere
    out = base.clone()
    centre = k // 2
    # Set the non-centre, non-zero taps to 1/phi
    for i in range(k):
        for j in range(k):
            if base[i, j] == 0.0:
                continue
            if i == centre and j == centre:
                out[i, j] = 1.0
            else:
                out[i, j] = 1.0 / PHI
    return out


def variance_preserving_gain(k: int = 3) -> float:
    """Closed-form He-init gain compensation for the phi-radial scale.

    A standard He-init for a 3x3 conv assumes 9 unit-variance taps.
    With phi-radial weighting active, the effective fan-in is
    1.0^2 + 6 * (1/phi)^2 = 1 + 6 * 0.382 ~= 3.29 -- so the variance
    is scaled by 3.29/9 ~= 0.366 of the dense case. To preserve
    variance, multiply the He-init std by sqrt(9 / 3.29) ~= 1.654.
    """
    if k != 3:
        raise NotImplementedError("variance_preserving_gain only defined for k=3")
    mask = hex_phi_radial_mask(k)
    sum_sq = (mask * mask).sum().item()
    dense_fan_in = k * k  # 9 for k=3
    return math.sqrt(dense_fan_in / sum_sq)


class PhiRadialHexConv2d(nn.Module):
    """HexConv2d + phi-radial energy distribution.

    Equivalent to:
        Conv2d(c_in, c_out, k) with weight *= hex_phi_radial_mask(k)
    applied at every forward (mask is non-trainable buffer).

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, padding
        Standard Conv2d plumbing.
    toroidal
        If True, use circular padding (torus topology) instead of zero
        padding -- composes with H22.
    apply_gain
        If True, init weights with the variance-preserving gain so the
        phi-radial scale does not shrink early-layer activations.
        Default False (matches H21 hypothesis doc § 5.1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        toroidal: bool = False,
        apply_gain: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.toroidal = toroidal
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if apply_gain:
            with torch.no_grad():
                self.weight.mul_(variance_preserving_gain(kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer("mask", hex_phi_radial_mask(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.toroidal:
            x = toroidal_pad(x, self.padding)
            pad = 0
        else:
            pad = self.padding
        w_eff = self.weight * self.mask  # (O, I, k, k)
        return F.conv2d(x, w_eff, self.bias, stride=self.stride, padding=pad)


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H21",
        short="hexagonal_phi_packing",
        primitives_touched=["hex_kernel_mask", "HexConv2d", "PHI"],
        flags_touched=["hex", "phi_radial"],
        active_taps=7,
        phi=PHI,
    )
