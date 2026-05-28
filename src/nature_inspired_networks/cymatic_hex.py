"""H28 — Cymatic Hex Resonance.

Wrap the static :class:`nature_inspired_networks.priors.HexConv2d` with a
time-varying **per-tap** modulation of the kernel weights. Each of the
``T`` active hex taps (7 at radius 1, 19 at radius 2) gets a distinct
golden-angle-spaced phase

    phase_k = 2 * pi * PHI * k / T     for k in {0, ..., T-1}

and the effective tap weight at forward time becomes

    w_eff[..., i, j] = w_static[..., i, j] * cos(omega * t + phase_k)

for the k-th active tap at position (i, j). Taps outside the hex mask
remain zero. The ``T`` golden-angle-spaced phases avoid mode beats —
this is the "most irrational" tap-phase schedule on the honeycomb.

References
----------
Chladni 1787 (cymatic patterns); Sussillo & Abbott 2009
'Generating Coherent Patterns of Activity from Chaotic Neural Networks'
(arXiv:0903.4537); Hoogeboom et al. 2018 ICML 'HexaConv'
(arXiv:1803.02108). See
``hypotheses/g3_topologies_graphs/H28_cymatic_hex_resonance.md``.

Post-G3-audit (H28 MAJOR fix) — earlier versions applied a single
per-output-channel modulation ``cos(omega * t + PHI * c * t)``; the doc
specifies per-tap golden-angle-spaced phases. This file now matches the
doc.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI, HexConv2d, toroidal_pad


def _hex_tap_phases(mask: torch.Tensor) -> torch.Tensor:
    """Return a (k, k) tensor of per-tap golden-angle phases.

    Active taps (where ``mask != 0``) are assigned phases
    ``phase_k = 2 * pi * PHI * k / T`` in row-major scan order, where
    ``T`` is the number of active taps. Inactive taps get phase 0 — they
    are multiplied by a zero mask anyway, so their phase is a don't-care.
    """
    k = mask.shape[0]
    active = (mask != 0)
    T = int(active.sum().item())
    if T == 0:
        return torch.zeros(k, k)
    # Row-major scan order over active positions.
    idx = torch.arange(T, dtype=torch.float32)
    phases_flat = 2.0 * math.pi * PHI * idx / float(T)
    phase_grid = torch.zeros(k, k)
    flat_active = active.reshape(-1)
    fill = torch.zeros(k * k)
    fill[flat_active] = phases_flat
    phase_grid = fill.reshape(k, k)
    return phase_grid


class CymaticHexConv(nn.Module):
    """Hex-masked Conv2d with learnable per-tap oscillatory modulation.

    The block wraps a :class:`HexConv2d` (so the 7-tap honeycomb mask
    behaviour, optional toroidal padding, and optional radius-2 19-tap
    mask are inherited unchanged). At every forward, the convolution
    weight is multiplied by a per-tap gate

        gate[i, j] = cos(omega * t + phase[i, j])

    where ``phase[i, j] = 2 * pi * PHI * k / T`` for the k-th active tap
    (row-major scan), and ``T`` is the number of active taps (7 for
    radius=1, 19 for radius=2). Inactive taps stay zero through the hex
    mask. ``omega, t`` are scalar nn.Parameters; ``omega`` defaults to 0
    so that at init every tap's gate is ``cos(phase_k)`` — the static
    kernel pattern is preserved up to a fixed phi-spaced per-tap
    reweighting. Training drifts ``t`` and ``omega`` from there.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size, stride, padding : int
        Forwarded to the inner HexConv2d.
    toroidal : bool, default False
        If True, use circular padding.
    bias : bool, default False
    hex_kernel_radius : int, default 1
        ``1`` → 3x3 7-tap mask (180-sym), ``2`` → 5x5 19-tap mask
        (true 6-fold isotropic). See :class:`HexConv2d`.
    omega_init : float, default 1.0
        Initial value for the learnable global frequency parameter.
    t_init : float, default 0.0
        Initial value for the learnable time scalar. At t=0 every tap's
        gate is ``cos(phase_k)``; the per-tap gates are then constant
        until training drifts t.
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
        hex_kernel_radius: int = 1,
        omega_init: float = 1.0,
        t_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hex = HexConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            toroidal=toroidal,
            bias=bias,
            hex_kernel_radius=hex_kernel_radius,
        )
        self.out_channels = out_channels
        # Learnable scalars.
        self.t = nn.Parameter(torch.tensor(float(t_init)))
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))
        # Per-tap golden-angle phases — derived once from the hex mask.
        # Shape: (k, k). Inactive taps get phase 0 (mask zeroes them out).
        phase_grid = _hex_tap_phases(self.hex.mask)
        self.register_buffer("tap_phases", phase_grid)
        # Also expose the count of active taps for tests / introspection.
        self.n_active_taps = int((self.hex.mask != 0).sum().item())

    def _modulation(self) -> torch.Tensor:
        """Compute the per-tap modulation grid.

        Returns a tensor of shape ``(k, k)`` containing
        ``cos(omega * t + phase[i, j])`` for each active tap and
        ``cos(omega * t)`` (which is then masked out) at inactive taps.
        """
        return torch.cos(self.omega * self.t + self.tap_phases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with per-tap oscillatory modulation."""
        hex_layer = self.hex
        if hex_layer.toroidal:
            x = toroidal_pad(x, hex_layer.padding)
        else:
            x = F.pad(x, [hex_layer.padding] * 4, mode="constant", value=0.0)

        # Per-tap gate (k, k), broadcast across (O, I).
        gate = self._modulation()  # (k, k)
        # Apply hex mask first (zeros corners), then multiply by per-tap
        # gate. Broadcasting: (O, I, k, k) * (k, k) -> (O, I, k, k).
        w = hex_layer.conv.weight * hex_layer.mask * gate

        y = F.conv2d(
            x, w, hex_layer.conv.bias,
            stride=hex_layer.stride, padding=0,
        )
        return y
