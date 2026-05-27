"""H28 — Cymatic Hex Resonance.

Wrap the static :class:`nature_inspired_networks.priors.HexConv2d` with a
time-varying per-channel modulation of the tap weights:

    w_eff[c, :, i, j] = w_static[c, :, i, j] * cos(omega * t + PHI * c * t)

where ``t`` is a learnable scalar time variable (init=0), ``omega`` is a
learnable global frequency (init=1), and ``c`` is the output channel
index. The per-channel phase increment uses the golden ratio
``PHI = (1+sqrt(5))/2`` so the channel oscillators are spaced at the
most-irrational angle — avoiding mode beats and giving golden-angle-
modulated harmonics.

This is an **oscillatory perturbation** around the static hex kernel:
at ``t = 0`` every channel's modulation factor is ``cos(0) = 1``, so
the block reduces *exactly* to the underlying static
:class:`HexConv2d`. Training drifts ``t`` (and ``omega``) away from
the static point as the loss prefers.

References
----------
Chladni 1787 (cymatic patterns); Sussillo & Abbott 2009
'Generating Coherent Patterns of Activity from Chaotic Neural Networks'
(arXiv:0903.4537); Hoogeboom et al. 2018 ICML 'HexaConv'
(arXiv:1803.02108). See
``hypotheses/g3_topologies_graphs/H28_cymatic_hex_resonance.md``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI, HexConv2d, toroidal_pad


class CymaticHexConv(nn.Module):
    """Hex-masked Conv2d with a learnable oscillatory tap-weight modulation.

    The block wraps a :class:`HexConv2d` (so the 7-tap honeycomb mask
    behaviour, optional toroidal padding, and optional radius-2 19-tap
    mask are inherited unchanged). At every forward, the convolution
    weight is multiplied by

        m_c = cos(omega * t + PHI * c * t)            # per output channel

    where ``omega, t`` are scalar nn.Parameters and ``c`` is the output
    channel index. Because ``t`` defaults to 0, the initial forward is
    bit-identical to the underlying static HexConv2d (with ``omega``
    initialised to 1 so that as ``t`` grows from 0 the modulation has
    a finite gradient w.r.t. ``omega``).

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
        Init at 1.0 (not 0.0) so the gradient through ``omega`` is
        non-zero at the t=0 starting point.
    t_init : float, default 0.0
        Initial value for the learnable time scalar. At t=0 every
        channel's modulation is cos(0) = 1 and the block reduces to
        the static HexConv2d.
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
        # Learnable scalars — t init=0 means the block starts as static hex.
        self.t = nn.Parameter(torch.tensor(float(t_init)))
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))
        # Per-output-channel index buffer for the phi-spaced phases.
        chan_idx = torch.arange(out_channels, dtype=torch.float32)
        self.register_buffer("chan_idx", chan_idx)

    def _modulation(self) -> torch.Tensor:
        """Compute the per-output-channel modulation factor.

        Returns a tensor of shape ``(out_channels,)`` containing
        ``cos(omega * t + PHI * c * t)`` for each channel ``c``.
        """
        # Vectorised: cos(omega * t + PHI * c * t) for c = 0..out_channels-1.
        phase = self.omega * self.t + PHI * self.chan_idx * self.t
        return torch.cos(phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with oscillatory tap-weight modulation."""
        # Reproduce HexConv2d.forward but with channel modulation applied
        # to the masked weight before convolution.
        hex_layer = self.hex
        if hex_layer.toroidal:
            x = toroidal_pad(x, hex_layer.padding)
        else:
            x = F.pad(x, [hex_layer.padding] * 4, mode="constant", value=0.0)

        w = hex_layer.conv.weight * hex_layer.mask  # (O, I, k, k)
        mod = self._modulation()  # (O,)
        # Broadcast per output channel: (O,) → (O, 1, 1, 1).
        w = w * mod.view(-1, 1, 1, 1)

        y = F.conv2d(
            x, w, hex_layer.conv.bias,
            stride=hex_layer.stride, padding=0,
        )
        return y
