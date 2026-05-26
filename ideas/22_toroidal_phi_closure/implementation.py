"""H22 - Toroidal phi-Closure - implementation module.

Re-exports the toroidal-padding primitive from
``nature_inspired_networks.priors`` and adds a thin phi-scaled wrapper
used by this hypothesis. No primitive duplication: the canonical
``toroidal_pad`` lives in ``src/nature_inspired_networks/priors.py``.

The single new primitive here is the phi-scaled wrap distance: the
effective pad is ``round(PHI * pad)`` rather than ``pad``. For a 3x3
kernel with pad=1 this rounds to 2 (i.e., the effective wrap is two
rows/columns), giving a kernel that sees the second-row neighborhood
across the wrap rather than just the immediate edge.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI, toroidal_pad  # noqa: F401


def phi_toroidal_pad(x: torch.Tensor, pad: int = 1, phi_scale: bool = True) -> torch.Tensor:
    """Circular pad with optional phi-scaled wrap distance.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(B, C, H, W)``.
    pad : int
        Base half-extent of the kernel (kernel_size // 2).
    phi_scale : bool
        If True, the effective pad is ``round(PHI * pad)``; otherwise
        identical to ``toroidal_pad``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, C, H + 2*eff, W + 2*eff)``.
    """
    eff = int(round(PHI * pad)) if phi_scale else pad
    if eff <= 0:
        return x
    return F.pad(x, (eff, eff, eff, eff), mode="circular")


class ToroidalConv2d(nn.Module):
    """Conv2d with toroidal (circular) padding, optionally phi-scaled.

    Drop-in replacement for ``nn.Conv2d(..., padding=k//2)`` that wraps
    the input circularly before convolving. We handle padding manually
    so the underlying ``nn.Conv2d`` is constructed with ``padding=0``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        phi_scale: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self._k = kernel_size // 2
        self.phi_scale = phi_scale
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = phi_toroidal_pad(x, pad=self._k, phi_scale=self.phi_scale)
        return self.conv(x)


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H22",
        short="toroidal_phi_closure",
        primitives_touched=["toroidal_pad", "phi_toroidal_pad", "ToroidalConv2d"],
        flags_touched=["toroidal"],
        phi_value=float(PHI),
    )
