"""H75 — Harmonic Cymatic SwiGLU (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H75_harmonic_cymatic_swiglu.md``.

A SwiGLU-style feed-forward block where:

  1. The **gate** activation is :class:`activations.PhiGELU`
     (x · σ(φ · x)) — H39 harmonic φ-activation.
  2. The **up-projection** weights are cymatic-initialised through
     :func:`priors.cymatic_init_` with ``orthonormalize=True,
     band=(2, 5)`` — H35.v2 orthonormal Chladni eigenmode init.
  3. The **down-projection** is a standard ``nn.Linear``.

The SwiGLU forward is ``down(gate(up_a · x) ⊙ up_b · x)``, where the
gating tensor uses PhiGELU and the up-projection ``up_a`` carries the
cymatic-orthonormal init.

The ``cymatic_init_`` helper operates on a ``nn.Conv2d`` weight tensor;
for a ``Linear(d_in, d_hidden)`` we wrap it as a transient
``Conv2d(d_in, d_hidden, 1, 1)`` to receive the init. The kernel is
1×1 so the orthonormalisation reduces to a scaled per-output-channel
sign-pattern over a single Chladni cell — which still decorrelates
output channels and respects the He-equivalent variance.

References (Citation Rigor)::

    Shazeer 2020 'GLU Variants Improve Transformer' (arXiv:2002.05202)
    — SwiGLU baseline.

    Hodgkin, Huxley 1952 J. Physiol. — biophysical dynamic-threshold
    precedent.

    Chladni 1787 — eigen-mode basis.

    Hendrycks, Gimpel 2016 'GELU' (arXiv:1606.08415) — GELU lineage
    that PhiGELU descends from.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import PhiGELU
from .priors import PHI, cymatic_init_


__all__ = ["HarmonicCymaticSwiGLU"]


def _cymatic_init_linear_(
    linear: nn.Linear,
    band: tuple[int, int] = (2, 5),
    seed: int = 0,
) -> None:
    """Apply :func:`cymatic_init_` to a Linear via a 5×5-Conv2d proxy.

    The Conv2d proxy lets us reuse the existing initialiser unchanged.
    We use a 5×5 kernel (>= band's max frequency) so the Chladni basis
    has genuine 2-D structure. After init we take the **central tap**
    (the spatial centre of each filter) and L2-normalise rows back to
    a He-equivalent norm. The central-tap reduction preserves all
    output channels (averaging risks collapsing channels whose modes
    are anti-symmetric in space) while collapsing the spatial dimension
    that Linear lacks.
    """
    out_f = linear.out_features
    in_f = linear.in_features
    k_proxy = max(5, band[1] + 1)
    if k_proxy % 2 == 0:
        k_proxy += 1
    proxy = nn.Conv2d(in_f, out_f, k_proxy, padding=k_proxy // 2, bias=False)
    cymatic_init_(proxy, orthonormalize=True, band=band, seed=seed)
    centre = k_proxy // 2
    with torch.no_grad():
        # Central-tap reduction → (out_f, in_f).
        w2d = proxy.weight[:, :, centre, centre].clone()
        fan_in = in_f
        he_std = math.sqrt(2.0 / fan_in)
        # Per-row normalise then rescale to He-std × sqrt(fan_in) so the
        # row's L2 ≈ He-norm.
        row_norm = w2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w2d = w2d / row_norm * he_std * math.sqrt(in_f)
        linear.weight.copy_(w2d)
        if linear.bias is not None:
            linear.bias.zero_()


class HarmonicCymaticSwiGLU(nn.Module):
    """SwiGLU FFN with PhiGELU gate and cymatic-orthonormal up-projection.

    Forward: ``(..., d_in)`` → ``(..., d_in)``.

    Architecture::

        gate = PhiGELU(up_a(x))     # φ-Swish gate
        value = up_b(x)
        h = gate * value            # element-wise GLU
        y = down(h)

    Parameters
    ----------
    d_in : int
        Input/output feature dim.
    d_hidden : int, optional
        SwiGLU hidden width. Defaults to ``round(d_in * PHI * 2)``
        (the Llama-style 2·φ multiplier).
    band : tuple[int, int], default (2, 5)
        Frequency band for the cymatic-orthonormal init of ``up_a``.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int | None = None,
        band: tuple[int, int] = (2, 5),
        cymatic_seed: int = 0,
    ) -> None:
        super().__init__()
        assert d_in >= 1
        d_hidden = int(round(d_in * PHI * 2)) if d_hidden is None else int(d_hidden)
        assert d_hidden >= 1
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.band = band
        # Two up-projections (SwiGLU gate + value); the gate path carries
        # the cymatic-orthonormal init.
        self.up_a = nn.Linear(d_in, d_hidden, bias=False)
        self.up_b = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_in, bias=False)
        # PhiGELU as the gate activation (H39 harmonic φ-activation).
        self.gate_act = PhiGELU()
        # Apply cymatic-orthonormal init to up_a.
        _cymatic_init_linear_(self.up_a, band=band, seed=cymatic_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2 or x.shape[-1] != self.d_in:
            raise ValueError(
                f"expected (..., {self.d_in}); got {tuple(x.shape)}"
            )
        gate = self.gate_act(self.up_a(x))
        value = self.up_b(x)
        return self.down(gate * value)
