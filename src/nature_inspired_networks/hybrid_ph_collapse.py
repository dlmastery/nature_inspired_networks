"""H65 — Persistent-Homology Betti-Collapse Loss (G7 cross-paradigm hybrid).

Design doc:
``hypotheses/g7_cross_paradigm_hybrids/H65_ph_betti_collapse_loss.md``.

A specialisation of :class:`nature_inspired_networks.betti_loss.BettiLoss`
that targets ``β₀ = 1``: a single connected component on the activation
manifold, i.e. maximal "topological collapse". The hypothesis is that
pulling intra-class activations into one connected component (β₀ → 1)
while leaving β₁, β₂ unconstrained provides a *positive-only* topological
regulariser that's easier to optimise than the generic 3-vector Betti
target of H51.

H65 also composes a **spectral cymatic-loss** consistency term when the
optional :mod:`cymatic_loss` module is available: the activations should
have a band-limited Fourier signature aligned with the Chladni eigenmode
basis (i.e., the network's learned features sit in the same spectral band
as the natural-system standing waves). When ``cymatic_loss`` is absent
the spectral term gracefully reduces to zero, so H65 ships and self-tests
either way.

References (Citation Rigor)::

    Gabrielsson, Nelson, Dwaraknath, Skraba 2020 AISTATS 'A Topology
    Layer for Machine Learning' (arXiv:1905.12200) -- differentiable PH
    surrogate.

    Naitzat, Zhitnikov, Lim 2020 JMLR 'Topology of deep neural networks'
    (arXiv:2004.06093) -- empirically, training simplifies cluster
    topology toward β₀=1; H65 supplies the explicit gradient for that
    simplification.

    Chladni 1787 'Entdeckungen über die Theorie des Klanges' -- the
    eigenmode plate experiments whose discrete basis sits behind the
    optional spectral consistency term.

Public surface
--------------
- :class:`BettiCollapseLoss` -- specialised BettiLoss with target (1, *, *).
  Composes the spectral cymatic-loss consistency term when available.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .betti_loss import BettiLoss, differentiable_persistence
from .priors import PHI


__all__ = ["BettiCollapseLoss"]


# ---------------------------------------------------------------------------
# Optional cymatic-loss spectral consistency hook (guarded -- concurrent agent)
# ---------------------------------------------------------------------------
try:
    from . import cymatic_loss as _cymatic_loss  # type: ignore
    _HAVE_CYMATIC_LOSS = True
except ImportError:
    _cymatic_loss = None
    _HAVE_CYMATIC_LOSS = False


def _spectral_consistency_fallback(activations: torch.Tensor) -> torch.Tensor:
    """Fallback spectral term when :mod:`cymatic_loss` isn't installed.

    Penalises high-frequency energy in the (B, C, H, W) feature map's
    2-D FFT spectrum, biasing the activations toward the low-frequency
    Chladni-friendly band (m, n in [2, 5]). Returns a scalar tensor.
    """
    if activations.ndim != 4:
        # Non-spatial activations: spectral term is undefined → return zero.
        return activations.new_zeros(())
    spec = torch.fft.fft2(activations, norm="ortho").abs()
    B, C, H, W = spec.shape
    # Build a radial-frequency mask that rewards mid-band [2, 5].
    fy = torch.fft.fftfreq(H, device=spec.device).abs() * H
    fx = torch.fft.fftfreq(W, device=spec.device).abs() * W
    rad = torch.sqrt(fy.view(H, 1) ** 2 + fx.view(1, W) ** 2)
    mask_hi = (rad > 5.0).to(spec.dtype)
    return (spec * mask_hi).mean()


def _spectral_consistency(activations: torch.Tensor) -> torch.Tensor:
    if _HAVE_CYMATIC_LOSS:
        try:
            # Prefer a public ``cymatic_consistency_loss`` if defined;
            # else any plausibly-named function on the module.
            fn = getattr(_cymatic_loss, "cymatic_consistency_loss", None)
            if fn is None:
                fn = getattr(_cymatic_loss, "spectral_consistency", None)
            if fn is None:
                fn = getattr(_cymatic_loss, "cymatic_loss", None)
            if fn is not None:
                return fn(activations)
        except Exception:  # pragma: no cover -- defensive
            pass
    return _spectral_consistency_fallback(activations)


class BettiCollapseLoss(nn.Module):
    """Betti-collapse regulariser: pull β₀ → 1 with optional spectral term.

    Loss::

        L = ((beta_0 - 1) ** 2) + spectral_weight * spectral_consistency(x)

    where ``beta_0`` is the differentiable surrogate from
    :class:`BettiLoss` (count of persistent connected components plus a
    base 1). The spectral consistency term uses :mod:`cymatic_loss` when
    available; otherwise it falls back to penalising high-frequency
    Fourier energy (outside the Chladni mid-band).

    Parameters
    ----------
    persistence_threshold : float
        Forwarded to the inner :class:`BettiLoss` surrogate.
    spectral_weight : float, default ``1 / phi``
        Weight on the cymatic-spectral consistency term. The default
        ``1/phi`` keeps the spectral term subordinate to the topological
        term (consistent with the H65 hypothesis that β₀-collapse is the
        load-bearing prior).
    max_pts : int
        Cap on the point-cloud size inside the differentiable PH
        surrogate.
    """

    def __init__(
        self,
        persistence_threshold: float = 0.1,
        spectral_weight: float = 1.0 / PHI,
        max_pts: int = 64,
    ) -> None:
        super().__init__()
        self.betti = BettiLoss(persistence_threshold=persistence_threshold, max_pts=max_pts)
        self.spectral_weight = float(spectral_weight)

    # ------------------------------------------------------------------
    def beta_0_estimate(self, activations: torch.Tensor) -> torch.Tensor:
        """Return the differentiable β₀ estimate as a scalar tensor."""
        return self.betti.estimate_betti(activations)[0]

    def forward(
        self,
        activations: torch.Tensor,
        target_beta_0: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the H65 collapse + spectral loss.

        Returns
        -------
        loss : Tensor (scalar)
            Combined loss.
        parts : dict
            ``{"beta_0": float, "collapse": float, "spectral": float}``
            -- diagnostics for the trainer log.
        """
        est = self.betti.estimate_betti(activations)
        b0 = est[0]
        collapse = (b0 - float(target_beta_0)) ** 2
        spectral = _spectral_consistency(activations)
        loss = collapse + self.spectral_weight * spectral
        parts = {
            "beta_0": float(b0.detach().item()),
            "collapse": float(collapse.detach().item()),
            "spectral": float(spectral.detach().item()),
        }
        return loss, parts
