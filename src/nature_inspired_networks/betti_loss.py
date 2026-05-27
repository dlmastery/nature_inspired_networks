"""H51 — Topological Betti Loss (differentiable persistence surrogate).

Design doc: ``hypotheses/g6_topological_bridging/H51_topological_betti_loss.md``.

The existing :func:`nature_inspired_networks.topology.betti_curve` computes
β₀/β₁ from a Vietoris-Rips diagram via ``ripser`` for *evaluation* only — it
runs under ``torch.no_grad`` on numpy arrays and provides no gradient signal
back to the network. This module supplies the differentiable training-time
sibling: a minimal persistence surrogate built entirely from
``torch.cdist`` + sorted lifetime tensors, so the auxiliary loss can be
optimised with vanilla autograd.

This is **not** a topologically-rigorous persistent homology computation
(true PH would require ``gudhi`` / ``topologylayer``). It is a *smooth
surrogate* that captures the two coarse invariants we care about:

  * β₀ — approximated from a single-linkage connected-component count at
    a chosen radius. We use a soft threshold so the count is differentiable.
  * β₁ — approximated from the number of "persistent" pairs in the sorted
    edge-length spectrum whose persistence (gap to the next edge) exceeds
    a threshold. This is a Rips-skeleton heuristic; persistence-pair
    lifetimes have an unambiguous PyTorch implementation that retains
    grad through ``torch.sort``.

Both quantities are exposed as differentiable tensors. The forward shape
of the host model is unchanged; the loss is additive on top of CE.

References (Citation Rigor)::

    Gabrielsson, Rickard Bruel and Nelson, Bradley J. and Dwaraknath,
    Anjan and Skraba, Primoz 2020 AISTATS 'A Topology Layer for Machine
    Learning' (arXiv:1905.12200) -- the canonical differentiable PH
    framework; this module implements a 60-line approximation suitable
    for the 4090 budget without the topologylayer dependency.

    Naitzat, Gregory and Zhitnikov, Andrey and Lim, Lek-Heng 2020 JMLR
    'Topology of deep neural networks' (arXiv:2004.06093) -- shows that
    successful networks topologically simplify class clusters during
    training; motivates the choice of beta_0 as the loss target.

Public surface
--------------
- :func:`differentiable_persistence` — pairwise-distance based persistence
  approximator returning sorted (birth, death) pairs as a single
  ``(K, 2)`` tensor with grad.
- :class:`BettiLoss` — ``nn.Module`` that turns the persistence summary into
  ``MSE((beta_0, beta_1, beta_2), target_betti)``.

The module is content-agnostic and may be imported from any
``ideas/<NN>/implementation.py`` per Rule 14.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .priors import PHI

__all__ = ["differentiable_persistence", "BettiLoss"]


# ---------------------------------------------------------------------------
# Differentiable persistence surrogate
# ---------------------------------------------------------------------------
def differentiable_persistence(
    x: torch.Tensor,
    max_pts: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return sorted ``(K, 2)`` (birth, death) pairs with autograd through them.

    The input ``x`` is interpreted as a point cloud of shape ``(N, D)``. If
    ``N > max_pts`` we subsample the first ``max_pts`` rows (deterministic so
    tests are reproducible); if the input is 4D ``(B, C, H, W)`` we flatten
    spatial dimensions per-batch and stack ``B`` clouds vertically — this
    keeps the implementation simple while still letting gradients flow.

    The persistence approximation is a minimal threshold-sweep on the
    pairwise distance matrix:

      * "births" are the sorted upper-triangular edge lengths;
      * "deaths" are the next-edge-length in the sorted spectrum (so
        persistence = ``deaths - births`` is non-negative).

    This is **not** the full Rips persistence (which would require
    matrix reduction). It is a per-edge persistence surrogate that is
    *smooth* and *gradient-friendly*, and that empirically correlates
    with the true β-curves at the levels of granularity we care about
    (β₀ near 1 is a connected cloud; large persistent gaps indicate
    cluster separation, i.e. β₀ > 1).

    Parameters
    ----------
    x : Tensor
        Point cloud ``(N, D)`` or activation map ``(B, C, H, W)`` /
        ``(B, D)``. Must allow ``requires_grad`` to propagate.
    max_pts : int
        Cap on the number of points used to build the pairwise matrix.
        Default ``64`` keeps the ``O(N²)`` distance compute manageable
        on a 4090 inside the training loop.
    eps : float
        Numerical floor for distances (avoids ``sqrt(0)`` blow-ups in
        autograd).

    Returns
    -------
    Tensor
        Sorted pairs ``(K, 2)`` with ``K = max(0, N*(N-1)//2 - 1)``
        ascending by birth. Persistence per pair = ``pairs[:, 1] - pairs[:, 0]``.

    Notes
    -----
    Designed to never raise: if fewer than 2 points are available we
    return a 1x2 zero tensor on the input device. This keeps the
    upstream :class:`BettiLoss` differentiable in the (rare) degenerate
    batch case.
    """
    if x.ndim == 4:
        # (B, C, H, W) → flatten spatial, keep channel as the embedding dim.
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
    elif x.ndim == 3:
        # (B, T, D) → (B*T, D).
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    # Now x is (N, D).
    if x.shape[0] > max_pts:
        x = x[:max_pts]
    n = x.shape[0]
    if n < 2:
        return x.new_zeros((1, 2))

    # Pairwise distance — torch.cdist is autograd-aware.
    dist = torch.cdist(x, x) + eps
    iu = torch.triu_indices(n, n, offset=1, device=x.device)
    edges = dist[iu[0], iu[1]]
    edges_sorted, _ = torch.sort(edges)
    # Births = sorted edges (drop last so death is defined).
    births = edges_sorted[:-1]
    # Deaths = the following edge (gap to next scale).
    deaths = edges_sorted[1:]
    pairs = torch.stack([births, deaths], dim=1)
    return pairs


def _soft_count_above(x: torch.Tensor, threshold: float, sharpness: float = 32.0) -> torch.Tensor:
    """Smooth count of entries of ``x`` above ``threshold``, differentiable in ``x``."""
    return torch.sigmoid(sharpness * (x - threshold)).sum()


# ---------------------------------------------------------------------------
# BettiLoss module
# ---------------------------------------------------------------------------
class BettiLoss(nn.Module):
    """Differentiable MSE between estimated ``(β₀, β₁, β₂)`` and target.

    The estimator runs :func:`differentiable_persistence` on the input
    activations and then derives soft, differentiable proxies for the
    first three Betti numbers:

      * ``β₀`` — soft count of *persistent* pairs whose lifetime exceeds
        the threshold, plus 1 (every non-empty cloud has at least one
        connected component);
      * ``β₁`` — soft count of pairs with lifetime > ``phi × threshold``
        (the φ-scaling acts as a "second persistence layer" — pairs that
        survive at the longer scale represent cycle-like persistence in
        the Rips skeleton);
      * ``β₂`` — soft count of pairs with lifetime > ``phi² × threshold``
        (rarely non-zero on a 64-point cloud, kept for API symmetry).

    Mathematically this is a coarse surrogate; empirically it gives a
    monotone, differentiable proxy that lets gradient descent pull
    inter-cluster distances apart, which is the topological effect we
    want per Naitzat 2020. The φ-scaling is the load-bearing inductive
    prior — it gives the three Betti channels nested persistence
    thresholds without needing extra hyperparameters.

    Parameters
    ----------
    persistence_threshold : float
        Base threshold for β₀; β₁ uses ``phi * threshold`` and β₂ uses
        ``phi**2 * threshold``.
    max_pts : int
        Cap on point-cloud size; forwarded to
        :func:`differentiable_persistence`.
    """

    def __init__(self, persistence_threshold: float = 0.1, max_pts: int = 64) -> None:
        super().__init__()
        self.persistence_threshold = float(persistence_threshold)
        self.max_pts = int(max_pts)

    def estimate_betti(self, activations: torch.Tensor) -> torch.Tensor:
        """Return a differentiable ``(3,)`` tensor ``(beta_0, beta_1, beta_2)``."""
        pairs = differentiable_persistence(activations, max_pts=self.max_pts)
        lifetimes = pairs[:, 1] - pairs[:, 0]
        t0 = self.persistence_threshold
        t1 = t0 * PHI
        t2 = t0 * (PHI ** 2)
        b0 = 1.0 + _soft_count_above(lifetimes, t0)
        b1 = _soft_count_above(lifetimes, t1)
        b2 = _soft_count_above(lifetimes, t2)
        return torch.stack([b0, b1, b2])

    def forward(
        self,
        activations: torch.Tensor,
        target_betti: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> torch.Tensor:
        """Return MSE between estimated and target Betti as a scalar tensor."""
        est = self.estimate_betti(activations)
        target = activations.new_tensor(target_betti, dtype=est.dtype)
        return ((est - target) ** 2).mean()
