"""H49 — Platonic Representation Alignment Loss (PRH auxiliary).

An auxiliary CKA-based loss that pulls a network's penultimate-layer
features toward a fixed "Platonic" target embedding (e.g., a
dodeca/icosa-vertex projection of a frozen CLIP encoder). The
hypothesis (Huh et al. 2024, arXiv:2405.07987) is that sufficiently
large networks implicitly converge to a universal embedding;
explicitly aligning short-circuits the convergence and accelerates
training.

Public API
----------
- :func:`cka_loss(feat_a, feat_b)` — centered kernel alignment loss
  in ``[0, 1]``. Returns ``1 - CKA(K_a, K_b)`` using linear kernels.
- :class:`PRHAlignmentLoss(nn.Module)` — wraps a fixed Platonic target
  tensor and (optionally) a learnable projection so the model feature
  dim can differ from the target dim. ``forward(feat)`` returns the
  CKA distance between the projected features and the target.

The target embedding is typically pre-cached on disk
(``data/prh_targets/cifar10_dodeca.pt``) and supplied as a constructor
argument; this module does not handle the CLIP precomputation step.

Lives in ``src/nature_inspired_networks/prh_loss.py`` per Rule 14.

References
----------
Huh, M. et al. 2024 ICML 'The Platonic Representation Hypothesis'
(arXiv:2405.07987); Kornblith, S. et al. 2019 ICML 'Similarity of
Neural Network Representations Revisited' (arXiv:1905.00414); Radford,
A. et al. 2021 ICML 'Learning Transferable Visual Models...'
(arXiv:2103.00020). See
``hypotheses/g5_optimization_init_reg_nas/H49_platonic_representation_alignment_loss.md``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CKA (linear-kernel, centered)
# ---------------------------------------------------------------------------
def _hsic_linear(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion for already-centered Gram
    matrices. Returns ``trace(K L) / (n - 1)^2`` (Kornblith 2019 eq. 3).
    """
    n = K.shape[0]
    # K, L are (n, n) centered. HSIC = trace(K L) / (n-1)^2 (biased estimator).
    return (K * L).sum() / max(1, (n - 1) ** 2)


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix: ``H K H`` with ``H = I - 1/n``."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.full(
        (n, n), 1.0 / n, device=K.device, dtype=K.dtype
    )
    return H @ K @ H


def cka_loss(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Centered Kernel Alignment loss: ``1 - CKA(feat_a, feat_b)``.

    Uses the linear kernel ``K = X X^T``. Returns a scalar in
    approximately ``[0, 1]`` (0 when features are identical up to an
    orthogonal linear transform, 1 when they are maximally
    decorrelated).

    Parameters
    ----------
    feat_a : torch.Tensor
        Shape ``(N, D_a)``. ``N`` is the batch dimension.
    feat_b : torch.Tensor
        Shape ``(N, D_b)``. Must share ``N`` with ``feat_a``; ``D_b``
        may differ from ``D_a`` (CKA is dim-agnostic via the Gram
        matrices).
    eps : float
        Numerical floor added to the denominator.

    Returns
    -------
    torch.Tensor
        Scalar CKA loss (``1 - CKA``).
    """
    assert feat_a.dim() == 2 and feat_b.dim() == 2, (
        f"cka_loss expects 2-D (N, D); got {tuple(feat_a.shape)} and {tuple(feat_b.shape)}"
    )
    assert feat_a.shape[0] == feat_b.shape[0], (
        f"cka_loss requires shared batch dim; got {feat_a.shape[0]} vs {feat_b.shape[0]}"
    )

    Ka = feat_a @ feat_a.t()  # (N, N)
    Kb = feat_b @ feat_b.t()  # (N, N)
    Kc = _center_gram(Ka)
    Lc = _center_gram(Kb)

    hsic_ab = _hsic_linear(Kc, Lc)
    hsic_aa = _hsic_linear(Kc, Kc)
    hsic_bb = _hsic_linear(Lc, Lc)

    denom = torch.sqrt(hsic_aa.clamp(min=0.0) * hsic_bb.clamp(min=0.0) + eps)
    cka = hsic_ab / denom.clamp(min=eps)
    # CKA can be slightly negative due to numerical noise on small N — clamp
    # to [0, 1] before flipping to a loss.
    cka_clamped = cka.clamp(min=0.0, max=1.0)
    return 1.0 - cka_clamped


# ---------------------------------------------------------------------------
# PRH alignment module
# ---------------------------------------------------------------------------
class PRHAlignmentLoss(nn.Module):
    """Auxiliary loss that pulls features toward a fixed Platonic target.

    The target embedding ``T`` (e.g., a 12-vertex dodecahedron or
    20-vertex icosahedron projection of a frozen CLIP image-encoder
    embedding) is stored as a non-trainable buffer. ``forward(feat)``:

    1. (optional) projects ``feat`` from ``feat_dim → target_dim``
       through a learnable linear head ``W ∈ R^{feat_dim × target_dim}``.
    2. selects the matching rows of ``T`` (either the full ``T`` if
       its batch dim equals ``feat``'s, or a supplied index tensor).
    3. returns ``cka_loss(projected_feat, T_rows)``.

    Parameters
    ----------
    target : torch.Tensor
        Pre-computed target embedding, shape ``(N_total, target_dim)``
        OR a single fixed embedding of shape ``(target_dim,)`` that
        will be broadcast to every batch row. The dodeca/icosa vertex
        coordinates of the Platonic anchor.
    feat_dim : int, default 168
        Dimensionality of the model's penultimate-layer features.
    project : bool, default True
        Whether to insert a learnable linear projection ``feat_dim →
        target_dim``. If False, ``feat_dim`` must equal ``target_dim``.
    """

    def __init__(
        self,
        target: torch.Tensor,
        feat_dim: int = 168,
        project: bool = True,
    ) -> None:
        super().__init__()
        assert target.dim() in (1, 2), (
            f"target must be 1-D (target_dim,) or 2-D (N_total, target_dim); "
            f"got {tuple(target.shape)}"
        )
        if target.dim() == 1:
            target = target.unsqueeze(0)  # (1, target_dim)
        target_dim = target.shape[1]
        self.feat_dim = feat_dim
        self.target_dim = target_dim
        # store target as a buffer so .to(device) moves it
        self.register_buffer("target", target.clone())
        if project:
            self.proj = nn.Linear(feat_dim, target_dim, bias=False)
        else:
            assert feat_dim == target_dim, (
                f"project=False requires feat_dim == target_dim; "
                f"got feat_dim={feat_dim}, target_dim={target_dim}"
            )
            self.proj = nn.Identity()

    def forward(
        self,
        feat: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CKA-distance from projected features to the target.

        Parameters
        ----------
        feat : torch.Tensor
            Shape ``(B, feat_dim)``.
        batch_idx : torch.Tensor | None
            Optional ``(B,)`` integer tensor of indices into
            ``self.target``. If None and ``self.target.shape[0] == 1``
            the single anchor is broadcast to every batch row. If None
            and ``self.target.shape[0] == B`` we use the target rows
            in order.
        """
        assert feat.dim() == 2 and feat.shape[1] == self.feat_dim, (
            f"PRHAlignmentLoss expects feat of shape (B, {self.feat_dim}); "
            f"got {tuple(feat.shape)}"
        )
        z = self.proj(feat)  # (B, target_dim)
        if batch_idx is not None:
            t = self.target[batch_idx]
        elif self.target.shape[0] == 1:
            t = self.target.expand(feat.shape[0], -1)
        elif self.target.shape[0] == feat.shape[0]:
            t = self.target
        else:
            raise ValueError(
                f"target has shape {tuple(self.target.shape)} but feat has "
                f"batch {feat.shape[0]} and no batch_idx was supplied"
            )
        return cka_loss(z, t)
