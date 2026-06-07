"""Modern-recipe Mixup (Zhang et al., 2018) — per-batch λ from Beta(α, α).

Used as part of the convergence-regime 11-trick recipe addressed by
PhD-critique items B and 2. The implementation follows the canonical
``mixup`` / ``timm`` convention:

  λ ~ Beta(α, α),   x_mix = λ·x + (1-λ)·x[perm]
  loss = λ·CE(logits, y) + (1-λ)·CE(logits, y[perm])

The 50/50 per-batch alternation with CutMix lives in ``train.py``; this
module is concerned only with the Mixup forward + loss compounding.

Skipping label smoothing when Mixup is on is a Bello-2021 recommendation
(label smoothing already softens the target; Mixup additionally
randomises the supervision) — the trainer enforces that policy.

References (Citation Rigor)::

    Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and
    Lopez-Paz, David 2018 ICLR 'mixup: Beyond Empirical Risk
    Minimization' (arXiv:1710.09412) -- defines the Beta(α, α)
    per-batch Mixup formulation used here.

    Bello, Irwan and Fedus, William and Du, Xianzhi and Cubuk, Ekin D.
    and Srinivas, Aravind and Lin, Tsung-Yi and Shlens, Jonathon and
    Zoph, Barret 2021 NeurIPS 'Revisiting ResNets: Improved Training
    and Scaling Strategies' (arXiv:2103.07579) -- the 11-trick modern
    recipe whose Mixup spec (α=0.2, mixup⊕cutmix per-batch alternation)
    is reproduced here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["mixup_batch", "mixup_loss"]


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to ``(x, y)`` and return ``(x_mix, y_a, y_b, lam)``.

    Parameters
    ----------
    x : Tensor (B, C, H, W)
        Image batch.
    y : Tensor (B,)
        Integer class labels.
    alpha : float
        Beta-distribution shape parameter. ``alpha <= 0`` disables
        Mixup and returns the input unchanged with ``lam=1.0``.

    Returns
    -------
    x_mix : Tensor (B, C, H, W)
        Mixed images.
    y_a : Tensor (B,)
        Original labels (weight ``lam``).
    y_b : Tensor (B,)
        Permuted labels (weight ``1-lam``).
    lam : float
        The Beta-sampled mixing coefficient, ∈ (0, 1).
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    # Synthesis-100 D1 (2026-06-06): IMPORTANT — this is NOT symmetric in
    # expectation. The fold ``lam = 1 - lam if lam < 0.5`` reflects every
    # below-0.5 sample to its above-0.5 mirror, producing a *folded*
    # Beta(alpha, alpha) on [0.5, 1.0]. Consequences:
    #   * The marginal distribution of lam is no longer Beta(alpha, alpha)
    #     -- the variance is approximately halved at alpha->0 and the
    #     mean shifts from 0.5 to ((mean of folded distribution)).
    #   * The label ``y_a`` always carries the dominant mix weight, which
    #     is the timm convention; ``y_b`` always carries the minority.
    #   * At alpha -> 0 the folded mean converges to (0.5 + 1.0)/2 = 0.75.
    # Callers (e.g. the trainer's ``train_top1`` computation) MUST treat
    # ``y_a`` as the dominant label. Removing the fold would restore the
    # raw Beta(alpha, alpha) distribution at the cost of breaking the
    # ``y_a == dominant`` invariant relied upon by ``train_top1`` and the
    # Synthesis-100 D7 generalization-gap fix.
    if lam < 0.5:
        lam = 1.0 - lam
    assert lam >= 0.5, "mixup.py: post-flip lam must be >= 0.5"
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], lam


def mixup_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute the standard ``λ·CE(y_a) + (1-λ)·CE(y_b)`` Mixup loss.

    Label smoothing is forwarded into both CE calls. Per Bello 2021 the
    trainer normally sets ``label_smoothing=0`` when Mixup is on; this
    helper still accepts it so the loss is well-defined under any
    combination.
    """
    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1.0 - lam) * loss_b
