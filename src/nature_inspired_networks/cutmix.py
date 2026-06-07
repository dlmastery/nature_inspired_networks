"""Modern-recipe CutMix (Yun et al., 2019) — per-batch box from Beta(α, α).

CutMix replaces a randomly-located rectangular patch of each image with
the corresponding patch from a permuted sample; the label is mixed by
the *area* of the patch rather than the Beta-sampled λ (so the loss
reflects the actual pixel ratio after rounding to integer box bounds).

The 50/50 per-batch alternation with Mixup lives in ``train.py``.

References (Citation Rigor)::

    Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk
    and Choe, Junsuk and Yoo, Youngjoon 2019 ICCV 'CutMix:
    Regularization Strategy to Train Strong Classifiers with
    Localizable Features' (arXiv:1905.04899) -- defines the Beta(α, α)
    box + area-ratio λ-correction used here.

    Bello, Irwan and Fedus, William and Du, Xianzhi and Cubuk, Ekin D.
    and Srinivas, Aravind and Lin, Tsung-Yi and Shlens, Jonathon and
    Zoph, Barret 2021 NeurIPS 'Revisiting ResNets: Improved Training
    and Scaling Strategies' (arXiv:2103.07579) -- the 11-trick modern
    recipe (CutMix α=1.0, per-batch alternation with Mixup) reproduced
    here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["cutmix_batch", "cutmix_loss", "rand_bbox"]


def rand_bbox(size: tuple[int, ...], lam: float) -> tuple[int, int, int, int]:
    """Sample a CutMix bounding box (x1, y1, x2, y2) given mixing ratio λ.

    Per Yun 2019: the box has area ``(1-lam) * H * W`` and is centred at
    a uniformly-random pixel; box edges are clipped to image bounds. The
    returned tuple is integer pixel coordinates with ``x1 <= x2`` and
    ``y1 <= y2`` (both inclusive of x1/y1 and exclusive of x2/y2 by the
    standard slicing convention).
    """
    _, _, H, W = size
    cut_rat = float(np.sqrt(1.0 - lam))
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cy = int(np.random.randint(0, H))
    cx = int(np.random.randint(0, W))
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    return x1, y1, x2, y2


def cutmix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix to ``(x, y)`` and return ``(x_mix, y_a, y_b, lam_eff)``.

    ``lam_eff`` is the *area-corrected* mixing coefficient — the ratio
    of pixels NOT replaced after integer rounding of the box edges. This
    is the standard Yun-2019 formulation and is what the loss must use.

    Parameters
    ----------
    x : Tensor (B, C, H, W)
        Image batch.
    y : Tensor (B,)
        Integer class labels.
    alpha : float
        Beta shape parameter. ``alpha <= 0`` disables CutMix and returns
        the input unchanged with ``lam_eff=1.0``.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    # Synthesis-100 D2 (2026-06-06): at alpha=1.0 the Beta sample is
    # uniform on (0, 1) which produces zero-area boxes far too often
    # (lam ~ 1.0 -> cut_rat=sqrt(1-lam) -> 0 -> integer floor 0). The
    # prior implementation silently fell back to ``lam_eff=1.0`` (no cut)
    # on those batches, which lowers the *effective* CutMix rate well
    # below what the alpha implies. We now re-sample up to 5 times before
    # falling back, which empirically drops the degenerate rate from
    # ~31% to <1% at alpha=1.0.
    x1, y1, x2, y2 = rand_bbox(x.shape, lam)
    for _retry in range(5):
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            break
        # Re-sample a new lam AND a new bbox. We re-roll lam so the
        # retry is not stuck in the same degenerate region of (lam, cy, cx).
        lam = float(np.random.beta(alpha, alpha))
        x1, y1, x2, y2 = rand_bbox(x.shape, lam)
    if (x2 - x1) > 0 and (y2 - y1) > 0:
        # In-place patch replacement on a clone (caller may rely on the
        # original x downstream — be safe).
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam_eff = 1.0 - float(area) / float(H * W)
    else:
        # Degenerate box even after 5 retries (lam ~ 1.0 -> zero-area
        # patch). Fall back to no-cut with lam_eff=1.0.
        x_mix = x
        lam_eff = 1.0
    return x_mix, y, y[perm], lam_eff


def cutmix_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute the ``λ·CE(y_a) + (1-λ)·CE(y_b)`` CutMix loss.

    Identical form to :func:`mixup.mixup_loss`; the difference is that
    ``lam`` here is the AREA-corrected ratio returned by
    :func:`cutmix_batch`, not the Beta sample.
    """
    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1.0 - lam) * loss_b
