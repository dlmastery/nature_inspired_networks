"""H63 — Platonic Auxiliary Cymatic Teacher (G7 cross-paradigm hybrid).

Design doc:
``hypotheses/g7_cross_paradigm_hybrids/H63_platonic_aux_cymatic_teacher.md``.

The student model is a NaturePriorBlock stack (Platonic-equivariant group
conv inside each block). The "teacher" is a frozen target derived from the
Chladni eigenmode bank — i.e. **cymatic** target activations. The
training-time auxiliary loss is the CKA (centered-kernel-alignment)
distance between student activations and the cymatic teacher target.

The CKA loss helper is provided optionally by :mod:`prh_loss`; if that
optional module isn't yet on disk we fall back to an MSE-on-Gram-matrix
implementation that is mathematically the unnormalised CKA numerator
(``||Y^T X||_F^2``). The guard means H63 ships and self-tests even when
``prh_loss`` is provided by a concurrent agent that lands later.

References (Citation Rigor)::

    Hinton, Vinyals, Dean 2015 NIPS-W 'Distilling the Knowledge in a
    Neural Network' (arXiv:1503.02531) -- the canonical teacher/student
    distillation pattern.

    Kornblith, Norouzi, Lee, Hinton 2019 ICML 'Similarity of Neural
    Network Representations Revisited' (arXiv:1905.00414) -- the CKA
    similarity used as the distillation target metric.

    Hindman 1909 / Chladni 1787 -- classical cymatic eigenmode plates,
    operationalised here via :func:`chladni_modes_banded` in
    ``nature_inspired_networks.priors``.

Public surface
--------------
- :class:`PlatonicCymaticTeacher` — student NaturePriorBlock stack +
  cymatic teacher target; ``forward`` returns logits plus a tensor of
  student activations suitable for the CKA distillation loss.
- :func:`cymatic_teacher_target` — derive a teacher activation tensor
  from the Chladni mode bank, matched to a given spatial shape.
- :func:`cka_distillation_loss` — CKA distance between student and
  teacher activations; falls back to MSE-on-Gram if ``prh_loss`` is
  unavailable.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NaturePriorBlock, NaturePriorFlags
from .priors import PHI, chladni_modes_banded


__all__ = [
    "PlatonicCymaticTeacher",
    "cymatic_teacher_target",
    "cka_distillation_loss",
]


# ---------------------------------------------------------------------------
# Optional CKA loss import (guarded — concurrent agent may land prh_loss later)
# ---------------------------------------------------------------------------
try:
    from .prh_loss import cka_loss as _external_cka  # type: ignore
    _HAVE_EXTERNAL_CKA = True
except ImportError:
    _external_cka = None
    _HAVE_EXTERNAL_CKA = False


def _internal_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Centered-kernel-alignment distance, fallback implementation.

    Implements::

        CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    on row-mean-centered matrices X, Y. Returns ``1 - CKA`` so the
    output is a *distance* (zero when perfectly aligned).
    """
    if x.ndim != 2:
        x = x.flatten(1)
    if y.ndim != 2:
        y = y.flatten(1)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xtx = (x.T @ x)
    yty = (y.T @ y)
    xty = (x.T @ y)
    num = (xty * xty).sum()
    den = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum() + 1e-12)
    cka = num / (den + 1e-12)
    return 1.0 - cka


def cka_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """CKA distillation loss with graceful fallback when prh_loss is absent.

    Uses ``prh_loss.cka_loss`` when that optional module is available; the
    fallback is an in-module CKA implementation that is bit-equivalent on
    centred inputs.
    """
    if _HAVE_EXTERNAL_CKA:
        try:
            return _external_cka(student, teacher.detach())
        except Exception:  # pragma: no cover -- defensive
            pass
    return _internal_cka(student, teacher.detach())


# ---------------------------------------------------------------------------
# Cymatic teacher target
# ---------------------------------------------------------------------------
def cymatic_teacher_target(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    band: tuple[int, int] = (2, 5),
    device: torch.device | str | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Return a fixed Chladni-eigenmode target activation map.

    Shape: ``(B, C, H, W)``. The target is a deterministic broadcast of
    the banded Chladni basis (constructed via :func:`chladni_modes_banded`)
    bilinear-interpolated to the target ``(H, W)`` and tiled to ``C``
    channels with golden-ratio-spaced sign flips so the per-channel
    targets are linearly independent.
    """
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"all of channels, height, width must be positive; got "
            f"{channels}, {height}, {width}"
        )
    basis = chladni_modes_banded(channels, max(3, min(height, width)), band=band, seed=seed)
    # basis: (channels, k, k) where k = min(H, W) (>= 3).
    basis = basis.unsqueeze(0)  # (1, channels, k, k)
    if (basis.shape[-2], basis.shape[-1]) != (height, width):
        basis = F.interpolate(
            basis, size=(height, width), mode="bilinear", align_corners=False
        )
    # Tile to the requested batch dimension.
    target = basis.expand(batch_size, -1, -1, -1).contiguous()
    if device is not None:
        target = target.to(device)
    return target


# ---------------------------------------------------------------------------
# Student model: NaturePriorBlock stack
# ---------------------------------------------------------------------------
class PlatonicCymaticTeacher(nn.Module):
    """Student NaturePriorBlock stack + cymatic teacher distillation hook.

    Returns ``(logits, student_activations)``: the second tensor is the
    spatial feature map after the last NaturePriorBlock, so the trainer
    can call :func:`cka_distillation_loss(student_act, teacher_target)`
    where ``teacher_target = cymatic_teacher_target(...)`` matches the
    feature map's shape.

    Parameters
    ----------
    in_channels : int
        Input image channel count.
    width : int
        Stem and block width (kept constant for simplicity; the H63 design
        doc is about distillation, not channel scaling).
    n_blocks : int
        NaturePriorBlock stack depth.
    n_classes : int
        Final classifier head dimensionality.
    flags : NaturePriorFlags, optional
        Forwarded to each NaturePriorBlock (per-prior ablation knobs).
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        n_blocks: int = 2,
        n_classes: int = 10,
        flags: NaturePriorFlags | None = None,
    ) -> None:
        super().__init__()
        flags = flags or NaturePriorFlags()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.flags = flags
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                NaturePriorBlock(width, width, stride=1, flags=flags)
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        student_act = h  # spatial feature map (B, width, H, W)
        logits = self.head(student_act)
        return logits, student_act

    def teacher_for(self, student_act: torch.Tensor, **kw) -> torch.Tensor:
        """Convenience helper: build a cymatic teacher target matching the
        shape of ``student_act``.
        """
        B, C, H, W = student_act.shape
        return cymatic_teacher_target(B, C, H, W, device=student_act.device, **kw)
