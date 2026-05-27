"""H70 — Cymatic Low-Data Curriculum (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H70_cymatic_low_data_curriculum.md``.

A callable curriculum loss that **anneals** from a cymatic Fourier-domain
MSE auxiliary objective (Chladni-eigenmode-grounded structured signal)
to a standard cross-entropy classification objective over the first
``warmup_epochs`` epochs of training.

The mix coefficient is ``α(epoch) = max(0, 1 - epoch / warmup_epochs)``.
At ``epoch = 0`` the loss is pure cymatic (α=1, CE weight 0);
at ``epoch = warmup_epochs`` the loss is pure CE (α=0). The schedule is
monotone non-increasing in α, so a model is biased toward learning the
structured Chladni signal first and only later toward the task objective.

The cymatic loss component uses the FFT spectrum of the input and
compares it against a *target* Chladni-eigenmode spectrum constructed
from :func:`chladni_modes_banded`. This is the LLM-track abstraction of
the H35 cymatic-init: where H35 *initialises* weights to Chladni modes,
H70 makes the *training signal itself* prefer Chladni structure.

The optional :mod:`nature_inspired_networks.cymatic_loss` module is
imported with ``try/except`` (other agents are concurrently landing it).

References (Citation Rigor)::

    Bengio, Louradour, Collobert, Weston 2009 ICML 'Curriculum Learning'
    (arXiv:0904.0102) — foundational curriculum scheduling.

    Chladni 1787 'Entdeckungen über die Theorie des Klanges' — the
    eigen-mode basis we compare against in the Fourier domain.

    Eldan, Li 2023 'TinyStories' (arXiv:2305.07759) — the low-data
    pretraining target.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI, chladni_modes_banded


# Optional concurrent module — guard with try/except per task spec.
try:  # pragma: no cover - optional, landing concurrently
    from .cymatic_loss import cymatic_loss as _ext_cymatic_loss  # type: ignore
    _HAS_EXT_CYMATIC = True
except Exception:
    _HAS_EXT_CYMATIC = False


__all__ = ["CymaticCurriculumLoss", "default_cymatic_loss"]


def default_cymatic_loss(
    feat: torch.Tensor,
    target_modes: torch.Tensor,
) -> torch.Tensor:
    """Fourier-domain MSE between ``feat`` and a Chladni target.

    Parameters
    ----------
    feat : torch.Tensor
        ``(B, C, H, W)`` activation tensor.
    target_modes : torch.Tensor
        ``(n_modes, H, W)`` orthonormal Chladni-mode basis.

    Returns
    -------
    torch.Tensor
        Scalar MSE between the magnitude spectra of ``feat`` averaged
        across channels and the magnitude spectrum of the *mean*
        target mode. The Fourier-domain compare is shift-invariant and
        roughly scale-invariant — appropriate for an auxiliary
        curriculum signal.
    """
    if feat.dim() != 4:
        raise ValueError(f"feat must be 4-D (B, C, H, W); got {tuple(feat.shape)}")
    if target_modes.dim() != 3:
        raise ValueError(
            f"target_modes must be 3-D (n_modes, H, W); got {tuple(target_modes.shape)}"
        )
    # Channel-mean magnitude spectrum of features.
    feat_mean = feat.mean(dim=1)  # (B, H, W)
    f_spec = torch.fft.fft2(feat_mean).abs()
    target = target_modes.mean(dim=0).to(feat.device, feat.dtype)  # (H, W)
    t_spec = torch.fft.fft2(target).abs()
    # Normalise both by their respective L2 norms for scale invariance.
    f_norm = f_spec / (f_spec.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
    t_norm = t_spec / (t_spec.norm() + 1e-8)
    return F.mse_loss(f_norm, t_norm.unsqueeze(0).expand_as(f_norm))


class CymaticCurriculumLoss(nn.Module):
    """Curriculum loss that anneals from cymatic-aux to standard CE.

    Forward signature::

        loss = CymaticCurriculumLoss(...)(feat, logits, targets, epoch)

    where ``feat`` is a ``(B, C, H, W)`` intermediate activation tensor
    used by the auxiliary cymatic loss, ``logits`` are the classifier
    logits, ``targets`` are integer class labels, and ``epoch`` is the
    current training epoch (int). The returned scalar loss is

        ``α(epoch) · cymatic_loss(feat) + (1 - α(epoch)) · CE(logits, y)``

    with ``α(epoch) = max(0, 1 - epoch / warmup_epochs)``.

    Parameters
    ----------
    warmup_epochs : int, default 5
        Number of epochs over which to anneal α from 1 to 0.
    k : int, default 8
        Spatial extent of the Chladni target modes (must match ``feat``
        spatial size at forward time).
    n_modes : int, default 4
        Number of Chladni basis modes.
    band : tuple[int, int], default (2, 5)
        Frequency band for :func:`chladni_modes_banded`. Mirrors the
        H35.v2 default cymatic_init.
    cymatic_fn : callable, optional
        Override the auxiliary loss function. By default uses the
        optional :mod:`nature_inspired_networks.cymatic_loss` module if
        available, else :func:`default_cymatic_loss`.
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        k: int = 8,
        n_modes: int = 4,
        band: tuple[int, int] = (2, 5),
        cymatic_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if warmup_epochs < 1:
            raise ValueError(f"warmup_epochs must be >= 1; got {warmup_epochs}")
        self.warmup_epochs = int(warmup_epochs)
        self.k = int(k)
        self.n_modes = int(n_modes)
        self.band = (int(band[0]), int(band[1]))
        # Fixed Chladni target basis, registered as a buffer for device moves.
        target = chladni_modes_banded(n_modes, k, band=band, seed=0)
        self.register_buffer("target_modes", target)
        if cymatic_fn is not None:
            self._cymatic_fn = cymatic_fn
        elif _HAS_EXT_CYMATIC:
            # Use the external module if it landed; adapt its signature
            # by ignoring unsupported kwargs.
            ext = _ext_cymatic_loss  # type: ignore

            def _wrap(feat: torch.Tensor, target_modes: torch.Tensor) -> torch.Tensor:
                # External cymatic_loss expects (H, W) or (B, C, H, W);
                # we collapse the (n_modes, H, W) basis to a single
                # mean-mode 2-D target (the mean of the orthonormal
                # band) for the wrapper signature.
                try:
                    tgt_2d = target_modes.mean(dim=0)
                    return ext(feat, tgt_2d)  # type: ignore
                except (TypeError, ValueError):
                    return default_cymatic_loss(feat, target_modes)

            self._cymatic_fn = _wrap
        else:
            self._cymatic_fn = default_cymatic_loss

    def alpha(self, epoch: int) -> float:
        """Return the current mix coefficient ``α(epoch)``."""
        a = 1.0 - float(epoch) / float(self.warmup_epochs)
        return max(0.0, a)

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        a = self.alpha(epoch)
        ce = F.cross_entropy(logits, targets)
        if a > 0.0:
            cym = self._cymatic_fn(feat, self.target_modes)
            return a * cym + (1.0 - a) * ce
        return ce
