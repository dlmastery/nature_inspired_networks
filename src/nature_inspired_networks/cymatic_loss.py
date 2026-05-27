"""H46 — Cymatic Loss (Fourier-domain MSE vs Chladni harmonics).

A small auxiliary loss term that pulls the **power-spectrum** of an
intermediate activation map toward the spectrum of the
:func:`chladni_modes_banded` basis. The intuition (Chladni 1787;
Rahaman et al. 2019 on spectral bias) is that natural images decompose
sparsely into low-(m, n) Chladni eigenmodes, so regularising the FFT
spectrum toward this target is the spectral analogue of Tikhonov
smoothness without ad-hoc spectral filtering.

Public API
----------
- :func:`power_spectrum_2d(x)` — ``|FFT2(x)|^2`` keeping the
  ``(B, C, H, W)`` layout.
- :func:`chladni_target_spectrum(h, w, n_modes=8, band=(2, 5))` —
  builds an ``(h, w)`` target power-spectrum from the
  orthonormalised :func:`chladni_modes_banded` basis.
- :func:`cymatic_loss(activations, target_spec)` — MSE between the
  per-batch / per-channel power-spectrum and the broadcast target.
- :class:`CymaticLossModule` — stateful wrapper that caches the target
  spectrum across forward calls (the target is shape-only deterministic
  for a given ``(h, w, n_modes, band, seed)``).

The loss is **training-only**: no inference cost. Compose with the
standard cross-entropy loss via
``loss = ce_loss + lam_cym * cymatic_loss(...)``.

Lives in ``src/nature_inspired_networks/cymatic_loss.py`` per Rule 14
(single import surface for shared primitives).

References
----------
Chladni, E.F.F. 1787 'Entdeckungen über die Theorie des Klanges';
Rahaman, N. et al. 2019 ICML 'On the Spectral Bias of Neural
Networks' (arXiv:1806.08734); Tancik et al. 2020 NeurIPS 'Fourier
Features ...' (arXiv:2006.10739). See
``hypotheses/g5_optimization_init_reg_nas/H46_cymatic_loss.md``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import chladni_modes_banded


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
def power_spectrum_2d(x: torch.Tensor) -> torch.Tensor:
    """Return the 2-D power spectrum of ``x``.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        ``|FFT2(x)|^2`` of shape ``(B, C, H, W)`` (complex magnitude
        squared, kept real). Float dtype matches ``x``.
    """
    assert x.dim() == 4, f"power_spectrum_2d expects (B, C, H, W); got {tuple(x.shape)}"
    spec = torch.fft.fft2(x).abs() ** 2
    return spec


def chladni_target_spectrum(
    h: int,
    w: int,
    n_modes: int = 8,
    band: tuple[int, int] = (2, 5),
    seed: int = 0,
) -> torch.Tensor:
    """Build an ``(h, w)`` Chladni-derived target power-spectrum.

    Construction:

    1. Sample ``n_modes`` orthonormal Chladni eigenmodes of shape
       ``(h, w)`` via :func:`chladni_modes_banded` with the given
       ``band`` and ``seed`` (must be square so use ``k = max(h, w)``
       then crop).
    2. Take the per-mode 2-D power spectrum and sum across modes.
    3. Normalise so the total spectrum mass sums to 1 — this makes
       :func:`cymatic_loss` scale-free and λ_cym directly comparable
       across layer resolutions.

    For ``h != w`` we build a square basis at ``k = max(h, w)`` and
    crop to ``(h, w)`` before computing the spectrum — the dominant
    low-frequency structure carries over.
    """
    assert h >= 2 and w >= 2, f"target spectrum requires h, w >= 2; got ({h}, {w})"
    lo, hi = band
    assert lo >= 1 and hi >= lo, f"invalid band {band}"
    k = max(h, w)
    basis = chladni_modes_banded(n_modes, k, band=band, seed=seed)  # (n_modes, k, k)
    if k != h or k != w:
        basis = basis[:, :h, :w].contiguous()
    spec = (torch.fft.fft2(basis).abs() ** 2).sum(dim=0)  # (h, w)
    total = spec.sum()
    if total.item() > 0:
        spec = spec / total
    return spec


def cymatic_loss(
    activations: torch.Tensor,
    target_spec: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE between the per-(B, C) power-spectrum of ``activations`` and
    a broadcastable ``target_spec``.

    Parameters
    ----------
    activations : torch.Tensor
        Shape ``(B, C, H, W)``.
    target_spec : torch.Tensor
        Either ``(H, W)`` (broadcast across batch + channels) or
        ``(B, C, H, W)`` (per-sample target). Will be detached from
        any autograd graph as we treat it as a fixed teacher.
    reduction : {"mean", "sum", "none"}
        Passed through to :func:`F.mse_loss`.

    Returns
    -------
    torch.Tensor
        Scalar (if reduction in {"mean", "sum"}) or per-element MSE
        tensor (if ``reduction="none"``).
    """
    assert activations.dim() == 4, (
        f"cymatic_loss expects (B, C, H, W); got {tuple(activations.shape)}"
    )
    spec = power_spectrum_2d(activations)
    # Per-(B, C) normalisation so the MSE is independent of feature scale.
    spec_flat = spec.reshape(spec.shape[0], spec.shape[1], -1)
    norm = spec_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    spec_norm = (spec_flat / norm).reshape(spec.shape)
    # Broadcast the target to (B, C, H, W).
    tgt = target_spec.detach()
    if tgt.dim() == 2:
        tgt = tgt.unsqueeze(0).unsqueeze(0).expand_as(spec_norm)
    elif tgt.dim() == 4:
        assert tgt.shape == spec_norm.shape, (
            f"target_spec shape {tuple(tgt.shape)} != activations spec shape {tuple(spec_norm.shape)}"
        )
    else:
        raise ValueError(f"target_spec must be 2-D or 4-D; got dim {tgt.dim()}")
    return F.mse_loss(spec_norm, tgt, reduction=reduction)


# ---------------------------------------------------------------------------
# Stateful module wrapper
# ---------------------------------------------------------------------------
class CymaticLossModule(nn.Module):
    """Stateful cymatic-loss with cached target spectrum.

    The target is cached as a non-persistent buffer on first call and
    re-used across subsequent calls as long as the spatial shape
    ``(H, W)`` matches. If a forward call sees a different shape (e.g.,
    spatial pooling changed the layer dims), the cache is rebuilt.

    Parameters
    ----------
    n_modes : int, default 8
        Number of Chladni eigenmodes to sum.
    band : tuple[int, int], default (2, 5)
        ``(m, n)`` frequency band for the basis. Matches H35.v2.
    seed : int, default 0
        RNG seed for the basis construction.
    reduction : str, default "mean"
    """

    def __init__(
        self,
        n_modes: int = 8,
        band: tuple[int, int] = (2, 5),
        seed: int = 0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.band = band
        self.seed = seed
        self.reduction = reduction
        # Lazily-built target spectrum. Registered as a buffer so it
        # moves with .to(device); but we cannot register-on-first-call
        # without a placeholder. Use a tiny 1x1 placeholder; rebuilt
        # on first forward.
        self.register_buffer("_target", torch.zeros(1, 1), persistent=False)
        self._cached_shape: tuple[int, int] | None = None

    def _ensure_target(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if self._cached_shape != (h, w) or self._target.shape != (h, w):
            tgt = chladni_target_spectrum(h, w, self.n_modes, self.band, self.seed)
            self._target = tgt.to(device)
            self._cached_shape = (h, w)
        elif self._target.device != device:
            self._target = self._target.to(device)
        return self._target

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        assert activations.dim() == 4, (
            f"CymaticLossModule expects (B, C, H, W); got {tuple(activations.shape)}"
        )
        _, _, h, w = activations.shape
        tgt = self._ensure_target(h, w, activations.device)
        return cymatic_loss(activations, tgt, reduction=self.reduction)
