"""H72 — Fractal Vesica FFN (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H72_fractal_vesica_ffn.md``.

Combines:

  1. **VesicaPiscisConv2d** (G4 multi-path overlap convs): the up-
     projection in a feed-forward block uses three parallel paths at
     different vesica radii whose masks overlap at half-radius.
  2. **Fractal multi-path summation** (G2 / FractalNet, Larsson 2017):
     the three vesica paths are summed inside a residual FFN block.
  3. **φ-channel widths**: the FFN hidden width follows
     ``round(c * PHI)`` so the parameter budget matches the
     classical 4× FFN multiplier when ``PHI ~ 1.618`` ≈ 4 / 2.5.

Two surfaces are exposed:

  * :class:`FractalVesicaFFN`     — 2-D conv-style FFN for image inputs
    ``(B, C, H, W)``.
  * :class:`FractalVesica1DFFN`   — 1-D token-mixing FFN for sequence
    inputs ``(B, N, C)``.

References (Citation Rigor)::

    Larsson, Maire, Shakhnarovich 2017 ICLR 'FractalNet'
    (arXiv:1605.07648) — fractal multi-path topology.

    Hoogeboom, Peters, Cohen, Welling 2018 ECCV 'HexaConv'
    (arXiv:1803.02108) — hex/vesica overlap motif.

    Shazeer 2020 'GLU Variants Improve Transformer'
    (arXiv:2002.05202) — SwiGLU FFN baseline.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI
from .vesica_piscis import VesicaPiscisConv2d


__all__ = ["FractalVesicaFFN", "FractalVesica1DFFN"]


def _vesica_radii(base: float, n_paths: int) -> list[float]:
    """Three fractal radii at successive 1/φ shrinks (b, b/φ, b/φ²)."""
    return [base * (PHI ** -i) for i in range(n_paths)]


class FractalVesicaFFN(nn.Module):
    """Conv-FFN with three parallel VesicaPiscisConv2d up-projections.

    Forward: ``(B, C, H, W)`` → ``(B, C, H, W)``.

    Architecture:

      1. Up-projection: 3 parallel :class:`VesicaPiscisConv2d` with
         radii ``r, r/φ, r/φ²`` (fractal shrink). Their outputs are
         summed with a learnable per-path scale initialised to 1/3.
      2. φ-GELU-style activation (``x · σ(φ·x)`` via :data:`PHI`).
      3. Down-projection: a 1×1 Conv2d back to ``c``.
      4. Residual addition.

    Parameters
    ----------
    c : int
        Channel count.
    c_hidden : int, optional
        FFN hidden width. Defaults to ``round(c * PHI)``.
    kernel_size : int, default 5
        Spatial extent of the vesica-piscis kernels.
    base_radius : float, default 2.0
        Radius of the largest vesica disc; subsequent paths shrink by
        successive 1/φ factors.
    """

    def __init__(
        self,
        c: int,
        c_hidden: int | None = None,
        kernel_size: int = 5,
        base_radius: float = 2.0,
    ) -> None:
        super().__init__()
        assert c >= 1
        c_hidden = int(round(c * PHI)) if c_hidden is None else int(c_hidden)
        assert c_hidden >= 1
        self.c = c
        self.c_hidden = c_hidden
        radii = _vesica_radii(base_radius, 3)
        self.radii = radii
        self.paths = nn.ModuleList(
            [
                VesicaPiscisConv2d(
                    c, c_hidden, kernel_size=kernel_size, n_circles=3,
                    radius=r, offset=1.0, bias=False,
                )
                for r in radii
            ]
        )
        # Per-path scale, init 1/3 so the un-trained sum is roughly identity-
        # gain at init.
        self.path_scales = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.down = nn.Conv2d(c_hidden, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got {tuple(x.shape)}")
        # Up-project via three fractal vesica paths.
        up: torch.Tensor | None = None
        for i, path in enumerate(self.paths):
            y = path(x)
            term = self.path_scales[i] * y
            up = term if up is None else up + term
        assert up is not None
        # φ-GELU activation.
        up = up * torch.sigmoid(PHI * up)
        out = self.down(up)
        return out + x


class FractalVesica1DFFN(nn.Module):
    """1-D token-mixing FFN where the up-projection is the 1-D analogue
    of :class:`VesicaPiscisConv2d`.

    Forward: ``(B, N, C)`` → ``(B, N, C)``.

    Architecture mirrors :class:`FractalVesicaFFN` but uses 1-D masked
    convolutions for the up-projection. Three paths at radii
    ``r, r/φ, r/φ²`` are summed before a φ-GELU and a linear down-
    projection.

    Parameters
    ----------
    c : int
        Token channel count.
    c_hidden : int, optional
        Hidden width. Defaults to ``round(c * PHI)``.
    kernel_size : int, default 5
        1-D conv kernel size (must be odd to keep N invariant).
    base_radius : float, default 2.0
        Largest 1-D disc radius; subsequent paths shrink by 1/φ.
    """

    def __init__(
        self,
        c: int,
        c_hidden: int | None = None,
        kernel_size: int = 5,
        base_radius: float = 2.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}")
        c_hidden = int(round(c * PHI)) if c_hidden is None else int(c_hidden)
        assert c >= 1 and c_hidden >= 1
        self.c = c
        self.c_hidden = c_hidden
        self.kernel_size = kernel_size
        radii = _vesica_radii(base_radius, 3)
        self.radii = radii
        # 1-D paths: per-radius binary mask × Conv1d.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(c, c_hidden, kernel_size, padding=kernel_size // 2, bias=False)
                for _ in range(3)
            ]
        )
        masks = self._make_1d_masks(kernel_size, radii)
        self.register_buffer("masks", masks)
        self.path_scales = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.down = nn.Linear(c_hidden, c)

    @staticmethod
    def _make_1d_masks(k: int, radii: Sequence[float]) -> torch.Tensor:
        """Return (n_paths, k) binary masks: |i - centre| < r."""
        cx = (k - 1) / 2.0
        xs = torch.arange(k, dtype=torch.float32)
        masks = torch.zeros(len(radii), k, dtype=torch.float32)
        for i, r in enumerate(radii):
            masks[i] = ((xs - cx).abs() <= float(r)).float()
        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, C); got {tuple(x.shape)}")
        B, N, C = x.shape
        if C != self.c:
            raise ValueError(f"channel mismatch: x.shape[-1]={C} != c={self.c}")
        # Transpose to (B, C, N) for Conv1d.
        x_t = x.transpose(1, 2)
        out: torch.Tensor | None = None
        for i, conv in enumerate(self.convs):
            mask = self.masks[i].view(1, 1, self.kernel_size)
            w = conv.weight * mask
            y = F.conv1d(x_t, w, conv.bias, padding=self.kernel_size // 2)
            term = self.path_scales[i] * y
            out = term if out is None else out + term
        assert out is not None
        out = out * torch.sigmoid(PHI * out)
        out = out.transpose(1, 2)  # (B, N, c_hidden)
        return self.down(out) + x
