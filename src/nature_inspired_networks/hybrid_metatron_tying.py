"""H74 — Metatron Overlap Weight-Tying (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H74_metatron_overlap_tying.md``.

A Conv2d-shaped surface whose weights are **tied across 13 circle-overlap
groups**: a single underlying weight tensor is shared by 13 Conv2d layers,
each multiplied by a learnable scaling coefficient ``α_c`` (one per
Metatron-Cube vertex / circle). The output is the sum across the 13
tied convs.

This is a massive intra-layer parameter-tying scheme. Compared to a
naive ensemble of 13 independent convs (which would carry 13× the
parameters), the tied surface costs only ``1 × (k×k×c_in×c_out) + 13``
parameters — a compression ratio of ≈1/13 relative to an untied bank.

The 13 circles correspond to the canonical Metatron-Cube vertices used
by :func:`platonic_graph.metatron_cube_adjacency`: 1 centre + 6 inner
hex + 6 outer hex. The 13 ``α_c`` scalars are initialised to 1/13 so
the un-trained sum has roughly identity gain.

References (Citation Rigor)::

    Hu et al. 2022 ICLR 'LoRA' (arXiv:2106.09685) — low-rank
    factorisation precedent.

    Lan et al. 2020 ICLR 'ALBERT' (arXiv:1909.11942) — cross-layer
    parameter sharing precedent; H74's intra-layer cross-circle
    sharing is a strict refinement.

    Press, Wolf 2017 EACL 'Using the Output Embedding to Improve LMs'
    (arXiv:1608.05859) — embedding tying.

    Hales 2001 Annals of Math. — hex optimality of the Metatron
    13-vertex pattern.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metatron_kernel import metatron_basis_kernels
from .priors import PHI


__all__ = ["MetatronTiedConv2d", "METATRON_N_CIRCLES"]


METATRON_N_CIRCLES: int = 13


class MetatronTiedConv2d(nn.Module):
    """Conv2d whose 13 "circle overlap" outputs share a single weight tensor.

    Forward: ``(B, C_in, H, W)`` → ``(B, C_out, H, W)`` at stride 1.

    The computation is::

        W_eff = sum_{c=0..12} α_c · (W ⊙ mask_c)
        y     = conv2d(x, W_eff, stride=stride, padding=padding)

    where ``mask_c`` is a fixed ``(k, k)`` binary circle mask drawn from
    :func:`metatron_basis_kernels` (the same 13-circle pattern used by
    H40's :class:`MetatronConv2d`). Because the 13 masks are spatially
    DISTINCT — one centre + 6 inner-hex + 6 outer-hex circles — the
    13 ``α_c`` scalars are not reparameterisation-redundant: setting
    ``α = [1, 0, 0, ...]`` produces a *different* effective kernel from
    ``α = [0, 1, 0, ...]`` and therefore a different output.

    G7-audit fix: the previous implementation computed ``W * Σα_c`` (a
    scalar gate on a single tensor) which made the 13 alphas collapse
    to their sum. The mask bank breaks that degeneracy.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int, default 3
        Square kernel size.
    stride, padding : int
    bias : bool, default False

    Attributes
    ----------
    weight : nn.Parameter
        Shape ``(out_channels, in_channels, k, k)``. The single tied
        weight matrix shared across all 13 circles.
    alphas : nn.Parameter
        Shape ``(13,)``. Per-circle learnable scaling coefficient.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert in_channels >= 1 and out_channels >= 1 and kernel_size >= 1
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = (kernel_size // 2) if padding is None else int(padding)
        # Single tied weight tensor (Kaiming-init for parity with Conv2d default).
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w)
        self.bias = (
            nn.Parameter(torch.zeros(out_channels)) if bias else None
        )
        # 13 per-circle scaling coefficients, init at 1/13 so the
        # un-trained sum has roughly identity-conv gain.
        self.alphas = nn.Parameter(torch.full((METATRON_N_CIRCLES,), 1.0 / METATRON_N_CIRCLES))
        # ---- 13 fixed per-circle binary masks (G7-audit fix) ----
        # ``metatron_basis_kernels`` returns ``(13, k, k)``. We use the
        # same circle bank as H40's MetatronConv2d so the two layers
        # share a consistent geometric basis. Registered as a
        # non-persistent buffer (not trainable, but rides .to(device)).
        masks = metatron_basis_kernels(
            k=int(self.kernel_size), n_circles=METATRON_N_CIRCLES
        )  # (13, k, k), float32, values in {0, 1}
        self.register_buffer("masks", masks, persistent=False)

    @staticmethod
    def n_circles() -> int:
        """Number of Metatron circles (=13)."""
        return METATRON_N_CIRCLES

    def effective_weight(self) -> torch.Tensor:
        """Return ``Σ_c α_c · (W ⊙ mask_c)`` — the effective conv kernel.

        Because the 13 masks are spatially distinct (one centre + 6
        inner-hex + 6 outer-hex circles), this sum is NOT equivalent to
        ``W · Σα_c``: per-spatial-position the contribution of circle
        ``c`` is gated by ``mask_c[y, x]``. The 13 alphas therefore
        encode genuinely independent spatial-mask weightings and do not
        collapse to a single scalar.
        """
        # masks: (13, k, k) -> (13, 1, 1, k, k)
        m = self.masks.to(self.weight.dtype, copy=False).view(
            METATRON_N_CIRCLES, 1, 1, self.kernel_size, self.kernel_size
        )
        # alphas: (13,) -> (13, 1, 1, 1, 1)
        a = self.alphas.view(METATRON_N_CIRCLES, 1, 1, 1, 1)
        # Sum over the circle axis: (out_c, in_c, k, k).
        combined_mask = (a * m).sum(dim=0)  # (1, 1, k, k)
        # Broadcast multiply with the shared (out_c, in_c, k, k) weight.
        return self.weight * combined_mask

    def param_compression_ratio(self) -> float:
        """Compression vs an untied bank of 13 independent Conv2d kernels.

        Returns ``1 - tied_params / untied_params`` where
        ``untied_params = 13 · k² · c_in · c_out`` (ignoring biases).
        At init the ratio is approximately
        ``1 - 1/13 ≈ 0.923`` (≈92 % parameter reduction relative to a
        naive 13-circle bank).
        """
        k2 = self.kernel_size * self.kernel_size
        kernel_params = k2 * self.in_channels * self.out_channels
        tied = kernel_params + METATRON_N_CIRCLES
        untied = METATRON_N_CIRCLES * kernel_params
        return 1.0 - tied / untied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B, {self.in_channels}, H, W); got {tuple(x.shape)}"
            )
        eff_w = self.effective_weight()
        return F.conv2d(
            x,
            eff_w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, n_circles={METATRON_N_CIRCLES}"
        )
