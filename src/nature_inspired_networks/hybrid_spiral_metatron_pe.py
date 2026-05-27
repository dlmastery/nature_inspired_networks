"""H73 — Golden Spiral × Metatron Positional Encoding (G7 hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H73_golden_spiral_metatron_pe.md``.

Concatenates two complementary positional signals and projects back to the
model dimension:

  1. **Golden-spiral PE** (G5 spiral): sequence-position encoded as a 2-D
     phyllotactic spiral ``r = √k, θ = k · GOLD_ANGLE`` with frequency
     features layered on top. The optional
     :mod:`nature_inspired_networks.spiral_pe` module is imported with
     ``try / except`` per task spec — when present we use
     ``spiral_pe.golden_spiral_pe``; otherwise we use a local fallback
     that implements the same Vogel construction.
  2. **Metatron-routed positional vector** (G3 platonic graph): a per-
     token 13-node feature vector aggregated through
     :class:`MetatronGraphLayer`. This captures **relational** rather
     than sequential position.

The two signals are concatenated along the feature dimension and projected
back to ``d_model`` by a final ``nn.Linear``.

References (Citation Rigor)::

    Su et al. 2021 'RoFormer' (arXiv:2104.09864) — the RoPE baseline
    whose absolute-position basis we extend.

    Vogel 1979 Math. Biosciences — golden-angle optimality.

    Hales 2001 Annals of Math. — hex optimality underpinning Metatron.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .priors import PHI
from .platonic_graph import MetatronGraphLayer


# Optional module — guard.
try:  # pragma: no cover - optional concurrent landing
    from .spiral_pe import golden_spiral_pe as _ext_golden_spiral_pe  # type: ignore
    _HAS_EXT_SPIRAL_PE = True
except Exception:
    _HAS_EXT_SPIRAL_PE = False


__all__ = ["SpiralMetatronPE", "fallback_golden_spiral_pe"]


_GOLDEN_ANGLE_RAD: float = 2.0 * math.pi * (1.0 - 1.0 / PHI)


def fallback_golden_spiral_pe(n: int, d: int) -> torch.Tensor:
    """Return a ``(n, d)`` golden-spiral positional encoding.

    Vogel construction:
      - radius   ``r_k = √k``
      - angle    ``θ_k = k · GOLDEN_ANGLE_RAD``
      - features layered as ``[r·cos(θ), r·sin(θ), cos(k·ω_i), sin(k·ω_i), ...]``
        with ``ω_i = PHI^(-i)``.

    This is a self-contained implementation used when the optional
    ``spiral_pe.golden_spiral_pe`` module is not yet importable.
    """
    if d < 2 or d % 2 != 0:
        raise ValueError(f"d must be an even number >= 2; got {d}")
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    k = torch.arange(n, dtype=torch.float32)
    r = (k + 1).sqrt()
    theta = k * _GOLDEN_ANGLE_RAD
    out = torch.empty(n, d, dtype=torch.float32)
    out[:, 0] = r * theta.cos()
    out[:, 1] = r * theta.sin()
    n_freq = (d - 2) // 2
    for i in range(n_freq):
        omega = PHI ** (-i)
        out[:, 2 + 2 * i] = (k * omega).cos()
        out[:, 2 + 2 * i + 1] = (k * omega).sin()
    return out


class SpiralMetatronPE(nn.Module):
    """Combine golden-spiral sequential PE with a Metatron-graph relational PE.

    Forward signature: ``SpiralMetatronPE(...)(positions)`` → ``(N, d_model)``.

    Parameters
    ----------
    d_model : int
        Model feature dimension. Must be even.
    n_max : int
        Maximum sequence length. The spiral PE is precomputed for
        positions ``0, ..., n_max - 1`` and cached as a buffer.
    d_spiral : int, optional
        Spiral-PE feature width. Defaults to ``d_model // 2``. Must be
        even.
    d_metatron_node : int, default 4
        Per-node feature width inside the Metatron graph layer.
    """

    def __init__(
        self,
        d_model: int,
        n_max: int,
        d_spiral: int | None = None,
        d_metatron_node: int = 4,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even; got {d_model}")
        if n_max < 1:
            raise ValueError(f"n_max must be >= 1; got {n_max}")
        self.d_model = int(d_model)
        self.n_max = int(n_max)
        self.d_spiral = int(d_spiral if d_spiral is not None else d_model // 2)
        if self.d_spiral % 2 != 0:
            raise ValueError("d_spiral must be even")
        self.d_metatron_node = int(d_metatron_node)
        d_metatron = 13 * d_metatron_node

        # Precompute the spiral PE for n_max positions.
        spiral = self._make_spiral_pe(n_max, self.d_spiral)
        self.register_buffer("spiral_pe", spiral)  # (n_max, d_spiral)

        # Per-position metatron seed: a learnable (n_max, 13, d_node) tensor.
        self.metatron_seed = nn.Parameter(
            torch.randn(n_max, 13, d_metatron_node) * 0.02
        )
        self.metatron = MetatronGraphLayer(d_metatron_node, d_metatron_node)
        # Final projection from concat(d_spiral + d_metatron) → d_model.
        self.proj = nn.Linear(self.d_spiral + d_metatron, d_model)

    @staticmethod
    def _make_spiral_pe(n: int, d: int) -> torch.Tensor:
        """Use external module when available, fallback otherwise."""
        if _HAS_EXT_SPIRAL_PE:
            try:
                pe = _ext_golden_spiral_pe(n, d)  # type: ignore
                if pe.shape == (n, d):
                    return pe
            except Exception:
                pass
        return fallback_golden_spiral_pe(n, d)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Look up the spiral+metatron PE for ``positions``.

        Parameters
        ----------
        positions : torch.Tensor
            Shape ``(N,)`` integer-valued. All values must be in
            ``[0, n_max)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, d_model)``.
        """
        if positions.dim() != 1:
            raise ValueError(
                f"positions must be 1-D; got {tuple(positions.shape)}"
            )
        idx = positions.to(dtype=torch.long, device=self.spiral_pe.device)
        if (idx < 0).any() or (idx >= self.n_max).any():
            raise ValueError(
                f"positions out of range [0, {self.n_max}); got "
                f"min={int(idx.min())} max={int(idx.max())}"
            )
        spiral = self.spiral_pe[idx]  # (N, d_spiral)
        # Metatron pass: (N, 13, d_node) → graph layer → flatten.
        seed = self.metatron_seed[idx]  # (N, 13, d_node)
        meta = self.metatron(seed)  # (N, 13, d_node)
        meta_flat = meta.reshape(meta.shape[0], -1)  # (N, 13 * d_node)
        cat = torch.cat([spiral, meta_flat], dim=-1)
        return self.proj(cat)
