"""H69 — KAN-Metatron Symbolic Head (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H69_kan_metatron_symbolic_head.md``.

Glue layer that combines:

  1. **MetatronGraphLayer** (G3 platonic graph): a 13-node Metatron-Cube
     adjacency for message passing between symbolic slots.
  2. A **per-edge KAN spline** (G7 KAN paradigm): each output neuron's
     input is routed through a learnable lookup-table activation
     (``KANEdge``) before the final linear projection.

The head replaces the standard ``nn.Linear`` classifier with a 13-circle
graph aggregation followed by per-edge spline activations. The spline is
implemented as a per-edge learnable 1-D lookup over a fixed knot grid
(``spline_pts=8`` by default), interpolated by clamped linear
interpolation so gradients flow naturally.

References (Citation Rigor)::

    Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark 2024
    NeurIPS 'KAN: Kolmogorov-Arnold Networks' (arXiv:2404.19756).

    Hales 2001 Annals of Math. 'The Honeycomb Conjecture' — the
    optimality of the hex packing that the 13-vertex Metatron Cube
    extends.

    Huh, Cheung, Wang, Isola 2024 ICML 'The Platonic Representation
    Hypothesis' (arXiv:2405.07987).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI
from .platonic_graph import MetatronGraphLayer, metatron_cube_adjacency


__all__ = ["KANEdge", "KANMetatronHead"]


class KANEdge(nn.Module):
    """A per-edge KAN spline implemented as a learnable lookup table.

    Stores ``spline_pts`` knots over the input range ``[lo, hi]`` and
    evaluates the activation by clamped linear interpolation. The knot
    values are initialised so the activation behaves like ``SiLU`` at
    init (so the un-trained head reduces to a standard MLP-style head).

    Parameters
    ----------
    spline_pts : int, default 8
        Number of knot points. Must be >= 2.
    lo, hi : float, default -3.0, 3.0
        Domain over which the lookup table is defined. Inputs outside the
        domain are clamped before interpolation (matches the KAN paper's
        boundary handling).
    """

    def __init__(self, spline_pts: int = 8, lo: float = -3.0, hi: float = 3.0) -> None:
        super().__init__()
        if spline_pts < 2:
            raise ValueError(f"spline_pts must be >= 2; got {spline_pts}")
        if not hi > lo:
            raise ValueError(f"hi must exceed lo; got lo={lo}, hi={hi}")
        self.spline_pts = int(spline_pts)
        self.lo = float(lo)
        self.hi = float(hi)
        # Initialise the lookup table so the activation == SiLU at the
        # knot positions. This makes the un-trained head equivalent to a
        # standard MLP head at init.
        xs = torch.linspace(lo, hi, spline_pts)
        ys = xs * torch.sigmoid(xs)
        self.knots = nn.Parameter(ys.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the spline activation element-wise."""
        # Map x ∈ [lo, hi] → t ∈ [0, spline_pts-1]
        x_clip = x.clamp(min=self.lo, max=self.hi)
        t = (x_clip - self.lo) / (self.hi - self.lo) * (self.spline_pts - 1)
        i_lo = t.floor().long().clamp(max=self.spline_pts - 2)
        frac = (t - i_lo.to(t.dtype))
        y_lo = self.knots[i_lo]
        y_hi = self.knots[i_lo + 1]
        return y_lo + frac * (y_hi - y_lo)


class KANMetatronHead(nn.Module):
    """KAN-style head with 13-circle Metatron aggregation.

    Forward pass:

      1. Linear ``in_proj`` projects the ``d_in``-vector into 13 node
         feature slots of width ``node_dim``.
      2. A :class:`MetatronGraphLayer` performs one round of
         13-vertex message passing over the Metatron-Cube adjacency.
      3. Each output neuron's pre-activation is filtered through its
         own :class:`KANEdge` spline activation.
      4. ``out_proj`` collapses the 13-slot vector to ``d_out``.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_out : int
        Output dimension.
    node_dim : int, default 4
        Per-node feature width inside the Metatron graph layer.
    spline_pts : int, default 8
        Number of KAN spline knots per edge.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        node_dim: int = 4,
        spline_pts: int = 8,
    ) -> None:
        super().__init__()
        assert d_in >= 1 and d_out >= 1 and node_dim >= 1
        self.d_in = d_in
        self.d_out = d_out
        self.node_dim = node_dim

        # Project (B, d_in) -> (B, 13 * node_dim) -> (B, 13, node_dim).
        self.in_proj = nn.Linear(d_in, 13 * node_dim)
        # Metatron graph message-passing.
        self.metatron = MetatronGraphLayer(node_dim, node_dim)
        # One KAN edge spline per output neuron — d_out edges total.
        self.edges = nn.ModuleList(
            [KANEdge(spline_pts=spline_pts) for _ in range(d_out)]
        )
        # Pre-edge linear: maps the 13-node aggregated vector to d_out
        # pre-activation scalars (each scalar then runs through its
        # own KANEdge).
        self.pre_edge = nn.Linear(13 * node_dim, d_out)

    @staticmethod
    def metatron_edge_count() -> int:
        """Return the number of non-zero (directed) entries in the
        Metatron-Cube adjacency. Used for verifying graph structure.
        """
        A = metatron_cube_adjacency()
        return int((A != 0).sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(..., d_in)`` → ``(..., d_out)``.

        Leading dimensions are flattened so the head is reusable as a
        classification top or a sequence-token head.
        """
        if x.dim() < 2:
            raise ValueError(f"expected at least 2-D input; got {tuple(x.shape)}")
        lead = x.shape[:-1]
        h = self.in_proj(x.reshape(-1, self.d_in))
        h = h.view(-1, 13, self.node_dim)
        h = self.metatron(h)  # (N, 13, node_dim)
        h = h.reshape(-1, 13 * self.node_dim)
        pre = self.pre_edge(h)  # (N, d_out)
        # Apply per-output KAN spline.
        out = torch.empty_like(pre)
        for j in range(self.d_out):
            out[:, j] = self.edges[j](pre[:, j])
        return out.view(*lead, self.d_out)
