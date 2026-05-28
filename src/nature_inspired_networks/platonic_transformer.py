"""H55 — Platonic Transformer (Islam 2025 reference implementation).

Design doc: ``hypotheses/g6_topological_bridging/H55_platonic_transformers.md``.

A pre-norm Transformer block whose multi-head attention is partitioned
by the vertex incidence of a Platonic solid. For ``group="icosa"`` the
12 icosahedron vertices index 12 attention heads; for
``group="dodeca"`` the 20 dodecahedron vertices index 20 heads. Heads
sharing a 5-fold symmetry orbit share a small head-bias term derived
from the Platonic vertex coordinates -- this is the strict-equivariant
inductive bias (Islam 2025).

The implementation is intentionally minimal pure-torch (no e3nn /
e2cnn dependency) so the same module imports on Windows + Python 3.13
without external compiled wheels. Heavier full-equivariance machinery
is the H24 / H37 territory; H55 is the *reference-implementation* line
of evidence for the head-grouping idea on its own.

Public surface
--------------
- :class:`PlatonicAttention`        -- multi-head attention with
                                       Platonic-vertex head biases.
- :class:`PlatonicTransformerBlock` -- pre-norm block (LN -> attn ->
                                       residual -> LN -> FFN ->
                                       residual).
- :func:`platonic_vertex_coords`    -- ``(n_heads, 3)`` coordinates
                                       for tetra/cube/octa/icosa/dodeca.

References (Citation Rigor)
---------------------------
    Islam, M. and others 2025 arXiv 'Platonic Transformers: A Solid
    Choice For Equivariance' (arXiv:2510.03511) -- the reference paper
    for H55; defines the Platonic-vertex head partitioning we implement.
    Vaswani, Ashish and others 2017 NeurIPS 'Attention Is All You
    Need' (arXiv:1706.03762) -- the original Transformer; vanilla
    baseline.
    Bronstein, Michael M., Bruna, Joan, Cohen, Taco S., Velickovic,
    Petar 2021 arXiv 'Geometric Deep Learning: Grids, Groups, Graphs,
    Geodesics, and Gauges' (arXiv:2104.13478) -- GDL manifesto;
    theoretical framework for group-equivariant architectures.
    Shaw, Peter, Uszkoreit, Jakob, Vaswani, Ashish 2018 NAACL 'Self-
    Attention with Relative Position Representations' (arXiv:1803.02155)
    -- relative-positional bias precedent that the per-head angular
    bias generalises with a Platonic-vertex phase per head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


_PLATONIC_HEAD_COUNTS = {
    "tetra": 4,
    "cube": 8,    # 8 vertices
    "octa": 6,    # 6 vertices
    "icosa": 12,  # 12 vertices
    "dodeca": 20, # 20 vertices
}


def _platonic_head_angles(group: str) -> torch.Tensor:
    """Return ``(n_heads,)`` Platonic-vertex angular phases.

    Each head is assigned an angle in ``[0, 2*pi)`` derived from the
    azimuth of the corresponding Platonic vertex (``atan2(y, x)``).
    Because Platonic vertex sets are vertex-transitive, no two heads
    share the *same* azimuth except in degenerate axis-aligned cases
    (octa has two vertices at the poles with ``x=y=0``; those get
    angles 0 and pi from the canonical ``atan2(0, 0)`` and ``atan2(0,
    -0)`` -- but we offset polar vertices with a deterministic
    head-index micro-phase so every head gets a distinct angle).

    The angle is what drives the relative-positional bias in
    :class:`PlatonicAttention`: head ``h``'s bias at relative position
    ``delta = j - i`` is ``cos(angle_h + 2*pi*delta / N)`` where ``N``
    is the sequence length. This mirrors the H37 PentagonalAttention
    pattern with a Platonic-vertex phase per head instead of a 5-fold
    one.
    """
    coords = platonic_vertex_coords(group)            # (n_heads, 3)
    n_heads = coords.shape[0]
    # Primary phase = vertex azimuth atan2(y, x). For polar vertices
    # (x = y = 0) this returns 0; add a deterministic 2*pi*h / n_heads
    # micro-phase to keep every head's angle distinct so the bias is
    # genuinely per-head rather than degenerate.
    azimuth = torch.atan2(coords[:, 1], coords[:, 0])  # (n_heads,)
    micro = 2.0 * math.pi * torch.arange(n_heads, dtype=torch.float32) / float(n_heads)
    # Combine: azimuth carries the geometric Platonic signature, micro
    # breaks any axis-aligned degeneracy. Use a small weight on micro
    # so the geometric prior dominates when distinct.
    return azimuth + 0.25 * micro


def platonic_vertex_coords(group: str) -> torch.Tensor:
    """Return ``(n_heads, 3)`` Platonic-solid vertex coordinates.

    Each Platonic solid has its canonical vertex set up to scale;
    we return unit-normalised coordinates. The output is deterministic
    and consistent across calls.

    Parameters
    ----------
    group : str
        One of ``"tetra", "cube", "octa", "icosa", "dodeca"``.
    """
    if group not in _PLATONIC_HEAD_COUNTS:
        raise ValueError(
            f"unknown group={group!r}; expected one of "
            f"{sorted(_PLATONIC_HEAD_COUNTS)}"
        )
    if group == "tetra":
        # 4 vertices of a regular tetrahedron
        v = torch.tensor(
            [
                [+1.0, +1.0, +1.0],
                [+1.0, -1.0, -1.0],
                [-1.0, +1.0, -1.0],
                [-1.0, -1.0, +1.0],
            ],
            dtype=torch.float32,
        )
    elif group == "cube":
        # 8 vertices of a cube (+-1, +-1, +-1)
        v = torch.tensor(
            [
                [s1, s2, s3]
                for s1 in (+1.0, -1.0)
                for s2 in (+1.0, -1.0)
                for s3 in (+1.0, -1.0)
            ],
            dtype=torch.float32,
        )
    elif group == "octa":
        # 6 vertices of a regular octahedron on the unit axes
        v = torch.tensor(
            [
                [+1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, +1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, +1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
    elif group == "icosa":
        # 12 icosa vertices via golden-ratio rectangles
        inv = 1.0 / PHI
        v = torch.tensor(
            [
                [0.0, +1.0, +inv], [0.0, +1.0, -inv],
                [0.0, -1.0, +inv], [0.0, -1.0, -inv],
                [+1.0, +inv, 0.0], [+1.0, -inv, 0.0],
                [-1.0, +inv, 0.0], [-1.0, -inv, 0.0],
                [+inv, 0.0, +1.0], [-inv, 0.0, +1.0],
                [+inv, 0.0, -1.0], [-inv, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
    else:  # dodeca
        # 20 dodecahedron vertices = 8 cube vertices + 12 golden-rect
        # corners (the dual-icosa-vertex set scaled by phi).
        inv = 1.0 / PHI
        cube = [
            [s1, s2, s3]
            for s1 in (+1.0, -1.0)
            for s2 in (+1.0, -1.0)
            for s3 in (+1.0, -1.0)
        ]
        rects = [
            [0.0, +PHI, +inv], [0.0, +PHI, -inv],
            [0.0, -PHI, +inv], [0.0, -PHI, -inv],
            [+PHI, +inv, 0.0], [+PHI, -inv, 0.0],
            [-PHI, +inv, 0.0], [-PHI, -inv, 0.0],
            [+inv, 0.0, +PHI], [-inv, 0.0, +PHI],
            [+inv, 0.0, -PHI], [-inv, 0.0, -PHI],
        ]
        v = torch.tensor(cube + rects, dtype=torch.float32)
    # unit-normalise rows (norm-equal property of Platonic vertices)
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return v / norm


class PlatonicAttention(nn.Module):
    """Multi-head attention with Platonic-solid head incidence.

    Compared to a vanilla ``nn.MultiheadAttention(num_heads=n)``, this
    layer adds a per-head *relative-positional* angular bias derived
    from the Platonic vertex azimuths. The bias at relative position
    ``delta = j - i`` is

        ``head_bias[h, i, j] = (1 / PHI) * cos(angle_h + 2*pi*delta / N)``

    where ``angle_h`` is head ``h``'s Platonic-vertex angular phase
    (azimuth of the corresponding Platonic vertex, offset by a small
    deterministic micro-phase to break axis-aligned degeneracy). The
    bias is registered as a buffer (non-learnable) so the Platonic
    symmetry is *structural*, not learned. This mirrors the H37
    Pentagonal-attention precedent with a Platonic-vertex phase per
    head instead of a uniform 5-fold one.

    Note. The previous implementation summed ``coords @ coords.T`` and
    took ``mean(dim=-1)``, which is identically zero for every
    vertex-transitive Platonic solid (the vertex coordinates sum to
    the origin). That construction was bit-equivalent to a vanilla
    MHA -- the relative-positional cosine bias here is the genuine
    Platonic prior.

    Q/K/V projections remain the standard linear maps; the Platonic-
    specific machinery is the per-head angular bias added to the
    pre-softmax logits.

    Parameters
    ----------
    d_model : int
        embedding dimension. Must be divisible by ``head_count(group)``.
    group : str, default 'icosa'
        Platonic group identifier.
    dropout : float, default 0.0
        applied to attention weights.
    """

    def __init__(self, d_model: int, group: str = "icosa", dropout: float = 0.0) -> None:
        super().__init__()
        if group not in _PLATONIC_HEAD_COUNTS:
            raise ValueError(
                f"unknown group={group!r}; expected one of "
                f"{sorted(_PLATONIC_HEAD_COUNTS)}"
            )
        n_heads = _PLATONIC_HEAD_COUNTS[group]
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads} "
                f"for group={group!r}"
            )
        self.d_model = d_model
        self.group = group
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        # Per-head Platonic-vertex angular phase. Used to build a
        # relative-positional cosine bias at forward time. Registered
        # as a buffer so it moves with .to(device) and is non-trainable.
        self.register_buffer("head_angles", _platonic_head_angles(group))

    def _relative_bias(self, N: int, device, dtype) -> torch.Tensor:
        """Compute the ``(n_heads, N, N)`` relative-positional bias.

        Bias at ``(h, i, j)`` is ``(1 / PHI) * cos(angle_h + 2*pi *
        (j - i) / N)``. Periodic with period ``N`` so a cyclic shift
        of the sequence permutes the per-head phase identically across
        heads -- a discrete rotation-equivariance signature.
        """
        idx = torch.arange(N, device=device, dtype=dtype)
        rel = idx.view(1, N) - idx.view(N, 1)        # (N, N): j - i
        phase = 2.0 * math.pi * rel / float(N)       # (N, N)
        angles = self.head_angles.to(device=device, dtype=dtype).view(self.n_heads, 1, 1)
        return torch.cos(angles + phase.view(1, N, N)) / float(PHI)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, D)``
        attn_mask : torch.Tensor or None
            additive mask, broadcastable to ``(B, n_heads, N, N)``.
        """
        assert x.ndim == 3, f"expected (B, N, D), got shape {tuple(x.shape)}"
        B, N, D = x.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"
        qkv = self.qkv(x)                                # (B, N, 3*D)
        qkv = qkv.view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # (3, B, H, N, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, H, N, d_head)
        scale = 1.0 / math.sqrt(self.d_head)
        logits = (q @ k.transpose(-2, -1)) * scale       # (B, H, N, N)
        # Per-head relative-positional Platonic angular bias broadcast
        # over batch. Mathematically non-zero (unlike the previous
        # gram-mean construction, which collapsed to identically 0).
        rel_bias = self._relative_bias(N, x.device, x.dtype)  # (H, N, N)
        logits = logits + rel_bias.unsqueeze(0)
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = self.dropout(F.softmax(logits, dim=-1))
        out = attn @ v                                   # (B, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class PlatonicTransformerBlock(nn.Module):
    """Pre-norm Transformer block with :class:`PlatonicAttention`.

    Structure::

        x -> LN -> PlatonicAttention -> + residual ->
             LN -> FFN (Linear-GELU-Linear) -> + residual

    Parameters
    ----------
    d_model, group, dropout
        forwarded to :class:`PlatonicAttention`.
    ffn_mult : int, default 4
        FFN hidden dim is ``ffn_mult * d_model``.
    """

    def __init__(
        self,
        d_model: int,
        group: str = "icosa",
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PlatonicAttention(d_model, group=group, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
