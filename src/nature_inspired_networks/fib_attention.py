"""H16 — Fibonacci Head Diversity (heads grouped by Fib counts and dilations).

Design doc: ``hypotheses/g2_layer_channel_neuron/H16_fibonacci_head_diversity.md``.

A multi-head attention variant in which attention heads are *allocated*
in Fibonacci counts ``[1, 1, 2, 3, 5, 8]`` (= 20 heads), and each
group's heads share a Fibonacci dilation drawn from ``[1, 2, 3, 5, 8,
13]`` -- i.e. one head at dilation 1, one at dilation 2, two at
dilation 3, three at dilation 5, five at dilation 8, eight at dilation
13. The hypothesis is that this matches the natural-system optimum for
multi-frequency cortical attention (the visual / auditory band-
allocation rule, H16 sec. 1) and reduces head redundancy relative to
the uniform-12-head ViT-Tiny baseline.

This is the *complement* of the existing H32 ``fibottention`` module:
H32 gives **each head** its own dilation; H16 instead allocates
multiple heads to each dilation with Fib counts -- the trade-off is
multiple heads at the same dilation can specialise on different sub-
patterns, while still keeping the natural-system Fib stair on the
dilation axis.

Public surface
--------------
- :func:`fib_head_counts`     canonical ``[1, 1, 2, 3, 5, 8]`` schedule.
- :func:`fib_head_dilations`  canonical ``[1, 2, 3, 5, 8, 13]`` schedule.
- :class:`FibMultiheadAttention`
                              MHA whose head allocation follows the
                              Fib counts + Fib dilations. ``fib=False``
                              falls back to uniform heads + dilation=1
                              for direct parity tests against the
                              standard ``nn.MultiheadAttention``.

References (Citation Rigor):
    Voita, Talbot, Moiseev, Sennrich, Titov 2019 ACL 'Analyzing Multi-
    Head Self-Attention' (arXiv:1905.09418) -- head-pruning motivation
    for non-uniform allocation.
    Michel, Levy, Neubig 2019 NeurIPS 'Are Sixteen Heads Really Better
    than One?' (arXiv:1905.10650) -- complementary head redundancy.
    Dosovitskiy et al 2021 ICLR 'An Image is Worth 16x16 Words: ViT'
    (arXiv:2010.11929) -- ViT-Tiny scaffold; default 12 heads at
    d_head=16, d_model=192.
    Rao, Rajagopalan, et al 2024 'Fibottention: Inceptive Visual
    Representation Learning' (arXiv:2406.19391) -- per-head Fib
    dilations; H16 reduces this to head count + dilation jointly.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI  # noqa: F401  (re-exported for docs/canonical PHI)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
def fib_head_counts() -> list[int]:
    """Return the canonical H16 Fibonacci head-count allocation.

    ``[1, 1, 2, 3, 5, 8]`` summing to 20 heads -- the design-doc
    default (sec. 5.2). The trailing 8 dominates so eight heads share
    the largest dilation (13) for long-range context, while a single
    head holds dilation 1 (the original local attention).
    """
    return [1, 1, 2, 3, 5, 8]


def fib_head_dilations() -> list[int]:
    """Return the canonical per-group dilation list ``[1, 2, 3, 5, 8, 13]``.

    Each entry is the dilation shared by all heads in the corresponding
    group from :func:`fib_head_counts`. Together the two lists encode
    the Fibonacci attention prior.
    """
    return [1, 2, 3, 5, 8, 13]


# ---------------------------------------------------------------------------
# Sparse mask builder (per-head dilated band)
# ---------------------------------------------------------------------------
def _per_head_dilation_mask(
    seq_len: int,
    head_counts: Sequence[int],
    head_dilations: Sequence[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a ``(n_heads, seq_len, seq_len)`` bool mask.

    Within each group, every head shares the same dilation. The
    per-head mask attends from query ``i`` to key ``j`` iff
    ``(j - i) % dilation == 0``. The self-position is always
    included (``i == j``).

    Returned tensor is True where attention is *kept*.
    """
    assert len(head_counts) == len(head_dilations), (
        f"head_counts and head_dilations must have same length; "
        f"got {len(head_counts)} vs {len(head_dilations)}"
    )
    n_heads = int(sum(head_counts))
    mask = torch.zeros(n_heads, seq_len, seq_len, dtype=torch.bool,
                       device=device)
    idx = torch.arange(seq_len, device=device)
    diff = idx.view(-1, 1) - idx.view(1, -1)  # (N, N)
    h = 0
    for count, d in zip(head_counts, head_dilations):
        d = max(1, int(d))
        keep = (diff % d == 0)
        for _ in range(int(count)):
            mask[h] = keep
            h += 1
    return mask


# ---------------------------------------------------------------------------
# FibMultiheadAttention
# ---------------------------------------------------------------------------
class FibMultiheadAttention(nn.Module):
    """Multi-head attention with Fibonacci head allocation + Fib dilations.

    Architecture::

        x : (B, N, D)
          -> Linear(D, 3*D)              (Q, K, V projection)
          -> split into n_heads of d_head = D / n_heads
          -> scaled dot-product attention WITH per-head dilation mask
          -> concat heads back to (B, N, D)
          -> Linear(D, D)                (output projection)

    Parameters
    ----------
    embed_dim : int
        Model width D. Must be divisible by the total head count
        ``sum(head_counts)`` -- for the default Fib allocation this is
        20, so ``embed_dim`` must be a multiple of 20.
    head_counts : sequence of int, optional
        Heads per dilation group. Defaults to :func:`fib_head_counts`
        (``[1, 1, 2, 3, 5, 8]``) so the total is 20.
    head_dilations : sequence of int, optional
        Dilation per group. Defaults to :func:`fib_head_dilations`
        (``[1, 2, 3, 5, 8, 13]``).
    fib : bool
        When True (default), use the Fib head counts and per-head
        dilations. When False, the module degenerates to uniform-head
        MHA at dilation 1, allowing a parity test against
        :class:`nn.MultiheadAttention`. The number of heads in the
        ``fib=False`` mode is taken from ``len(head_counts)`` (6 in the
        canonical schedule) so the user can still benefit from the
        Fib count divisor.
    bias : bool
        Whether QKV / output projections include bias. Default True.

    Notes
    -----
    Uses :func:`torch.nn.functional.scaled_dot_product_attention` with
    an ``attn_mask`` derived from the per-head dilation pattern (True =
    attend, False = block). Standard ViT positional encoding is the
    caller's responsibility; this module only handles the attention
    head allocation.
    """

    def __init__(
        self,
        embed_dim: int,
        head_counts: Sequence[int] | None = None,
        head_dilations: Sequence[int] | None = None,
        fib: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if head_counts is None:
            head_counts = fib_head_counts()
        if head_dilations is None:
            head_dilations = fib_head_dilations()
        if len(head_counts) != len(head_dilations):
            raise ValueError(
                f"head_counts and head_dilations must have same length; "
                f"got {len(head_counts)} vs {len(head_dilations)}"
            )
        # Non-fib mode: collapse the count allocation so each group
        # contributes a single head, dilation locked to 1.
        if not fib:
            n_heads_eff = len(head_counts)
            head_counts = [1] * n_heads_eff
            head_dilations = [1] * n_heads_eff
        else:
            n_heads_eff = int(sum(head_counts))
        if embed_dim % n_heads_eff != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by total heads "
                f"{n_heads_eff}"
            )
        self.fib = bool(fib)
        self.embed_dim = int(embed_dim)
        self.head_counts: list[int] = [int(c) for c in head_counts]
        self.head_dilations: list[int] = [int(d) for d in head_dilations]
        self.n_heads = n_heads_eff
        self.d_head = embed_dim // n_heads_eff
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        """Return the per-head bool attention mask, or None if all heads
        attend everywhere (the fib=False, all-dilation-1 case)."""
        if all(d == 1 for d in self.head_dilations):
            return None
        return _per_head_dilation_mask(
            seq_len, self.head_counts, self.head_dilations, device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward (B, N, D) -> (B, N, D).

        The QKV projection produces ``(B, N, 3*D)`` which is reshaped to
        ``(B, n_heads, N, d_head)``. The attention mask (when the
        Fibonacci dilations are active) is broadcast over the batch as
        ``(1, n_heads, N, N)``.
        """
        if x.ndim != 3:
            raise ValueError(f"FibMultiheadAttention expects (B, N, D); got {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"last dim {D} != embed_dim {self.embed_dim}")
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = self._attn_mask(N, device=x.device)
        if mask is not None:
            # SDPA accepts a bool attn_mask where True == KEEP.
            mask = mask.unsqueeze(0)  # (1, H, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Runner wiring (TODO; left for the integration pass)
# ---------------------------------------------------------------------------
# TODO runner wiring:
#   - FibMultiheadAttention is a ViT-track primitive. The CIFAR runner
#     currently builds ResNet-20 / NaturePriorNet (CNN). A ViT-Tiny
#     dispatcher (build_vit_model) needs to be added to models.py to
#     enable the H16 sweep row -- left for the integration pass.
#   - When wired, the ViT-Tiny config defaults to d_model=200 (matches
#     20 heads at d_head=10) instead of the standard 192 -- the 20-head
#     allocation requires the slight d_model bump for clean divisibility.
#     Alternatively d_model=240 keeps d_head=12.
#   - Per Rule 1 the H16 sweep is atomic: the only change vs ViT-Tiny is
#     the attention module's head allocation + dilations. Patch embed,
#     MLP, classification head all default.
