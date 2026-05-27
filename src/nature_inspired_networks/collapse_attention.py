"""H83 — CollapseGatedAttention.

Multi-head self-attention with a LEARNABLE softmax temperature ``tau`` that
interpolates between diffuse (high-tau, "wave") and peaked (low-tau,
"particle") attention, plus a forward-time ``collapse`` knob that can
additionally sharpen the attention toward its argmax.

Neutral framing
---------------
Temperature- and sparsity-controlled attention is a well-studied, real
technique: dividing the logits by a temperature before softmax controls how
peaked the distribution is (entropy regularisation; sparsemax/entmax replace
softmax with sparsity-controlled maps entirely). Making the temperature a
learnable positive scalar (via softplus) lets each attention module discover
how concentrated its attention should be. The esoteric origin (the double-slit
"observer collapse" from wave to particle) is acknowledged only as the source
intuition for naming the high-/low-temperature regimes; the operational object
is standard scaled-dot-product attention with a learnable temperature.

Class
-----
- :class:`CollapseGatedAttention` — ``nn.Module`` taking ``(B, N, D)`` →
  ``(B, N, D)``; softmax over ``QKᵀ / (sqrt(d_head) * tau)`` with learnable
  positive ``tau`` and an optional ``collapse`` interpolation toward argmax.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseGatedAttention(nn.Module):
    """Multi-head attention with a learnable softmax temperature.

    Parameters
    ----------
    embed_dim : int
        Model / embedding dimension ``D``. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    tau_init : float
        Initial softmax temperature (must be ``> 0``). ``1.0`` reproduces
        standard scaled-dot-product attention at init. The temperature is
        stored as a raw parameter passed through ``softplus`` so the effective
        ``tau`` is always strictly positive. Low ``tau`` → peaked ("particle")
        attention; high ``tau`` → diffuse ("wave") attention.
    bias : bool
        Whether the q/k/v/out projections carry a bias.

    Notes
    -----
    The effective temperature is ``tau = softplus(tau_raw)``; ``tau_raw`` is
    initialised so that ``softplus(tau_raw) == tau_init`` at construction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        tau_init: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert tau_init > 0, "tau_init must be positive"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Invert softplus so softplus(tau_raw) == tau_init:
        #   softplus(z) = log(1 + e^z)  =>  z = log(e^tau - 1).
        tau_raw = math.log(math.expm1(tau_init))
        self.tau_raw = nn.Parameter(torch.tensor(float(tau_raw)))

    # Numerical floor on the effective temperature so 1/(sqrt(d)*tau) cannot
    # overflow to inf (which would make the softmax produce NaNs) when tau_raw
    # is driven strongly negative during training.
    _TAU_FLOOR = 1e-3

    @property
    def tau(self) -> torch.Tensor:
        """The effective, strictly-positive softmax temperature (floored)."""
        return F.softplus(self.tau_raw) + self._TAU_FLOOR

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        collapse: float = 0.0,
        need_weights: bool = False,
    ):
        """Self-attention over ``x`` of shape ``(B, N, D)``.

        Parameters
        ----------
        x : (B, N, D) tensor
        attn_mask : optional additive mask broadcastable to (B, H, N, N) or
            (N, N); added to the scaled logits before softmax (use ``-inf`` for
            disallowed positions, e.g. a causal mask).
        collapse : float in [0, 1]
            Additional sharpening toward the argmax applied AFTER the softmax:
            ``attn <- (1 - collapse) * attn + collapse * onehot(argmax(attn))``.
            ``0`` (default) leaves the temperature-softmax untouched (full
            "wave"); ``1`` fully collapses each row onto its argmax ("particle").
        need_weights : bool
            If ``True``, also return the (B, H, N, N) attention probabilities.
        """
        B, N, D = x.shape
        H, dh = self.num_heads, self.head_dim

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, H, dh).transpose(1, 2)  # (B, H, N, dh)

        q = reshape(self.q_proj(x))
        k = reshape(self.k_proj(x))
        v = reshape(self.v_proj(x))

        tau = F.softplus(self.tau_raw) + self._TAU_FLOOR
        scale = 1.0 / (math.sqrt(dh) * tau)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = torch.softmax(logits, dim=-1)

        if collapse > 0.0:
            assert 0.0 <= collapse <= 1.0, "collapse must be in [0, 1]"
            idx = attn.argmax(dim=-1, keepdim=True)
            onehot = torch.zeros_like(attn).scatter_(-1, idx, 1.0)
            attn = (1.0 - collapse) * attn + collapse * onehot

        out = torch.matmul(attn, v)              # (B, H, N, dh)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        if need_weights:
            return out, attn
        return out


# TODO runner wiring:
# To integrate without touching models.py / blocks.py here, the lead adds a
# `flags: {collapse_attention: true, tau_init: 1.0}` branch that swaps the
# standard `nn.MultiheadAttention` (or the repo's attention block) for
# `CollapseGatedAttention(embed_dim, num_heads, tau_init=cfg.tau_init)` in each
# Transformer/ViT block. tau_init=1.0 reproduces baseline attention at step 0,
# making it a clean one-flag ablation (Rule 1). The learned `tau` per layer can
# be logged to history.json to observe which layers prefer peaked vs. diffuse
# attention. The `collapse` forward arg is left at 0 during training; a separate
# eval row can sweep collapse in {0, 0.5, 1.0} to test inference-time sharpening.
