"""ViT-Tiny model for the Control 4 (H71 IcosaRoPE3D) smoke.

Minimal Vision Transformer Tiny re-implementation sized for the
reviewer-flagged H71 control sweep (controls/PLAN.md item 4).

Architectural choices (documented per the directive)
-----------------------------------------------------

We choose **embed_dim=198, num_heads=6, head_dim=33** rather than the
canonical ViT-Tiny (embed_dim=192, num_heads=3, head_dim=64) because
H71's :class:`IcosaRoPE3D` rotates triples of channels via Rodrigues'
formula and therefore requires ``head_dim % 3 == 0``:

  - 64 % 3 = 1     -- FAILS (the canonical configuration).
  - 33 % 3 = 0     -- OK (6 heads x 33 = 198 embed_dim).
  - 48 % 3 = 0     -- OK alternative (4 heads x 48 = 192).

The 6 x 33 = 198 option preserves heads=6 (closer to canonical
multi-head budget) at the cost of a 3.1 % wider embed_dim (198 vs 192);
the 4 x 48 alternative preserves embed_dim=192 at the cost of fewer
heads. The control YAML pins 6 x 33 = 198, which is what we build here.

References (Citation Rigor):

  Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner,
  Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby 2021 ICLR
  'An Image is Worth 16x16 Words: Transformers for Image Recognition
  at Scale' (arXiv:2010.11929) -- ViT base architecture.

  Touvron, Cord, Douze, Massa, Sablayrolles, Jegou 2021 ICML
  'Training data-efficient image transformers & distillation through
  attention' (arXiv:2012.12877) -- ViT-Tiny / DeiT-Tiny canonical
  config.

  Su, Lu, Pan, Murtadha, Wen, Liu 2021 'RoFormer' (arXiv:2104.09864)
  -- 1-D RoPE baseline.

  Cohen, Weiler, Kicanaoglu, Welling 2019 ICML 'Icosahedral CNN'
  (arXiv:1902.04615) -- icosahedral group structure used by H71.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_icosa_rope import IcosaRoPE3D


__all__ = [
    "ViTTiny",
    "build_vit_tiny",
]


def _apply_rope1d(q: torch.Tensor, k: torch.Tensor,
                  positions: torch.Tensor,
                  base: float = 10_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard 1-D rotary positional embedding (RoFormer).

    Rotates pairs of channels by position-dependent angles. The first
    ``2 * (D // 2)`` channels are split into real / imag halves; any
    trailing channel (when ``D`` is odd) passes through untouched.

    The pass-through tail is essential for Control 4: the reviewer-
    flagged head-to-head compares rope1d vs icosa3d at THE SAME ViT
    architecture. icosa3d requires ``head_dim % 3 == 0``; the chosen
    Control 4 head_dim=33 (=6 x 33 = 198 embed) is odd, so the 1-D
    RoPE rotates 32 of the 33 channels and leaves channel 32 alone.
    The single un-rotated channel is a deterministic implementation
    artifact, not a defect -- the H71 design doc allows asymmetric
    encoders and the equivariance benchmark sums over all heads.
    """
    B, H, N, D = q.shape
    even_d = (D // 2) * 2
    if even_d == 0:
        return q, k
    half = even_d // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=q.dtype, device=q.device) / half)
    )
    pos = positions.to(dtype=q.dtype, device=q.device)
    angles = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (N, half)
    cos = angles.cos().view(1, 1, N, half)
    sin = angles.sin().view(1, 1, N, half)

    def _rot(t: torch.Tensor) -> torch.Tensor:
        # Split into rotated even-D prefix + un-rotated odd-tail.
        t_rot = t[..., :even_d]
        t_tail = t[..., even_d:]
        t_real, t_imag = t_rot[..., :half], t_rot[..., half:]
        out_rot = torch.cat(
            [t_real * cos - t_imag * sin, t_real * sin + t_imag * cos],
            dim=-1,
        )
        return torch.cat([out_rot, t_tail], dim=-1)

    return _rot(q), _rot(k)


class _MultiHeadSelfAttention(nn.Module):
    """MHSA with pluggable RoPE: ``rope_kind`` in {"none", "rope1d", "icosa3d"}."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        rope_kind: Literal["none", "rope1d", "icosa3d"] = "none",
        rope_base: float = 10_000.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by "
                    f"num_heads ({num_heads}) when head_dim is None"
                )
            head_dim = embed_dim // num_heads
        if num_heads * head_dim != embed_dim:
            raise ValueError(
                f"num_heads ({num_heads}) * head_dim ({head_dim}) "
                f"must equal embed_dim ({embed_dim})"
            )
        if rope_kind == "icosa3d" and head_dim % 3 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 3 for "
                f"icosa3d RoPE (Rodrigues triple-rotation)"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_kind = rope_kind
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.icosa_rope: IcosaRoPE3D | None = None
        if rope_kind == "icosa3d":
            # IcosaRoPE3D's design-doc default is base=PHI (geometric
            # decay across the 12 icosa axes); we forward the caller's
            # rope_base only if they explicitly override the rope1d
            # default (10_000.0). This keeps the H71 "icosa with PHI"
            # contract intact while still allowing a deliberate sweep
            # over base.
            icosa_base = rope_base if rope_base != 10_000.0 else None
            self.icosa_rope = (
                IcosaRoPE3D(head_dim=head_dim, base=icosa_base)
                if icosa_base is not None
                else IcosaRoPE3D(head_dim=head_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(dim=0)
        if self.rope_kind != "none":
            positions = torch.arange(N, device=x.device, dtype=q.dtype)
            if self.rope_kind == "rope1d":
                q, k = _apply_rope1d(q, k, positions, base=self.rope_base)
            else:  # icosa3d
                assert self.icosa_rope is not None
                q, k = self.icosa_rope(q, k, positions)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class _MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(round(embed_dim * mlp_ratio))
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int,
                 mlp_ratio: float, rope_kind: str, rope_base: float,
                 dropout: float = 0.0, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim,
            rope_kind=rope_kind, rope_base=rope_base,
            attn_dropout=attn_dropout, proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLPBlock(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """ViT-Tiny for CIFAR-shaped inputs.

    Parameters
    ----------
    img_size : int, default 32
        Input spatial size. With ``patch_size=4`` the network sees
        ``(img_size / patch_size) ** 2`` tokens (64 at 32x32 / patch 4).
    patch_size : int, default 4
        Non-overlapping patch side. 4 is the canonical CIFAR-scaled
        choice (so 32x32 → 8x8 = 64 patches).
    in_chans : int, default 3
        Input channels.
    num_classes : int, default 10
    embed_dim : int, default 198
        Token / channel dimension. 198 = 6 heads x head_dim 33 so
        ``head_dim % 3 == 0`` (Control 4 requirement for IcosaRoPE3D).
    num_heads : int, default 6
    head_dim : int, default 33
        ``head_dim % 3 == 0`` is enforced when ``rope_kind='icosa3d'``.
    depth : int, default 12
    mlp_ratio : float, default 4.0
    rope_kind : {'none', 'rope1d', 'icosa3d'}, default 'rope1d'
        Position-encoding mechanism inside each attention block. The
        Control 4 reference row uses ``'rope1d'``; the H71 headline row
        uses ``'icosa3d'``.
    rope_base : float, default 10000.0
        Base of the per-pair frequency decay for 1-D RoPE. The
        IcosaRoPE3D module uses its own default (``PHI``) when
        ``rope_kind='icosa3d'`` -- the value here is only consulted
        for the 1-D RoPE path.
    dropout, attn_dropout : float, default 0.0
        Standard ViT residual / attention dropouts.
    use_cls_token : bool, default True
        Prepend a learnable CLS token used by the classifier head.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 198,
        num_heads: int = 6,
        head_dim: int = 33,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        rope_kind: Literal["none", "rope1d", "icosa3d"] = "rope1d",
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        # head_dim % 3 == 0 is mandatory for icosa3d RoPE, and we enforce
        # it unconditionally so a future swap to icosa3d does not silently
        # mis-align the rotor (Rule 25 -- Q&A test correspondence: the
        # docstring above asserts this constraint).
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if num_heads * head_dim != embed_dim:
            raise ValueError(
                f"num_heads * head_dim ({num_heads * head_dim}) must "
                f"equal embed_dim ({embed_dim})"
            )
        if head_dim % 3 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 3 -- "
                f"required for IcosaRoPE3D triple-rotation; use "
                f"6 heads x 33 (=198) or 4 heads x 48 (=192)"
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be a multiple of "
                f"patch_size ({patch_size})"
            )

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.depth = int(depth)
        self.use_cls_token = bool(use_cls_token)
        self.rope_kind = rope_kind

        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            n_tokens = n_patches + 1
        else:
            self.cls_token = None
            n_tokens = n_patches
        # Learnable position embedding (additive to the patch tokens).
        # Used together with RoPE per the standard ViT recipe; the RoPE
        # only injects relative position into Q/K, so the additive PE
        # is what carries the absolute CLS-vs-patch distinction.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads,
                head_dim=head_dim, mlp_ratio=mlp_ratio,
                rope_kind=rope_kind, rope_base=rope_base,
                dropout=dropout, attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Patch embed: (B, embed_dim, H/P, W/P) → (B, N, embed_dim).
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # CLS-token (or global-avg) head.
        if self.cls_token is not None:
            head_token = x[:, 0]
        else:
            head_token = x.mean(dim=1)
        return self.fc(head_token)


def build_vit_tiny(num_classes: int = 10,
                   embed_dim: int = 198,
                   num_heads: int = 6,
                   head_dim: int = 33,
                   depth: int = 12,
                   patch_size: int = 4,
                   img_size: int = 32,
                   mlp_ratio: float = 4.0,
                   rope_kind: str = "rope1d",
                   rope_base: float = 10_000.0) -> ViTTiny:
    """Thin factory used by :func:`models.build_model` to construct a
    ViT-Tiny for Control 4 (the H71 IcosaRoPE3D smoke).

    All kwargs accept the cfg-key names emitted by
    ``controls/configs/control4_h71_vit_tiny.yaml``.
    """
    return ViTTiny(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        rope_kind=rope_kind,  # type: ignore[arg-type]
        rope_base=rope_base,
    )
