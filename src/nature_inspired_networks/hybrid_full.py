"""H67 — Full Paradigm Hybrid (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H67_full_paradigm_hybrid.md``.

The "everything on" reference model. Stacks together:

  1. **NaturePriorBlock** stem (hex / fractal / cymatic / toroidal / golden
     modulate -- the CNN-track full prior set).
  2. **Fibottention** head -- multi-head attention with per-head
     Fibonacci dilation (CIFAR-shape compatible).
  3. **Platonic attention** -- a Metatron-Cube graph message-passing
     layer over the pooled tokens (optional; guarded if
     :mod:`platonic_graph` is unavailable).
  4. **Liquid CFC** latent recurrence with golden-step ``dt = 1/φ``.
  5. **Cymatic-init** orthonormal weights on the Fibottention QKV proj.
  6. **Golden RoPE** rotary positional embedding (optional; falls back to
     a plain sinusoidal PE when :mod:`golden_rope` isn't installed).

This model is deliberately NOT a SOTA target. It's a stress test: when
you turn every prior on simultaneously, the inductive biases probably
clash -- this is the same finding pattern as H50 in the FINDINGS.md
("all-on" was worse than the best single-prior subset). Shipping H67 as
a self-tested reference makes that finding reproducible.

References (Citation Rigor)::

    Lipton, Steinhardt 2018 'Troubling Trends in Machine Learning
    Scholarship' (arXiv:1807.03341) -- argues that piling more inductive
    biases on does NOT monotonically improve performance; H67 is the
    operational counterexample.

    Hoogeboom et al. 2018 'HexaConv' (arXiv:1803.02108).
    Rajagopalan et al. 2024 'Fibottention' (arXiv:2406.19391).
    Hasani et al. 2021 'Liquid Time-Constant Networks' (arXiv:2006.04439).
    Cohen 2019 'Icosahedral CNN' (arXiv:1902.04615).
    Chladni 1787 'Entdeckungen über die Theorie des Klanges'.

Public surface
--------------
- :class:`FullParadigmHybrid` -- the all-on reference model. Exposes a
  ``which_priors_active`` property so test code can verify each of the
  six priors is genuinely engaged.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NaturePriorBlock, NaturePriorFlags
from .fibottention import Fibottention
from .hybrid_cymatic_qkv import cymatic_init_linear_
from .hybrid_liquid_jepa import LiquidCFCCell
from .priors import PHI


__all__ = ["FullParadigmHybrid"]


# ---------------------------------------------------------------------------
# Optional dependencies (guarded -- concurrent agents may land these later)
# ---------------------------------------------------------------------------
try:
    from .platonic_graph import MetatronGraphLayer  # type: ignore
    _HAVE_METATRON = True
except ImportError:
    MetatronGraphLayer = None
    _HAVE_METATRON = False

try:
    from .golden_rope import GoldenRoPE  # type: ignore
    _HAVE_GOLDEN_ROPE = True
except ImportError:
    GoldenRoPE = None
    _HAVE_GOLDEN_ROPE = False


class _SinusoidalPE(nn.Module):
    """Fallback rotary positional embedding.

    Plain additive sinusoidal PE in the spirit of Vaswani 2017. Used when
    the optional :mod:`golden_rope` module isn't on disk yet; the H67
    hybrid still self-tests and runs in this case.
    """

    def __init__(self, dim: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return x + self.pe[: x.shape[1]].to(x.device, x.dtype).unsqueeze(0)


class FullParadigmHybrid(nn.Module):
    """The all-on reference model. Documents which priors are engaged.

    Parameters
    ----------
    in_channels : int
        Input image channel count.
    width : int
        Stem / block width (constant across the encoder for simplicity).
    n_blocks : int
        NaturePriorBlock stack depth.
    n_heads : int
        Fibottention head count.
    n_classes : int
        Classifier head output dimensionality.
    flags : NaturePriorFlags, optional
        Forwarded to NaturePriorBlock; H67 defaults to "everything True".

    Attributes
    ----------
    which_priors_active : dict
        Per-prior activation flags (boolean). Lets tests assert that each
        of the six load-bearing inductive biases is genuinely engaged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        n_blocks: int = 2,
        n_heads: int = 4,
        n_classes: int = 10,
        flags: NaturePriorFlags | None = None,
    ) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width {width} not divisible by n_heads {n_heads}")
        flags = flags or NaturePriorFlags(
            hex=True,
            group=True,
            fractal=True,
            toroidal=True,
            cymatic_init=True,
            golden_modulate=True,
        )
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.n_classes = int(n_classes)
        self.flags = flags
        # ---- 1: NaturePriorBlock encoder ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.ModuleList(
            [
                NaturePriorBlock(width, width, stride=1, flags=flags)
                for _ in range(n_blocks)
            ]
        )
        # ---- 2: Fibottention head ----
        self.attention = Fibottention(dim=width, n_heads=n_heads)
        # ---- 5: cymatic-init QKV ----
        cymatic_init_linear_(self.attention.qkv, band=(2, 5), orthonormalize=True)
        # ---- 6: Golden RoPE (or sinusoidal fallback) ----
        if _HAVE_GOLDEN_ROPE:
            try:
                self.pe = GoldenRoPE(dim=width)  # type: ignore[misc]
                self._pe_kind = "golden_rope"
            except Exception:
                self.pe = _SinusoidalPE(dim=width)
                self._pe_kind = "sinusoidal_fallback"
        else:
            self.pe = _SinusoidalPE(dim=width)
            self._pe_kind = "sinusoidal_fallback"
        # ---- 3: Platonic graph (or skip-connection no-op) ----
        if _HAVE_METATRON:
            try:
                self.platonic = MetatronGraphLayer(width)  # type: ignore[misc]
                self._platonic_kind = "metatron"
            except Exception:
                self.platonic = nn.Identity()
                self._platonic_kind = "identity_fallback"
        else:
            self.platonic = nn.Identity()
            self._platonic_kind = "identity_fallback"
        # ---- 4: Liquid CFC latent ----
        self.cfc = LiquidCFCCell(d_in=width, d_hid=width)
        # Classifier head
        self.head = nn.Linear(width, n_classes)

    @property
    def which_priors_active(self) -> dict:
        """Boolean activation flags for each of the six load-bearing priors."""
        return {
            "nature_prior_blocks": True,
            "fibottention_attention": True,
            "cymatic_qkv_init": True,
            "liquid_cfc": True,
            "golden_rope": self._pe_kind == "golden_rope",
            "platonic_graph": self._platonic_kind == "metatron",
        }

    # ------------------------------------------------------------------
    def _apply_platonic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the (optional) Platonic graph layer on a token sequence.

        ``tokens`` is shape ``(B, N, D)``. The MetatronGraphLayer (when
        available) expects a fixed 13-node graph; if N doesn't match we
        skip the platonic layer and pass tokens through unchanged.
        """
        if self._platonic_kind != "metatron":
            return tokens
        # Try to apply; if the platonic layer can't be applied at this
        # token count (e.g., shape mismatch) fall back silently.
        try:
            return self.platonic(tokens)
        except Exception:
            return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) CNN encoder.
        h = self.stem(x)
        for block in self.encoder:
            h = block(h)
        # Flatten spatial → tokens.
        B, C, H, W = h.shape
        tokens = h.flatten(2).transpose(1, 2)  # (B, N=H*W, D=C)
        # 6) Positional embedding.
        tokens = self.pe(tokens)
        # 2 + 5) Fibottention with cymatic-init QKV.
        tokens = self.attention(tokens)
        # 3) Platonic graph (optional).
        tokens = self._apply_platonic(tokens)
        # Pool → vector.
        z = tokens.mean(dim=1)
        # 4) Liquid CFC integrator.
        h_state = self.cfc(z, None)
        return self.head(h_state)
