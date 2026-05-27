"""H61 — Sacred Liquid JEPA Hybrid (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H61_sacred_liquid_jepa.md``.

Thin glue layer combining three already-implemented inductive biases into a
single decoder block:

  1. A **NaturePriorBlock** stack as the spatial encoder
     (CNN-track priors: hex / fractal / cymatic init / golden modulation).
  2. A **Liquid-Time-Constant** cell as the latent recurrence — a learnable
     damped-oscillator with φ-spaced integration step ``dt = 1/φ``.
  3. A **JEPA-style predictor** head that maps the recurrent latent to the
     *next* latent embedding; the training objective is MSE between the
     predicted and (detached) target latent.

The module is content-agnostic per Rule 14: it imports the encoder from
``nature_inspired_networks.blocks`` and the LTC cell + JEPA head live here.
This is the *reference implementation* of the H61 hypothesis — no training
loop, no dataset, just the model surface and a JEPA loss helper.

References (Citation Rigor)::

    Hasani, Lechner, Amini, Rus, Grosu 2021 AAAI 'Liquid Time-Constant
    Networks' (arXiv:2006.04439) -- the continuous-time recurrence we
    instantiate with φ-spaced dt.

    Bardes, Garrido, Ponce, Chen, Ballas, LeCun 2024 'Revisiting Feature
    Prediction for Learning Visual Representations from Video' (V-JEPA,
    arXiv:2404.08471) -- the joint-embedding predictive architecture
    pattern whose loss we adopt verbatim.

    LeCun, Assran, Ballas, Bardes 2025 arXiv 'seq-JEPA / Sequential JEPA'
    (arXiv:2506.09985) -- the causal sequential variant of JEPA.

Public surface
--------------
- :class:`LiquidCFCCell` — simplified Liquid CFC recurrence with φ step.
- :class:`SacredLiquidJEPA` — encoder + CFC + JEPA predictor.
- :func:`jepa_loss` — MSE between predicted and (detached) target latent.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NaturePriorBlock, NaturePriorFlags
from .priors import PHI


__all__ = [
    "LiquidCFCCell",
    "SacredLiquidJEPA",
    "jepa_loss",
]


# ---------------------------------------------------------------------------
# Liquid Time-Constant cell (simplified CFC variant)
# ---------------------------------------------------------------------------
class LiquidCFCCell(nn.Module):
    """Simplified Liquid Closed-Form-Continuous (CFC) recurrence cell.

    Implements the damped-oscillator update::

        h_{t+1} = h_t + dt * (-h_t / tau + W * tanh(x_t) + b)

    with ``dt = 1 / PHI`` by default (the H61 sacred-geometric integration
    step). ``tau`` is a learnable per-channel time constant (positive via
    softplus parameterisation) and ``W`` is a learnable linear map from the
    input dimension to the hidden dimension.

    This is the minimal LTC cell that is differentiable and stateful; it
    deliberately omits the rich neuromodulation of the full CFC paper so it
    plays nicely with the existing single-GPU 4090 budget.
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        dt: float | None = None,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_hid <= 0:
            raise ValueError(f"d_in={d_in}, d_hid={d_hid} must be positive")
        self.d_in = int(d_in)
        self.d_hid = int(d_hid)
        # dt defaults to the H61 sacred 1/φ ≈ 0.618 step.
        self.dt = float(dt) if dt is not None else 1.0 / PHI
        self.W = nn.Linear(d_in, d_hid, bias=True)
        # Softplus-parameterise tau so it stays positive without clamping.
        # Initial raw_tau = 0 → softplus(0) ≈ 0.693, a sensible default.
        self.raw_tau = nn.Parameter(torch.zeros(d_hid))

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + 1e-3  # strictly positive

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One CFC step.

        Parameters
        ----------
        x : Tensor of shape ``(B, d_in)``
            Input activations at the current time step.
        h : Tensor of shape ``(B, d_hid)``, optional
            Previous hidden state. ``None`` → zero-init.

        Returns
        -------
        h_next : Tensor of shape ``(B, d_hid)``
        """
        if x.dim() != 2 or x.shape[-1] != self.d_in:
            raise ValueError(
                f"expected (B, {self.d_in}), got {tuple(x.shape)}"
            )
        if h is None:
            h = x.new_zeros((x.shape[0], self.d_hid))
        drive = self.W(torch.tanh(x))
        decay = -h / self.tau
        return h + self.dt * (decay + drive)


# ---------------------------------------------------------------------------
# Sacred Liquid JEPA model
# ---------------------------------------------------------------------------
class SacredLiquidJEPA(nn.Module):
    """NaturePriorBlock encoder + Liquid CFC latent + JEPA predictor head.

    Forward semantics
    -----------------
    Input is an image tensor ``(B, C, H, W)``. The encoder stack returns a
    feature map which is pooled to a vector ``z`` of dimension ``d_latent``.
    The CFC cell consumes ``z`` and updates an internal latent ``h``; the
    JEPA predictor maps ``h`` to ``z_pred``, a prediction of the *next*
    encoder output.

    During training the JEPA loss is :func:`jepa_loss(z_pred, z_target)``
    where ``z_target`` is produced by running the encoder on a paired
    (next-time-step / masked / augmented) view; here we expose ``forward``
    such that any of those pairings can be passed in.

    Parameters
    ----------
    in_channels : int
        Input channel count (3 for CIFAR-10).
    d_latent : int
        Encoder pooled feature dimension and CFC hidden width.
    n_blocks : int
        Number of NaturePriorBlock instances in the encoder stack.
    flags : NaturePriorFlags, optional
        Forwarded to every NaturePriorBlock for ablation control.
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_latent: int = 64,
        n_blocks: int = 2,
        flags: NaturePriorFlags | None = None,
    ) -> None:
        super().__init__()
        flags = flags or NaturePriorFlags()
        self.in_channels = int(in_channels)
        self.d_latent = int(d_latent)
        self.flags = flags

        # Stem: lift in_channels → d_latent
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_latent, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_latent),
            nn.ReLU(inplace=True),
        )
        # Encoder: NaturePriorBlock stack at constant width (so JEPA latent
        # has a well-defined embedding dim regardless of n_blocks).
        self.encoder = nn.ModuleList(
            [
                NaturePriorBlock(d_latent, d_latent, stride=1, flags=flags)
                for _ in range(n_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # CFC latent recurrence — dt = 1/φ by default.
        self.cfc = LiquidCFCCell(d_latent, d_latent)
        # JEPA predictor — small MLP, BN-free per V-JEPA practice.
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward → pooled feature vector ``(B, d_latent)``."""
        h = self.stem(x)
        for block in self.encoder:
            h = block(h)
        return self.pool(h).flatten(1)  # (B, d_latent)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder → CFC → predictor.

        Returns
        -------
        z : Tensor (B, d_latent)
            The encoder output for ``x``.
        h : Tensor (B, d_latent)
            The updated CFC latent state.
        z_pred : Tensor (B, d_latent)
            JEPA prediction of the *next* latent.
        """
        z = self.encode(x)
        h = self.cfc(z, h_prev)
        z_pred = self.predictor(h)
        return z, h, z_pred


# ---------------------------------------------------------------------------
# JEPA loss
# ---------------------------------------------------------------------------
def jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and (stop-grad) target latent.

    Per Bardes et al. 2024 V-JEPA, the target is detached from the
    computation graph so the encoder is *only* updated by gradients
    flowing through ``z_pred``.
    """
    if z_pred.shape != z_target.shape:
        raise ValueError(
            f"shape mismatch: z_pred {tuple(z_pred.shape)} vs z_target {tuple(z_target.shape)}"
        )
    return F.mse_loss(z_pred, z_target.detach())
