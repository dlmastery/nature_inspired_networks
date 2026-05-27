"""H68 — On-Device World Model (G7 cross-paradigm hybrid).

Design doc: ``hypotheses/g7_cross_paradigm_hybrids/H68_ondevice_world_model.md``.

Thin glue layer combining three already-implemented inductive biases into a
single JEPA-style world-model module:

  1. A **small Fibonacci-channel encoder** (G1: φ/Fibonacci scaling).
  2. **Golden-skip residuals** (G2: ``GoldenSkipBlock`` with 1/φ skip scale).
  3. **Drop-path regularization** (G5: optional ``drop_path`` import, with a
     local fallback when the optional module is absent).

The recurrent predictor maps the current latent ``z_t`` to a prediction of
``z_{t+1}`` (a JEPA-style next-latent objective). All shape, parameter and
forward semantics are exercised by ``tests/test_hybrid_world_model.py``.

References (Citation Rigor)::

    Bardes, Garrido, Ponce, Chen, Ballas, LeCun 2024 'V-JEPA: Revisiting
    Feature Prediction' (arXiv:2404.08471) -- the next-latent objective.

    Larsson, Maire, Shakhnarovich 2017 ICLR 'FractalNet' (arXiv:1605.07648)
    -- drop-path regularisation we re-use here.

    Hayou, Doucet, Rousseau 2021 ICML 'Stable ResNet' -- the per-skip
    scalar that ``GoldenSkipBlock`` instantiates with 1/φ.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI, fibonacci_channels
from .phi_scaling import GoldenSkipBlock


# Optional import — other agents are landing concurrently, so guard.
try:
    from .drop_path import DropPath as _DropPath  # type: ignore
    _HAS_DROP_PATH = True
except Exception:  # pragma: no cover - optional import landing
    _HAS_DROP_PATH = False


class _FallbackDropPath(nn.Module):
    """Minimal stochastic-depth drop-path fallback used when the optional
    :mod:`nature_inspired_networks.drop_path` module is not yet importable.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= p < 1.0, f"drop_prob must be in [0, 1); got {p}"
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


def _make_drop_path(p: float) -> nn.Module:
    if _HAS_DROP_PATH:
        return _DropPath(p=p)  # type: ignore[name-defined]
    return _FallbackDropPath(p=p)


__all__ = [
    "OnDeviceWorldModel",
    "world_model_jepa_loss",
]


class OnDeviceWorldModel(nn.Module):
    """Small on-device JEPA-style world model.

    Architecture
    ------------
    1. Encoder: 3-stage CNN whose stage widths follow
       :func:`fibonacci_channels(c0, 3, mode='fib')`. Each stage is a
       :class:`GoldenSkipBlock` (1/φ skip scaling) wrapped in drop-path.
    2. Global-average pool → linear projection to a ``d_latent`` latent
       vector ``z``.
    3. Latent predictor: a single-layer GRU cell that maps ``z_t`` to
       ``ẑ_{t+1}``.

    Forward modes
    -------------
    * ``encode(x)``        → ``(B, d_latent)`` latent.
    * ``predict(z, h=None)`` → ``(ẑ_{t+1}, h_{t+1})``.
    * ``forward(seq)``     → ``(z_seq, z_pred_seq)`` for a sequence input
      of shape ``(B, T, C, H, W)``. ``z_pred_seq[..., t]`` is the predicted
      next-latent given the current ``z_seq[..., t]``; the natural target
      is ``z_seq[..., t+1].detach()``.

    Parameters
    ----------
    in_channels : int
        Input channel count (e.g. 3 for RGB frames).
    c0 : int, default 8
        Base channel count of the Fibonacci schedule. Stage widths are
        ``fibonacci_channels(c0, 3, mode='fib')``.
    d_latent : int, default 32
        Latent dimensionality.
    drop_path : float, default 0.1
        Stochastic-depth drop probability per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        c0: int = 8,
        d_latent: int = 32,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        assert in_channels >= 1
        assert c0 >= 4
        assert d_latent >= 1
        widths = fibonacci_channels(c0, 3, mode="fib")
        self.widths = widths
        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(widths[0])

        # Stage 0: width preservation (no spatial down)
        self.stage0 = GoldenSkipBlock(widths[0], widths[0], stride=1)
        # Stage 1: widen + downsample 2x
        self.stage1 = GoldenSkipBlock(widths[0], widths[1], stride=2)
        # Stage 2: widen + downsample 2x
        self.stage2 = GoldenSkipBlock(widths[1], widths[2], stride=2)
        self.dp0 = _make_drop_path(drop_path)
        self.dp1 = _make_drop_path(drop_path)
        self.dp2 = _make_drop_path(drop_path)

        self.proj = nn.Linear(widths[-1], d_latent)
        self.d_latent = d_latent

        # JEPA-style next-latent predictor — a GRUCell maps z_t (and a
        # hidden state) to ẑ_{t+1}. We use a single recurrent step so the
        # surface stays minimal.
        self.predictor = nn.GRUCell(d_latent, d_latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` → ``(B, d_latent)``."""
        assert x.dim() == 4, f"encode expects 4-D input; got {tuple(x.shape)}"
        h = F.relu(self.bn_stem(self.stem(x)), inplace=True)
        h = self.dp0(self.stage0(h))
        h = self.dp1(self.stage1(h))
        h = self.dp2(self.stage2(h))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)  # (B, widths[-1])
        return self.proj(h)

    def predict(
        self,
        z: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict ẑ_{t+1} from z_t and an optional hidden state.

        Returns ``(z_pred, h_new)`` where ``z_pred`` and ``h_new`` are both
        ``(B, d_latent)``. ``z_pred = h_new`` for the GRUCell-based
        predictor (we return both for API parity with multi-layer
        predictors).
        """
        assert z.dim() == 2 and z.shape[-1] == self.d_latent, (
            f"predict expects (B, {self.d_latent}); got {tuple(z.shape)}"
        )
        if h is None:
            h = torch.zeros_like(z)
        h_new = self.predictor(z, h)
        return h_new, h_new

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence forward.

        Parameters
        ----------
        frames : torch.Tensor
            Shape ``(B, T, C, H, W)`` with ``T >= 2``.

        Returns
        -------
        z_seq : torch.Tensor, ``(B, T, d_latent)``
            Per-frame encoded latents.
        z_pred_seq : torch.Tensor, ``(B, T - 1, d_latent)``
            Predicted next-latent for ``t = 0, ..., T - 2``. The natural
            JEPA target is ``z_seq[:, 1:].detach()``.
        """
        assert frames.dim() == 5, (
            f"forward expects (B, T, C, H, W); got {tuple(frames.shape)}"
        )
        B, T, C, H, W = frames.shape
        assert T >= 2, f"need at least 2 timesteps; got T={T}"
        z_list = []
        for t in range(T):
            z_list.append(self.encode(frames[:, t]))
        z_seq = torch.stack(z_list, dim=1)  # (B, T, d)

        # Roll the GRU predictor forward across the (T-1) input latents.
        h = torch.zeros(B, self.d_latent, dtype=z_seq.dtype, device=z_seq.device)
        preds = []
        for t in range(T - 1):
            h = self.predictor(z_seq[:, t], h)
            preds.append(h)
        z_pred_seq = torch.stack(preds, dim=1)  # (B, T-1, d)
        return z_seq, z_pred_seq


def world_model_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """JEPA-style cosine-distance loss with target stop-grad.

    ``1 - mean(cos(z_pred, z_target.detach()))`` averaged across the
    leading dimensions. Shape-mismatch is rejected with ``ValueError``.
    """
    if z_pred.shape != z_target.shape:
        raise ValueError(
            f"shape mismatch: z_pred {tuple(z_pred.shape)} vs "
            f"z_target {tuple(z_target.shape)}"
        )
    z_t = z_target.detach()
    cos = F.cosine_similarity(z_pred, z_t, dim=-1)
    return 1.0 - cos.mean()
