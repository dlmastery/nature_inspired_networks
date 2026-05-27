"""H19 — phi-Neuron Activation Threshold (PhiReLU + PhiAdaptiveReLU).

Drop-in ReLU replacements whose firing threshold is tuned by the local
phi-ratio. Two complementary formulations are provided:

* :class:`PhiReLU` — *static-init* per-channel learnable threshold
  ``tau`` initialised at ``1/phi ~= 0.618`` (cortical-interneuron
  threshold proxy, He 2015 / Trottier 2017 family). Forward:
  ``out = max(0, x - tau)``.

* :class:`PhiAdaptiveReLU` — *fully dynamic* phi-thresholded ReLU per
  the assignment hint ``out = where(x > phi * mean(x), x, 0)``. The
  threshold tracks the per-channel running mean via an exponential
  moving average so ``tau_c = phi * mu_c``. No learnable parameters;
  the threshold floats with the activation distribution and can
  surpass ``1/phi`` for high-magnitude channels.

Both modules accept 2/3/4-D tensors and broadcast the per-channel
threshold over the non-channel axes. The ``num_channels`` argument is
the channel-dimension size.

This module lives in its own file (``phi_threshold.py``) rather than
``activations.py`` to avoid touching the existing H39 :class:`PhiGELU`
implementation. It re-exports nothing from there.

Wire-in: imports register a ``"natureprior_phi_relu"`` model variant
with :func:`build_model` so the sweep row ``sg_only_phi_relu`` resolves
through the standard runner. The variant is a NaturePriorNet whose
stem ReLU is replaced with :class:`PhiReLU`; intra-block ReLUs remain
standard (Rule 1 atomicity).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


PHI_RECIPROCAL: float = 1.0 / PHI  # 0.6180339...


class PhiReLU(nn.Module):
    """Per-channel learnable-threshold ReLU.

    The threshold ``tau`` is a learnable :class:`~torch.nn.Parameter`
    of shape ``(num_channels,)`` initialised to ``init``
    (default ``1/phi``). The forward pass broadcasts ``tau`` over the
    spatial / temporal axes based on the input rank:

    * 2-D input ``(B, C)``: ``tau`` broadcasts as ``(1, C)``.
    * 3-D input ``(B, T, C)``: ``tau`` broadcasts as ``(1, 1, C)``.
    * 4-D input ``(B, C, H, W)``: ``tau`` broadcasts as
      ``(1, C, 1, 1)`` — the canonical conv-NCHW layout.

    The output is ``F.relu(x - tau)``; gradients flow through ``tau``.
    """

    def __init__(self, num_channels: int,
                 init: float = PHI_RECIPROCAL) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1; got {num_channels}")
        self.num_channels = int(num_channels)
        self.init = float(init)
        self.tau = nn.Parameter(torch.full((num_channels,), float(init)))

    def _broadcast(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.tau.view(1, -1)
        if x.ndim == 3:
            # (B, T, C) — channel-last sequence layout
            return self.tau.view(1, 1, -1)
        if x.ndim == 4:
            return self.tau.view(1, -1, 1, 1)
        if x.ndim == 5:
            return self.tau.view(1, -1, 1, 1, 1)
        raise ValueError(
            f"PhiReLU expects 2/3/4/5-D input; got {x.ndim}-D"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self._broadcast(x)
        return F.relu(x - tau)

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, init={self.init:.6f}"


class PhiAdaptiveReLU(nn.Module):
    """Fully-dynamic phi-thresholded ReLU.

    Implements the per-channel rule

        ``out = where(x > phi * mu_c, x, 0)``

    where ``mu_c`` is the per-channel running mean tracked via an
    exponential moving average (momentum ``ema_momentum``). At
    inference time the EMA buffer is frozen.

    In training mode the per-batch channel mean is used to update the
    EMA; the threshold applied to the forward pass is
    ``phi * batch_mean``. In eval mode the EMA buffer is used directly.

    No learnable parameters — the threshold is non-parametric. The
    output is gated by a binary mask (``x > tau``) so gradients flow
    only through the surviving entries (a hard ReLU-style gate).
    """

    def __init__(self, num_channels: int, phi: float = PHI,
                 ema_momentum: float = 0.1,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1; got {num_channels}")
        if not (0.0 < ema_momentum <= 1.0):
            raise ValueError(
                f"ema_momentum must be in (0, 1]; got {ema_momentum}"
            )
        self.num_channels = int(num_channels)
        self.phi = float(phi)
        self.ema_momentum = float(ema_momentum)
        self.eps = float(eps)
        # Running per-channel mean (frozen at eval time, like BatchNorm).
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer(
            "num_batches_tracked", torch.zeros(1, dtype=torch.long)
        )

    @staticmethod
    def _channel_axis(ndim: int) -> int:
        if ndim == 2:
            return 1
        if ndim == 3:
            return 2  # (B, T, C)
        if ndim in (4, 5):
            return 1  # (B, C, H, W[, D])
        raise ValueError(
            f"PhiAdaptiveReLU expects 2/3/4/5-D input; got {ndim}-D"
        )

    def _reduce_dims(self, x: torch.Tensor) -> tuple[int, ...]:
        c_axis = self._channel_axis(x.ndim)
        return tuple(d for d in range(x.ndim) if d != c_axis)

    def _broadcast(self, vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return vec.view(1, -1)
        if x.ndim == 3:
            return vec.view(1, 1, -1)
        if x.ndim == 4:
            return vec.view(1, -1, 1, 1)
        return vec.view(1, -1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            reduce_dims = self._reduce_dims(x)
            batch_mean = x.detach().mean(dim=reduce_dims)
            # Update EMA in-place
            m = self.ema_momentum
            self.running_mean.mul_(1 - m).add_(batch_mean, alpha=m)
            self.num_batches_tracked += 1
            mu = batch_mean
        else:
            mu = self.running_mean
        tau = self._broadcast(self.phi * mu, x)
        # Hard gate: keep x where x > tau, else 0.
        return torch.where(x > tau, x, torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Drop-in NaturePriorNet variant with PhiReLU stem activation
# ---------------------------------------------------------------------------
class PhiReLUNaturePriorNet(nn.Module):
    """NaturePriorNet whose stem ReLU is replaced with :class:`PhiReLU`.

    Per Rule 1 (one config change per experiment) only the stem
    activation is swapped — intra-block ReLUs remain standard so this
    isolates the activation-threshold prior from the geometry priors.
    """

    def __init__(self, num_classes: int = 10, channel_mode: str = "fib",
                 flags=None) -> None:
        super().__init__()
        from .blocks import NaturePriorFlags
        from .models import NaturePriorConfig, NaturePriorNet
        cfg = NaturePriorConfig(
            num_classes=num_classes, channel_mode=channel_mode,
            flags=flags or NaturePriorFlags(),
        )
        base = NaturePriorNet(cfg)
        self.cfg = base.cfg
        self.widths = base.widths
        stem_conv: nn.Conv2d = base.stem[0]  # type: ignore[assignment]
        stem_bn: nn.BatchNorm2d = base.stem[1]  # type: ignore[assignment]
        self.stem = nn.Sequential(
            stem_conv,
            stem_bn,
            PhiReLU(num_channels=stem_bn.num_features),
        )
        self.stages = base.stages
        self.pool = base.pool
        self.fc = base.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# build_model registration
# ---------------------------------------------------------------------------
def _register_phi_relu_variant() -> None:
    from . import models as _models
    from . import runner as _runner

    original = _models.build_model
    if getattr(original, "_phi_relu_wrapped", False):
        return

    def build_model(name: str, num_classes: int, flags=None,
                    channel_mode: str = "fib"):
        if name.lower() in {"natureprior_phi_relu", "phi_relu_natureprior"}:
            return PhiReLUNaturePriorNet(
                num_classes=num_classes, channel_mode=channel_mode,
                flags=flags,
            )
        return original(name, num_classes, flags=flags,
                        channel_mode=channel_mode)

    build_model._phi_relu_wrapped = True  # type: ignore[attr-defined]
    build_model._original = original  # type: ignore[attr-defined]
    _models.build_model = build_model  # type: ignore[assignment]
    _runner.build_model = build_model  # type: ignore[assignment]


_register_phi_relu_variant()
