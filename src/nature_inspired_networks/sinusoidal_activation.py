"""H81 — SinusoidalHarmonicActivation.

A SIREN-style sinusoidal activation ``sin(omega * x)`` with a learnable
frequency ``omega``, plus a helper that walks a model swapping every
``nn.ReLU`` for it.

Neutral framing
---------------
Sinusoidal activations are a real, well-cited tool: Sitzmann et al. 2020
(SIREN) showed that periodic activations let networks represent signals and
their derivatives far more faithfully than ReLU/Tanh, with the per-layer
frequency ``omega`` (often initialised to 30 for the first layer) being the
key hyper-parameter. The esoteric "vibration / harmonic" framing is
acknowledged only as the source intuition; the operational object is exactly
the SIREN activation with a *learnable* frequency.

Classes / functions
--------------------
- :class:`SinusoidalActivation` — ``sin(omega * x)`` with learnable ``omega``
  (scalar or per-channel) initialised to ``omega_init`` (default 1.0; SIREN
  uses 30.0 for the first layer).
- :func:`swap_relu_with_sine(model, omega_init=1.0)` — recursively replace
  every ``nn.ReLU`` module in ``model`` with a fresh ``SinusoidalActivation``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalActivation(nn.Module):
    """Periodic activation ``sin(omega * x)`` with a learnable frequency.

    Parameters
    ----------
    omega_init : float
        Initial frequency. ``1.0`` is a conservative default that begins close
        to the identity for small inputs (``sin(x) ≈ x`` near 0); SIREN's
        canonical first-layer value is ``30.0``.
    num_channels : int | None
        If ``None`` (default), ``omega`` is a single learnable scalar shared
        across all units. If an int ``C`` is given, ``omega`` is a per-channel
        vector of length ``C`` applied along ``dim`` (broadcasts over the
        remaining dims), so each feature map / feature has its own frequency.
    dim : int
        Channel dimension along which a per-channel ``omega`` is applied.
        Default ``1`` (the conv ``C`` axis of ``(B, C, H, W)``; also the
        feature axis of ``(B, C)``). Ignored when ``num_channels is None``.
    learnable : bool
        If ``True`` (default), ``omega`` is an ``nn.Parameter``; if ``False``
        it is a fixed buffer (the original SIREN treats ``omega`` as a fixed
        hyper-parameter — this flag recovers that behaviour for ablation).
    """

    def __init__(
        self,
        omega_init: float = 1.0,
        num_channels: int | None = None,
        dim: int = 1,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.dim = dim
        self.learnable = learnable
        if num_channels is None:
            init = torch.tensor(float(omega_init))
        else:
            init = torch.full((num_channels,), float(omega_init))
        if learnable:
            self.omega = nn.Parameter(init)
        else:
            self.register_buffer("omega", init)

    def _omega_view(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_channels is None:
            return self.omega
        # Reshape per-channel omega so it broadcasts along `dim`.
        shape = [1] * x.dim()
        shape[self.dim] = self.num_channels
        return self.omega.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self._omega_view(x) * x)


def swap_relu_with_sine(
    model: nn.Module,
    omega_init: float = 1.0,
    learnable: bool = True,
) -> nn.Module:
    """Recursively replace every ``nn.ReLU`` in ``model`` with a
    :class:`SinusoidalActivation`.

    Operates in place and also returns ``model`` for convenience. A fresh
    (scalar-frequency) ``SinusoidalActivation`` is created for each replaced
    ReLU so the activations do not share a frequency parameter. Modules other
    than ``nn.ReLU`` are left untouched.

    Returns
    -------
    nn.Module
        The same ``model`` object, mutated.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(
                model, name,
                SinusoidalActivation(omega_init=omega_init, learnable=learnable),
            )
        else:
            swap_relu_with_sine(child, omega_init=omega_init, learnable=learnable)
    return model


# TODO runner wiring:
# To integrate without touching models.py / blocks.py here, the lead adds a
# `flags: {sine_activation: true, omega_init: 1.0}` branch that, after building
# the standard model, calls `swap_relu_with_sine(model, omega_init=cfg.omega_init)`
# once before training. This is a single-flag ablation (Rule 1). For the SIREN
# canonical recipe the first stem activation may instead use omega_init=30.0; a
# second sweep row flips that one number. The learned per-activation omega values
# can be logged to history.json to observe which layers prefer high vs. low
# frequency.
