"""φ-derived optimizers (H41 Golden Ratio Optimizer).

This module hosts optimizer primitives whose hyperparameters are derived
from the golden ratio φ rather than from Adam's empirical magic numbers
(0.9, 0.999, 1e-8). The aim is to swap-in a single dimensionless
natural constant for three independently-tuned floats and ablate the
training trajectory.

H41 GoldenRatioAdamW
--------------------
Subclass of :class:`torch.optim.AdamW` that overrides the constructor
defaults with φ-derived values:

    β1 = 1/φ     ≈ 0.6180339887  (gradient-EMA, was 0.9)
    β2 = 1/φ²    ≈ 0.3819660113  (squared-gradient-EMA, was 0.999)
    eps = 1/φ⁴  ≈ 0.1458980338  (denominator floor, was 1e-8)

The β-derivation follows Jaeger 2020 (arXiv:2006.04751) — see H41 design
doc for the full edge-of-chaos motivation. The eps choice is **not**
load-bearing for the hypothesis (Adam's eps is mostly a safety floor);
it is set to a φ-derived value purely so the optimizer has a single
dimensionless source. Callers wanting to test β alone can pass
``eps=1e-8`` explicitly to recover Adam's eps.

The optimizer is otherwise the stock AdamW step (same code path); only
the defaults differ. Wired via ``optimizer: 'golden_adam'`` in the run
config.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW

from .priors import PHI


# Pre-computed φ-derived defaults (module-level constants so tests can
# assert the exact value rather than re-deriving). All four are pure
# functions of PHI — no other free parameter enters.
GOLDEN_BETA1: float = 1.0 / PHI            # ≈ 0.6180339887
GOLDEN_BETA2: float = 1.0 / (PHI ** 2)     # ≈ 0.3819660113
GOLDEN_EPS: float = 1.0 / (PHI ** 4)       # ≈ 0.1458980338


class GoldenRatioAdamW(AdamW):
    """AdamW with φ-derived (β1, β2, eps) defaults.

    Drop-in replacement for :class:`torch.optim.AdamW`. Only the default
    values of ``betas`` and ``eps`` change; the optimizer step is the
    inherited AdamW implementation, so any subsequent torch upgrade
    (fused / foreach kernels) is picked up for free.

    Parameters mirror :class:`torch.optim.AdamW`. Callers may override
    ``betas`` or ``eps`` explicitly to recover stock AdamW (useful for
    H41 ablation control runs).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (GOLDEN_BETA1, GOLDEN_BETA2),
        eps: float = GOLDEN_EPS,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    def __repr__(self) -> str:  # pragma: no cover — diagnostic only
        return (
            f"GoldenRatioAdamW(lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, eps={self.defaults['eps']})"
        )


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Factory used by the runner to select between AdamW and
    GoldenRatioAdamW from a single config kwarg.

    Wired via ``optimizer: 'golden_adam'`` (or ``'adamw'`` for the
    classical baseline). Unknown names raise :class:`ValueError` so the
    sweep fails loudly rather than silently degrading.
    """
    n = (name or "adamw").lower()
    if n in {"golden_adam", "golden_adamw", "phi_adam", "phi_adamw"}:
        return GoldenRatioAdamW(params, lr=lr, weight_decay=weight_decay)
    if n in {"adamw", "adam_w"}:
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(
        f"unknown optimizer '{name}'; expected one of "
        f"{{'adamw', 'golden_adam'}}"
    )
