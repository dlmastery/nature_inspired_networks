"""Phi-decay learning-rate and momentum schedulers (H10, H48).

H10: PhiDecayLR — drop-in replacement for
``torch.optim.lr_scheduler.CosineAnnealingLR`` with multiplicative
phi-decay per step.

H48: GoldenMomentumScheduler — multiplicative phi-decay of an
Adam-family optimizer's β1 (or SGD's momentum) with a 1/φ² floor.

Refs:
  Loshchilov, Hutter 2017 ICLR 'SGDR: Stochastic Gradient Descent with
    Warm Restarts' (arXiv:1608.03983)
  Smith 2017 WACV 'Cyclical Learning Rates' (arXiv:1506.01186)
  Jaeger 2020 'Echo State Networks at the Edge of Chaos'
    (arXiv:2006.04751) — edge-of-chaos β = 1/φ² motivation.
  Sutskever et al. 2013 ICML 'On the Importance of Initialization and
    Momentum in Deep Learning' — variable-momentum schedules.
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from .priors import PHI


class PhiDecayLR(_LRScheduler):
    """LR scheduler with multiplicative phi-decay per step.

    Args:
        optimizer: any ``torch.optim.Optimizer``.
        T_max: decay-unit length. After ``T_max`` steps the LR is
            divided by exactly ``phi``. With ``T_max=1`` the schedule is
            the raw ``base_lr * phi^{-k}`` rule from H10 sec. 5.1.
        lr_floor: hard lower bound on the produced LR (avoids
            denormals at long horizons; default 1e-6 per H10 sec. 5.1).
        last_epoch: standard ``_LRScheduler`` resume hook.

    Notes:
        Unlike cosine annealing this schedule has **no warmup** and
        **no eta_min cliff** — early steps emit ``base_lr`` directly
        (the H10 ergonomic claim) and late steps asymptote at
        ``lr_floor``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int = 1,
        lr_floor: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        if lr_floor < 0:
            raise ValueError(f"lr_floor must be >= 0, got {lr_floor}")
        self.T_max = int(T_max)
        self.lr_floor = float(lr_floor)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        # ``self.last_epoch`` is the step index (0 on the first call).
        k = max(0, self.last_epoch)
        factor = PHI ** (-k / float(self.T_max))
        return [max(base_lr * factor, self.lr_floor) for base_lr in self.base_lrs]
