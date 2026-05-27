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


# ---------------------------------------------------------------------------
# H48 — Golden Momentum Scheduler
# ---------------------------------------------------------------------------
GOLDEN_MOMENTUM_INIT: float = 1.0 / PHI            # ≈ 0.6180339887
GOLDEN_MOMENTUM_FLOOR: float = 1.0 / (PHI ** 2)    # ≈ 0.3819660113


class GoldenMomentumScheduler:
    """Multiplicative φ-decay scheduler for Adam β1 (or SGD momentum).

    On each :meth:`step` call the current β1 (or momentum) is updated
    via ``new = max(floor, current * (1/φ))``. After enough steps the
    value asymptotes to ``floor`` and further calls are no-ops.

    The scheduler is stateless on the optimizer side: it reads/writes
    ``param_groups[i]['betas'][0]`` for Adam-family optimizers, or
    ``param_groups[i]['momentum']`` for SGD. Detection is automatic.

    Parameters
    ----------
    optimizer
        Any :class:`torch.optim.Optimizer`. Adam-family optimizers
        (whose ``param_groups`` carry ``'betas'``) use ``betas[0]``;
        otherwise ``'momentum'`` is updated.
    floor
        Lower bound that β1 / momentum will not be decayed below.
        Defaults to ``1/φ² ≈ 0.382`` (H48 design-doc asymptote).
    init
        Initial β1 / momentum value seeded at construction time. If
        ``None`` the optimizer's current value is preserved; otherwise
        every param-group is overwritten so the schedule starts cleanly
        from the φ-derived initial point.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        floor: float = GOLDEN_MOMENTUM_FLOOR,
        init: float | None = GOLDEN_MOMENTUM_INIT,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer; got "
                f"{type(optimizer)!r}"
            )
        self.optimizer = optimizer
        self.floor = float(floor)
        # Detect the field name (betas[0] for Adam-family, momentum for SGD).
        self._uses_betas = "betas" in optimizer.param_groups[0]
        if init is not None:
            self._write(float(init))
        self._step_count = 0
        self._history: list[float] = [self._read()]

    def _read(self) -> float:
        """Return the current first-moment coefficient (β1 / momentum)."""
        g = self.optimizer.param_groups[0]
        if self._uses_betas:
            return float(g["betas"][0])
        return float(g.get("momentum", 0.0))

    def _write(self, value: float) -> None:
        """Set β1 (or momentum) on every param_group to ``value``."""
        for g in self.optimizer.param_groups:
            if self._uses_betas:
                b1, b2 = g["betas"]
                g["betas"] = (float(value), b2)
            else:
                g["momentum"] = float(value)

    def step(self) -> float:
        """Apply one φ-decay step; returns the new β1 / momentum."""
        cur = self._read()
        new = max(self.floor, cur / PHI)
        self._write(new)
        self._step_count += 1
        self._history.append(new)
        return new

    def get_last(self) -> float:
        """Most recent β1 / momentum value (post-step)."""
        return self._history[-1]

    def state_dict(self) -> dict:  # pragma: no cover — convenience
        return {
            "floor": self.floor,
            "step_count": self._step_count,
            "history": list(self._history),
            "uses_betas": self._uses_betas,
        }

    def load_state_dict(self, state: dict) -> None:  # pragma: no cover
        self.floor = float(state["floor"])
        self._step_count = int(state["step_count"])
        self._history = list(state["history"])
        self._uses_betas = bool(state.get("uses_betas", self._uses_betas))


def golden_momentum_curve(epochs: int, tau: float = 5.0) -> list[float]:
    """Reference H48 closed-form curve:

        β1(e) = 1/φ² + (1/φ - 1/φ²) · exp(-e / tau)

    Returns a length-``epochs`` list of float β1 values. Not used by
    :class:`GoldenMomentumScheduler` (which is multiplicative) but kept
    here for plotting / regression tests against the design-doc curve.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    asymp = GOLDEN_MOMENTUM_FLOOR
    span = GOLDEN_MOMENTUM_INIT - GOLDEN_MOMENTUM_FLOOR
    return [asymp + span * math.exp(-e / tau) for e in range(epochs)]
