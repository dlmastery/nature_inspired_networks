"""Phi-decay learning-rate and momentum schedulers (H10, H48).

H10: PhiDecayLR — drop-in replacement for
``torch.optim.lr_scheduler.CosineAnnealingLR`` with multiplicative
phi-decay per step.

H48: GoldenMomentumScheduler — closed-form exponential decay (default)
or multiplicative phi-decay (legacy) of an Adam-family optimizer's β1
(or SGD's momentum) with a 1/φ² floor.

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
from typing import Literal

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
    """φ-decay scheduler for Adam β1 (or SGD momentum).

    Two decay modes are supported:

    * ``mode='exponential'`` (default, PAPER_GAP_G5-correct) — closed-
      form ``β(e) = floor + span · exp(-e/τ)`` where ``span = init -
      floor`` and ``τ = total_epochs / 3`` by default (or
      ``τ=4.0`` if ``total_epochs`` is not supplied — legacy fallback).
      ``β(0) = init = 1/φ``; ``β(∞) → floor = 1/φ²``. The schedule is
      advanced by :meth:`set_epoch` (Trainer hook) — :meth:`step`
      delegates to ``set_epoch(self._step_count + 1)`` so both APIs
      work.
    * ``mode='multiplicative'`` (legacy / gentle-φ) — each :meth:`step`
      multiplies the current β1 by ``φ^(-1/T_max)``. Over ``T_max``
      steps the cumulative multiplier equals ``1/φ`` (so β drops from
      ``1/φ`` to ``1/φ²`` end-to-end). Retained for reproducing
      pre-PAPER_GAP_G5 numbers.

    Rationale for the mode switch
    -----------------------------
    The ORIGINAL release applied a hard ``× 1/φ`` per epoch, which
    saturated to the ``1/φ²`` floor after a SINGLE epoch (because
    ``(1/φ) × (1/φ) = 1/φ²``), making β1 effectively constant from
    epoch 2 onward — a one-shot overwrite, not a smooth curriculum.
    The G5 audit fix introduced the gentler multiplicative
    ``φ^(-1/T_max)`` schedule; the PAPER_GAP_G5 audit then replaced
    that with the closed-form exponential decay so the curve is
    monotonically smooth and reaches the floor only as ``e → ∞``,
    matching the design-doc reference curve in
    :func:`golden_momentum_curve`.

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
    T_max
        Legacy multiplicative knob: number of :meth:`step` invocations
        over which the cumulative decay should equal ``1/φ`` (only used
        when ``mode='multiplicative'``). Must be >= 1.
    mode
        ``'exponential'`` (default) or ``'multiplicative'``.
    total_epochs
        Used only in ``'exponential'`` mode to derive the default time
        constant ``τ = total_epochs / 3``. If ``None`` the constructor
        falls back to ``tau=4.0`` (legacy reference curve).
    tau
        Optional explicit override for the exponential time constant.
        When set, overrides the ``total_epochs / 3`` default.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        floor: float = GOLDEN_MOMENTUM_FLOOR,
        init: float | None = GOLDEN_MOMENTUM_INIT,
        T_max: int = 30,
        mode: Literal["exponential", "multiplicative"] = "exponential",
        total_epochs: int | None = None,
        tau: float | None = None,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer; got "
                f"{type(optimizer)!r}"
            )
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        if mode not in {"exponential", "multiplicative"}:
            raise ValueError(
                f"mode must be 'exponential' or 'multiplicative', "
                f"got {mode!r}"
            )
        self.optimizer = optimizer
        self.floor = float(floor)
        self.T_max = int(T_max)
        self.mode = mode
        # Initial value remembered so set_epoch() can reset deterministically
        # from any epoch index (the exponential curve is closed-form).
        self._init = (
            float(init) if init is not None else None
        )
        # Per-step multiplicative factor (only used in multiplicative mode).
        # Over T_max steps this composes to exactly 1/φ.
        self._decay = PHI ** (-1.0 / float(T_max))
        # Exponential-mode time constant.
        if tau is not None:
            self._tau = float(tau)
        elif total_epochs is not None and int(total_epochs) > 0:
            self._tau = float(total_epochs) / 3.0
        else:
            self._tau = 4.0  # legacy reference (matches golden_momentum_curve)
        self.total_epochs = (
            int(total_epochs) if total_epochs is not None else None
        )
        # Detect the field name (betas[0] for Adam-family, momentum for SGD).
        self._uses_betas = "betas" in optimizer.param_groups[0]
        # Resolve the effective initial value once (defaults to optimizer's
        # current value if init was None) — needed for exponential closed
        # form to be reset-from-epoch-0.
        if init is None:
            self._init = self._read()
        else:
            self._write(float(init))
            self._init = float(init)
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

    def _exponential_beta(self, epoch: int) -> float:
        """Closed-form β(e) = floor + span · exp(-e / tau)."""
        e = max(0, int(epoch))
        span = float(self._init) - self.floor
        # span may be negative if a caller seeded init < floor; clamp.
        beta = self.floor + span * math.exp(-e / self._tau)
        return max(self.floor, beta)

    def set_epoch(self, epoch: int) -> float:
        """Set the schedule to epoch ``epoch`` (exponential mode).

        In ``'exponential'`` mode this re-derives β from the closed form
        β(epoch) = floor + span · exp(-epoch/τ) and writes it to the
        optimizer. In ``'multiplicative'`` mode this falls back to
        repeated multiplicative :meth:`step` calls (the legacy path).
        Returns the new β1 / momentum.
        """
        if self.mode == "exponential":
            new = self._exponential_beta(epoch)
            self._write(new)
            self._step_count = int(max(0, epoch))
            self._history.append(new)
            return new
        # Legacy multiplicative: re-derive from init by composing the
        # per-step factor ``epoch`` times.
        decayed = float(self._init) * (self._decay ** max(0, int(epoch)))
        new = max(self.floor, decayed)
        self._write(new)
        self._step_count = int(max(0, epoch))
        self._history.append(new)
        return new

    def step(self) -> float:
        """Advance the schedule by one epoch; returns the new β1.

        In exponential mode this calls ``set_epoch(self._step_count + 1)``.
        In multiplicative mode this applies the per-step decay factor.
        """
        if self.mode == "exponential":
            return self.set_epoch(self._step_count + 1)
        # Legacy multiplicative path.
        cur = self._read()
        new = max(self.floor, cur * self._decay)
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
            "T_max": self.T_max,
            "mode": self.mode,
            "tau": self._tau,
            "init": self._init,
            "total_epochs": self.total_epochs,
        }

    def load_state_dict(self, state: dict) -> None:  # pragma: no cover
        self.floor = float(state["floor"])
        self._step_count = int(state["step_count"])
        self._history = list(state["history"])
        self._uses_betas = bool(state.get("uses_betas", self._uses_betas))
        if "T_max" in state:
            self.T_max = int(state["T_max"])
            self._decay = PHI ** (-1.0 / float(self.T_max))
        if "mode" in state:
            self.mode = state["mode"]
        if "tau" in state:
            self._tau = float(state["tau"])
        if "init" in state and state["init"] is not None:
            self._init = float(state["init"])
        if "total_epochs" in state:
            self.total_epochs = state["total_epochs"]


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
