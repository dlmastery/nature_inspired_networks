"""φ-derived optimizers (H41 Golden Ratio Optimizer).

This module hosts optimizer primitives whose hyperparameters are derived
from the golden ratio φ rather than from Adam's empirical magic numbers
(0.9, 0.999, 1e-8). The aim is to swap-in a single dimensionless
natural constant for three independently-tuned floats and ablate the
training trajectory.

H41 GoldenRatioAdamW
--------------------
Subclass of :class:`torch.optim.AdamW` that overrides the constructor
defaults with φ-derived values for the β coefficients:

    β1 = 1/φ     ≈ 0.6180339887  (gradient-EMA, was 0.9)
    β2 = 1/φ²    ≈ 0.3819660113  (squared-gradient-EMA, was 0.999)

**eps is intentionally NOT φ-derived.** The original release defaulted
``eps`` to ``1/φ⁴ ≈ 0.1459`` which — at CIFAR-scale gradient magnitudes
of ~1e-3 — dominates the Adam denominator and effectively scales the
learning rate by ~6.85×. This conflated the β-only hypothesis with a
massive implicit LR change and was the dominant cause of the -32.82 pp
CIFAR-10 collapse reported in the H41 falsification. The default is
now the standard ``eps = 1e-8`` so the β experiment is clean. Callers
who want the original (φ-derived) eps for backward-compat reproducibility
must opt in explicitly via ``phi_eps=True`` (which sets eps = 1/φ⁴) or
pass ``eps=`` directly.

The β-derivation follows Jaeger 2020 (arXiv:2006.04751) — see H41
design doc for the full edge-of-chaos motivation.

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
# assert the exact value rather than re-deriving). β1 and β2 are pure
# functions of PHI. GOLDEN_EPS is exported only for callers opting into
# the phi_eps=True backward-compat path; the default eps for
# GoldenRatioAdamW is now the standard Adam eps (1e-8), see __init__
# docstring for the rationale.
GOLDEN_BETA1: float = 1.0 / PHI            # ≈ 0.6180339887
GOLDEN_BETA2: float = 1.0 / (PHI ** 2)     # ≈ 0.3819660113
GOLDEN_EPS: float = 1.0 / (PHI ** 4)       # ≈ 0.1458980338 — opt-in only
STOCK_ADAM_EPS: float = 1e-8


class GoldenRatioAdamW(AdamW):
    """AdamW with φ-derived (β1, β2) defaults; eps stays at the Adam stock.

    Drop-in replacement for :class:`torch.optim.AdamW`. Only the default
    values of ``betas`` change; the optimizer step is the inherited
    AdamW implementation, so any subsequent torch upgrade (fused /
    foreach kernels) is picked up for free.

    eps is intentionally NOT φ-derived; the original ``eps = 1/φ⁴`` was
    the dominant cause of the -32.82 pp CIFAR-10 collapse reported for
    H41, conflating the β-change with an implicit ~6.85× LR change. The
    default is now ``eps = 1e-8`` so the β-only experiment is clean.

    Parameters
    ----------
    params, lr, betas, eps, weight_decay, amsgrad
        Mirror :class:`torch.optim.AdamW`.
    phi_eps
        Opt-in flag for backward-compat reproducibility. When ``True``,
        and ``eps`` was not explicitly overridden by the caller, the
        eps default is set to ``1/φ⁴`` (the legacy buggy value). Callers
        running the original H41 falsification rerun should set this.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (GOLDEN_BETA1, GOLDEN_BETA2),
        eps: float = STOCK_ADAM_EPS,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        phi_eps: bool = False,
    ) -> None:
        # If the caller asks for phi_eps=True but did not explicitly set
        # eps, swap in the φ-derived value. Explicit eps always wins.
        if phi_eps and eps == STOCK_ADAM_EPS:
            eps = GOLDEN_EPS
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
    eps: float | None = None,
    phi_eps: bool = False,
) -> torch.optim.Optimizer:
    """Factory used by the runner to select between AdamW and
    GoldenRatioAdamW from a single config kwarg.

    Wired via ``optimizer: 'golden_adam'`` (or ``'adamw'`` for the
    classical baseline). Unknown names raise :class:`ValueError` so the
    sweep fails loudly rather than silently degrading.

    Parameters
    ----------
    eps
        Optional explicit eps. If ``None``, the optimizer's own default
        is used (1e-8 for AdamW; ``STOCK_ADAM_EPS`` for GoldenRatioAdamW
        unless ``phi_eps=True``).
    phi_eps
        For GoldenRatioAdamW only: opt-in to the legacy ``eps=1/φ⁴``
        behaviour for backward-compat reproducibility.
    """
    n = (name or "adamw").lower()
    if n in {"golden_adam", "golden_adamw", "phi_adam", "phi_adamw"}:
        kwargs: dict = dict(lr=lr, weight_decay=weight_decay, phi_eps=phi_eps)
        if eps is not None:
            kwargs["eps"] = eps
        return GoldenRatioAdamW(params, **kwargs)
    if n in {"adamw", "adam_w"}:
        kwargs = dict(lr=lr, weight_decay=weight_decay)
        if eps is not None:
            kwargs["eps"] = eps
        return AdamW(params, **kwargs)
    raise ValueError(
        f"unknown optimizer '{name}'; expected one of "
        f"{{'adamw', 'golden_adam'}}"
    )
