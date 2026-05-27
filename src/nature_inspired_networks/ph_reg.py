"""H54 — Persistent Homology Activation Regularization (hierarchical).

Design doc: ``hypotheses/g6_topological_bridging/H54_persistent_homology_activation_reg.md``.

H51 (:mod:`nature_inspired_networks.betti_loss`) shapes a single stage's
topology via a differentiable Betti loss. H54 extends this to a
*hierarchy* of stages: register forward hooks on every named stage of
an arbitrary model, compute the differentiable persistence surrogate
on each captured activation, and accumulate a per-stage MSE penalty
against a stage-specific target Betti tuple. The result is an auxiliary
loss the training loop can add to CE — the host model is untouched.

Public surface
--------------
- :class:`PHActivationRegularizer` — hookable ``nn.Module`` that owns a
  list of forward hooks and exposes ``loss()`` after each forward pass.
- :func:`register_ph_hooks` — convenience wrapper that instantiates a
  :class:`PHActivationRegularizer` and registers hooks against the named
  stage attributes of a model.

References (Citation Rigor)::

    Naitzat, Gregory and Zhitnikov, Andrey and Lim, Lek-Heng 2020 JMLR
    'Topology of deep neural networks' (arXiv:2004.06093) -- the
    empirical staircase observation that motivates per-stage targets.

    Gabrielsson, Rickard Bruel and Nelson, Bradley J. and Dwaraknath,
    Anjan and Skraba, Primoz 2020 AISTATS 'A Topology Layer for Machine
    Learning' (arXiv:1905.12200) -- the differentiable PH machinery
    whose surrogate we ship in betti_loss.py and reuse here.

    Hofer, Christoph and Kwitt, Roland and Niethammer, Marc 2019 ICML
    'Connectivity-Optimized Representation Learning via Persistent
    Homology' (arXiv:1906.00722) -- the precedent for connectivity-
    preserving PH losses; same idea applied per-stage.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .betti_loss import BettiLoss

__all__ = ["PHActivationRegularizer", "register_ph_hooks"]


# Default staircase target: beta_0 collapses from 2*n_classes (early raw
# clusters) down to n_classes (one cluster per class) as depth grows. We use
# n_classes=10 (CIFAR-10) at the late stage and double it at the early
# stage. β₁ targets are 0 throughout because the differentiable surrogate is
# not reliable for higher-order homology on small clouds — the registered
# defaults are conservative.
_DEFAULT_STAGE_TARGETS: Tuple[Tuple[float, float, float], ...] = (
    (20.0, 0.0, 0.0),
    (15.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
)


class PHActivationRegularizer(nn.Module):
    """Hook-driven hierarchical PH regularizer.

    Usage::

        reg = PHActivationRegularizer(stage_targets=[(20, 0, 0), (10, 0, 0)])
        reg.register(model, stage_attr_names=("stage1", "stage2"))
        # training step:
        out = model(x)
        loss = ce(out, y) + lam * reg.loss()
        loss.backward()
        reg.clear()

    The regularizer captures each stage's activation in a buffer during
    the forward pass via ``register_forward_hook``. After the forward,
    :meth:`loss` walks the buffer, runs each captured tensor through
    :class:`BettiLoss` with the matching stage target, weights by
    ``lambdas[i]``, and returns the sum.

    Parameters
    ----------
    stage_targets : Sequence[Tuple[float, float, float]]
        Target ``(beta_0, beta_1, beta_2)`` per stage. Length sets the
        expected number of registered hooks.
    lambdas : Optional[Sequence[float]]
        Per-stage weight on the MSE term. Defaults to a deepening
        schedule ``(0.05, 0.10, 0.20)`` matching the H54 design doc;
        if ``stage_targets`` is shorter / longer the schedule is
        truncated / linearly extrapolated.
    persistence_threshold : float
        Forwarded to :class:`BettiLoss`.
    max_pts : int
        Forwarded to :class:`BettiLoss` — caps the point-cloud size used
        for the per-stage persistence computation.
    """

    def __init__(
        self,
        stage_targets: Sequence[Tuple[float, float, float]] = _DEFAULT_STAGE_TARGETS,
        lambdas: Optional[Sequence[float]] = None,
        persistence_threshold: float = 0.1,
        max_pts: int = 64,
    ) -> None:
        super().__init__()
        if not stage_targets:
            raise ValueError("stage_targets must be non-empty")
        self.stage_targets: List[Tuple[float, float, float]] = [tuple(t) for t in stage_targets]
        if lambdas is None:
            # Deepening schedule per H54: deeper stages weighted higher.
            n = len(self.stage_targets)
            self.lambdas: List[float] = [0.05 + 0.15 * k / max(n - 1, 1) for k in range(n)]
        else:
            self.lambdas = [float(l) for l in lambdas]
            if len(self.lambdas) != len(self.stage_targets):
                raise ValueError(
                    f"len(lambdas)={len(self.lambdas)} must match len(stage_targets)={len(self.stage_targets)}"
                )
        self.betti_loss = BettiLoss(
            persistence_threshold=persistence_threshold,
            max_pts=max_pts,
        )
        # Activation buffer keyed by stage index in the registered order.
        self._captured: Dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._n_registered: int = 0

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------
    def register(
        self,
        model: nn.Module,
        stage_attr_names: Sequence[str] = ("stage1", "stage2", "stage3"),
    ) -> None:
        """Register forward hooks on the listed attribute names.

        Each listed attribute must be a child ``nn.Module`` of ``model``.
        Missing attributes are silently skipped so a 2-stage model can
        be used with the default 3-stage target list; the truncated
        registration is reflected in :attr:`_n_registered`.
        """
        self.clear_hooks()
        idx = 0
        for name in stage_attr_names:
            module = getattr(model, name, None)
            if module is None or not isinstance(module, nn.Module):
                continue
            handle = module.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)
            idx += 1
            if idx >= len(self.stage_targets):
                break
        self._n_registered = idx

    def _make_hook(self, idx: int):
        def _hook(_module, _inputs, output):
            # Store the (possibly non-leaf) tensor; this preserves grad
            # so the auxiliary loss can backprop through the activations.
            self._captured[idx] = output
        return _hook

    def clear_hooks(self) -> None:
        """Remove all forward hooks (idempotent)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._n_registered = 0
        self._captured.clear()

    def clear(self) -> None:
        """Drop captured activations (call after each training step)."""
        self._captured.clear()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(self) -> torch.Tensor:
        """Return the weighted-sum auxiliary loss across registered stages.

        Returns a scalar tensor with grad if any activations were captured;
        otherwise a zero scalar on CPU (useful as a no-op when no hooks
        are registered yet).
        """
        if not self._captured:
            return torch.zeros((), requires_grad=False)
        total: Optional[torch.Tensor] = None
        for idx, feat in self._captured.items():
            if idx >= len(self.stage_targets):
                continue
            target = self.stage_targets[idx]
            lam = self.lambdas[idx] if idx < len(self.lambdas) else self.lambdas[-1]
            term = lam * self.betti_loss(feat, target_betti=target)
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), requires_grad=False)
        return total

    # ------------------------------------------------------------------
    # Convenience forward — invoke `loss()` so callers can write
    #   aux = reg(); aux.backward()
    # interchangeably with reg.loss().
    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:  # type: ignore[override]
        return self.loss()

    def __len__(self) -> int:
        return self._n_registered


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def register_ph_hooks(
    model: nn.Module,
    stage_attr_names: Sequence[str] = ("stage1", "stage2", "stage3"),
    stage_targets: Optional[Sequence[Tuple[float, float, float]]] = None,
    lambdas: Optional[Sequence[float]] = None,
    persistence_threshold: float = 0.1,
    max_pts: int = 64,
) -> PHActivationRegularizer:
    """Instantiate + register a :class:`PHActivationRegularizer` in one step.

    This is the canonical entry point for the trainer-side wiring: the
    trainer holds the returned regularizer, adds ``lam * reg.loss()`` to
    its CE loss, and calls ``reg.clear()`` at the end of each step. The
    host model is unchanged.
    """
    targets = stage_targets if stage_targets is not None else _DEFAULT_STAGE_TARGETS
    reg = PHActivationRegularizer(
        stage_targets=targets,
        lambdas=lambdas,
        persistence_threshold=persistence_threshold,
        max_pts=max_pts,
    )
    reg.register(model, stage_attr_names=stage_attr_names)
    return reg
