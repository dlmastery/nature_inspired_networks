"""H44 — Golden Regularization (φ-decay per-layer weight decay).

Implements *Golden Regularization* — a per-layer weight-decay schedule
``λ_k = λ_0 / φ^k`` where ``k`` is the layer index counted from the
shallowest learnable layer (``k = 0``). The schedule is realised as a
list of optimiser ``param_groups`` so it composes naturally with
``torch.optim.AdamW`` and any other decoupled-weight-decay optimiser.

Two entry-points:

:func:`build_phi_decay_param_groups`
    Walks an arbitrary ``nn.Module`` in registration order and groups
    parameters by their containing top-level child (stem, blocks, head,
    …). Each child receives ``weight_decay = base_wd / φ^k`` so the
    deepest blocks are regularised exponentially less than the
    shallowest.

:class:`GoldenRegularizer`
    A thin adapter wrapping any optimiser whose ``param_groups`` are
    pre-populated; it (re-)sets each group's ``weight_decay`` to
    ``base_wd / φ^k`` and exposes the resulting schedule via
    :meth:`schedule`.

The H44 ``regularizers.py`` namespace is owned by H47 (PhiDropout), so
this module is sibling rather than sub-module. The two are imported
together from ``__init__`` if needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .priors import PHI


def _named_layer_groups(model: nn.Module) -> list[tuple[str, list[nn.Parameter]]]:
    """Return ordered ``[(name, params)]`` where each entry is a
    top-level named child of ``model``.

    The model's ``stem`` (shallowest, k=0) ends up first and the head
    (deepest, k=K-1) ends up last, matching the H44 schedule. Param
    de-duplication is by id() — a parameter shared between two children
    only joins the first group (the deeper child loses it).
    """
    seen: set[int] = set()
    groups: list[tuple[str, list[nn.Parameter]]] = []
    for name, child in model.named_children():
        params: list[nn.Parameter] = []
        for p in child.parameters(recurse=True):
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            params.append(p)
        if params:
            groups.append((name, params))
    return groups


def build_phi_decay_param_groups(
    model: nn.Module,
    base_wd: float = 1e-2,
    phi: float = PHI,
    block_attr: str | None = None,
) -> list[dict]:
    """Construct AdamW-ready param groups with per-layer φ-decay.

    Parameters
    ----------
    model : nn.Module
        Any module. Layer index ``k`` is the order in which top-level
        children are registered, *except* when ``block_attr`` is
        supplied, in which case the children of
        ``getattr(model, block_attr)`` (a ``ModuleList`` /
        ``nn.Sequential``) form the indexed schedule. Non-block
        children (stem, head, etc.) keep ``base_wd``.
    base_wd : float, default 1e-2
        The decay coefficient applied at ``k=0``.
    phi : float, default :data:`PHI`
        Decay ratio. Layer ``k`` gets ``base_wd / phi**k``.
    block_attr : str, optional
        Attribute name (e.g. ``"stages"`` or ``"blocks"``) whose
        children form the indexed schedule. When None, all top-level
        children are indexed.

    Returns
    -------
    list of dict
        Each dict has keys ``params``, ``weight_decay``, and
        ``layer_name`` (for diagnostics).
    """
    if base_wd < 0:
        raise ValueError(f"base_wd must be >= 0, got {base_wd}")
    if phi <= 0:
        raise ValueError(f"phi must be > 0, got {phi}")

    groups: list[dict] = []
    if block_attr is not None and hasattr(model, block_attr):
        block_container = getattr(model, block_attr)
        # 1) non-block children get base_wd
        seen: set[int] = set()
        for name, params in _named_layer_groups(model):
            if name == block_attr:
                continue
            for p in params:
                seen.add(id(p))
            groups.append(dict(
                params=params, weight_decay=base_wd, layer_name=name,
            ))
        # 2) block children get the phi schedule
        for k, block in enumerate(block_container):
            params = [
                p for p in block.parameters(recurse=True)
                if p.requires_grad and id(p) not in seen
            ]
            for p in params:
                seen.add(id(p))
            if not params:
                continue
            lam = base_wd / (phi ** k)
            groups.append(dict(
                params=params, weight_decay=lam,
                layer_name=f"{block_attr}[{k}]",
            ))
    else:
        for k, (name, params) in enumerate(_named_layer_groups(model)):
            lam = base_wd / (phi ** k)
            groups.append(dict(
                params=params, weight_decay=lam, layer_name=name,
            ))
    if not groups:
        raise RuntimeError(
            "build_phi_decay_param_groups produced no groups -- "
            "model has no learnable parameters"
        )
    return groups


def phi_decay_param_groups(
    model: nn.Module,
    base_wd: float = 1e-2,
    phi: float = PHI,
    block_attr: str | None = None,
) -> list[dict]:
    """Public alias for :func:`build_phi_decay_param_groups`.

    Wired by the runner: when ``cfg.phi_decay_wd`` is set, the trainer
    builds AdamW param groups via this function so each top-level block
    receives weight-decay ``base_wd / phi^k`` (k = block index). Kept as
    a thin wrapper so the runner imports a stable name even if the
    underlying implementation moves.
    """
    return build_phi_decay_param_groups(
        model, base_wd=base_wd, phi=phi, block_attr=block_attr,
    )


class GoldenRegularizer:
    """Apply φ-decay weight-decay to an existing optimiser's groups.

    The wrapper does NOT subclass ``Optimizer`` — it merely mutates the
    ``param_groups`` in place so a downstream training loop is free to
    call ``opt.step()`` as usual. This is the minimal surface change
    required by H44 (Rule 1: one config change).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        An optimiser whose ``param_groups`` are already populated (one
        group per layer / block in shallow-to-deep order).
    base_wd : float
        The decay applied to group 0. Subsequent groups receive
        ``base_wd / phi**k`` where ``k`` is the group index.
    phi : float, default :data:`PHI`
        Decay ratio. With ``phi=1.0`` every group receives ``base_wd``
        (uniform-λ baseline; useful for ablations).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_wd: float,
        phi: float = PHI,
    ) -> None:
        if not optimizer.param_groups:
            raise ValueError("optimizer has no param_groups to schedule")
        if base_wd < 0:
            raise ValueError(f"base_wd must be >= 0, got {base_wd}")
        if phi <= 0:
            raise ValueError(f"phi must be > 0, got {phi}")
        self.optimizer = optimizer
        self.base_wd = float(base_wd)
        self.phi = float(phi)
        self._apply()

    def _apply(self) -> None:
        for k, g in enumerate(self.optimizer.param_groups):
            g["weight_decay"] = self.base_wd / (self.phi ** k)

    def schedule(self) -> list[float]:
        """Return the per-group ``weight_decay`` schedule as a list."""
        return [float(g["weight_decay"]) for g in self.optimizer.param_groups]

    def __repr__(self) -> str:
        sched = ", ".join(f"{wd:.3e}" for wd in self.schedule())
        return (
            f"GoldenRegularizer(base_wd={self.base_wd:.3e}, "
            f"phi={self.phi:.6f}, schedule=[{sched}])"
        )
