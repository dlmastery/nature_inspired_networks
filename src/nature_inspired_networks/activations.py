"""φ-parameterised activations.

H39 — :class:`PhiGELU` / :func:`phi_act`
    A Swish/SiLU variant with the slope-at-origin coefficient fixed to
    ``β = φ ≈ 1.618``:

        PhiGELU(x) = x · sigmoid(x · φ)

    This sits between Swish's ``β = 1`` and the next-larger value that
    produces unstable activations, and (per H39's motivation) aligns
    multiplicatively with the φ-scaled width / depth progressions used
    elsewhere in the codebase. The activation is a drop-in replacement
    for ``nn.ReLU`` / ``nn.GELU``: zero at zero, monotonic for ``x > 0``,
    and gradient-continuous through the origin.

A free function :func:`phi_act` and an ``nn.Module`` wrapper are
provided so the call-site can choose between the two without rebinding
the ``β`` buffer.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .priors import PHI


def phi_act(x: torch.Tensor, beta: float = PHI) -> torch.Tensor:
    """Pure-functional φ-GELU / φ-Swish.

    ``phi_act(x) = x · sigmoid(beta · x)``. With the default
    ``beta = PHI`` this is Swish with β = φ; with ``beta = 1.0`` it is
    the standard SiLU (a property used by
    ``tests.test_activations.test_phi_act_reduces_to_swish``).
    """
    return x * torch.sigmoid(beta * x)


class PhiGELU(nn.Module):
    """φ-parameterised GELU/Swish-style activation.

    Parameters
    ----------
    learnable : bool, default False
        When True, ``beta`` is an ``nn.Parameter`` and gradients flow
        through it during training. When False, ``beta`` is a frozen
        buffer so the activation has zero learnable parameters.
    beta_init : float, default :data:`PHI`
        Initial value of ``beta``. Set ``learnable=True, beta_init=1.0``
        to obtain a Swish module whose β can drift toward φ.
    inplace : bool, default False
        Accepted for API parity with ``nn.ReLU(inplace=True)`` — the
        sigmoid path forbids true in-place execution, so this flag is
        ignored and exists only to keep call-sites uniform.
    """

    def __init__(
        self,
        learnable: bool = False,
        beta_init: float = PHI,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.learnable = bool(learnable)
        self.inplace = bool(inplace)  # noqa: parity-only
        beta_t = torch.tensor(float(beta_init), dtype=torch.float32)
        if learnable:
            self.beta = nn.Parameter(beta_t)
        else:
            self.register_buffer("beta", beta_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

    def extra_repr(self) -> str:
        b = self.beta.item() if self.beta.numel() == 1 else float("nan")
        return f"beta={b:.6f}, learnable={self.learnable}"


def swap_relu_with_phigelu(module: nn.Module,
                           learnable: bool = False,
                           beta_init: float = PHI) -> nn.Module:
    """Recursively replace every ``nn.ReLU`` in ``module`` with a fresh
    :class:`PhiGELU`. Functional ``F.relu`` calls inside hand-written
    forward methods are NOT touched -- this helper only addresses the
    submodule tree. Returns the module for chaining.

    Wired by the runner for the H39 ``sg_only_phi_activation`` row.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, PhiGELU(learnable=learnable,
                                          beta_init=beta_init))
        else:
            swap_relu_with_phigelu(child, learnable=learnable,
                                    beta_init=beta_init)
    return module


def swap_relu_with(module: nn.Module,
                   factory: Callable[[], nn.Module]) -> nn.Module:
    """Recursively replace every ``nn.ReLU`` in ``module`` with a fresh
    activation produced by ``factory()``.

    Generic counterpart to :func:`swap_relu_with_phigelu` and
    :func:`sinusoidal_activation.swap_relu_with_sine`. A FRESH module
    is created per replacement so the activations do not share state
    (each call ``factory()`` returns a new instance). Returns the
    module for chaining.

    Used by the runner's ``slot_activation`` cfg dispatch (Control 2,
    reviewer-flagged): swap ReLU for {tanh, softplus, gelu, swish}
    under the same SLOT recipe used by ``slot_act_sine``.

    Parameters
    ----------
    module : nn.Module
        The model whose submodule tree is mutated in-place.
    factory : Callable[[], nn.Module]
        Zero-arg callable returning a fresh activation module per call
        (e.g., ``lambda: nn.Tanh()``).

    Notes
    -----
    Functional ``F.relu(...)`` calls inside hand-written forward
    methods are NOT touched -- this helper only addresses the
    submodule tree. The same caveat applies to all sibling
    ``swap_relu_with_*`` helpers in this package.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, factory())
        else:
            swap_relu_with(child, factory)
    return module


# Factories used by the runner's ``slot_activation`` cfg dispatch.
# Single source of truth for the activation aliases the reviewer's
# Control 2 matrix requires: {tanh, softplus, gelu, swish}.
# ``sine`` and ``phi`` remain routed via their dedicated helpers
# (sinusoidal_activation.swap_relu_with_sine / swap_relu_with_phigelu).
SLOT_ACTIVATION_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "tanh":     lambda: nn.Tanh(),
    "softplus": lambda: nn.Softplus(),
    "gelu":     lambda: nn.GELU(),
    "swish":    lambda: nn.SiLU(),     # canonical "Swish-1" = SiLU
    "silu":     lambda: nn.SiLU(),     # alias of swish
}
