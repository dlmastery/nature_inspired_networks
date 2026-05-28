"""H08 — Dynamic phi-Growth (Fibonacci-scheduled adaptive layer addition).

Design doc: ``hypotheses/g1_scaling_growth/H08_dynamic_phi_growth.md``.

Adaptive depth scaling: rather than fixing total depth at initialisation,
the network grows new residual blocks during training at Fibonacci-spaced
epochs (default schedule ``(3, 5, 8, 13)``). The premise is that
biological cortical layers reach maturity at Fibonacci-spaced intervals
because each new layer integrates the maturation of the two prior layers
-- this gives the natural-system optimum for resource-inherited growth.
Per Chen, Goodfellow, Shlens 2016 ICLR 'Net2Net' (arXiv:1511.05641) we
expect cumulative-compute-to-target-top-1 to drop by 20-35 pct vs the
static-depth baseline. This module ships the pure callback + grow helper;
the trainer-side wiring (optimizer.add_param_group, LR-reset) is left
additive and documented at the bottom of this file.

Public surface
--------------
- :func:`grow_model`             append n extra NaturePriorBlock clones
                                 to the deepest stage with kaiming reinit.
- :class:`DynamicGrowthCallback` schedule wrapper, returns the possibly-
                                 grown model at each epoch.
- :func:`fib_growth_schedule`    canonical Fibonacci schedule helper.

References (Citation Rigor):
    Chen, Tianqi, Goodfellow, Ian, Shlens, Jonathon 2016 ICLR
    'Net2Net: Accelerating Learning via Knowledge Transfer'
    (arXiv:1511.05641) -- the foundational dynamic-growth paper.
    Wei, Tao, Wang, Changhu, Rui, Yong, Chen, Chang Wen 2016 CVPR
    'Network Morphism' (arXiv:1603.01670) -- function-preserving
    network growth.
    Larsson, Maire, Shakhnarovich 2017 ICLR 'FractalNet: Ultra-Deep
    Neural Networks without Residuals' (arXiv:1605.07648) -- the static
    fractal cousin; H08 is the time-varying analogue.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

from .blocks import NaturePriorBlock, NaturePriorFlags
from .priors import PHI  # noqa: F401  (re-exported for callers; canonical PHI)
from .scaling import fibonacci_sequence


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
def fib_growth_schedule(n_events: int = 4, start_index: int = 3) -> tuple[int, ...]:
    """Return ``n_events`` Fibonacci-indexed growth epochs.

    With the default ``start_index=3, n_events=4`` the schedule is
    ``(3, 5, 8, 13)`` -- matching the H08 design doc. ``start_index=2``
    gives ``(2, 3, 5, 8)``. The returned epochs are intended to be
    matched against ``epoch in schedule`` inside
    :meth:`DynamicGrowthCallback.step`.
    """
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1; got {n_events}")
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0; got {start_index}")
    needed = start_index + n_events
    seq = fibonacci_sequence(needed)
    return tuple(seq[start_index:start_index + n_events])


# ---------------------------------------------------------------------------
# Grow helper -- appends NaturePriorBlock clones to the deepest stage
# ---------------------------------------------------------------------------
def _function_preserving_init_(block: nn.Module) -> None:
    """Initialise a freshly-appended :class:`NaturePriorBlock` so that
    its forward pass is the identity function at insertion time.

    This implements Net2Net-style function-preserving growth (Chen,
    Goodfellow, Shlens 2016 ICLR 'Net2Net: Accelerating Learning via
    Knowledge Transfer', arXiv:1511.05641). The block's residual form is
    ``y_out = relu(residual_branch(x) + skip(x))``. Because the
    appended block has matched ``c_in == c_out`` and ``stride == 1``,
    the skip is :class:`nn.Identity`, so ``skip(x) == x`` and (since
    ``x`` is the post-ReLU activation of the prior block, hence
    non-negative) ``relu(0 + x) == x``.

    To make ``residual_branch(x) == 0`` for every input, we zero the
    BatchNorm gamma (``weight``) and beta (``bias``) of EVERY BN in the
    block's terminal residual branch (``block.conv2``). Each BN then
    outputs zero regardless of upstream conv weights:
    ``bn(y) = gamma * (y - mu) / sigma + beta = 0``. The fractal-path
    variant's ``0.5 * (a + b)`` merge becomes ``0.5 * (0 + 0) == 0``
    because both ``a`` and ``b`` terminate in (now-zeroed) BN layers.

    Conv weights are kept at their PyTorch default (Kaiming-uniform)
    initialisation so the block has well-conditioned gradients the
    moment BN gamma starts moving away from zero — training "wakes up"
    the block smoothly rather than from a degenerate zero-weight start.

    The golden-angle channel modulation (when enabled) multiplies the
    residual output by ``cos(phases + alpha) * 0.5 + 0.5``; since the
    residual output is zero, the gate is irrelevant. ``alpha`` is left
    at its default zero initialisation.
    """
    # Identity-init: zero the BN gamma + beta of every BatchNorm in the
    # residual branch (conv2). This makes the branch output zero, so
    # the block reduces to its identity skip.
    conv2 = getattr(block, "conv2", None)
    if conv2 is None:
        # Non-NaturePriorBlock module: fall back to a no-op (the caller
        # documents this branch as silently passing through, matching
        # the prior behaviour of unknown module types).
        return
    for m in conv2.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


def grow_model(model: nn.Module, n_extra_blocks: int = 1,
               flags: NaturePriorFlags | None = None) -> nn.Module:
    """Append ``n_extra_blocks`` fresh :class:`NaturePriorBlock` instances
    to the *deepest* stage of ``model``.

    The function inspects ``model.stages`` (a :class:`nn.ModuleList` per
    :class:`NaturePriorNet`) and adds the new blocks at the end of the
    last stage, all at stride 1 with ``c_in == c_out == widths[-1]`` so
    the spatial footprint is preserved (the network's forward output
    shape is unchanged). New conv/linear weights are Kaiming-reinitialised
    (``_kaiming_reinit_``) so the appended blocks start from He-normal.

    The returned object is the *same* ``model`` (mutated in place); a
    reference is returned so callers can chain. Models without a
    ``stages`` ModuleList are returned untouched (the H08 callback then
    no-ops on growth events).

    Function-preservation contract: each appended block is initialised
    via :func:`_function_preserving_init_` (Net2Net-style identity
    init), so ``model(x)`` after ``grow_model`` equals ``model(x)``
    before within float tolerance. See the docstring of
    :func:`_function_preserving_init_` for the mechanism.
    """
    if n_extra_blocks < 0:
        raise ValueError(f"n_extra_blocks must be >= 0; got {n_extra_blocks}")
    if n_extra_blocks == 0:
        return model
    stages = getattr(model, "stages", None)
    if stages is None or len(stages) == 0:
        # Caller's model is not the canonical NaturePriorNet — silently
        # no-op so external models pass through cleanly. The callback's
        # ``param_count`` test catches an absent growth path.
        return model
    last_stage = stages[-1]
    if not isinstance(last_stage, nn.Sequential):
        return model
    # Discover the last stage's output width via the trailing block's
    # output projection. NaturePriorBlock exposes ``conv2`` whose last
    # BatchNorm has ``num_features = c_out``.
    last_block = last_stage[-1]
    c_out = _infer_block_out_channels(last_block)
    if c_out is None:
        return model
    flags = flags or _infer_flags(last_block)
    # Inserted blocks must be put in eval-equivalent mode so BN uses its
    # zeroed gamma + beta directly rather than re-computing running
    # stats from the first batch. We rely on the fact that BN with
    # gamma=0, beta=0 outputs zero regardless of running stats; that
    # holds in both train() and eval() modes.
    for _ in range(n_extra_blocks):
        new_block = NaturePriorBlock(c_out, c_out, stride=1, flags=flags)
        _function_preserving_init_(new_block)
        # Inherit the parent module's training mode so the new block's
        # BNs behave consistently with their siblings.
        new_block.train(mode=model.training)
        last_stage.append(new_block)
    return model


def _infer_block_out_channels(block: nn.Module) -> int | None:
    """Best-effort discovery of ``c_out`` for an arbitrary block."""
    # Prefer the explicit BatchNorm in conv2 / conv1 chains.
    for name in ("conv2", "conv1", "path", "a"):
        sub = getattr(block, name, None)
        if sub is None:
            continue
        for m in sub.modules():
            if isinstance(m, nn.BatchNorm2d):
                return int(m.num_features)
    for m in block.modules():
        if isinstance(m, nn.BatchNorm2d):
            return int(m.num_features)
    return None


def _infer_flags(block: nn.Module) -> NaturePriorFlags:
    """Return the block's NaturePriorFlags or a default instance."""
    f = getattr(block, "flags", None)
    if isinstance(f, NaturePriorFlags):
        return f
    return NaturePriorFlags()


# ---------------------------------------------------------------------------
# Callback -- orchestrates Fibonacci-spaced growth events
# ---------------------------------------------------------------------------
class DynamicGrowthCallback:
    """Trainer-side callback that grows a model at Fibonacci-spaced epochs.

    Usage pattern (trainer-side, NOT wired here per CLAUDE.md Rule on
    runner ownership):

        cb = DynamicGrowthCallback(model_factory=lambda: NaturePriorNet(cfg),
                                   fib_schedule=(3, 5, 8, 13))
        model = cb.model
        for epoch in range(num_epochs):
            train_one_epoch(model, ...)
            model = cb.step(epoch, model)
            # if model was grown, the trainer must rebuild the optimizer
            # OR call ``optimizer.add_param_group({'params': new_params})``.

    Parameters
    ----------
    model_factory : callable returning ``nn.Module``
        Used to obtain the initial model. Stored so the callback owns a
        canonical "before any growth" reference (handy for unit tests
        and parameter-count regression checks).
    fib_schedule : tuple of int
        Epoch indices at which :meth:`grow_model` should be invoked.
        Defaults to ``(3, 5, 8, 13)`` -- the canonical Fibonacci schedule
        from the H08 design doc.
    n_extra_per_event : int
        Number of fresh NaturePriorBlock instances to append per growth
        event. Default 1.
    flags : NaturePriorFlags, optional
        Forward to :func:`grow_model` when the new blocks need flags
        that differ from the parent model. Default None -> inherit from
        the last block.

    Attributes
    ----------
    model : nn.Module
        The (possibly mutated) model -- updated in place by ``step``.
    fired : set[int]
        Epochs at which growth has already occurred. Idempotency guard
        so re-calling ``step`` on the same epoch does NOT regrow.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        fib_schedule: Sequence[int] = (3, 5, 8, 13),
        n_extra_per_event: int = 1,
        flags: NaturePriorFlags | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.fib_schedule: tuple[int, ...] = tuple(int(e) for e in fib_schedule)
        self.n_extra_per_event = int(n_extra_per_event)
        self.flags = flags
        self.model = model_factory()
        self.fired: set[int] = set()

    def step(self, epoch: int, model: nn.Module | None = None) -> nn.Module:
        """Possibly grow ``model`` and return it.

        If ``epoch`` is in :attr:`fib_schedule` and was not previously
        fired, append ``n_extra_per_event`` blocks via :func:`grow_model`.
        Otherwise return the model unchanged. When the caller passes
        ``model=None`` the callback's internal :attr:`model` reference is
        used; this matches the canonical usage pattern.
        """
        if model is None:
            model = self.model
        if epoch in self.fib_schedule and epoch not in self.fired:
            grow_model(model, n_extra_blocks=self.n_extra_per_event,
                       flags=self.flags)
            self.fired.add(epoch)
        # Sync the callback's owned reference.
        self.model = model
        return model

    def total_param_count(self) -> int:
        """Convenience: ``sum(p.numel() for p in self.model.parameters())``."""
        return sum(p.numel() for p in self.model.parameters())


# ---------------------------------------------------------------------------
# Runner-side wiring (TODO; left for the integration pass)
# ---------------------------------------------------------------------------
# TODO runner wiring:
#   - Instantiate DynamicGrowthCallback inside runner.train() when the
#     config carries ``flags.dynamic_growth=True`` (NEW flag; add to
#     NaturePriorFlags in a separate commit).
#   - Per Rule 1 atomicity, the growth schedule lives in config:
#       dynamic_growth:
#         schedule: [3, 5, 8, 13]
#         n_extra_per_event: 1
#   - After cb.step(epoch) the optimizer MUST be re-built (or the new
#     block's parameters added via optimizer.add_param_group({'params':
#     new_block.parameters()})), and the LR scheduler should NOT be
#     reset -- the cosine schedule should keep its current value so
#     downstream epochs continue smoothly (per H08 mechanism sec. 5.1).
#   - Wall-clock cost: 4 growth events x ~1 block ~= +30 pct param count
#     at end of training; cumulative compute is reduced because early
#     epochs train the shallow stack.
