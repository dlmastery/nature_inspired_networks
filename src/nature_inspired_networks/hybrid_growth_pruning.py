"""H64 — Dynamic Growth + Pruning Cycle (G7 cross-paradigm hybrid).

Design doc:
``hypotheses/g7_cross_paradigm_hybrids/H64_dynamic_growth_pruning.md``.

Combines two existing Fibonacci-scheduled callbacks:

  * :class:`nature_inspired_networks.dynamic_growth.DynamicGrowthCallback`
    -- appends NaturePriorBlock clones at Fibonacci-spaced *grow epochs*
    ``(3, 5, 8, 13)``.
  * :func:`nature_inspired_networks.pruning.fibonacci_prune`
    -- magnitude-prunes the model at Fibonacci-spaced *prune epochs*
    ``(4, 6, 9, 14)`` -- i.e., one epoch *after* each growth event.

The H64 hypothesis is that a strict grow→prune alternation captures
Hebbian "synaptic pruning" -- biological networks grow new connections
early in critical periods and then prune the weakest ones cycle-by-cycle.
The two-callback wiring is encoded in :class:`GrowthPruningSchedule` as a
thin orchestration layer that delegates to the upstream primitives.

References (Citation Rigor)::

    Chen, Goodfellow, Shlens 2016 ICLR 'Net2Net' (arXiv:1511.05641) --
    function-preserving growth; H64 uses the dynamic_growth wrapper of
    this idea.

    Han, Mao, Dally 2016 ICLR 'Deep Compression: Pruning, Trained
    Quantization and Huffman Coding' (arXiv:1510.00149) -- the iterative
    magnitude pruning whose schedule we set to Fibonacci ratios.

    Frankle, Carbin 2019 ICLR 'The Lottery Ticket Hypothesis'
    (arXiv:1803.03635) -- grow-then-prune produces sparse trainable
    subnetworks; H64 instantiates the schedule with biological
    Fibonacci-spaced intervals.

Public surface
--------------
- :class:`GrowthPruningSchedule` -- combines DynamicGrowthCallback and
  fibonacci_prune on disjoint Fibonacci-spaced epoch sets. ``step(epoch,
  model)`` is the single entry point.
- :data:`DEFAULT_GROW_EPOCHS`, :data:`DEFAULT_PRUNE_EPOCHS` -- the
  canonical H64 schedules.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import NaturePriorFlags
from .dynamic_growth import DynamicGrowthCallback
from .pruning import fibonacci_prune


__all__ = [
    "DEFAULT_GROW_EPOCHS",
    "DEFAULT_PRUNE_EPOCHS",
    "GrowthPruningSchedule",
]


# Canonical H64 schedules: Fibonacci numbers for grow,
# (Fibonacci + 1) offsets for prune so the two never collide.
DEFAULT_GROW_EPOCHS: tuple[int, ...] = (3, 5, 8, 13)
DEFAULT_PRUNE_EPOCHS: tuple[int, ...] = (4, 6, 9, 14)


class GrowthPruningSchedule(nn.Module):
    """Combined Fibonacci-grow + Fibonacci-prune callback wrapper.

    Wraps a :class:`DynamicGrowthCallback` (for the grow events) and a
    direct :func:`fibonacci_prune` call (for the prune events). The two
    schedules are disjoint by construction so a single epoch is never
    both grown and pruned.

    The class subclasses :class:`nn.Module` purely so it lives nicely
    inside the rest of the package's typing conventions; it has no
    learnable parameters of its own.

    Usage::

        sched = GrowthPruningSchedule(
            model_factory=lambda: NaturePriorNet(cfg),
            grow_epochs=DEFAULT_GROW_EPOCHS,
            prune_epochs=DEFAULT_PRUNE_EPOCHS,
        )
        model = sched.model
        for epoch in range(num_epochs):
            train_one_epoch(model, ...)
            model, event = sched.step(epoch, model)
            # event is one of {"grow", "prune", "none"}; the trainer
            # rebuilds the optimizer when event == "grow".

    Parameters
    ----------
    model_factory : callable returning ``nn.Module``
        Forwarded to :class:`DynamicGrowthCallback`. Must be callable
        with no arguments and return a model with a ``stages`` attribute
        for growth to be effective (the inner callback no-ops otherwise).
    grow_epochs : sequence of int, default ``DEFAULT_GROW_EPOCHS``
        Epochs at which to grow the model.
    prune_epochs : sequence of int, default ``DEFAULT_PRUNE_EPOCHS``
        Epochs at which to magnitude-prune the model.
    n_extra_blocks : int, default 1
        Number of blocks appended at each grow event.
    prune_schedule_length : int, default 4
        Number of Fibonacci prune ratios. The default (4) means
        prune fractions of (1, 2, 3, 5) / 11 are applied successively
        across the 4 prune epochs.
    flags : NaturePriorFlags, optional
        Forwarded into newly grown blocks.
    """

    def __init__(
        self,
        model_factory,
        grow_epochs: Sequence[int] = DEFAULT_GROW_EPOCHS,
        prune_epochs: Sequence[int] = DEFAULT_PRUNE_EPOCHS,
        n_extra_blocks: int = 1,
        prune_schedule_length: int = 4,
        flags: NaturePriorFlags | None = None,
    ) -> None:
        super().__init__()
        grow_set = set(int(e) for e in grow_epochs)
        prune_set = set(int(e) for e in prune_epochs)
        overlap = grow_set & prune_set
        if overlap:
            raise ValueError(
                f"grow and prune epochs must be disjoint; overlap = {sorted(overlap)}"
            )
        self.grow_epochs = tuple(sorted(grow_set))
        self.prune_epochs = tuple(sorted(prune_set))
        self.prune_schedule_length = int(prune_schedule_length)
        # The underlying growth callback (handles model construction + grow).
        # NB: upstream API uses ``n_extra_per_event`` (per-event count).
        self._grower = DynamicGrowthCallback(
            model_factory=model_factory,
            fib_schedule=tuple(self.grow_epochs),
            n_extra_per_event=int(n_extra_blocks),
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def model(self) -> nn.Module:
        return self._grower.model

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, epoch: int, model: nn.Module) -> tuple[nn.Module, str]:
        """Take one schedule step.

        Returns ``(model, event)``:

        * ``event == "grow"`` — the model was grown by the inner
          DynamicGrowthCallback. Trainer should rebuild the optimizer
          to include the new parameters.
        * ``event == "prune"`` — :func:`fibonacci_prune` was applied
          with the matched Fibonacci-ratio. Trainer keeps the optimizer.
        * ``event == "none"`` — no scheduled action this epoch.
        """
        e = int(epoch)
        if e in self.grow_epochs:
            new_model = self._grower.step(e, model)
            return new_model, "grow"
        if e in self.prune_epochs:
            # Translate the epoch into the 1-indexed Fibonacci gate used
            # by fibonacci_prune. We map our prune epoch to the same
            # ordinal Fibonacci ratio: 4 → idx 0, 6 → idx 1, 9 → idx 2,
            # 14 → idx 3. fibonacci_prune itself uses (1, 2, 3, 5, 8, ...)
            # so we synthesize a "fake epoch" by setting (epoch_arg + 1)
            # to FIB_SCHEDULE[idx].
            try:
                idx = self.prune_epochs.index(e)
            except ValueError:  # pragma: no cover -- guarded above
                return model, "none"
            from .pruning import FIB_SCHEDULE
            fake_epoch = FIB_SCHEDULE[idx] - 1
            fibonacci_prune(
                model,
                epoch=fake_epoch,
                schedule_length=self.prune_schedule_length,
                make_permanent=False,
            )
            return model, "prune"
        return model, "none"
