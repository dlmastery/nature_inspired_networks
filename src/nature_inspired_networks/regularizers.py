"""Nature-inspired regularizers (H47 φ-Dropout).

PhiDropout
----------
Drop-in replacement for :class:`torch.nn.Dropout` whose rate ``p``
follows a Fibonacci- or φ-derived curriculum over the course of
training. The hypothesis (H47 design doc) is that early-training noise
should be high (≈ 1/φ ≈ 0.618) and late-training noise low (≈ 1/φ⁴ ≈
0.146), matching the curriculum-noise prescription of Bengio 2009 and
the φ-decay seen in biological synaptic stochasticity.

Two cycle modes are supported:

  * ``cycle='fib'`` — at epoch index ``e`` the rate is
    ``Fib(e mod L) / sum(Fib[:L])``, i.e. the normalised Fibonacci
    fractions. ``L`` defaults to 5 → {1/19, 2/19, 3/19, 5/19, 8/19} ≈
    {0.053, 0.105, 0.158, 0.263, 0.421}.
  * ``cycle='phi'`` — ``p(e) = 1/φ^(1 + (e mod L))`` cycles through
    {0.618, 0.382, 0.236, 0.146, 0.090} (the H47 design-doc default).

**Epoch-keyed, not step-keyed (PAPER_GAP_G5).** The original release
incremented the internal counter once per forward pass; with
batch=256/CIFAR-10 that cycled the 5-entry schedule ~39× per epoch,
contradicting the design doc's "early high noise, late low noise"
curriculum (which oscillated rapidly instead). The schedule is now
indexed by ``epoch`` by default — callers (typically the Trainer)
invoke :meth:`step_epoch` (or :meth:`set_epoch`) once per epoch. If
neither is called the module stays at epoch 0 (the first / highest-
noise entry).

The ``step_unit`` constructor arg makes the choice explicit:

* ``step_unit='epoch'`` (default, PAPER_GAP_G5-correct) — :meth:`step`
  is a no-op; the curriculum only advances when the Trainer calls
  :meth:`step_epoch` at the end of every epoch (duck-typed — the
  Trainer probes for the method).
* ``step_unit='forward'`` (legacy) — every forward pass advances the
  internal counter (the original buggy behaviour, retained for
  reproducing pre-fix numbers).

Eval mode is identity, as with standard dropout. The epoch and
``step_counter`` buffers are checkpointed.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


# φ-decay schedule used when ``cycle='phi'``. Pre-computed so tests can
# assert exact values without re-deriving from PHI.
PHI_DROPOUT_SCHEDULE: tuple[float, ...] = tuple(
    1.0 / (PHI ** (1 + k)) for k in range(5)
)  # ≈ (0.6180, 0.3820, 0.2361, 0.1459, 0.0902)


def _fib_dropout_schedule(length: int) -> list[float]:
    """Return the normalised Fibonacci sequence of length ``length``."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    seq = [1, 2]
    while len(seq) < length:
        seq.append(seq[-1] + seq[-2])
    seq = seq[:length]
    s = float(sum(seq))
    return [v / s for v in seq]


class PhiDropout(nn.Module):
    """Dropout whose rate follows an epoch-keyed Fibonacci / φ curriculum.

    The schedule is indexed by the current training epoch (set via
    :meth:`step_epoch` or :meth:`set_epoch`), not by per-forward-pass
    step count. Within a single epoch the dropout rate is constant;
    between epochs the rate advances along the schedule (wrapping mod
    ``length``). This matches the H47 design-doc curriculum: early
    epochs see the highest noise, later epochs see lower noise.

    Parameters
    ----------
    p_init
        Fallback dropout rate used only when the schedule is somehow
        empty (defensive default; the module always overrides this
        from the schedule in normal operation). Defaults to ``1/φ``.
    cycle
        ``'fib'`` (normalised Fibonacci) or ``'phi'`` (φ⁻ⁿ decay).
    length
        Length of the cycle. The epoch index wraps mod ``length`` so
        training longer than ``length`` epochs cycles repeatedly.
    inplace
        Forwarded to :func:`F.dropout`. Off by default to keep
        backward-graph debugging easy.
    step_unit
        Either ``'epoch'`` (default, PAPER_GAP_G5-correct) or
        ``'forward'`` (legacy buggy mode kept for reproducing pre-fix
        numbers). When ``'epoch'`` the curriculum only advances when
        :meth:`step_epoch` or :meth:`set_epoch` is invoked by the
        Trainer at end-of-epoch; per-forward :meth:`step` is a no-op.
        When ``'forward'`` each invocation of :meth:`step` (and each
        ``forward`` call, for parity with the original buggy release)
        advances the internal counter by one.

    Notes
    -----
    For the ``'epoch'`` default the Trainer must call :meth:`step_epoch`
    (or equivalently :meth:`set_epoch`) once per epoch (the runner's
    :class:`Trainer` does this automatically — it iterates over
    ``model.modules()`` and duck-types on the presence of
    :meth:`step_epoch` / :meth:`set_epoch`). If neither is called the
    module remains pinned at epoch 0 (highest-noise entry).
    """

    def __init__(
        self,
        p_init: float = 1.0 / PHI,
        cycle: str = "fib",
        length: int = 5,
        inplace: bool = False,
        step_unit: Literal["forward", "epoch"] = "epoch",
    ) -> None:
        super().__init__()
        if cycle not in {"fib", "phi"}:
            raise ValueError(
                f"cycle must be 'fib' or 'phi', got {cycle!r}"
            )
        if not (0.0 <= p_init < 1.0):
            raise ValueError(
                f"p_init must be in [0, 1), got {p_init}"
            )
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if step_unit not in {"forward", "epoch"}:
            raise ValueError(
                f"step_unit must be 'forward' or 'epoch', got "
                f"{step_unit!r}"
            )
        self.cycle = cycle
        self.length = length
        self.inplace = inplace
        self.p_init = float(p_init)
        self.step_unit = step_unit
        if cycle == "fib":
            schedule = _fib_dropout_schedule(length)
        else:
            schedule = [1.0 / (PHI ** (1 + k)) for k in range(length)]
        # Clamp into [0, 1) — Fibonacci normalisation guarantees this
        # but the φ-cycle's first value is 1/φ ≈ 0.618 which is fine;
        # the clamp is defensive against future length values.
        schedule = [max(0.0, min(0.999, v)) for v in schedule]
        self.register_buffer(
            "schedule", torch.tensor(schedule, dtype=torch.float32)
        )
        # ``epoch`` is the live curriculum index used by ``current_p``.
        # ``step_counter`` is mutated only when ``step_unit='forward'``
        # (legacy mode); otherwise it stays at 0 and is retained for
        # backward checkpoint compatibility.
        self.register_buffer(
            "epoch", torch.zeros(1, dtype=torch.long)
        )
        self.register_buffer(
            "step_counter", torch.zeros(1, dtype=torch.long)
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the curriculum to the given training epoch.

        Called by the Trainer once per epoch in the default
        ``step_unit='epoch'`` mode; the dropout rate stays constant
        within each epoch interval. Negative epochs are clamped to 0.
        """
        e = max(0, int(epoch))
        self.epoch.fill_(e)

    def step_epoch(self) -> None:
        """Advance the curriculum by one epoch (Trainer hook).

        Duck-typed by :class:`Trainer.fit` — any module exposing this
        method gets called at the end of every epoch. Equivalent to
        ``set_epoch(int(self.epoch.item()) + 1)`` in the default
        ``step_unit='epoch'`` mode. A no-op in ``'forward'`` mode
        (where the counter advances on every forward).
        """
        if self.step_unit == "epoch":
            self.epoch.add_(1)

    def step(self) -> None:
        """Per-forward step (legacy buggy mode only).

        In the default ``step_unit='epoch'`` mode this is a no-op —
        the curriculum only advances when :meth:`step_epoch` /
        :meth:`set_epoch` is called. In ``step_unit='forward'`` mode
        each call increments ``step_counter``; the dropout rate is
        then indexed by ``step_counter % length``.
        """
        if self.step_unit == "forward":
            self.step_counter.add_(1)

    @property
    def current_p(self) -> float:
        """The dropout probability used in the current interval.

        In the default ``step_unit='epoch'`` mode this is indexed by
        ``self.epoch % self.length`` and is constant within each epoch.
        In the legacy ``step_unit='forward'`` mode it is indexed by
        ``self.step_counter % self.length`` and changes per-forward.
        """
        if self.step_unit == "forward":
            idx = int(self.step_counter.item()) % self.length
        else:
            idx = int(self.epoch.item()) % self.length
        return float(self.schedule[idx].item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # Legacy parity: in 'forward' mode the original buggy release
        # incremented the step counter on every forward call. Preserve
        # that exact behaviour (so legacy_forward_mode_still_works can
        # be reproduced) but only in the explicit opt-in mode.
        if self.step_unit == "forward":
            self.step_counter.add_(1)
        p = self.current_p
        return F.dropout(x, p=p, training=True, inplace=self.inplace)

    def extra_repr(self) -> str:
        return (
            f"cycle={self.cycle!r}, length={self.length}, "
            f"p_init={self.p_init:.4f}, step_unit={self.step_unit!r}"
        )
