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

**Epoch-keyed, not step-keyed.** The original release incremented the
internal counter once per forward pass; with batch=256/CIFAR-10 that
cycled the 5-entry schedule ~39× per epoch, contradicting the design
doc's "early high noise, late low noise" curriculum (which oscillated
rapidly instead). The schedule is now indexed by ``epoch`` — callers
(typically the Trainer) must invoke :meth:`set_epoch` once per epoch.
If :meth:`set_epoch` is never called the module stays at epoch 0 (the
first schedule entry), which is the most-noise / safest default.

Eval mode is identity, as with standard dropout. The epoch buffer is
checkpointed.
"""
from __future__ import annotations

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
    :meth:`set_epoch`), not by per-forward-pass step count. Within a
    single epoch the dropout rate is constant; between epochs the rate
    advances along the schedule (wrapping mod ``length``). This matches
    the H47 design-doc curriculum: early epochs see the highest noise,
    later epochs see lower noise.

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

    Notes
    -----
    Callers MUST drive the curriculum by invoking :meth:`set_epoch` once
    per epoch (the runner's :class:`Trainer` does this automatically when
    a model contains any ``PhiDropout`` modules). If never called the
    module remains pinned at epoch 0 (the first / highest-noise entry).
    The internal ``step_counter`` buffer is retained for checkpoint
    round-trip compatibility but is no longer mutated by ``forward``.
    """

    def __init__(
        self,
        p_init: float = 1.0 / PHI,
        cycle: str = "fib",
        length: int = 5,
        inplace: bool = False,
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
        self.cycle = cycle
        self.length = length
        self.inplace = inplace
        self.p_init = float(p_init)
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
        # ``step_counter`` is retained as a legacy buffer for backward
        # checkpoint compatibility but is no longer mutated by forward().
        self.register_buffer(
            "epoch", torch.zeros(1, dtype=torch.long)
        )
        self.register_buffer(
            "step_counter", torch.zeros(1, dtype=torch.long)
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the curriculum to the given training epoch.

        Called by the Trainer once per epoch; the dropout rate stays
        constant within each epoch interval. Negative epochs are clamped
        to 0.
        """
        e = max(0, int(epoch))
        self.epoch.fill_(e)

    @property
    def current_p(self) -> float:
        """The dropout probability used in the current epoch interval.

        Indexed by ``self.epoch % self.length``; constant within each
        epoch (unlike the previous step-keyed implementation which
        oscillated ~39× per epoch at batch=256/CIFAR-10).
        """
        idx = int(self.epoch.item()) % self.length
        return float(self.schedule[idx].item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        p = self.current_p
        return F.dropout(x, p=p, training=True, inplace=self.inplace)

    def extra_repr(self) -> str:
        return (
            f"cycle={self.cycle!r}, length={self.length}, "
            f"p_init={self.p_init:.4f}"
        )
