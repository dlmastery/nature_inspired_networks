"""Nature-inspired regularizers (H47 φ-Dropout).

PhiDropout
----------
Drop-in replacement for :class:`torch.nn.Dropout` whose rate ``p``
cycles through a Fibonacci-normalised sequence over the course of
training. The hypothesis (H47 design doc) is that early-training noise
should be high (≈ 1/φ ≈ 0.618) and late-training noise low (≈ 1/φ⁴ ≈
0.146), matching the curriculum-noise prescription of Bengio 2009 and
the φ-decay seen in biological synaptic stochasticity.

Two cycle modes are supported:

  * ``cycle='fib'`` — at internal step ``t`` the rate is
    ``Fib(t mod L) / sum(Fib[:L])``, i.e. the normalised Fibonacci
    fractions. ``L`` defaults to 5 → {1/19, 2/19, 3/19, 5/19, 8/19} ≈
    {0.053, 0.105, 0.158, 0.263, 0.421}.
  * ``cycle='phi'`` — ``p(t) = 1/φ^(1 + (t mod L))`` cycles through
    {0.618, 0.382, 0.236, 0.146, 0.090} (the H47 design-doc default).

The module increments its internal step counter once per forward pass
during training (and only during training). Eval mode is identity, as
with standard dropout. Counter state is exposed as a buffer so
checkpointing round-trips it.
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
    """Dropout whose rate cycles through Fibonacci / φ-derived values.

    Parameters
    ----------
    p_init
        Fallback dropout rate used only when the schedule is somehow
        empty (defensive default; the module always overrides this
        from the schedule in normal operation). Defaults to ``1/φ``.
    cycle
        ``'fib'`` (normalised Fibonacci) or ``'phi'`` (φ⁻ⁿ decay).
    length
        Length of the cycle. The internal step counter wraps mod
        ``length`` so training of any duration sees the full cycle
        repeatedly.
    inplace
        Forwarded to :func:`F.dropout`. Off by default to keep
        backward-graph debugging easy.
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
        self.register_buffer(
            "step_counter", torch.zeros(1, dtype=torch.long)
        )

    @property
    def current_p(self) -> float:
        """The dropout probability that *would* be used on the next
        training forward pass. Read-only convenience for tests / logs.
        """
        idx = int(self.step_counter.item()) % self.length
        return float(self.schedule[idx].item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        p = self.current_p
        # advance counter AFTER reading so step 0 sees schedule[0]
        self.step_counter += 1
        return F.dropout(x, p=p, training=True, inplace=self.inplace)

    def extra_repr(self) -> str:
        return (
            f"cycle={self.cycle!r}, length={self.length}, "
            f"p_init={self.p_init:.4f}"
        )
