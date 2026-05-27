"""H20 — Fibonacci Ensemble (weighted average of last K checkpoints).

An inference-time wrapper that averages the last ``K`` checkpoints with
Fibonacci weights ``(F1, F2, ..., FK) = (1, 1, 2, 3, 5, 8, 13, 21)``.
Sits between Stochastic Weight Averaging (Izmailov 2018, uniform
weighting) and Exponential Moving Average (DINO, geometric weighting).

The mechanism is **post-training**: training proceeds normally; the
ensemble buffer records the last ``K`` state dicts (or runs an O(1)
running EMA with the analytical Fibonacci recursion) and the averaged
weights are loaded for evaluation. There is no training-loop
modification — the test suite covers the math only, per the
assignment.

Two complementary implementations:

* :class:`FibEnsemble` — explicit buffer of the last K state dicts.
  ``averaged_state_dict()`` returns the Fibonacci-weighted mean.
  Memory ``= K * model_size``.

* :class:`FibEMA` — O(1)-memory running EMA whose new-checkpoint
  weight is ``F_K / sum(F_1..F_K)``. Approximates the explicit
  ensemble in steady state.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Mapping

import torch


def fibonacci(n: int) -> list[int]:
    """Return the first ``n`` Fibonacci numbers in the (1, 1, 2, 3, 5, ...)
    convention used by the H20 design document.

    ``fibonacci(0) -> []``; ``fibonacci(1) -> [1]``;
    ``fibonacci(2) -> [1, 1]``; ``fibonacci(8) -> [1, 1, 2, 3, 5, 8, 13, 21]``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    if n == 0:
        return []
    if n == 1:
        return [1]
    out = [1, 1]
    for _ in range(2, n):
        out.append(out[-1] + out[-2])
    return out


def fib_weights(K: int, normalise: bool = False) -> list[float]:
    """Return Fibonacci weights of length ``K``.

    If ``normalise=True``, the weights are divided by their sum so the
    output is a probability distribution summing to 1.
    """
    raw = [float(x) for x in fibonacci(K)]
    if normalise:
        s = sum(raw)
        if s <= 0:
            raise ValueError("sum of Fibonacci weights is zero")
        return [w / s for w in raw]
    return raw


class FibEnsemble:
    """Explicit-buffer Fibonacci-weighted ensemble.

    The buffer is a FIFO of the last ``K`` state dicts. Order
    convention: ``checkpoints[0]`` is the *oldest* surviving snapshot,
    ``checkpoints[-1]`` is the most recent. The corresponding
    Fibonacci weight is ``F_{i+1}`` so the newest snapshot gets the
    largest weight (``F_K = 21`` for ``K=8``).

    Use ``update(state_dict)`` after each epoch (or at any cadence) and
    call ``averaged_state_dict()`` at inference time. The output is a
    plain ``OrderedDict`` of CPU tensors with the same keys and shapes
    as the inputs.
    """

    def __init__(self, K: int = 8) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        self.K = int(K)
        self.weights = fib_weights(K)
        self.total = float(sum(self.weights))
        self.checkpoints: list[Mapping[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.checkpoints)

    @property
    def is_full(self) -> bool:
        return len(self.checkpoints) >= self.K

    def update(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Push a deep-cloned snapshot. Oldest is evicted at K."""
        snap = OrderedDict(
            (k, v.detach().clone()) for k, v in state_dict.items()
        )
        self.checkpoints.append(snap)
        if len(self.checkpoints) > self.K:
            self.checkpoints.pop(0)

    def _active_weights(self) -> list[float]:
        """Use the last ``len(self.checkpoints)`` Fibonacci weights so
        the newest snapshot still receives the heaviest weight even when
        the buffer is not yet full.
        """
        n = len(self.checkpoints)
        if n == 0:
            raise RuntimeError("no checkpoints to average")
        return self.weights[-n:]

    def averaged_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        if not self.checkpoints:
            raise RuntimeError("no checkpoints to average")
        w = self._active_weights()
        total = float(sum(w))
        keys = list(self.checkpoints[0].keys())
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k in keys:
            ref = self.checkpoints[0][k]
            if not torch.is_tensor(ref):
                out[k] = ref
                continue
            if not torch.is_floating_point(ref):
                # Integer buffers (e.g. BN num_batches_tracked) — take
                # the most recent value rather than averaging.
                out[k] = self.checkpoints[-1][k].detach().clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for wi, cp in zip(w, self.checkpoints):
                acc.add_(cp[k].to(torch.float32), alpha=float(wi))
            acc.div_(total)
            out[k] = acc.to(ref.dtype)
        return out

    def load_into(self, model: torch.nn.Module) -> None:
        """Load the averaged state-dict into ``model`` in place."""
        model.load_state_dict(self.averaged_state_dict(), strict=True)


class FibEMA:
    """O(1)-memory Fibonacci-decay running average.

    Maintains a single tensor of running parameters. On ``update``:

        ``running <- (1 - w_new) * running + w_new * new``

    where ``w_new = F_K / sum(F_1..F_K)`` (``= 21/54 ~= 0.389`` for
    ``K = 8``). Approximates the steady-state behaviour of the
    explicit :class:`FibEnsemble` while using only one model's worth
    of memory.
    """

    def __init__(self, K: int = 8) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        self.K = int(K)
        self.weights = fib_weights(K)
        self.total = float(sum(self.weights))
        self.w_new = self.weights[-1] / self.total
        self.avg: OrderedDict[str, torch.Tensor] | None = None

    def update(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        if self.avg is None:
            self.avg = OrderedDict(
                (k, v.detach().clone().to(torch.float32)
                    if torch.is_tensor(v) and torch.is_floating_point(v)
                    else v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in state_dict.items()
            )
            self._orig_dtypes = {
                k: (v.dtype if torch.is_tensor(v) else None)
                for k, v in state_dict.items()
            }
            return
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            if not torch.is_floating_point(v):
                # most-recent semantics for integer buffers
                self.avg[k] = v.detach().clone()
                continue
            self.avg[k].mul_(1 - self.w_new).add_(
                v.detach().to(torch.float32), alpha=self.w_new
            )

    def averaged_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        if self.avg is None:
            raise RuntimeError("no update() calls yet")
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in self.avg.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                target_dtype = self._orig_dtypes.get(k, v.dtype)
                out[k] = v.to(target_dtype) if target_dtype is not None else v
            else:
                out[k] = v
        return out
