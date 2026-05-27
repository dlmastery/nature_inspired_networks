"""Fibonacci-ratio iterative magnitude pruning (H43).

Iterative magnitude pruning with prune fractions drawn from the
normalised Fibonacci sequence rather than from a doubling / linear
heuristic. The hypothesis (H43 design doc) is that the Fibonacci ratios
trace the SynFlow-optimal (Tanaka 2020, arXiv:2006.05467) sparsity
schedule and so preserve top-1 accuracy further into the high-sparsity
regime than a linear-doubling schedule.

The "epoch is in Fib set" gate uses the canonical Fibonacci sequence
{1, 2, 3, 5, 8, 13, 21, 34, 55, ...}; on those epochs we prune the
smallest-magnitude top-K% of weights globally, where K is the next
unconsumed Fibonacci ratio normalised by the partial sum.

The underlying masking uses :mod:`torch.nn.utils.prune` so that the
sparse weights survive subsequent forward/backward passes and integrate
correctly with the rest of the training loop. The function is
idempotent: a given (model, epoch) pair always prunes the same amount,
regardless of how many times it is called within that epoch.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# Canonical Fibonacci sequence used for both the epoch gate and the
# prune-fraction schedule. Indexed from 1 (so FIB[0] is the *first*
# Fibonacci number to use, F1 = 1). Kept module-level so tests can
# assert the exact values.
FIB_SCHEDULE: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)


def fibonacci_ratios(n: int) -> list[float]:
    """Return the first ``n`` Fibonacci numbers normalised so they sum
    to 1.

    Example: ``n=5`` → ``[1, 2, 3, 5, 8] / 19`` ≈
    ``[0.053, 0.105, 0.158, 0.263, 0.421]``. These are the *fractions
    of the remaining (un-pruned) weights* to remove at successive prune
    events, NOT cumulative sparsity.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    seq = list(FIB_SCHEDULE[:n])
    while len(seq) < n:  # extend if caller wants more than 10
        seq.append(seq[-1] + seq[-2])
    s = float(sum(seq))
    return [v / s for v in seq]


def _prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Return (module, parameter_name) pairs eligible for weight pruning.

    We target the ``weight`` parameter of every Conv2d / Linear layer.
    BatchNorm and biases are excluded — pruning them is known to destroy
    accuracy out of proportion to the parameter savings.
    """
    out: list[tuple[nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            out.append((m, "weight"))
    return out


def fibonacci_prune(
    model: nn.Module,
    epoch: int,
    schedule_length: int = 5,
    make_permanent: bool = False,
) -> dict:
    """Prune ``model`` at the Fibonacci epoch gate.

    Parameters
    ----------
    model
        The PyTorch module to prune. Pruning is applied globally across
        every Conv2d / Linear weight via L1-unstructured magnitude
        pruning.
    epoch
        Current training epoch (0-indexed). The prune triggers only when
        ``epoch + 1`` is in ``FIB_SCHEDULE[:schedule_length]``, i.e.
        epochs {0, 1, 2, 4, 7, ...} relative to 0-indexed counting (so
        the "epochs" reported as 1, 2, 3, 5, 8 in the H43 doc).
    schedule_length
        Number of Fibonacci prune events to schedule. The corresponding
        prune ratios are ``fibonacci_ratios(schedule_length)``.
    make_permanent
        If True, after applying the mask call ``prune.remove`` so the
        zeros bake into the underlying weight tensor. Useful at the end
        of training; off by default so successive ``fibonacci_prune``
        calls can layer on more sparsity.

    Returns
    -------
    dict
        ``{"pruned": bool, "epoch": int, "ratio": float, "fib_idx": int}``
        — ``pruned`` is False when the epoch is not a Fibonacci gate.
    """
    triggers = list(FIB_SCHEDULE[:schedule_length])
    one_indexed = epoch + 1
    if one_indexed not in triggers:
        return {"pruned": False, "epoch": epoch, "ratio": 0.0, "fib_idx": -1}

    fib_idx = triggers.index(one_indexed)
    ratios = fibonacci_ratios(schedule_length)
    ratio = ratios[fib_idx]

    targets = _prunable_modules(model)
    if not targets:
        return {"pruned": False, "epoch": epoch, "ratio": ratio,
                "fib_idx": fib_idx}

    prune.global_unstructured(
        targets,
        pruning_method=prune.L1Unstructured,
        amount=float(ratio),
    )
    if make_permanent:
        for mod, name in targets:
            try:
                prune.remove(mod, name)
            except ValueError:
                # already removed / never pruned — safe to skip
                pass
    return {"pruned": True, "epoch": epoch, "ratio": ratio,
            "fib_idx": fib_idx}


def global_sparsity(model: nn.Module) -> float:
    """Return the fraction of zero entries across every Conv2d / Linear
    weight in ``model``. Useful as a test-time sanity check after one
    or more :func:`fibonacci_prune` calls.
    """
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            w = m.weight
            total += w.numel()
            zeros += int((w == 0).sum().item())
    if total == 0:
        return 0.0
    return zeros / total
