"""nature-inspired compound-scaling primitives for the G1 design space.

This module hosts the closed-form scaling rules used by H01 (phi-compound
scaling) and H02 (Fibonacci depth progression). The functions are
intentionally tiny and dependency-free so that other modules / ideas can
import them as primitives without pulling the whole model graph.

Refs:
  Tan, Le 2019 ICML 'EfficientNet' (arXiv:1905.11946)
  Larsson, Maire, Shakhnarovich 2017 ICLR 'FractalNet' (arXiv:1605.07648)
  He et al. 2016 CVPR 'Deep Residual Learning' (arXiv:1512.03385)
"""
from __future__ import annotations

from typing import Sequence

from .priors import PHI, _round8


# ---------------------------------------------------------------------------
# H01 — phi-compound scaling: replace EfficientNet alpha/beta/gamma with
# successive phi-powers for depth, width, and resolution.
# ---------------------------------------------------------------------------
def phi_compound(
    k: int,
    base_depth: int = 18,
    base_width: int = 64,
    base_res: int = 32,
) -> tuple[int, int, int]:
    """Return ``(depth, width, resolution)`` for phi-compound stage index k.

    Following H01 sec. 5.1:

        depth      = base_depth * phi**k
        width      = base_width * phi**(k/2)   (rounded to multiples of 8)
        resolution = base_res   * phi**(k/4)   (rounded to multiples of 16)

    For k = 0 the triple reduces to ``(base_depth, base_width, base_res)``
    so a k=0 configuration reproduces the unscaled baseline exactly. ``k``
    may be negative for sub-scaling (e.g. ``k=-1`` shrinks the family for
    small-budget ablations) — channel/resolution floors guard against
    sub-8 / sub-16 values.
    """
    if not isinstance(k, int):
        raise TypeError(f"phi_compound expects int k, got {type(k).__name__}")
    if base_depth < 1 or base_width < 1 or base_res < 1:
        raise ValueError("phi_compound base_* must be >= 1")
    d = max(1, int(round(base_depth * (PHI ** k))))
    w_raw = base_width * (PHI ** (k / 2.0))
    w = max(8, 8 * int(round(w_raw / 8.0)))
    r_raw = base_res * (PHI ** (k / 4.0))
    r = max(16, 16 * int(round(r_raw / 16.0)))
    return d, w, r


def phi_compound_channels(
    k: int,
    n_stages: int = 3,
    base_width: int = 16,
) -> list[int]:
    """Channel schedule for an n-stage backbone at phi-compound scale k.

    The stem width is ``phi_compound(k, base_width=base_width)`` and each
    successive stage multiplies by phi (the H01 width recurrence). The
    function is the H01 analogue of ``priors.fibonacci_channels`` and
    keeps the same div-by-8 rounding contract.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    _, w0, _ = phi_compound(k, base_width=base_width)
    return [_round8(int(round(w0 * (PHI ** s)))) for s in range(n_stages)]


# ---------------------------------------------------------------------------
# H02 — Fibonacci depth progression: stage block counts follow Fib seq.
# ---------------------------------------------------------------------------
def fibonacci_sequence(n: int) -> list[int]:
    """Return the first n terms of the canonical Fibonacci sequence
    starting at 1, 1, 2, 3, 5, 8, ...

    H02 references the sequence variant starting (1, 1); the
    ``fibonacci_depths`` helper below picks a shifted slice so that the
    canonical CIFAR depth schedule is ``[3, 5, 8, 13]``.
    """
    if n < 1:
        return []
    out = [1]
    if n >= 2:
        out.append(1)
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out


def fibonacci_depths(n_stages: int, start_index: int = 3) -> list[int]:
    """Per-stage block counts following Fibonacci.

    ``start_index`` selects the offset into the (1, 1, 2, 3, 5, 8, 13,
    21, 34) sequence at which the first stage begins. With the default
    ``start_index=3`` and ``n_stages=4`` the schedule is the canonical
    H02 spec ``[3, 5, 8, 13]`` (total depth 29). For 5 stages with
    ``start_index=2`` the schedule is ``[2, 3, 5, 8, 13]``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    needed = start_index + n_stages
    seq = fibonacci_sequence(needed)
    return seq[start_index:start_index + n_stages]


def resolve_blocks_schedule(
    blocks_per_stage: int | Sequence[int],
    n_stages: int,
    mode: str = "uniform",
    fib_start: int = 3,
) -> list[int]:
    """Resolve a depth schedule from either a scalar, a list, or a mode.

    - ``mode='uniform'``: replicate ``blocks_per_stage`` n_stages times
      (existing behaviour, preserves legacy configs byte-for-byte).
    - ``mode='fib'``  : H02 Fibonacci schedule via ``fibonacci_depths``.
    - ``mode='linear'``: control schedule ``[base, base+1, base+2, ...]``.
    - If a sequence is passed, use it verbatim (length must match
      ``n_stages``).
    """
    if isinstance(blocks_per_stage, (list, tuple)):
        sched = list(blocks_per_stage)
        if len(sched) != n_stages:
            raise ValueError(
                f"blocks_per_stage list length {len(sched)} != n_stages {n_stages}"
            )
        return [int(b) for b in sched]
    base = int(blocks_per_stage)
    if base < 1:
        raise ValueError("blocks_per_stage must be >= 1")
    if mode == "uniform":
        return [base] * n_stages
    if mode == "fib":
        return fibonacci_depths(n_stages, start_index=fib_start)
    if mode == "linear":
        return [base + k for k in range(n_stages)]
    raise ValueError(f"unknown blocks_mode '{mode}'")
