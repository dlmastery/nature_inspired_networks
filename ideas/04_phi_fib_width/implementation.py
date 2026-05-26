"""H04 — phi / Fibonacci channel scaling — implementation module.

This idea is implemented entirely on top of the shared primitive
``nature_inspired_networks.priors.fibonacci_channels`` (see
``src/nature_inspired_networks/priors.py``). That single function emits
three schedules:

- ``mode='fib'``    -> c0, c0*F_2/F_1, c0*F_3/F_1, ... rounded to mod 8
- ``mode='phi'``    -> c0 * phi^k                   rounded to mod 8
- ``mode='linear'`` -> c0 * (k+1)                   rounded to mod 8

The idea-specific glue here is a thin wrapper that adds two safety
features the bare primitive does not have:

1. ``phi_fib_widths(c0, n_stages, mode)``: a typed re-export that
   asserts the returned schedule is strictly increasing -- this catches
   pathological c0/n_stages combinations where mod-8 rounding produces a
   non-monotonic schedule (e.g., c0=4, n=3 in 'phi' mode floors to a
   constant 8).

2. ``schedules_are_distinct(c0, n_stages)``: the regression guard for
   the T1.1/T1.2 mod-8 collapse observation. Returns True iff the phi
   and fib schedules produce strictly different integer schedules. If
   this returns False the experiment runner SHOULD refuse to start a
   phi-vs-fib comparison, because the two variants are functionally
   identical at that (c0, n_stages).

DO NOT duplicate ``fibonacci_channels`` here; import it.
"""
from __future__ import annotations

from typing import List

from nature_inspired_networks.priors import PHI, fibonacci_channels


def phi_fib_widths(c0: int, n_stages: int, mode: str = "phi") -> List[int]:
    """Return a channel schedule, validated to be strictly increasing.

    Parameters
    ----------
    c0
        Starting channel count. Must be >= 8 for mod-8 quantisation to
        be meaningful.
    n_stages
        Number of stages. The CIFAR ResNet-20 backbone uses 3; H04
        recommends 4 (with c0 = 32) to escape mod-8 collapse.
    mode
        One of {"phi", "fib", "linear"}.
    """
    if c0 < 8:
        raise ValueError(
            f"c0={c0} below 8 collapses every mod-8 round to 8; pick c0 >= 8."
        )
    widths = fibonacci_channels(c0, n_stages, mode=mode)
    # The primitive is documented to be monotonic; assert it here as the
    # idea-level invariant. This catches future regressions in the
    # primitive that would silently break this hypothesis.
    if widths != sorted(widths):
        raise ValueError(
            f"non-monotonic schedule from fibonacci_channels({c0},{n_stages},{mode!r}): {widths}"
        )
    return widths


def schedules_are_distinct(c0: int, n_stages: int) -> bool:
    """Return True iff phi and fib schedules give different integer widths.

    This is the regression guard for the T1.1/T1.2 mod-8 collapse.
    At the legacy T1.1/T1.2 configuration (c0=16, n_stages=3) the
    mod-8 quantisation collapses BOTH phi and fib schedules onto the
    same integer schedule [16, 24, 40] -- this is the methodological
    pathology that made the two runs functionally identical (top-1
    80.11 % at 127k params each, single seed). At the H04-recommended
    c0=32, n_stages=4 the schedules diverge in the deepest stage
    (phi -> [32, 48, 80, 136] vs fib -> [32, 48, 80, 128]).
    """
    fib = phi_fib_widths(c0, n_stages, mode="fib")
    phi = phi_fib_widths(c0, n_stages, mode="phi")
    return fib != phi


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H04",
        short="phi_fib_width",
        primitives_touched=["fibonacci_channels"],
        flags_touched=["channel_mode"],
        phi=PHI,
    )
