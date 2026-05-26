"""H05 — Fractal phi-Recursion — implementation module.

This idea is implemented entirely on top of the shared building blocks
in ``nature_inspired_networks.blocks``:

- ``NaturePriorBlock`` with ``flags.fractal=True`` swaps the second conv
  for a depth-N recursive sub-block (``_FractalPath``).
- ``_FractalPath`` itself implements the depth-recursion (binary
  fractal a la Larsson 2017): at depth=1 a single conv, at depth=2 two
  parallel branches averaged by 0.5.

The idea-specific glue here is:

1. ``build_fractal_block(c_in, c_out, stride, depth)``: a thin builder
   that returns a ``NaturePriorBlock`` with all OTHER priors OFF, so
   the fractal effect is isolated for ablation. This is the
   ``sg_only_fractal`` configuration from the previous CIFAR sweep
   (T1.5).

2. ``predicted_param_factor(depth)``: a closed-form prediction of the
   param-count blow-up vs. the priors-off baseline as a function of
   ``fractal_depth``. Used by the audit and as a sanity check in the
   tests.

DO NOT duplicate ``_FractalPath`` here; import it.
"""
from __future__ import annotations

import torch.nn as nn

from nature_inspired_networks.blocks import NaturePriorBlock, NaturePriorFlags


def fractal_only_flags() -> NaturePriorFlags:
    """The 'sg_only_fractal' flag combo from the previous CIFAR sweep.

    Every other prior is off; only the fractal sub-block is enabled.
    This is the configuration that T1.5 ran on a single seed and that
    H05 needs to reproduce at 3 seeds.
    """
    return NaturePriorFlags(
        hex=False,
        group=False,
        fractal=True,
        toroidal=False,
        cymatic_init=False,
        golden_modulate=False,
        group_reduce="max",
    )


def build_fractal_block(
    c_in: int,
    c_out: int,
    stride: int = 1,
    depth: int = 2,
) -> nn.Module:
    """Return a NaturePriorBlock with ONLY the fractal prior enabled.

    Parameters
    ----------
    c_in, c_out, stride
        Standard residual-block plumbing.
    depth
        Fractal recursion depth. ``depth=1`` reduces to a plain
        ``_GenericConv`` (no recursion). ``depth=2`` is the H05 default
        and matches T1.5 `sg_only_fractal`.
    """
    if depth < 1:
        raise ValueError(f"fractal depth must be >= 1, got {depth}")
    return NaturePriorBlock(
        c_in=c_in,
        c_out=c_out,
        stride=stride,
        flags=fractal_only_flags(),
        fractal_depth=depth,
    )


def predicted_param_factor(depth: int) -> float:
    """Closed-form param-blow-up factor relative to a non-fractal block.

    At ``depth=1`` the block degenerates to plain conv1 + conv2, so the
    factor is 1.0. At ``depth=2`` the second conv is replaced by a
    fractal path with two branches (a + b1 + b2(depth=1)), which is
    3 convs instead of 1 -- factor ~ (1 + 3) / (1 + 1) = 2.0 at the
    conv-count level. The empirical T1.5 observation was +104 %, i.e.
    factor 2.04. At depth=3 the recursion compounds again: factor ~
    (1 + 5) / 2 = 3.0.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if depth == 1:
        return 1.0
    # Each extra fractal level adds 2 convs to the second branch:
    # depth=2 -> 3 inner convs (a, b1, b2-as-conv), block total 4 convs
    # depth=3 -> 5 inner convs, block total 6 convs
    # baseline (no fractal) -> 2 convs (conv1, conv2)
    inner = 2 * depth - 1
    return (1 + inner) / 2.0


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H05",
        short="fractal_phi_recursion",
        primitives_touched=["_FractalPath", "NaturePriorBlock"],
        flags_touched=["fractal", "fractal_depth"],
    )
