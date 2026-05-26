"""H58 - C4 group conv avg-pool reduction (DISCARDED).

This module is intentionally minimal: it re-exports
``nature_inspired_networks.priors.GroupConv2d`` with the
``reduce='mean'`` configuration that H58 hypothesised would fix the
dominant negative term in the H50 full-hybrid failure.

============================== DISCARD VERDICT ==============================
The H58 experiment was run on 2026-05-27 (`sg_only_group_avg`,
`sg_full_fib_avg`). Both rows underperformed their max-pool
counterparts:

  * `sg_only_group_avg`  top-1 65.38% vs `sg_only_group`  69.84%  (Δ -4.46pp)
  * `sg_full_fib_avg`    top-1 66.86% vs `sg_full_fib`    73.24%  (Δ -6.38pp)

The intuition "max-pool over the C4 orbit throws away 75% of the
signal" treated the 4 orbit channels as independent. They are not -
they are correlated rotated copies of the same convolution. `amax`
over orientations is a SOFT ARGMAX: at each spatial location it picks
the rotation whose receptive field best matches the input. Mean-pool
DILUTES that response with the three non-matching orientations,
degrading contrast at every layer.

The actual fix is the DATA, not the reduction operator: C4
equivariance pays off when the data has rotational variance
(rotated CIFAR, IcoMNIST, spherical MNIST). On canonically oriented
CIFAR-10 the equivariance prior is unused by the task, and no choice
of reduction makes that go away.

See ``FINDINGS.md`` § "H58 follow-up - the avg-pool fix DISCARDED"
and ``ideas/58_group_avg_pool/IDEA.md`` for the full record.
=============================================================================
"""
from __future__ import annotations

import torch.nn as nn

from nature_inspired_networks.priors import GroupConv2d


def make_avg_pool_group_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    group: str = "c4",
    bias: bool = False,
) -> nn.Module:
    """Construct a ``GroupConv2d`` with ``reduce='mean'`` (the H58 variant).

    This factory exists so call-sites do not pass the (DISCARDED) string
    "mean" directly - they go through this function, which has a docstring
    explaining the verdict. Use ``GroupConv2d(..., reduce='max')`` for the
    default, which the empirical sweep shows is the better choice on
    canonically-oriented data.
    """
    return GroupConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        group=group,
        bias=bias,
        reduce="mean",
    )


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H58",
        short="group_avg_pool",
        primitives_touched=["GroupConv2d"],
        flags_touched=["group", "group_reduce"],
        # Empirical numbers from
        # experiments/cifar10/sg_only_group_avg_seed0/metrics.json
        # experiments/cifar10/sg_full_fib_avg_seed0/metrics.json
        falsifier_status="discarded",
        sg_only_group_max_top1=0.6984,
        sg_only_group_avg_top1=0.6538,
        sg_only_group_delta_top1=-0.0446,
        sg_full_fib_max_top1=0.7324,
        sg_full_fib_avg_top1=0.6686,
        sg_full_fib_delta_top1=-0.0638,
        # The future direction: rotate the DATA, not the operator
        future_direction="rotated_cifar10",
    )
