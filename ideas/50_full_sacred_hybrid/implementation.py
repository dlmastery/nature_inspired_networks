"""H50 - Full Sacred-Geometry Hybrid - implementation module.

This is the "all priors on" composition that was empirically falsified
in T1.9 (`sg_full_fib`, composite 0.6966 - the WORST single row in
the 11-run CIFAR-10 sweep, Δ -0.1169 vs the `sg_chan_fib` reference).

The implementation re-exports the canonical block and flags from
``nature_inspired_networks.blocks`` and provides a single factory
that returns the falsified configuration. **Do not duplicate the
block here** - the failure mode is in the COMPOSITION, not the
primitives.
"""
from __future__ import annotations

from nature_inspired_networks.blocks import NaturePriorBlock, NaturePriorFlags
from nature_inspired_networks.priors import GroupConv2d  # noqa: F401


def full_hybrid_flags(group_reduce: str = "max") -> NaturePriorFlags:
    """Return the flag combination that produced the T1.9 negative result.

    Parameters
    ----------
    group_reduce : {"max", "mean"}
        The H58 follow-up tried `reduce="mean"` to see if the +5 to +10 pp
        prediction would recover the top-1; it did NOT (see
        `ideas/58_group_avg_pool/`). Default stays `"max"` to match
        the falsified T1.9 archive bit-for-bit.
    """
    return NaturePriorFlags(
        hex=True,
        group=True,
        fractal=True,
        toroidal=True,
        cymatic_init=True,
        golden_modulate=True,
        group_reduce=group_reduce,
    )


def build_full_hybrid_block(c_in: int, c_out: int, stride: int = 1,
                            group_reduce: str = "max") -> NaturePriorBlock:
    """Build one `NaturePriorBlock` with the falsified six-prior combo."""
    return NaturePriorBlock(c_in, c_out, stride=stride,
                            flags=full_hybrid_flags(group_reduce=group_reduce))


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H50",
        short="full_sacred_hybrid",
        primitives_touched=[],  # no new primitives; pure composition
        flags_touched=[
            "hex", "group", "fractal", "toroidal",
            "cymatic_init", "golden_modulate",
        ],
        # Empirical numbers from `experiments/cifar10/sg_full_fib_seed0/metrics.json`
        falsifier_status="disproved",
        observed_top1=0.7324,
        observed_composite=0.6966,
        reference_top1=0.8011,
        reference_composite=0.8135,
        delta_composite=-0.1169,
    )
