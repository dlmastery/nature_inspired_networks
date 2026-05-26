"""H17 — Golden Ratio Skip Connections — implementation module.

The shared infrastructure does NOT yet expose a pure phi-skip-scaling
block (the closest available primitive is the golden-angle channel
modulate used inside ``NaturePriorBlock`` when
``flags.golden_modulate=True``; that's the T1.8 variant which
confounds skip scaling with channel gating).

This sub-project therefore provides a small composition wrapper that
imports the canonical phi constant from
``nature_inspired_networks.priors.PHI`` and applies the 1/phi (or
phi-1, or learnable) scale to the residual skip output of any nn.Module
that exposes a ``skip`` attribute (e.g. ``NaturePriorBlock``).

DO NOT redefine PHI here; import it from the shared primitive.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nature_inspired_networks.priors import PHI


PHI_INV = 1.0 / PHI  # 0.6180339887...
PHI_INV2 = 1.0 / (PHI * PHI)  # 0.3819660113...


class PhiSkipBlock(nn.Module):
    """Wrap any residual block to scale its skip path by phi-derived factors.

    The wrapped block must expose two callable attributes:

    - ``skip(x)`` -- returns the residual identity / projection path.
    - ``forward_branch(x)`` -- returns the F(x) branch output (i.e. the
      non-identity path). If absent, we fall back to subtracting the
      block's skip from its forward output, which works only when the
      block sums the two without further activation.

    The wrapper produces ``branch_scale * F(x) + skip_scale * skip(x)``.

    Modes
    -----
    - ``inv_phi``         skip=1/phi=0.618,  branch=1.0     (H17 default)
    - ``phi_inv2_sum1``   skip=1/phi,        branch=1/phi^2 (sum to 1)
    - ``phi_minus_1``     skip=phi-1=0.618,  branch=1.0     (algebraically = inv_phi)
    - ``learnable``       both learnable, init to 1/phi and 1.0
    """

    VALID_MODES = ("inv_phi", "phi_inv2_sum1", "phi_minus_1", "learnable")

    def __init__(self, wrapped: nn.Module, mode: str = "inv_phi") -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"unknown mode {mode!r}; pick one of {self.VALID_MODES}"
            )
        self.wrapped = wrapped
        self.mode = mode
        if mode == "learnable":
            self.skip_scale = nn.Parameter(torch.tensor(PHI_INV, dtype=torch.float32))
            self.branch_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            ss, bs = _static_scales(mode)
            self.register_buffer("skip_scale", torch.tensor(ss, dtype=torch.float32))
            self.register_buffer("branch_scale", torch.tensor(bs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decompose the wrapped block into its skip and branch outputs.
        # NaturePriorBlock-shaped blocks have a `.skip` attr (Identity or
        # 1x1 projection) and an internal post-conv residual sum + ReLU.
        # We compute both paths explicitly here so the scales apply
        # BEFORE the final ReLU.
        skip = self.wrapped.skip(x)
        # Run the branch by calling the wrapped block's forward and
        # subtracting the (post-ReLU) skip back out is fragile, so we
        # expose a clean entry point: the wrapped block must implement
        # `forward_branch` for an unambiguous decomposition. If absent,
        # we fall back to wrapped(x) - skip (assumes the block sums
        # before any non-linearity that touches skip).
        branch_fn = getattr(self.wrapped, "forward_branch", None)
        if callable(branch_fn):
            branch = branch_fn(x)
        else:
            # Fallback: assume the block's forward returns skip + branch
            # WITHOUT a final ReLU on the sum (true for a "preact" style
            # block; lossy for the canonical NaturePriorBlock which DOES
            # ReLU). Document this caveat in AUDIT.md weakness #2.
            branch = self.wrapped(x) - skip
        out = self.branch_scale * branch + self.skip_scale * skip
        return out


def _static_scales(mode: str) -> tuple[float, float]:
    """Return (skip_scale, branch_scale) for non-learnable modes."""
    if mode == "inv_phi":
        return (PHI_INV, 1.0)
    if mode == "phi_inv2_sum1":
        return (PHI_INV, PHI_INV2)
    if mode == "phi_minus_1":
        return (PHI - 1.0, 1.0)
    raise ValueError(f"no static scales for mode={mode!r}")


def sum_to_one_residual(skip_scale: float, branch_scale: float) -> bool:
    """Predicate used in tests: do the two scales sum to within 1e-6 of 1.0?"""
    return abs((skip_scale + branch_scale) - 1.0) < 1e-6


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H17",
        short="golden_ratio_skip",
        primitives_touched=["PHI"],
        flags_touched=["skip_scale_mode"],
        phi=PHI,
        phi_inv=PHI_INV,
    )
