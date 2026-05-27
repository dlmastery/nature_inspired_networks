"""H76 — TetrahedralDualPathBlock.

A residual block with TWO complementary C4 group-equivariant convolutional
paths whose outputs are fused by a learnable convex (balanced) merge.

Neutral framing
---------------
The two paths reuse the already-validated :class:`GroupConv2d`:

- **Path A** uses ``reduce='max'`` — a soft argmax over the 4-rotation orbit
  that selects the strongest orientation response at every spatial location.
- **Path B** uses ``reduce='mean'`` — the complementary pooling that averages
  responses over the orbit.

A single learnable scalar ``beta`` (squashed to ``[0, 1]`` via a sigmoid)
convex-combines them: ``out = beta * A + (1 - beta) * B``. The block keeps the
spatial / channel shape of a standard residual block so it is CNN-droppable.

The H58 finding established that the two C4 orbit reductions behave
differently (max dominates mean by 4-6 pp on CIFAR-10). Rather than committing
to one pooling, this block tests a *balanced* fusion in which the network
learns the mixing coefficient. The esoteric origin (the "Merkaba"
star-tetrahedron dual-polarity figure of two interpenetrating tetrahedra) is
acknowledged only as the source intuition for pairing two complementary
orientation aggregations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .priors import GroupConv2d


class TetrahedralDualPathBlock(nn.Module):
    """Dual complementary C4-equivariant conv paths with a learnable convex merge.

    Parameters
    ----------
    in_channels, out_channels : int
        Standard conv channel counts. ``out_channels`` is the merged output
        channel count (both paths produce ``out_channels`` and are combined
        element-wise, so the merge does not change channel count).
    kernel_size, stride, padding : int
        Passed to both :class:`GroupConv2d` paths.
    group : str
        ``"c4"`` (default) or ``"d4"``; passed to both paths.
    beta_init : float
        Initial value of the convex-merge coefficient in ``[0, 1]``. The
        underlying raw parameter is initialised to ``logit(beta_init)`` so that
        ``sigmoid(beta_raw) == beta_init`` at construction. Default ``0.5``
        (perfectly balanced).
    residual : bool
        If ``True`` (default) and ``in_channels == out_channels`` and
        ``stride == 1``, add an identity skip connection so the block is a
        drop-in residual block. Otherwise the merged dual-path output is
        returned directly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        group: str = "c4",
        beta_init: float = 0.5,
        residual: bool = True,
    ) -> None:
        super().__init__()
        assert 0.0 < beta_init < 1.0, "beta_init must be strictly inside (0, 1)"
        # Path A: max-pool orbit reduction (selects strongest orientation).
        self.path_max = GroupConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, group=group, reduce="max",
        )
        # Path B: mean-pool orbit reduction (averages orientations).
        self.path_mean = GroupConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, group=group, reduce="mean",
        )
        # Learnable convex-merge coefficient, stored as a raw logit so that
        # sigmoid(beta_raw) lands inside (0, 1) without clamping.
        beta_logit = torch.logit(torch.tensor(float(beta_init)))
        self.beta_raw = nn.Parameter(beta_logit)
        self.use_residual = residual and (in_channels == out_channels) and (stride == 1)

    @property
    def beta(self) -> torch.Tensor:
        """The convex-merge coefficient in ``[0, 1]`` (``sigmoid(beta_raw)``)."""
        return torch.sigmoid(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.path_max(x)   # strongest-orientation path
        b = self.path_mean(x)  # averaged-orientation path
        beta = torch.sigmoid(self.beta_raw)
        out = beta * a + (1.0 - beta) * b
        if self.use_residual:
            out = out + x
        return out


# TODO runner wiring:
# To integrate with the existing model factory WITHOUT touching models.py here,
# the lead would add a `channel_mode` / `block` branch that, when a config flag
# such as `flags: {tetra_dualpath: true}` is set, swaps the standard residual
# block constructor for `TetrahedralDualPathBlock(in_c, out_c, stride=stride,
# group="c4", beta_init=0.5, residual=True)`. A single sweep row in
# `configs/cifar10_quick.yaml` flips that one flag (Rule 1: one config change
# per experiment). The `beta` property can be logged each epoch to `history.json`
# so the learned max/mean mixing coefficient is observable in the dashboard.
