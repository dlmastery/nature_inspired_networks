"""H13 — Golden Neuron Connectivity (phi-sparse Linear / Conv2d wrappers).

Sparse linear and conv layers whose binary connectivity mask keeps a
fraction ``density = 1/phi`` (~ 0.618) of weights, mirroring the cortical
microcircuit recurrence probability (Markram 2015; Loomba 2022). The
complementary sparsity ``1 - 1/phi = 2 - phi`` ~ 0.382 is the
phi-modulated dropout-during-init prior described in
``hypotheses/g2_layer_channel_neuron/H13_golden_neuron_connectivity.md``.

Two mask-generation strategies are supported:

* ``"random"``  — Bernoulli mask with probability ``density``. Used at
  init when no warm-up dense weights are available.
* ``"magnitude"`` — top-``density`` of ``|W|`` (post-warmup magnitude
  pruning, Han et al 2015). Available via
  :func:`magnitude_prune_to_phi`.

The mask is a registered buffer (binary float in {0, 1}); gradients flow
to the masked-out positions as zero, so the optimiser does not move them
but they remain allocated tensors. Effective parameter count is
``round(numel * density)``.

A drop-in NaturePriorNet variant (``"natureprior_phi_sparse"``) swaps
the classifier head ``nn.Linear`` for :class:`PhiSparseLinear` and
registers itself with :func:`build_model` at import time so the
``scripts/run_sweep.py`` row ``sg_only_phi_sparse`` resolves without
editing ``models.py``.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PHI


DEFAULT_DENSITY: float = 1.0 / PHI  # 0.6180339...


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------
def phi_sparse_mask(shape: Iterable[int], density: float = DEFAULT_DENSITY,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Bernoulli binary mask of given ``shape`` with E[mask]=``density``.

    Each entry is independently sampled; expected sparsity is
    ``1 - density``. For the canonical golden-neuron prior,
    ``density = 1/phi``.
    """
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1]; got {density}")
    if generator is None:
        u = torch.rand(*shape)
    else:
        u = torch.rand(*shape, generator=generator)
    return (u < density).float()


def magnitude_prune_mask(weight: torch.Tensor,
                         density: float = DEFAULT_DENSITY) -> torch.Tensor:
    """Top-``density`` magnitude mask. Han et al 2015 magnitude pruning."""
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1]; got {density}")
    flat = weight.detach().abs().flatten()
    k = max(1, int(round(flat.numel() * density)))
    if k >= flat.numel():
        return torch.ones_like(weight)
    threshold = torch.topk(flat, k, largest=True).values.min()
    mask = (weight.detach().abs() >= threshold).float()
    return mask


# ---------------------------------------------------------------------------
# Sparse layers
# ---------------------------------------------------------------------------
class PhiSparseLinear(nn.Module):
    """Dense Linear layer with a fixed binary connectivity mask.

    Forward: ``y = (W * M) @ x + b`` where ``M`` is binary with
    ``E[M] = density``. The mask is a buffer (no gradients) but the
    masked weights still participate in autograd via the multiplication.

    The mask can be reset post-warmup via :meth:`reset_mask_magnitude`,
    matching the recipe in section 5.1 of the H13 design doc.
    """

    def __init__(self, in_features: int, out_features: int,
                 density: float = DEFAULT_DENSITY, bias: bool = True,
                 strategy: str = "random",
                 generator: torch.Generator | None = None) -> None:
        super().__init__()
        if strategy not in {"random", "ones"}:
            raise ValueError(
                f"strategy must be 'random' or 'ones'; got {strategy!r}. "
                "Use magnitude_prune_to_phi() for post-warmup magnitude masks."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.density = float(density)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        if strategy == "ones":
            mask = torch.ones(out_features, in_features)
        else:
            mask = phi_sparse_mask((out_features, in_features),
                                   density=density, generator=generator)
        self.register_buffer("mask", mask)

    @property
    def effective_param_count(self) -> int:
        """Number of weight entries the mask keeps active (mask==1)."""
        return int(self.mask.sum().item())

    def reset_mask_magnitude(self, density: float | None = None) -> None:
        """Replace the mask with the magnitude-pruned version of the
        current dense weight. Call AFTER a brief dense warm-up.
        """
        d = self.density if density is None else float(density)
        new = magnitude_prune_mask(self.weight, density=d)
        # Buffers are not parameters; replace in place to keep the
        # registered-buffer status intact.
        self.mask.copy_(new)
        self.density = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"density={self.density:.6f}, "
                f"bias={self.bias is not None}")


class PhiSparseConv2d(nn.Module):
    """Conv2d with a fixed binary mask over the full weight tensor.

    The mask shape matches ``(out, in, k, k)``; each entry is Bernoulli
    with probability ``density``. Same forward contract as
    :class:`PhiSparseLinear`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 density: float = DEFAULT_DENSITY, bias: bool = False,
                 generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.density = float(density)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        mask = phi_sparse_mask(self.weight.shape, density=density,
                               generator=generator)
        self.register_buffer("mask", mask)

    @property
    def effective_param_count(self) -> int:
        return int(self.mask.sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, stride=self.stride,
                        padding=self.padding)


# ---------------------------------------------------------------------------
# Dense -> sparse converter (post-warmup magnitude pruning)
# ---------------------------------------------------------------------------
def magnitude_prune_to_phi(linear: nn.Linear,
                           density: float = DEFAULT_DENSITY) -> PhiSparseLinear:
    """Replace a dense ``nn.Linear`` with a :class:`PhiSparseLinear`
    whose mask keeps the top-``density`` fraction of weights by ``|W|``
    (Han et al 2015 magnitude pruning).
    """
    has_bias = linear.bias is not None
    out = PhiSparseLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        density=density,
        bias=has_bias,
        strategy="ones",
    )
    with torch.no_grad():
        out.weight.copy_(linear.weight)
        if has_bias:
            out.bias.copy_(linear.bias)
        mask = magnitude_prune_mask(linear.weight, density=density)
        out.mask.copy_(mask)
    out.density = float(density)
    return out


# ---------------------------------------------------------------------------
# Drop-in NaturePriorNet variant with phi-sparse classifier head
# ---------------------------------------------------------------------------
class PhiSparseNaturePriorNet(nn.Module):
    """NaturePriorNet whose classifier head is a :class:`PhiSparseLinear`.

    The hypothesis is that intra-layer connectivity at density 1/phi
    matches the dense baseline within +/- 0.3 pp top-1 while shedding
    ~38 pct of head params (H13). This wrapper does not modify the
    backbone, keeping the change Rule-1 atomic: one head swap.
    """

    def __init__(self, num_classes: int = 10, channel_mode: str = "fib",
                 flags=None, density: float = DEFAULT_DENSITY) -> None:
        super().__init__()
        # Local import to avoid a circular dep at module load time.
        from .blocks import NaturePriorFlags
        from .models import NaturePriorConfig, NaturePriorNet
        cfg = NaturePriorConfig(
            num_classes=num_classes, channel_mode=channel_mode,
            flags=flags or NaturePriorFlags(),
        )
        base = NaturePriorNet(cfg)
        self.cfg = base.cfg
        self.widths = base.widths
        self.stem = base.stem
        self.stages = base.stages
        self.pool = base.pool
        # Swap the dense classifier head for the phi-sparse variant
        old_fc: nn.Linear = base.fc  # type: ignore[assignment]
        self.fc = PhiSparseLinear(
            in_features=old_fc.in_features,
            out_features=old_fc.out_features,
            density=density,
            bias=old_fc.bias is not None,
        )
        # Carry over the initialised dense weights so the head starts at the
        # same point as a dense run, with only the mask differing.
        with torch.no_grad():
            self.fc.weight.copy_(old_fc.weight)
            if old_fc.bias is not None:
                self.fc.bias.copy_(old_fc.bias)
        self.density = float(density)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Self-register the variant with build_model so the sweep row resolves
# without editing models.py (Rule 14 — shared primitives, single import).
# ---------------------------------------------------------------------------
def _register_phi_sparse_variant() -> None:
    from . import models as _models
    from . import runner as _runner

    original = _models.build_model
    if getattr(original, "_phi_sparse_wrapped", False):
        return

    def build_model(name: str, num_classes: int, flags=None,
                    channel_mode: str = "fib"):
        if name.lower() in {"natureprior_phi_sparse", "phi_sparse_natureprior"}:
            return PhiSparseNaturePriorNet(
                num_classes=num_classes, channel_mode=channel_mode,
                flags=flags, density=DEFAULT_DENSITY,
            )
        return original(name, num_classes, flags=flags,
                        channel_mode=channel_mode)

    build_model._phi_sparse_wrapped = True  # type: ignore[attr-defined]
    build_model._original = original  # type: ignore[attr-defined]
    _models.build_model = build_model  # type: ignore[assignment]
    # runner imports build_model by name at module load — patch that
    # binding too so the runner sees the wrapped dispatcher.
    _runner.build_model = build_model  # type: ignore[assignment]


_register_phi_sparse_variant()
