"""H35 - Cymatic Wavelet Kernels - implementation module.

Re-exports the Chladni-mode basis and the in-place cymatic-init from
``nature_inspired_networks.priors`` and adds a Gram-Schmidt orthonormal
variant that fixes the two bugs identified in T1.7
(`sg_only_cymatic_init` on CIFAR-10):

  1. all output channels were initialised from the same low-frequency
     mode and ended up highly correlated;
  2. the (1,1) mode dominates and is effectively DC on a 3x3 kernel.

The corrected init draws (m, n) from a configurable band, builds N
distinct modes for the N output channels, Gram-Schmidt orthonormalises
across channels, and rescales by the He standard deviation so
BatchNorm sees the same variance it would after Kaiming init.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from nature_inspired_networks.priors import chladni_modes, cymatic_init_  # noqa: F401


def chladni_modes_banded(
    n_modes: int,
    k: int,
    band: tuple[int, int] = (2, 5),
    seed: int | None = 0,
) -> torch.Tensor:
    """Build `n_modes` Chladni eigenmodes whose (m, n) frequencies are
    sampled uniformly from `[band[0], band[1]]`, then Gram-Schmidt
    orthonormalize across modes.

    Returns a tensor of shape (n_modes, k, k) with unit Frobenius norm
    per mode and pairwise inner-product approximately zero.
    """
    lo, hi = band
    assert lo >= 1 and hi >= lo, f"invalid band {band}"
    rng = random.Random(seed)
    xs = torch.linspace(0, math.pi, k + 2)[1:-1]
    ys = torch.linspace(0, math.pi, k + 2)[1:-1]
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    modes: list[torch.Tensor] = []
    for _ in range(n_modes):
        m = rng.randint(lo, hi)
        n = rng.randint(lo, hi)
        mode = torch.sin(m * X) * torch.sin(n * Y)
        modes.append(mode)
    stack = torch.stack(modes, dim=0)  # (n_modes, k, k)

    # Gram-Schmidt across the `n_modes` axis. QR on the (k*k, n_modes)
    # matrix gives an orthonormal column basis, which we reshape back.
    flat = stack.reshape(n_modes, -1).t()  # (k*k, n_modes)
    q, _ = torch.linalg.qr(flat, mode="reduced")  # (k*k, min(n_modes, k*k))
    n_keep = min(n_modes, k * k)
    q = q[:, :n_keep]
    out = q.t().reshape(n_keep, k, k)
    if n_keep < n_modes:
        # If we asked for more modes than k*k can support, cycle the
        # orthonormal columns back through.
        reps = (n_modes + n_keep - 1) // n_keep
        out = out.repeat(reps, 1, 1)[:n_modes]
    return out


def cymatic_init_ortho_(
    conv: nn.Conv2d,
    band: tuple[int, int] = (2, 5),
    seed: int = 0,
) -> None:
    """Initialize Conv2d weights from BAND-randomised, ORTHONORMALISED
    Chladni eigenmodes, then rescale to the He-normal variance.

    This is the corrected variant queued for the H35 re-run after T1.7's
    -2.67 pp negative on CIFAR-10. The key differences vs the legacy
    ``cymatic_init_`` (which lives in ``priors.py``):

    - distinct (m, n) pairs per output channel (was: same modes
      reused across channels);
    - Gram-Schmidt orthonormalisation across the n_out channels (was:
      no orthonormalisation, channels were highly correlated);
    - frequency band defaults to (2, 5) instead of (1, 1) - the latter
      is effectively DC on a 3x3 kernel and adds no spatial structure.
    """
    out_c, in_c, kh, kw = conv.weight.shape
    assert kh == kw, "cymatic init expects square kernel"
    basis = chladni_modes_banded(out_c, kh, band=band, seed=seed)  # (out_c, k, k)
    fan_in = in_c * kh * kw
    he_std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        for o in range(out_c):
            mode = basis[o]
            for i in range(in_c):
                # use a deterministic sign per (o, i) so different input
                # channels see anti-symmetric copies, not literal repeats
                sign = 1.0 if ((o + i) % 2 == 0) else -1.0
                w = sign * mode
                # rescale to unit Frobenius norm then He-std
                w = w / (w.norm() + 1e-8)
                conv.weight[o, i].copy_(w * he_std * math.sqrt(kh * kw))
        if conv.bias is not None:
            conv.bias.zero_()


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H35",
        short="cymatic_wavelet",
        primitives_touched=[
            "chladni_modes", "cymatic_init_",
            "chladni_modes_banded", "cymatic_init_ortho_",
        ],
        flags_touched=["cymatic_init"],
        legacy_t17_top1=0.7744,
        legacy_t17_composite=0.7883,
    )
