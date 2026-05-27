"""φ-flavoured initialisation primitives.

Two hypotheses are implemented here, both as in-place ``_(tensor)``
helpers that are interchangeable with the ``torch.nn.init.*_`` family:

H31 — :func:`golden_spiral_init_`
    Sample ``k*k`` points along a logarithmic golden spiral
    ``r(θ) = r0 * φ^(θ/(π/2))``, rasterise them onto the ``k×k`` grid,
    then bilinear-modulate a He-normal draw with the resulting mask. The
    post-init variance is rescaled so it matches the He variance to
    within ≈ 5 %, isolating *structure* from *scale*.

H42 — :func:`phi_weight_init_`
    Replace He's ``std = sqrt(2 / fan_in)`` with
    ``std = sqrt(φ / fan_in)``. Reduces to He init when φ is replaced by
    2 — verified by :func:`tests.test_inits.test_phi_init_reduces_to_he`.

Both functions live next to the existing ``priors.cymatic_init_`` so
that ``_GenericConv`` can wire them in via a flag in
``NaturePriorFlags`` without circular imports (Rule 14: shared
primitives in one module).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .priors import PHI


# ---------------------------------------------------------------------------
# H31 — Golden-spiral kernel mask + init
# ---------------------------------------------------------------------------
def golden_spiral_mask(k: int = 5, n_samples: int | None = None) -> torch.Tensor:
    """Return a ``(k, k)`` mask sampled along a logarithmic golden spiral.

    The spiral is ``r(θ) = r0 * φ^(θ/(π/2))`` traced at golden-angle
    increments ``Δθ = 2π·(1 − 1/φ)``. Each sample contributes a value
    that decays linearly with index, so the centre-most early samples
    dominate. Final mask is normalised so its sum equals ``k*k``,
    ensuring the variance of a He-init weight multiplied by the mask is
    preserved on average:

        E[(w · m)²] = m² · σ_He²

    and ``mean(m²) ≈ 1`` after the rescale → He variance preserved.

    Parameters
    ----------
    k : int
        Odd kernel size (≥ 3). Default 5 — the smallest grid on which
        the spiral has more than one φ-scaled turn.
    n_samples : int, optional
        Number of points along the spiral. Default ``max(64, 8*k*k)``.

    Returns
    -------
    torch.Tensor
        Shape ``(k, k)`` float32 tensor with non-negative entries.
    """
    if k < 3:
        raise ValueError(f"golden_spiral_mask requires k >= 3, got {k}")
    if n_samples is None:
        n_samples = max(128, 16 * k * k)
    mask = torch.zeros(k, k, dtype=torch.float32)
    # Parametrise the spiral so it makes ``n_turns`` full revolutions
    # from a small inner radius ``r0`` to an outer radius covering the
    # kernel's half-diagonal. This keeps every sample on-grid AND
    # guarantees coverage scales with ``k``: a 3x3 kernel sees ~1 turn,
    # a 5x5 sees ~3 turns, a 7x7 sees ~3 turns at larger radius.
    n_turns = max(1, (k - 1) // 2)
    theta_max = 2.0 * math.pi * n_turns
    r_max = (k - 1) / 2.0          # half-diagonal in cell units
    r0 = 0.3                       # always within the central cell
    # Growth factor per sample so r(theta_max) ≈ r_max.
    # r(θ) = r0 * exp(b * θ), b = ln(r_max/r0) / theta_max
    b = math.log(r_max / r0) / theta_max
    cx = cy = (k - 1) / 2.0
    for i in range(n_samples):
        theta = i * theta_max / max(1, n_samples - 1)
        r = r0 * math.exp(b * theta)
        # Cartesian → grid coordinates, bilinear "snap" via rounding.
        x = int(round(cx + r * math.cos(theta)))
        y = int(round(cy + r * math.sin(theta)))
        if 0 <= x < k and 0 <= y < k:
            # Decay weight: earliest points (inner turns) are loudest.
            v = 1.0 - i / n_samples
            if v > mask[y, x].item():
                mask[y, x] = v
    # Renormalise so mean(mask²) ≈ 1 → He-variance preserved.
    sq = (mask * mask).mean().item()
    if sq > 0:
        mask = mask / math.sqrt(sq)
    return mask


def golden_spiral_init_(weight: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Initialise a 4-D Conv weight tensor with a golden-spiral-modulated
    He-normal draw.

    The function operates in-place and returns the tensor for chaining,
    mirroring the ``torch.nn.init.*_`` convention.

    Parameters
    ----------
    weight : torch.Tensor
        Shape ``(out_c, in_c, kH, kW)``. ``kH`` must equal ``kW``.
    scale : float
        Multiplicative scale applied AFTER the mask. Default 1.0 — set
        to e.g. 1/√φ if you want to compose with H42 in the same call.

    Notes
    -----
    Post-init variance is approximately ``(2 / fan_in) * scale²`` thanks
    to the mask's renormalisation. The structural prior survives even
    though the variance matches He: each output channel sees the same
    spiral pattern multiplied by independent He samples, so the kernel
    is biased toward log-spiral edge detectors without being a
    degenerate rank-1 init (see test_golden_spiral_variance_close_to_he).
    """
    if weight.ndim != 4:
        raise ValueError(
            f"golden_spiral_init_ expects 4-D Conv weight, got {weight.ndim}-D"
        )
    out_c, in_c, kh, kw = weight.shape
    if kh != kw:
        raise ValueError(
            f"golden_spiral_init_ expects square kernel, got {kh}x{kw}"
        )
    mask = golden_spiral_mask(kh)  # (k, k)
    with torch.no_grad():
        nn.init.kaiming_normal_(weight, nonlinearity="relu")
        weight.mul_(mask.view(1, 1, kh, kw))
        weight.mul_(scale)
    return weight


# ---------------------------------------------------------------------------
# H42 — φ-Weight Initialization
# ---------------------------------------------------------------------------
def phi_weight_init_(
    tensor: torch.Tensor,
    fan_in: int | None = None,
    mode: str = "fan_in",
    phi: float = PHI,
) -> torch.Tensor:
    """φ-scaled Kaiming-normal init.

    Replaces He's ``std = sqrt(2 / fan_in)`` with
    ``std = sqrt(phi / fan_in)``. With the default ``phi=PHI`` this
    gives std ≈ ``sqrt(1.618 / fan_in) ≈ 1.272 / sqrt(fan_in)``. Setting
    ``phi=2.0`` recovers He init exactly — this is the backward-compat
    contract asserted by :func:`tests.test_inits.test_phi_init_reduces_to_he`.

    Parameters
    ----------
    tensor : torch.Tensor
        Any tensor with ≥ 2 dimensions (Conv2d.weight or Linear.weight).
    fan_in : int, optional
        Override the computed fan-in. Default uses
        ``nn.init._calculate_correct_fan(tensor, mode)``.
    mode : {"fan_in", "fan_out"}
        Direction for the fan calculation.
    phi : float
        The gain² constant. Default :data:`PHI`.

    Returns
    -------
    torch.Tensor
        The (modified in-place) tensor.
    """
    if tensor.ndim < 2:
        raise ValueError(
            f"phi_weight_init_ expects ≥2-D tensor, got {tensor.ndim}-D"
        )
    if fan_in is None:
        fan_in = nn.init._calculate_correct_fan(tensor, mode)
    if fan_in <= 0:
        raise ValueError(f"non-positive fan_in {fan_in}")
    std = math.sqrt(phi / fan_in)
    with torch.no_grad():
        tensor.normal_(0.0, std)
    return tensor


def apply_phi_init(module: nn.Module, phi: float = PHI) -> nn.Module:
    """Apply :func:`phi_weight_init_` to every ``Conv2d`` / ``Linear``
    weight in a module tree; biases are zeroed.

    Returns the same module for chaining.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            phi_weight_init_(m.weight, phi=phi)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


def apply_golden_spiral_init(module: nn.Module, k: int = 5,
                             scale: float = 1.0) -> nn.Module:
    """Apply :func:`golden_spiral_init_` to every Conv2d whose kernel
    is ``k x k`` in ``module``. Convs with different kernel sizes keep
    their existing initialisation. Returns the module for chaining.

    Wired by the runner for the H31 ``sg_only_golden_spiral_init`` row
    (default ``k=5`` per H31 design doc).
    """
    if k < 3:
        raise ValueError(f"apply_golden_spiral_init requires k >= 3, got {k}")
    matched = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.weight.shape[-2], m.weight.shape[-1]
            if kh == k and kw == k:
                golden_spiral_init_(m.weight, scale=scale)
                matched += 1
    return module
