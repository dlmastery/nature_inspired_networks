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

    Geometry (post G4-audit fix, 2026-05-27)
    ----------------------------------------
    The spiral is the **true** φ-logarithmic spiral

        r(θ) = r0 · φ^(θ / (π/2)) = r0 · exp(b · θ),
        b    = ln(φ) / (π/2) ≈ 0.30634

    Successive samples are placed at the **golden-angle** azimuthal
    step (Vogel / phyllotaxis spacing)

        Δθ = 2π · (1 − 1/φ) ≈ 137.508°

    so consecutive points form the canonical Fibonacci sunflower lattice
    (Vogel 1979 'A better way to construct the sunflower head';
    Mandelbrot 1982). r0 is set so the spiral starts inside the central
    cell; points whose rasterised (x, y) falls outside the ``k×k`` grid
    are simply clipped — we deliberately do NOT rescale b to make the
    spiral fit the kernel because that would destroy the φ-growth-rate
    invariant the hypothesis tests for (G4 audit MAJOR finding: prior
    implementation used b = ln(r_max/r0)/theta_max ≈ 0.151, which is
    a generic log spiral, not a golden one).

    Each sample contributes a value that decays linearly with index so
    the centre-most early samples dominate. Final mask is renormalised
    so ``mean(m²) ≈ 1`` → He-variance is preserved when the mask
    multiplies a He-normal draw:

        E[(w · m)²] = m² · σ_He²,  mean(m²) ≈ 1.

    Parameters
    ----------
    k : int
        Odd kernel size (≥ 3). Default 5 — the smallest grid on which
        the spiral has more than one φ-scaled turn.
    n_samples : int, optional
        Number of points along the spiral. Default ``max(128, 16*k*k)``
        — enough samples that the off-grid clip doesn't starve any
        interior cell.

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
    # --- φ-fixed spiral geometry --------------------------------------
    # b is FIXED by the φ-growth invariant. r0 keeps the first sample
    # within the central cell. The spiral's terminal radius is whatever
    # it naturally reaches at the last golden-angle step — out-of-grid
    # points are clipped by the bounds check below. (G4 audit fix.)
    b = math.log(PHI) / (math.pi / 2.0)        # ≈ 0.30634
    delta_theta = 2.0 * math.pi * (1.0 - 1.0 / PHI)  # ≈ 137.508° golden angle
    # r0 small enough that early samples (which dominate via linear
    # decay) cluster inside the central cell, then bilinear splatting
    # spreads each sample's weight to its 4 surrounding integer cells
    # so the mask covers a meaningful neighbourhood despite the true
    # φ-spiral's fast radial escape from a small k×k grid.
    r0 = 0.1
    cx = cy = (k - 1) / 2.0
    for i in range(n_samples):
        theta = i * delta_theta
        r = r0 * math.exp(b * theta)
        # Decay weight: earliest points (inner turns) are loudest.
        v = 1.0 - i / n_samples
        if v <= 0.0:
            break
        # Continuous Cartesian → bilinear splat over the 4 surrounding
        # integer cells; "max-merge" so the brightest contribution per
        # cell wins (matches the original nearest-neighbour semantics).
        xc = cx + r * math.cos(theta)
        yc = cy + r * math.sin(theta)
        x0 = int(math.floor(xc))
        y0 = int(math.floor(yc))
        fx = xc - x0
        fy = yc - y0
        for dx, dy, w in (
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ):
            xi = x0 + dx
            yi = y0 + dy
            if 0 <= xi < k and 0 <= yi < k:
                contrib = w * v
                if contrib > mask[yi, xi].item():
                    mask[yi, xi] = contrib
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
