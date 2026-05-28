"""nature-inspired priors as small, composable PyTorch utilities.

Each function/class implements ONE nature-inspired prior so the NaturePriorBlock
can ablate them independently. References to the literature are inline.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PHI = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio ≈ 1.618


# ---------------------------------------------------------------------------
# φ / Fibonacci channel scaling (Fibonacci-Net 2025; EfficientNet compound)
# ---------------------------------------------------------------------------
def fibonacci_channels(c0: int, n_stages: int, mode: str = "fib") -> list[int]:
    """Return a channel schedule of length n_stages.

    mode='fib'          → c0, c0*F2/F1, c0*F3/F1, ...  (Fibonacci ratios, /8)
    mode='phi'          → c0 * φ^k        (geometric golden growth)
    mode='linear'       → c0 * (k+1)      (control: arithmetic growth)
    mode='phi_compound' → H01 phi-compound width recurrence; identical to
                          'phi' at k=0 but kept as a distinct mode so the
                          H01 sweep row carries a non-overlapping tag.
    """
    out: list[int] = []
    if mode == "fib":
        # Generate Fibonacci(>=1): 1, 2, 3, 5, 8, 13, 21, 34, 55
        fib = [1, 2]
        while len(fib) < n_stages + 2:
            fib.append(fib[-1] + fib[-2])
        base = fib[1]
        for k in range(n_stages):
            c = int(round(c0 * fib[k + 1] / base))
            out.append(_round8(c))
    elif mode == "phi":
        for k in range(n_stages):
            out.append(_round8(int(round(c0 * (PHI ** k)))))
    elif mode == "phi_compound":
        # H01: phi-compound stage widths (see scaling.phi_compound_channels).
        # Kept here so build_model + fibonacci_channels share a single
        # branching surface — adding a parallel path in models.py would
        # duplicate the stem/stage glue and violate Rule 14.
        for k in range(n_stages):
            out.append(_round8(int(round(c0 * (PHI ** k)))))
    elif mode == "linear":
        for k in range(n_stages):
            out.append(_round8(c0 * (k + 1)))
    else:
        raise ValueError(f"unknown scaling mode '{mode}'")
    return out


def _round8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)


# ---------------------------------------------------------------------------
# Toroidal padding (Pittorino 2022, TopoCN)
# ---------------------------------------------------------------------------
def toroidal_pad(x: torch.Tensor, pad: int, phi_scaled: bool = False) -> torch.Tensor:
    """Circular padding on the last two spatial dims → torus topology.

    H22.v2 (post-G3-audit) — when ``phi_scaled=True`` the wrapped values
    on the four boundary strips are multiplied by ``1/phi`` (≈ 0.618),
    making the periodic boundary "softer": contributions from across the
    wrap are damped relative to the interior. This implements the H22
    doc claim of a phi-scaled periodic boundary while preserving the
    spatial shape (still ``pad`` pixels of wrap on each side).

    Default ``phi_scaled=False`` reproduces plain circular padding
    byte-for-byte so the existing T1.6 ``sg_only_toroidal`` row is
    unchanged.
    """
    if pad <= 0:
        return x
    out = F.pad(x, (pad, pad, pad, pad), mode="circular")
    if not phi_scaled:
        return out
    # Damp the four wrapped border strips by 1/phi. The interior region
    # (the original tensor's content) is left untouched at full unit
    # weight; only the wrap-padded pixels are attenuated.
    scale = 1.0 / PHI
    H, W = out.shape[-2], out.shape[-1]
    # Avoid aliasing the input tensor — clone so we don't mutate a buffer
    # that's still referenced upstream.
    out = out.clone()
    out[..., :pad, :] = out[..., :pad, :] * scale          # top strip
    out[..., H - pad:, :] = out[..., H - pad:, :] * scale  # bottom strip
    out[..., :, :pad] = out[..., :, :pad] * scale          # left strip
    out[..., :, W - pad:] = out[..., :, W - pad:] * scale  # right strip
    return out


# ---------------------------------------------------------------------------
# Hexagonal kernel mask (HexaConv 2018, HexagDLy 2019)
# ---------------------------------------------------------------------------
def hex_kernel_mask(k: int = 3) -> torch.Tensor:
    """3x3 hex-shaped mask on a square kernel: keep the 7 'honeycomb'
    positions, zero out the two corners. For k=3 the pattern is

        [ 1 1 0 ]
        [ 1 1 1 ]
        [ 0 1 1 ]

    which gives 6 neighbors + center (axial-coordinate hex on offset rows).
    For k=5 we extend to a hexagon of radius 2 (19 positions).
    """
    m = torch.ones(k, k)
    if k == 3:
        m[0, 2] = 0.0
        m[2, 0] = 0.0
    elif k == 5:
        # radius-2 hex mask (19 cells)
        for i in range(k):
            for j in range(k):
                # axial: q = j - 2, r = i - 2; require |q|+|r|+|q+r| <= 4
                q, r = j - 2, i - 2
                if abs(q) + abs(r) + abs(q + r) > 4:
                    m[i, j] = 0.0
    else:
        raise ValueError("hex_kernel_mask supports k in {3, 5}")
    return m


def hex_phi_radial_mask(k: int = 3) -> torch.Tensor:
    """Hex mask weighted by phi^{-r}, where r is the hex distance from
    the centre tap. Center (r=0) is weighted 1.0; nearest-neighbour
    taps (r=1) are weighted 1/phi (≈ 0.618); the radius-2 ring (r=2,
    only used when k=5) is weighted 1/phi^2 (≈ 0.382). Taps outside the
    honeycomb (the corner positions zeroed by ``hex_kernel_mask``)
    remain exactly zero, so the receptive field shape is unchanged —
    only the active taps get a phi-radial reweighting.

    For k=3 the mask layout
        [ 1 1 0 ]
        [ 1 1 1 ]
        [ 0 1 1 ]
    represents the centre + its 6 hex-nearest neighbours; all 6 active
    non-centre cells are at hex distance 1 by construction (the two
    zeroed corners are the only cells at hex distance 2). For k=5 we
    use axial-coordinate distance ``(|q| + |r| + |q+r|) / 2`` to
    separate the radius-1 ring (6 cells) from the radius-2 ring (12
    cells).

    This implements the H21 doc claim of ``phi^{-r/r_max}`` weighting
    on hex-neighbours (Hoogeboom 2018 HexaConv; Kepler-conjecture
    phi-packing).
    """
    m = torch.zeros(k, k)
    if k == 3:
        # By construction, every non-centre cell of the 7-tap honeycomb
        # is a hex-nearest neighbour of the centre. The two cells at
        # hex distance 2 are precisely the ones zeroed by hex_kernel_mask.
        keep = hex_kernel_mask(3)
        inv_phi = float(PHI) ** -1.0
        for i in range(k):
            for j in range(k):
                if keep[i, j].item() == 0.0:
                    continue
                if i == 1 and j == 1:
                    m[i, j] = 1.0
                else:
                    m[i, j] = inv_phi
    elif k == 5:
        keep = hex_kernel_mask(5)
        for i in range(k):
            for j in range(k):
                if keep[i, j].item() == 0.0:
                    continue
                q, r = j - 2, i - 2
                # axial hex distance: (|q| + |r| + |q+r|) / 2
                dist = (abs(q) + abs(r) + abs(q + r)) // 2
                m[i, j] = float(PHI) ** (-float(dist))
    else:
        raise ValueError("hex_phi_radial_mask supports k in {3, 5}")
    return m


# ---------------------------------------------------------------------------
# Cymatic / Chladni wavelet init (cymatic-resonance hypothesis)
# ---------------------------------------------------------------------------
def chladni_modes_banded(
    n_modes: int,
    k: int,
    band: tuple[int, int] = (2, 5),
    seed: int | None = 0,
) -> torch.Tensor:
    """Build ``n_modes`` Chladni eigenmodes with (m, n) sampled uniformly
    from ``[band[0], band[1]]``, then Gram-Schmidt orthonormalise across
    modes.

    Returns a tensor of shape (n_modes, k, k) with unit Frobenius norm
    per mode and (for the first ``min(n_modes, k*k)`` slots) pairwise
    inner-product approximately zero. Promoted from
    ``ideas/35_cymatic_wavelet/implementation.py`` (Code-Y) to the shared
    primitives module per Rule 14.

    Post-G4-audit (H35 fix) — earlier versions tiled the orthonormal
    block via ``out.repeat(...)`` when ``n_modes > k*k``, silently
    producing literal-duplicate slices. The new behaviour:

    1. Sample ``n_modes`` raw sin·sin modes from a *deterministic* RNG
       that prefers unique ``(m, n)`` pairs within ``[lo, hi]^2`` before
       allowing wrap-around.
    2. QR-orthonormalise across the first ``min(n_modes, k*k)`` modes —
       these slots are mutually orthonormal.
    3. For any remaining slots (``n_modes > k*k``), fall back to the raw
       sin·sin patterns (unit Frobenius-normalised) drawn from the
       *unused* portion of the band enumeration. These slots are no
       longer orthogonal to the QR block — they cannot be, by linear
       dimension — but they are guaranteed pairwise distinct.
    """
    import random as _random

    lo, hi = band
    assert lo >= 1 and hi >= lo, f"invalid band {band}"
    rng = _random.Random(seed)
    xs = torch.linspace(0, math.pi, k + 2)[1:-1]
    ys = torch.linspace(0, math.pi, k + 2)[1:-1]
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    # Step 1 — deterministic enumeration of (m, n) pairs. Iterate the
    # full Cartesian band once (in a shuffled order so the seed still
    # matters); only wrap around if n_modes exceeds the band size.
    band_pairs = [(m, n) for m in range(lo, hi + 1) for n in range(lo, hi + 1)]
    rng.shuffle(band_pairs)
    n_band = len(band_pairs)

    def _pair(i: int) -> tuple[int, int]:
        # Pull unique pairs until exhausted, then wrap.
        return band_pairs[i % n_band]

    modes: list[torch.Tensor] = []
    for i in range(n_modes):
        m, n = _pair(i)
        mode = torch.sin(m * X) * torch.sin(n * Y)
        modes.append(mode)
    stack = torch.stack(modes, dim=0)  # (n_modes, k, k)

    # Step 2 — QR-orthonormalise the first min(n_modes, k*k) modes.
    n_keep = min(n_modes, k * k)
    flat = stack[:n_keep].reshape(n_keep, -1).t()  # (k*k, n_keep)
    q, _ = torch.linalg.qr(flat, mode="reduced")  # (k*k, n_keep)
    q = q[:, :n_keep]
    out = q.t().reshape(n_keep, k, k)

    if n_keep < n_modes:
        # Step 3 — fallback: unique raw sin·sin patterns for the
        # remaining slots. They are distinct from the QR block because
        # they retain their (m, n) "fingerprint" before QR's mixing, and
        # they are distinct from each other because we keep walking the
        # shuffled enumeration. If n_modes > n_band we DO wrap, but
        # the wrap is across raw-mode index i — each subsequent wrap
        # slot is multiplied by a (-1)^j sign factor so even wrapped
        # copies are bit-distinct from their predecessors.
        extras: list[torch.Tensor] = []
        for i in range(n_keep, n_modes):
            raw = stack[i]
            norm = raw.norm() + 1e-8
            raw = raw / norm
            # Deterministic sign flip per wrap-around so wrapped slots
            # are anti-symmetric copies (still distinct bit-for-bit).
            wrap_layer = i // n_band
            if wrap_layer % 2 == 1:
                raw = -raw
            extras.append(raw)
        out = torch.cat([out, torch.stack(extras, dim=0)], dim=0)
    return out


def chladni_modes(k: int, n_modes: int = 4) -> torch.Tensor:
    """Return `n_modes` orthogonal Chladni-plate basis patterns of size k×k.

    Each pattern = sin(m·π·x/(k+1)) · sin(n·π·y/(k+1)) for (m,n) drawn from
    low frequencies — these are the eigenmodes of the 2-D wave equation
    on a square plate (Chladni patterns).
    """
    xs = torch.linspace(0, math.pi, k + 2)[1:-1]
    ys = torch.linspace(0, math.pi, k + 2)[1:-1]
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    modes: list[torch.Tensor] = []
    pairs = []
    # use lowest-frequency mode pairs first (m+n small)
    for s in range(2, 2 + n_modes * 2):
        for m in range(1, s):
            n = s - m
            pairs.append((m, n))
            if len(pairs) >= n_modes:
                break
        if len(pairs) >= n_modes:
            break
    for m, n in pairs[:n_modes]:
        mode = torch.sin(m * X) * torch.sin(n * Y)
        mode = mode / (mode.abs().max() + 1e-8)
        modes.append(mode)
    return torch.stack(modes, dim=0)  # (n_modes, k, k)


def cymatic_init_(
    conv: nn.Conv2d,
    n_modes: int | None = None,
    orthonormalize: bool = False,
    band: tuple[int, int] = (1, 1),
    seed: int = 0,
) -> None:
    """Initialize Conv2d weights from Chladni eigenmodes.

    Default (``orthonormalize=False, band=(1, 1)``) preserves the legacy
    behaviour exactly: a random convex combination over the low-frequency
    Chladni basis, broadcast across in/out channels with He fan-in
    scaling. This keeps the existing ``sg_only_cymatic_init`` smoke row
    bit-identical.

    H35.v2 — when ``orthonormalize=True`` (typical companion:
    ``band=(2, 5)``) the corrected variant validated by Code-Y is used
    instead:

    - distinct (m, n) pairs per output channel (was: same modes reused
      across channels in the legacy path);
    - Gram-Schmidt orthonormalisation across the ``out_c`` channels via
      :func:`chladni_modes_banded` (was: highly correlated channels);
    - frequency band defaults to (2, 5) instead of (1, 1) — the latter
      is effectively DC on a 3x3 kernel and adds no spatial structure.

    This addresses the T1.7 negative (`sg_only_cymatic_init` -2.67 pp on
    CIFAR-10) by giving each output channel a distinct, decorrelated
    Chladni eigenmode at He-normal variance.
    """
    out_c, in_c, kh, kw = conv.weight.shape
    assert kh == kw, "cymatic init expects square kernel"
    fan_in = in_c * kh * kw
    he_std = math.sqrt(2.0 / fan_in)

    if orthonormalize:
        basis = chladni_modes_banded(out_c, kh, band=band, seed=seed)  # (out_c, k, k)
        with torch.no_grad():
            for o in range(out_c):
                mode = basis[o]
                for i in range(in_c):
                    # deterministic sign per (o, i) so different input
                    # channels see anti-symmetric copies, not literal repeats
                    sign = 1.0 if ((o + i) % 2 == 0) else -1.0
                    w = sign * mode
                    # unit Frobenius norm * He-std * sqrt(k*k) keeps post-init
                    # variance He-equivalent for downstream BatchNorm.
                    w = w / (w.norm() + 1e-8)
                    conv.weight[o, i].copy_(w * he_std * math.sqrt(kh * kw))
            if conv.bias is not None:
                conv.bias.zero_()
        return

    # Legacy path — preserved byte-for-byte for backward compatibility.
    n = n_modes or min(8, max(2, kh * kh // 2))
    basis = chladni_modes(kh, n_modes=n)  # (n, k, k)
    g = torch.Generator().manual_seed(0xC1A171C)
    scale = he_std  # He-style
    with torch.no_grad():
        for o in range(out_c):
            for i in range(in_c):
                # random convex combination over modes + sign
                coef = torch.randn(n, generator=g)
                coef = coef / (coef.norm() + 1e-8)
                w = (basis * coef.view(n, 1, 1)).sum(0)
                conv.weight[o, i].copy_(w * scale)
        if conv.bias is not None:
            conv.bias.zero_()


# ---------------------------------------------------------------------------
# C4 / D4 group-equivariant conv (proxy for Platonic equivariance)
# ---------------------------------------------------------------------------
class GroupConv2d(nn.Module):
    """Cohen-Welling style group convolution over C4 (4 rotations) or D4
    (4 rotations + flips). Weight-shared across group, output channels =
    `out_channels`. This is a *light* Platonic prior cheap enough for ablations.

    The ``reduce`` parameter selects the orbit aggregation:
      - ``"max"`` — per-position soft-argmax over the 4 (or 8) rotated copies.
      - ``"mean"`` — average over the orbit (the H58 hypothesised "fix").

    **Empirical verdict (CIFAR-10, 12 epochs, seed 0):** ``max`` is *better*
    than ``mean`` by 4-6 pp top-1. The original intuition "max throws away
    75 % of the signal" was wrong; max acts as a soft argmax over
    orientations, preserving the strongest response at every spatial
    location, while mean dilutes discriminative features. See
    ``FINDINGS.md`` "H58 follow-up — the avg-pool fix DISCARDED" for the
    full per-row table. The correct cure for H24 is *data*, not the
    reduction operator — test on rotated CIFAR-10 / IcoMNIST where the
    equivariance prior is data-aligned.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        group: str = "c4",
        bias: bool = False,
        mask: torch.Tensor | None = None,
        reduce: str = "max",
    ) -> None:
        super().__init__()
        assert group in {"c4", "d4"}
        assert reduce in {"max", "mean"}
        self.group = group
        self.n_orbits = 4 if group == "c4" else 8
        self.stride = stride
        self.padding = padding
        self.reduce = reduce
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer(
            "mask",
            mask if mask is not None else torch.ones_like(self.weight[0, 0]),
        )

    def _orbit(self) -> torch.Tensor:
        w = self.weight * self.mask  # (O, I, k, k)
        outs = [w]
        for r in (1, 2, 3):
            outs.append(torch.rot90(w, k=r, dims=(2, 3)))
        if self.group == "d4":
            wf = torch.flip(w, dims=(3,))
            outs.append(wf)
            for r in (1, 2, 3):
                outs.append(torch.rot90(wf, k=r, dims=(2, 3)))
        return torch.stack(outs, dim=0)  # (G, O, I, k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orbit = self._orbit()  # (G, O, I, k, k)
        G, O, I, k, _ = orbit.shape
        w = orbit.reshape(G * O, I, k, k)
        y = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        # Group orbit reduction. Max-pool (legacy) is the dominant negative in
        # the prior CIFAR sweep — it throws away 75% of the signal at C4. The
        # H58 fix is mean-pool, which preserves the orbit's full signal while
        # still giving rotation invariance after averaging.
        y = y.view(x.shape[0], G, O, y.shape[-2], y.shape[-1])
        if self.reduce == "mean":
            y = y.mean(dim=1)
        else:
            y = y.amax(dim=1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


# ---------------------------------------------------------------------------
# Toroidal-padded hex conv (drop-in Conv2d replacement)
# ---------------------------------------------------------------------------
class HexConv2d(nn.Module):
    """Hex-masked Conv2d with optional toroidal padding. The mask zeroes
    out the two corner taps so the receptive field is the 7-cell honeycomb
    (HexaConv 2018) emulated on a square lattice.

    H21.v2 — ``hex_kernel_radius`` selects the hex radius:

    - ``radius=1`` (default): kernel_size 3, 7-tap mask, only **180-deg**
      symmetric (the two corners are zeroed but the lattice is not truly
      6-fold isotropic). Preserves all legacy behaviour byte-for-byte.
    - ``radius=2``: kernel_size 5, 19-tap mask, **true 6-fold isotropic**
      via the existing ``hex_kernel_mask(5)``.

    When ``radius=2`` the constructor overrides the supplied
    ``kernel_size`` and ``padding`` to 5 and 2 so the receptive field
    matches a radius-2 hexagon and the output spatial shape is unchanged.

    H21.v3 (post-G3-audit) — ``phi_radial=True`` replaces the boolean
    hex mask with a float mask whose active taps are weighted by
    ``phi^{-r}`` where ``r`` is the axial-hex distance from the centre
    tap (centre→1.0; nearest-6→1/φ≈0.618; radius-2 ring→1/φ²≈0.382).
    Default ``False`` preserves the binary-mask behaviour byte-for-byte
    so existing CIFAR-10 smoke rows are unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        toroidal: bool = False,
        bias: bool = False,
        hex_kernel_radius: int = 1,
        phi_radial: bool = False,
    ) -> None:
        super().__init__()
        assert hex_kernel_radius in {1, 2}, (
            f"hex_kernel_radius must be 1 (k=3, 180-sym) or 2 (k=5, 6-fold isotropic); "
            f"got {hex_kernel_radius}"
        )
        if hex_kernel_radius == 2:
            kernel_size = 5
            # symmetric padding so spatial shape is preserved at stride=1
            padding = 2
        self.hex_kernel_radius = hex_kernel_radius
        self.phi_radial = phi_radial
        self.stride = stride
        self.padding = padding
        self.toroidal = toroidal
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias,
        )
        if phi_radial:
            # Float mask: phi^{-r} on the honeycomb taps, zero elsewhere.
            # The receptive-field shape is unchanged vs the binary mask;
            # only the active taps get a phi-radial reweighting.
            self.register_buffer("mask", hex_phi_radial_mask(kernel_size))
        else:
            self.register_buffer("mask", hex_kernel_mask(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.toroidal:
            x = toroidal_pad(x, self.padding)
        else:
            x = F.pad(x, [self.padding] * 4, mode="constant", value=0.0)
        # Apply mask each forward — keeps gradients flowing to all weights
        w = self.conv.weight * self.mask
        y = F.conv2d(x, w, self.conv.bias, stride=self.stride, padding=0)
        return y


# ---------------------------------------------------------------------------
# Golden-angle positional / rotary modulation (output stage)
# ---------------------------------------------------------------------------
def golden_angle_phases(n: int) -> torch.Tensor:
    """Return n golden-angle phases in [0, 2π). Used for rotary-style
    modulation of channel groups (Metatron projection proxy).
    """
    golden_angle = 2 * math.pi * (1 - 1 / PHI)  # ≈ 2.3999632...
    return torch.arange(n, dtype=torch.float32) * golden_angle % (2 * math.pi)
