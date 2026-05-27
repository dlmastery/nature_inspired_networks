"""Unit tests for H34 — Golden-Angle Rotary (RoPE-φ).

Asserts:

* :func:`golden_angle_rope_freqs` returns ``(dim//2,)`` frequency and
  phase-offset tensors with the expected golden-angle progression
  ``k * GOLDEN_ANGLE`` (mod 2π) and φ-based exponential decay.
* :func:`apply_golden_rope` preserves the q, k tensor shapes.
* The rotation preserves the per-pair complex norm: applying RoPE-φ
  must leave ``||(q_{2i}, q_{2i+1})|| = ||(q'_{2i}, q'_{2i+1})||`` for
  every pair, since a 2-D rotation is norm-preserving.
* RoPE-φ produces numerically different outputs from standard
  base-10000 RoPE — the prior is not vacuously equal to the baseline.
* Relative-position equivariance: the dot product ``q(p1)·k(p2)`` after
  rotation depends only on ``p2 - p1`` for a constant query/key, i.e.
  shifting both positions by a constant offset leaves the inner
  product invariant. This is the property that RoPE-φ inherits from
  RoPE despite the phase offset (the offset cancels in differences).
* Even-dim guard and shape-validation guards raise loudly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.golden_rope import (  # noqa: E402
    GOLDEN_ANGLE,
    apply_golden_rope,
    golden_angle_rope_freqs,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


# ---------------------------------------------------------------------------
# golden_angle_rope_freqs
# ---------------------------------------------------------------------------
def test_freqs_shape_and_phase_progression():
    dim = 16
    freqs, phases = golden_angle_rope_freqs(dim)
    assert freqs.shape == (dim // 2,)
    assert phases.shape == (dim // 2,)
    # Phase k = k * GOLDEN_ANGLE mod 2π
    for k in range(dim // 2):
        expected = (k * GOLDEN_ANGLE) % (2 * math.pi)
        assert abs(phases[k].item() - expected) < 1e-5, (k, phases[k].item(), expected)


def test_freqs_phi_base_geometric_decay():
    dim = 16
    freqs, _ = golden_angle_rope_freqs(dim, base=PHI)
    # freq_k = phi^(-2k/dim) => freq_{k+1} / freq_k = phi^(-2/dim)
    ratio = PHI ** (-2.0 / dim)
    for k in range(len(freqs) - 1):
        assert abs(freqs[k + 1].item() / freqs[k].item() - ratio) < 1e-5


def test_freqs_rejects_odd_dim():
    try:
        golden_angle_rope_freqs(15)
    except ValueError:
        return
    raise AssertionError("expected ValueError on odd dim")


# ---------------------------------------------------------------------------
# apply_golden_rope — shape & norm preservation
# ---------------------------------------------------------------------------
def test_apply_golden_rope_shape_unchanged():
    B, H, N, D = 2, 4, 16, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    freqs, phases = golden_angle_rope_freqs(D)
    pos = torch.arange(N)
    q_rot, k_rot = apply_golden_rope(q, k, freqs, pos, phases)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()


def test_apply_golden_rope_preserves_pairwise_norm():
    """A 2-D rotation is norm-preserving — the norm of every consecutive
    pair ``(q_{2i}, q_{2i+1})`` must be unchanged."""
    B, H, N, D = 1, 2, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    freqs, phases = golden_angle_rope_freqs(D)
    pos = torch.arange(N)
    q_rot, _ = apply_golden_rope(q, k, freqs, pos, phases)
    # Pair-wise norms across the D axis.
    pairs_in = q.reshape(B, H, N, D // 2, 2).norm(dim=-1)
    pairs_out = q_rot.reshape(B, H, N, D // 2, 2).norm(dim=-1)
    assert torch.allclose(pairs_in, pairs_out, atol=1e-5), (
        (pairs_in - pairs_out).abs().max()
    )


def test_apply_golden_rope_differs_from_standard_rope_numerically():
    """RoPE-φ with golden-angle phase offset and φ-base frequency must
    produce a *different* numerical result from standard base-10000 RoPE
    (zero phase offset).
    """
    B, H, N, D = 1, 1, 16, 16
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    pos = torch.arange(N)

    # RoPE-φ: φ-base + golden-angle phase
    freqs_phi, phases_phi = golden_angle_rope_freqs(D, base=PHI)
    q_phi, _ = apply_golden_rope(q, k, freqs_phi, pos, phases_phi)

    # Standard RoPE: base-10000 + zero phase
    half = D // 2
    kidx = torch.arange(half, dtype=torch.float32)
    freqs_std = 1.0 / (10000.0 ** (2 * kidx / D))
    zero_phase = torch.zeros(half)
    q_std, _ = apply_golden_rope(q, k, freqs_std, pos, zero_phase)
    assert not torch.allclose(q_phi, q_std, atol=1e-3)


# ---------------------------------------------------------------------------
# Relative-position equivariance
# ---------------------------------------------------------------------------
def test_apply_golden_rope_relative_angle_stable_under_shift():
    """Q·K dot product after RoPE-φ depends only on relative position.

    The phyllotactic phase offset ``k · GOLDEN_ANGLE`` is constant in
    position, so it cancels in the difference between two rotation
    angles at positions p1 and p2:

        angle(p2, k) - angle(p1, k) = (p2 - p1) · freq_k

    Therefore shifting BOTH q-position and k-position by the same
    integer offset leaves the dot product q(p+δ)·k(p+δ) invariant
    (within float tol).
    """
    B, H, D = 1, 1, 16
    torch.manual_seed(123)
    # Use the SAME query/key vector at two different position pairs.
    base_q = torch.randn(B, H, 1, D)
    base_k = torch.randn(B, H, 1, D)
    freqs, phases = golden_angle_rope_freqs(D)

    def dot_at(p_q: int, p_k: int) -> torch.Tensor:
        # Stack one query at p_q, one key at p_k into N=2 so RoPE can be
        # applied uniformly.
        q2 = torch.cat([base_q, base_q], dim=2)
        k2 = torch.cat([base_k, base_k], dim=2)
        pos = torch.tensor([p_q, p_k], dtype=torch.long)
        q_rot, k_rot = apply_golden_rope(q2, k2, freqs, pos, phases)
        # q at position p_q (index 0) vs k at position p_k (index 1).
        return (q_rot[..., 0, :] * k_rot[..., 1, :]).sum()

    # Two pairs with the same relative offset must give the same dot.
    a = dot_at(0, 5)
    b = dot_at(7, 12)  # also offset = 5
    assert torch.allclose(a, b, atol=1e-4), (a.item(), b.item())


def test_apply_golden_rope_shape_validation():
    B, H, N, D = 1, 2, 4, 8
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    freqs, phases = golden_angle_rope_freqs(D)
    # Wrong positions shape
    try:
        apply_golden_rope(q, k, freqs, torch.arange(N + 1), phases)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on positions shape mismatch")
    # Wrong q vs k shape
    try:
        apply_golden_rope(q, k[:, :, :-1], freqs, torch.arange(N), phases)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on q,k shape mismatch")


if __name__ == "__main__":
    import inspect

    fns = [
        v for k, v in globals().items()
        if k.startswith("test_") and inspect.isfunction(v)
    ]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
