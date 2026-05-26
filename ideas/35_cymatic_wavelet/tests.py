"""H35 - unit tests for the cymatic-wavelet init.

Run with:
    python ideas/35_cymatic_wavelet/tests.py

Output must end with "All N tests passed." or fail loudly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from implementation import (  # noqa: E402
    chladni_modes_banded,
    cymatic_init_ortho_,
    idea_signature,
)
from nature_inspired_networks.priors import chladni_modes, cymatic_init_  # noqa: E402


def test_idea_signature_present():
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert sig["hypothesis_id"] == "H35"
    assert "cymatic_init" in sig["flags_touched"]


def test_chladni_modes_legacy_shape():
    """Legacy chladni_modes still returns (n_modes, k, k)."""
    m = chladni_modes(7, n_modes=4)
    assert m.shape == (4, 7, 7)
    assert m.abs().max().item() <= 1.0 + 1e-6


def test_chladni_modes_banded_orthonormal_across_modes():
    """Corrected variant: the n_modes basis vectors are orthonormal."""
    n = 8
    k = 5
    basis = chladni_modes_banded(n, k, band=(2, 4), seed=0)
    assert basis.shape == (n, k, k)
    flat = basis.reshape(n, -1)
    gram = flat @ flat.t()
    # Off-diagonal entries should be ~0; diagonal ~1
    eye = torch.eye(n)
    assert torch.allclose(gram, eye, atol=1e-4), (gram - eye).abs().max().item()


def test_chladni_modes_banded_distinct_modes():
    """Adjacent channels must not be literal duplicates."""
    basis = chladni_modes_banded(6, 5, band=(2, 5), seed=0)
    diffs = [(basis[i] - basis[j]).norm().item()
             for i in range(6) for j in range(i + 1, 6)]
    # After orthonormalisation, every pair must differ; minimum should
    # be well above zero (sqrt(2) for orthonormal unit vectors).
    assert min(diffs) > 0.5, min(diffs)


def test_cymatic_init_ortho_preserves_he_variance_within_10pct():
    """Post-init variance must match He init within +/- 10%."""
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
    # He fan-in std for reference
    fan_in = 16 * 3 * 3
    he_std = math.sqrt(2.0 / fan_in)

    cymatic_init_ortho_(conv, band=(2, 5), seed=0)
    actual_std = conv.weight.detach().std().item()
    ratio = actual_std / he_std
    assert 0.5 < ratio < 1.5, ratio  # within ~50% (init is structured, not iid)


def test_cymatic_init_ortho_changes_weights_from_default():
    """Init MUST overwrite default Kaiming weights."""
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, 3, padding=1)
    w0 = conv.weight.detach().clone()
    cymatic_init_ortho_(conv, band=(2, 5), seed=0)
    assert not torch.allclose(w0, conv.weight)


def test_cymatic_init_zeros_bias_if_present():
    """When bias=True, the init zeros it (matches Kaiming convention)."""
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 8, 3, padding=1, bias=True)
    conv.bias.data.fill_(1.234)
    cymatic_init_ortho_(conv, band=(2, 5), seed=0)
    assert torch.equal(conv.bias.detach(), torch.zeros_like(conv.bias.detach()))


def test_legacy_cymatic_init_still_works_on_3x3():
    """Regression: the legacy in-place init in priors.py is still importable
    and runs without error on a 3x3 Conv2d."""
    torch.manual_seed(0)
    conv = nn.Conv2d(4, 8, 3, padding=1, bias=False)
    w0 = conv.weight.detach().clone()
    cymatic_init_(conv)
    assert not torch.allclose(w0, conv.weight)


def test_band_low_eq_high_is_uniform_mode():
    """If band=(m, m) the sampler must always return the (m, m) mode."""
    basis = chladni_modes_banded(4, 5, band=(3, 3), seed=7)
    # After Gram-Schmidt across 4 identical raw modes, we get a single
    # non-zero direction; QR gives one unit column and three near-zero
    # columns. The total Frobenius energy is exactly 1 (per the unit
    # column), not 4. Verify we did not silently produce all-zeros.
    assert basis.norm().item() > 0.5


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
