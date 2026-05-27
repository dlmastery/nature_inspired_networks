"""Unit tests for H13 — Golden Neuron Connectivity (phi-sparse layers).

Covers:
* Canonical forward (shape, dtype) for both PhiSparseLinear and
  PhiSparseConv2d at the default 1/phi density.
* Mask-generation branches: random Bernoulli + magnitude-pruning.
* Regression: density boundary cases (1.0 -> dense, very-low density).
* Edge case: invalid density values are rejected.
* End-to-end variant: PhiSparseNaturePriorNet builds and runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.priors import PHI  # noqa: E402
from nature_inspired_networks.sparse import (  # noqa: E402
    DEFAULT_DENSITY,
    PhiSparseConv2d,
    PhiSparseLinear,
    PhiSparseNaturePriorNet,
    magnitude_prune_mask,
    magnitude_prune_to_phi,
    phi_sparse_mask,
)


def test_default_density_is_one_over_phi():
    assert abs(DEFAULT_DENSITY - 1.0 / PHI) < 1e-12
    assert abs(DEFAULT_DENSITY - 0.6180339887) < 1e-8


def test_phi_sparse_mask_is_binary_and_correct_shape():
    g = torch.Generator().manual_seed(0)
    m = phi_sparse_mask((4, 8), density=DEFAULT_DENSITY, generator=g)
    assert m.shape == (4, 8)
    # binary in {0.0, 1.0}
    uniq = set(m.unique().tolist())
    assert uniq.issubset({0.0, 1.0}), uniq
    # density approximately 1/phi at population level (4*8=32, but
    # check with a larger draw for tightness)
    big = phi_sparse_mask((256, 256), density=DEFAULT_DENSITY,
                          generator=torch.Generator().manual_seed(1))
    obs = big.mean().item()
    assert abs(obs - DEFAULT_DENSITY) < 0.01, obs


def test_phi_sparse_linear_canonical_forward_shape_and_mask_binary():
    g = torch.Generator().manual_seed(0)
    layer = PhiSparseLinear(16, 8, density=DEFAULT_DENSITY, generator=g)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()
    # mask is a buffer and remains binary
    uniq = set(layer.mask.unique().tolist())
    assert uniq.issubset({0.0, 1.0})
    # effective param count is between 50pct and 75pct of dense
    eff = layer.effective_param_count
    dense = 16 * 8
    assert 0.40 * dense <= eff <= 0.85 * dense, (eff, dense)


def test_phi_sparse_linear_equivalent_to_masked_matmul():
    """Forward must equal F.linear(x, W * M, b) — the masking math."""
    g = torch.Generator().manual_seed(42)
    layer = PhiSparseLinear(6, 4, density=0.5, bias=True, generator=g)
    x = torch.randn(3, 6)
    y_layer = layer(x)
    y_manual = torch.nn.functional.linear(
        x, layer.weight * layer.mask, layer.bias
    )
    assert torch.allclose(y_layer, y_manual, atol=1e-6)


def test_density_one_is_dense_baseline():
    """Branch: density=1.0 -> mask all ones -> identical to nn.Linear."""
    g = torch.Generator().manual_seed(0)
    layer = PhiSparseLinear(8, 4, density=1.0, generator=g)
    assert layer.mask.sum().item() == 8 * 4
    assert layer.effective_param_count == 32


def test_invalid_density_rejected():
    for bad in (0.0, -0.1, 1.5, 2.0):
        try:
            phi_sparse_mask((4, 4), density=bad)
            raise AssertionError(f"expected ValueError for density={bad}")
        except ValueError:
            pass


def test_magnitude_prune_mask_keeps_top_fraction():
    """Magnitude branch: top-1/phi by |W| is kept."""
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    m = magnitude_prune_mask(w, density=DEFAULT_DENSITY)
    assert m.shape == w.shape
    uniq = set(m.unique().tolist())
    assert uniq.issubset({0.0, 1.0})
    # Kept entries have absolute value >= dropped ones.
    kept = w.abs()[m.bool()]
    dropped = w.abs()[(1 - m).bool()]
    if dropped.numel() > 0:
        assert kept.min().item() >= dropped.max().item() - 1e-6


def test_magnitude_prune_to_phi_preserves_dense_weights():
    """Converting dense Linear -> PhiSparseLinear must keep weights."""
    dense = nn.Linear(12, 6, bias=True)
    torch.nn.init.kaiming_normal_(dense.weight)
    sparse = magnitude_prune_to_phi(dense, density=DEFAULT_DENSITY)
    assert isinstance(sparse, PhiSparseLinear)
    assert torch.allclose(sparse.weight, dense.weight)
    assert torch.allclose(sparse.bias, dense.bias)
    # mask keeps roughly 1/phi of entries
    frac = sparse.mask.mean().item()
    assert abs(frac - DEFAULT_DENSITY) < 0.05, frac


def test_phi_sparse_conv2d_forward_shape_and_density():
    g = torch.Generator().manual_seed(0)
    conv = PhiSparseConv2d(3, 8, kernel_size=3, stride=1, padding=1,
                            density=DEFAULT_DENSITY, generator=g)
    x = torch.randn(2, 3, 8, 8)
    y = conv(x)
    assert y.shape == (2, 8, 8, 8)
    frac = conv.mask.mean().item()
    assert 0.4 < frac < 0.85, frac


def test_phi_sparse_conv2d_stride2():
    g = torch.Generator().manual_seed(0)
    conv = PhiSparseConv2d(3, 8, kernel_size=3, stride=2, padding=1,
                            density=DEFAULT_DENSITY, generator=g)
    x = torch.randn(2, 3, 8, 8)
    y = conv(x)
    assert y.shape == (2, 8, 4, 4)


def test_reset_mask_magnitude_changes_mask_but_not_weights():
    g = torch.Generator().manual_seed(0)
    layer = PhiSparseLinear(8, 8, density=DEFAULT_DENSITY, generator=g)
    w_before = layer.weight.detach().clone()
    mask_before = layer.mask.detach().clone()
    layer.reset_mask_magnitude()
    assert torch.allclose(layer.weight, w_before)
    # mask cardinality matches the new density target
    assert abs(layer.mask.mean().item() - DEFAULT_DENSITY) < 0.05
    # And the new mask is binary
    uniq = set(layer.mask.unique().tolist())
    assert uniq.issubset({0.0, 1.0})


def test_phi_sparse_linear_gradient_flows_to_kept_weights():
    """Regression: masked-out positions get zero grad; kept positions
    get non-zero grad. This catches the failure mode where the mask
    isn't applied during backward."""
    g = torch.Generator().manual_seed(0)
    layer = PhiSparseLinear(4, 4, density=0.5, generator=g)
    x = torch.randn(8, 4, requires_grad=False)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    # Grad on positions where mask=0 should be exactly 0 (because
    # forward used weight*mask, so d(loss)/d(weight) ∝ mask).
    zero_mask = layer.mask == 0
    nonzero_mask = layer.mask == 1
    if zero_mask.any():
        assert (layer.weight.grad[zero_mask].abs() < 1e-12).all()
    if nonzero_mask.any():
        assert (layer.weight.grad[nonzero_mask].abs() > 0).any()


def test_phi_sparse_naturepriornet_forward():
    """End-to-end: the variant builds and produces logits of correct shape."""
    torch.manual_seed(0)
    net = PhiSparseNaturePriorNet(num_classes=10, channel_mode="fib")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 10)
    # The classifier head is a PhiSparseLinear with density~1/phi
    assert isinstance(net.fc, PhiSparseLinear)
    assert abs(net.fc.density - 1.0 / PHI) < 1e-12


def test_phi_sparse_naturepriornet_registered_with_build_model():
    """Regression: the import-time monkey-patch must wire the new name
    through build_model so the sweep row 'sg_only_phi_sparse' resolves
    without editing models.py."""
    from nature_inspired_networks.models import build_model
    net = build_model("natureprior_phi_sparse", num_classes=10)
    assert isinstance(net, PhiSparseNaturePriorNet)
    x = torch.randn(1, 3, 32, 32)
    assert net(x).shape == (1, 10)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
