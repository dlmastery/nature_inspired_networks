"""Unit tests for H11 — Pure Fibonacci MLP.

Asserts:
  - forward shape (B, input_dim) -> (B, output_dim);
  - default hidden_sizes match the canonical Fibonacci diamond;
  - parameter count matches the closed-form sum across linear layers;
  - drop-in regression: identical output to a vanilla MLP when sizes
    are equal and activation is shared.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.fib_mlp import (  # noqa: E402
    FibMLP,
    default_fib_hidden,
)


def test_fib_mlp_forward_shape():
    """(B, input_dim) -> (B, output_dim) on default Higgs-style sizing."""
    torch.manual_seed(0)
    m = FibMLP(input_dim=28, output_dim=2)
    x = torch.randn(16, 28)
    y = m(x)
    assert y.shape == (16, 2), y.shape


def test_fib_mlp_default_hidden_sizes_are_canonical_fib_diamond():
    """default_fib_hidden() returns the canonical [8,13,21,34,21,13,8]."""
    sizes = default_fib_hidden()
    assert sizes == [8, 13, 21, 34, 21, 13, 8], sizes
    m = FibMLP(input_dim=10, output_dim=3)
    assert m.hidden_sizes == [8, 13, 21, 34, 21, 13, 8]


def test_fib_mlp_param_count_matches_closed_form():
    """Param count = sum_i (in_i * out_i + out_i) over consecutive linears."""
    input_dim = 10
    output_dim = 3
    sizes = [8, 13, 21, 34, 21, 13, 8]
    m = FibMLP(input_dim=input_dim, hidden_sizes=sizes,
               output_dim=output_dim, bias=True)
    # Expected = sum over each Linear of (in*out + out).
    all_dims = [input_dim] + sizes + [output_dim]
    expected = 0
    for k in range(len(all_dims) - 1):
        in_k, out_k = all_dims[k], all_dims[k + 1]
        expected += in_k * out_k + out_k
    got = m.param_count()
    assert got == expected, (got, expected)


def test_fib_mlp_drop_in_regression_uniform_sizes():
    """When hidden_sizes are uniform AND we use a stateless activation
    factory, FibMLP must produce identical output to a hand-written
    nn.Sequential MLP with the same Linear weights (drop-in compatibility).
    """
    torch.manual_seed(0)
    input_dim = 16
    sizes = [64, 64, 64]
    output_dim = 4
    # Build the FibMLP with nn.ReLU (stateless), matching the standard MLP
    m_fib = FibMLP(input_dim=input_dim, hidden_sizes=sizes,
                   output_dim=output_dim, activation=nn.ReLU, bias=True)
    # Hand-built equivalent
    m_std = nn.Sequential(
        nn.Linear(input_dim, sizes[0]), nn.ReLU(),
        nn.Linear(sizes[0], sizes[1]), nn.ReLU(),
        nn.Linear(sizes[1], sizes[2]), nn.ReLU(),
        nn.Linear(sizes[2], output_dim),
    )
    # Copy weights from m_fib into m_std so they are identical
    fib_lins = [m for m in m_fib.net if isinstance(m, nn.Linear)]
    std_lins = [m for m in m_std if isinstance(m, nn.Linear)]
    assert len(fib_lins) == len(std_lins) == 4
    with torch.no_grad():
        for a, b in zip(fib_lins, std_lins):
            b.weight.copy_(a.weight)
            b.bias.copy_(a.bias)
    x = torch.randn(8, input_dim)
    y_fib = m_fib(x)
    y_std = m_std(x)
    assert torch.allclose(y_fib, y_std, atol=1e-6), (y_fib - y_std).abs().max()


def test_fib_mlp_rejects_invalid_inputs():
    """input_dim < 1, output_dim < 1, or non-positive hidden size are rejected."""
    for bad in (0, -1):
        try:
            FibMLP(input_dim=bad)
            raise AssertionError(f"expected ValueError for input_dim={bad}")
        except ValueError:
            pass
        try:
            FibMLP(input_dim=10, output_dim=bad)
            raise AssertionError(f"expected ValueError for output_dim={bad}")
        except ValueError:
            pass
    try:
        FibMLP(input_dim=10, hidden_sizes=[8, 0, 13])
        raise AssertionError("expected ValueError for hidden_sizes containing 0")
    except ValueError:
        pass


def test_fib_mlp_gradient_flow():
    """A forward+backward must produce finite gradients on every parameter."""
    torch.manual_seed(0)
    m = FibMLP(input_dim=10, output_dim=2)
    x = torch.randn(4, 10)
    target = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    logits = m(x)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
