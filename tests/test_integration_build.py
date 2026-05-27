"""Phase-A integration tests: build_model dispatches to every new
hypothesis primitive registered in src/nature_inspired_networks/.

Each test_* function builds a model via the central ``build_model``
entry point and runs a (2, 3, 32, 32) -> (2, 10) forward pass. The
backing factories live in:

* phi_scaling.py — H06 golden_bottleneck, H09 phi_budget, H17.pure golden_skip
* sparse.py      — H13 natureprior_phi_sparse  (self-registers)
* stride.py      — H18 natureprior_fib_stride  (self-registers)
* phi_threshold.py — H19 natureprior_phi_relu  (self-registers)

Run with the canonical ``__main__`` pattern -- no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks import models  # noqa: E402
from nature_inspired_networks.models import build_model  # noqa: E402


def _forward(model, batch: int = 2, c: int = 3, hw: int = 32,
             num_classes: int = 10) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(batch, c, hw, hw))
    assert out.shape == (batch, num_classes), out.shape
    return out


def test_build_model_golden_bottleneck_forward():
    """H06: ``golden_bottleneck`` builds + (2,3,32,32) -> (2,10)."""
    m = build_model("golden_bottleneck", num_classes=10)
    _forward(m)


def test_build_model_golden_bottleneck_inverted_kw_routes():
    """H06: ``phi_inverted`` cfg key flows through build_model into
    GoldenBottleneckNet.inverted -- a behaviour test, not just shape.
    """
    m_norm = build_model("golden_bottleneck", num_classes=10,
                         phi_inverted=False)
    m_inv = build_model("golden_bottleneck", num_classes=10,
                        phi_inverted=True)
    assert getattr(m_norm, "inverted", False) is False
    assert getattr(m_inv, "inverted", False) is True
    _forward(m_inv)


def test_build_model_phi_budget_forward_and_routes_kw():
    """H09: ``phi_budget`` forwards ``phi_budget_total / n_stages /
    mode`` into PhiBudgetNet so different budgets produce different
    param counts.
    """
    m_small = build_model("phi_budget", num_classes=10,
                          phi_budget_total=120_000,
                          phi_budget_n_stages=3,
                          phi_budget_mode="phi")
    m_big = build_model("phi_budget", num_classes=10,
                        phi_budget_total=400_000,
                        phi_budget_n_stages=3,
                        phi_budget_mode="phi")
    p_small = sum(p.numel() for p in m_small.parameters())
    p_big = sum(p.numel() for p in m_big.parameters())
    assert p_big > p_small, (p_small, p_big)
    _forward(m_small)
    _forward(m_big)


def test_build_model_golden_skip_forward_and_init_routes():
    """H17.pure: ``phi_skip_init`` / ``phi_skip_trainable`` flow into
    GoldenSkipResNet so the alpha is initialised + trainable as
    requested.
    """
    m = build_model("golden_skip", num_classes=10,
                    phi_skip_init=0.5, phi_skip_trainable=True)
    _forward(m)
    # Every GoldenSkipBlock alpha should equal the requested init
    # (within fp32 precision).
    from nature_inspired_networks.phi_scaling import GoldenSkipBlock
    found_blocks = [m_ for m_ in m.modules() if isinstance(m_, GoldenSkipBlock)]
    assert found_blocks, "no GoldenSkipBlock found"
    for b in found_blocks:
        assert b.trainable is True
        assert abs(float(b.alpha.detach().item()) - 0.5) < 1e-5

    m_frozen = build_model("golden_skip", num_classes=10,
                           phi_skip_init=None, phi_skip_trainable=False)
    frozen_blocks = [m_ for m_ in m_frozen.modules()
                     if isinstance(m_, GoldenSkipBlock)]
    assert all(b.trainable is False for b in frozen_blocks)


def test_build_model_natureprior_phi_sparse_forward():
    """H13: ``natureprior_phi_sparse`` self-registers via sparse.py."""
    m = build_model("natureprior_phi_sparse", num_classes=10)
    _forward(m)
    # Head should be a PhiSparseLinear with density ~ 1/phi.
    from nature_inspired_networks.sparse import PhiSparseLinear, DEFAULT_DENSITY
    assert isinstance(m.fc, PhiSparseLinear)
    assert abs(m.fc.density - DEFAULT_DENSITY) < 1e-9


def test_build_model_natureprior_fib_stride_forward():
    """H18: ``natureprior_fib_stride`` self-registers via stride.py."""
    m = build_model("natureprior_fib_stride", num_classes=10)
    _forward(m)
    # The 3-stage stride schedule should be (1, 2, 3).
    assert m.strides == (1, 2, 3)


def test_build_model_natureprior_phi_relu_forward():
    """H19: ``natureprior_phi_relu`` self-registers via phi_threshold.py."""
    m = build_model("natureprior_phi_relu", num_classes=10)
    _forward(m)
    from nature_inspired_networks.phi_threshold import PhiReLU
    found = [m_ for m_ in m.modules() if isinstance(m_, PhiReLU)]
    assert len(found) == 1, found  # stem only


def test_build_model_unknown_phi_kwargs_silently_ignored():
    """Forward-compat: unrelated kwargs do not raise on the legacy
    model paths -- build_model accepts ``**kwargs`` so the runner can
    pass every sweep key uniformly.
    """
    m = build_model("resnet20", num_classes=10,
                    phi_inverted=True, phi_budget_total=999_999,
                    something_unknown="ignored")
    _forward(m)


def test_build_model_resnet20_unchanged_legacy_path():
    """Backward-compat: legacy resnet20 path is unchanged."""
    m = build_model("resnet20", num_classes=10)
    _forward(m)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
