"""H41 — Unit tests for GoldenRatioAdamW.

Per CLAUDE.md Rule 12: ≥ 4 assertions, regression test named for H41,
edge case, parameter-value & state checks. File ends with the canonical
"All N tests passed." gate.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.optimizers import (  # noqa: E402
    GOLDEN_BETA1,
    GOLDEN_BETA2,
    GOLDEN_EPS,
    GoldenRatioAdamW,
    build_optimizer,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def _tiny_model() -> nn.Module:
    """Tiny model used by every step / state test below."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))


def test_h41_default_betas_are_phi_derived():
    """Regression test for H41: defaults must be (1/φ, 1/φ²)."""
    opt = GoldenRatioAdamW(_tiny_model().parameters(), lr=1e-3)
    b1, b2 = opt.defaults["betas"]
    assert math.isclose(b1, 1.0 / PHI, abs_tol=1e-12), b1
    assert math.isclose(b2, 1.0 / (PHI ** 2), abs_tol=1e-12), b2
    # eps is 1/φ^4 by design (see optimizers.py docstring).
    assert math.isclose(opt.defaults["eps"], 1.0 / (PHI ** 4), abs_tol=1e-12)
    # And the module-level constants stay in sync.
    assert GOLDEN_BETA1 == 1.0 / PHI
    assert GOLDEN_BETA2 == 1.0 / (PHI ** 2)
    assert GOLDEN_EPS == 1.0 / (PHI ** 4)


def test_h41_step_updates_params():
    """A single step() on non-zero grads must change every parameter."""
    model = _tiny_model()
    opt = GoldenRatioAdamW(model.parameters(), lr=1e-2)
    pre = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(3, 8)
    y = torch.tensor([0, 1, 0])
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in model.parameters()]
    # Every parameter must have moved (otherwise the optimizer is a no-op).
    for a, b in zip(pre, post):
        assert not torch.equal(a, b), "parameter did not update on step()"


def test_h41_state_has_exp_avg_after_step():
    """Adam-family optimizers should populate per-tensor state on step."""
    model = _tiny_model()
    opt = GoldenRatioAdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 8)
    y = torch.tensor([1, 0])
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    found_state = False
    for p in model.parameters():
        st = opt.state[p]
        if "exp_avg" in st and "exp_avg_sq" in st:
            found_state = True
            # Shapes match the parameter; values are finite.
            assert st["exp_avg"].shape == p.shape
            assert st["exp_avg_sq"].shape == p.shape
            assert torch.isfinite(st["exp_avg"]).all()
            assert torch.isfinite(st["exp_avg_sq"]).all()
    assert found_state, "no Adam state tensors populated after step()"


def test_h41_factory_dispatches_correctly():
    """build_optimizer must route 'golden_adam' to GoldenRatioAdamW
    and 'adamw' to plain AdamW; unknown names must raise."""
    model = _tiny_model()
    g = build_optimizer("golden_adam", model.parameters(), lr=1e-3,
                        weight_decay=1e-2)
    assert isinstance(g, GoldenRatioAdamW)
    assert math.isclose(g.defaults["betas"][0], GOLDEN_BETA1)

    a = build_optimizer("adamw", _tiny_model().parameters(), lr=1e-3,
                        weight_decay=1e-2)
    assert isinstance(a, torch.optim.AdamW)
    # Vanilla AdamW retains the original 0.9 / 0.999 defaults.
    assert math.isclose(a.defaults["betas"][0], 0.9)

    try:
        build_optimizer("does_not_exist", _tiny_model().parameters(),
                        lr=1e-3, weight_decay=1e-2)
    except ValueError:
        pass
    else:  # pragma: no cover — failure path
        raise AssertionError("unknown optimizer name should raise ValueError")


def test_h41_betas_overridable_for_ablation():
    """Edge case: caller can override betas back to vanilla Adam, in case
    they want to ablate eps alone. We accept the override and the
    optimizer still steps."""
    model = _tiny_model()
    opt = GoldenRatioAdamW(model.parameters(), lr=1e-3,
                           betas=(0.9, 0.999), eps=1e-8)
    assert math.isclose(opt.defaults["betas"][0], 0.9)
    assert math.isclose(opt.defaults["betas"][1], 0.999)
    assert math.isclose(opt.defaults["eps"], 1e-8)
    # And it still steps (no NaN / no exception).
    x = torch.randn(2, 8); y = torch.tensor([1, 0])
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
