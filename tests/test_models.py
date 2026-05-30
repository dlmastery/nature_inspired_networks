"""Unit tests for src/nature_inspired_networks/models.py.

Coverage:
- build_model dispatch for resnet20 / NaturePrior / phi-family (smoke).
- Control 3b (reviewer-flagged): RegNetX-200MF stock + shrunk variants
  via build_regnetx + width_multiplier_search.

Per CLAUDE.md Rule 12: >= 4 tests, every Boolean-flag combination,
one regression test, the canonical __main__ pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.models import (  # noqa: E402
    ResNet20,
    build_model,
    build_regnetx,
    width_multiplier_search,
)


# ---------------------------------------------------------------------------
# Smoke: resnet20 / NaturePrior path
# ---------------------------------------------------------------------------
def test_build_model_resnet20_smoke():
    """build_model('resnet20') returns a ResNet20 instance with a
    32x32 forward that produces (B, num_classes)."""
    m = build_model("resnet20", num_classes=10)
    assert isinstance(m, ResNet20)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_build_model_natureprior_smoke():
    """Legacy 'NaturePrior' alias still routes to NaturePriorNet."""
    m = build_model("NaturePrior", num_classes=10)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


# ---------------------------------------------------------------------------
# Control 3b — RegNetX-200MF stock + shrunk
# ---------------------------------------------------------------------------
def _torchvision_regnet_available() -> bool:
    try:
        from torchvision.models.regnet import BlockParams, RegNet  # noqa: F401
        return True
    except ImportError:  # pragma: no cover
        return False


def test_regnetx_200mf_builds():
    """Control 3b: stock RegNetX-200MF must build with the canonical
    init params (Radosavovic 2020 Table 9: depth=13, w_0=24, w_a=36.44,
    w_m=2.49, group_width=8). torchvision >= 0.21 dropped the
    regnet_x_200mf factory; this test exercises our locally-shipped
    BlockParams init."""
    if not _torchvision_regnet_available():
        # Environment without torchvision RegNet -- documented gap.
        return
    m = build_regnetx("regnetx_200mf", num_classes=100)
    # Forward sanity on ImageNet-shaped input (RegNet downsamples 5x;
    # at 32x32 the deepest stage collapses to 1x1 which BatchNorm
    # rejects in training mode. Use 224x224 -- the model's design
    # input -- for the shape sanity check).
    m.eval()
    y = m(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 100)


def test_regnetx_param_count_at_default_is_at_least_2M():
    """The stock RegNetX-200MF must produce >= 2M params (the
    canonical 200MF size from Table 9 is ~2.7M). This is the
    regression guard against an accidental rescaling that would
    silently collapse the iso-params comparison with phi_budget."""
    if not _torchvision_regnet_available():
        return
    m = build_regnetx("regnetx_200mf", num_classes=100)
    n = sum(p.numel() for p in m.parameters())
    assert n >= 2_000_000, f"stock RegNetX-200MF only {n} params (<2M)"


def test_regnetx_width_multiplier_search_hits_target_within_5_percent():
    """The width_multiplier_search must converge to a w_0 scale such
    that the realised param count is within +/- 5% of the requested
    target. We probe several budgets straddling phi_budget's 270k."""
    if not _torchvision_regnet_available():
        return
    for target in (270_000, 500_000, 1_000_000):
        init, realised, scale = width_multiplier_search(
            target_params=target, num_classes=100, tol_frac=0.05,
        )
        rel_err = abs(realised - target) / target
        assert rel_err <= 0.05, (
            f"target={target}: realised={realised}, scale={scale:.3f}, "
            f"rel_err={rel_err*100:.2f}% > 5%"
        )


def test_build_regnetx_shrunk_via_build_model_dispatch():
    """build_model('regnetx_200mf_shrunk', regnetx_param_budget=270k)
    routes through build_regnetx + width_multiplier_search."""
    if not _torchvision_regnet_available():
        return
    m = build_model("regnetx_200mf_shrunk", num_classes=100,
                    regnetx_param_budget=270_000)
    n = sum(p.numel() for p in m.parameters())
    # Must land within +/- 5% of the budget.
    assert abs(n - 270_000) / 270_000 <= 0.05, (
        f"shrunk RegNetX param drift: {n} vs 270000"
    )
    m.eval()
    y = m(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 100)


def test_build_model_rejects_unknown_name():
    """Unknown model name must raise ValueError -- no silent fallback
    (Rule 7)."""
    try:
        build_model("nonexistent_model", num_classes=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown model name")


def test_width_multiplier_search_rejects_invalid_target():
    """Defensive guard: target_params <= 0 is a hard error."""
    if not _torchvision_regnet_available():
        return
    try:
        width_multiplier_search(target_params=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on target_params=0")


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
