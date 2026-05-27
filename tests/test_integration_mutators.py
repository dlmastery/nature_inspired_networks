"""Phase-B integration tests: post_build_mutators applies the H07 /
H31 / H39 / H42 / H47 overrides to a freshly built model.

Each test_* function exercises one mutator path through the runner-
side helper :func:`nature_inspired_networks.runner.post_build_mutators`.
The mutators are independent — flipping one cfg flag changes one set
of modules / weights and leaves the rest alone.

Run with the canonical ``__main__`` pattern -- no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.activations import PhiGELU  # noqa: E402
from nature_inspired_networks.models import build_model  # noqa: E402
from nature_inspired_networks.regularizers import PhiDropout  # noqa: E402
from nature_inspired_networks.runner import post_build_mutators  # noqa: E402


def _fresh_natureprior():
    """Build a no-prior NaturePriorNet (channel_mode=linear, no flags)."""
    from nature_inspired_networks.blocks import NaturePriorFlags
    return build_model(
        "NaturePrior", num_classes=10,
        flags=NaturePriorFlags(
            hex=False, group=False, fractal=False, toroidal=False,
            cymatic_init=False, golden_modulate=False,
        ),
        channel_mode="linear",
    )


def _fresh_resnet20():
    return build_model("resnet20", num_classes=10)


def _has_phigelu(model: nn.Module) -> bool:
    return any(isinstance(m, PhiGELU) for m in model.modules())


def test_post_build_no_op_when_no_flags_set():
    """Identity guarantee: empty cfg returns the model unchanged + no
    PhiGELU / PhiDropout injected.
    """
    m = _fresh_natureprior()
    m2 = post_build_mutators(m, cfg={})
    assert m2 is m
    assert not _has_phigelu(m2)
    assert not any(isinstance(c, PhiDropout) for c in m2.modules())


def test_phi_activation_swaps_relu_with_phigelu():
    """H39: ``phi_activation=True`` replaces every nn.ReLU in the model
    tree with a PhiGELU; the count drops to zero for ReLU and rises for
    PhiGELU.
    """
    m = _fresh_natureprior()
    n_relu_before = sum(1 for c in m.modules() if isinstance(c, nn.ReLU))
    assert n_relu_before > 0, "no ReLU to swap — test premise broken"
    m2 = post_build_mutators(m, cfg={"phi_activation": True})
    n_relu_after = sum(1 for c in m2.modules() if isinstance(c, nn.ReLU))
    n_phigelu = sum(1 for c in m2.modules() if isinstance(c, PhiGELU))
    assert n_relu_after == 0, n_relu_after
    assert n_phigelu == n_relu_before, (n_phigelu, n_relu_before)
    # Forward still works.
    m2.eval()
    with torch.no_grad():
        y = m2(torch.zeros(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_phi_init_changes_conv_weights():
    """H42: ``phi_init=True`` re-initialises every Conv2d / Linear weight
    so the per-tensor mean of |w| differs from the He-init draw.
    """
    m = _fresh_resnet20()
    before = []
    for c in m.modules():
        if isinstance(c, (nn.Conv2d, nn.Linear)):
            before.append(c.weight.detach().clone())
    m2 = post_build_mutators(m, cfg={"phi_init": True})
    after = []
    for c in m2.modules():
        if isinstance(c, (nn.Conv2d, nn.Linear)):
            after.append(c.weight.detach().clone())
    assert len(before) == len(after) > 0
    diffs = [not torch.allclose(a, b) for a, b in zip(before, after)]
    assert any(diffs), "phi_init did not change any weight tensor"


def test_golden_spiral_init_changes_5x5_conv_weights():
    """H31: ``golden_spiral_init=True`` only re-inits Conv2ds whose
    kernel matches ``golden_spiral_kernel`` (default 5). A pure
    ResNet-20 has no 5x5 convs, so we construct a Sequential of one
    5x5 conv and confirm the weight changes.
    """
    seq = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=5, padding=2),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),  # 3x3 stays untouched
    )

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = seq
            self.fc = nn.Linear(8, 10)

        def forward(self, x):
            return self.fc(self.body(x).mean(dim=(2, 3)))

    m = _Wrap()
    w5_before = m.body[0].weight.detach().clone()
    w3_before = m.body[1].weight.detach().clone()
    m2 = post_build_mutators(
        m, cfg={"golden_spiral_init": True, "golden_spiral_kernel": 5},
    )
    w5_after = m2.body[0].weight.detach()
    w3_after = m2.body[1].weight.detach()
    assert not torch.allclose(w5_before, w5_after)
    # 3x3 conv must be unchanged
    assert torch.allclose(w3_before, w3_after)


def test_phi_fpn_wrap_returns_wrapped_model_with_phispacedfpn():
    """H07: ``phi_fpn=True`` wraps a NaturePriorNet in a PhiSpacedFPN
    head; the wrapped model still does (2,3,32,32) -> (2,10).
    """
    from nature_inspired_networks.phi_scaling import PhiSpacedFPN

    m = _fresh_natureprior()
    m2 = post_build_mutators(
        m, cfg={"phi_fpn": True, "phi_fpn_c0": 16, "phi_fpn_levels": 3},
    )
    # The wrap must carry a PhiSpacedFPN submodule somewhere.
    assert any(isinstance(c, PhiSpacedFPN) for c in m2.modules())
    m2.eval()
    with torch.no_grad():
        y = m2(torch.zeros(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_phi_dropout_inserts_module_before_fc():
    """H47: ``dropout='phi_dropout'`` injects exactly one PhiDropout
    immediately before the model's final fc Linear.
    """
    m = _fresh_resnet20()
    m2 = post_build_mutators(
        m,
        cfg={"dropout": "phi_dropout", "dropout_cycle": "fib",
             "dropout_length": 5},
    )
    # m2.fc should now be a Sequential[PhiDropout, Linear]
    assert isinstance(m2.fc, nn.Sequential)
    assert isinstance(m2.fc[0], PhiDropout)
    assert isinstance(m2.fc[1], nn.Linear)
    # Forward still works.
    m2.eval()
    with torch.no_grad():
        y = m2(torch.zeros(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_combined_phi_init_and_phi_activation_compose():
    """Phase-B mutators compose: enabling both flips ReLU -> PhiGELU AND
    re-inits every Conv2d / Linear weight.
    """
    m = _fresh_resnet20()
    w_before = [c.weight.detach().clone()
                for c in m.modules() if isinstance(c, nn.Conv2d)]
    m2 = post_build_mutators(
        m, cfg={"phi_init": True, "phi_activation": True},
    )
    assert _has_phigelu(m2)
    w_after = [c.weight.detach() for c in m2.modules()
               if isinstance(c, nn.Conv2d)]
    assert any(not torch.allclose(a, b) for a, b in zip(w_before, w_after))


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
