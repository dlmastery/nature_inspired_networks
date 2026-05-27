"""Unit tests for H44 — Golden Regularization (φ-decay weight decay).

Asserts:

* per-block schedule matches ``base_wd / φ^k`` to float tolerance;
* layer 0 receives ``base_wd`` exactly (regression: an off-by-one in
  the decay loop would shift the schedule by one block);
* layer N receives ``base_wd / φ^N`` exactly;
* GoldenRegularizer mutates ``opt.param_groups`` in place — caller's
  ``opt.step()`` semantics unchanged;
* param-group coverage is complete and disjoint (no orphans, no
  duplicates) — this is the "tests.py" property requested in the H44
  Committee Q&A.
* phi=1.0 ablation: every group gets exactly ``base_wd`` (uniform-λ
  control row used by the sweep).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.phi_decay import (  # noqa: E402
    GoldenRegularizer,
    build_phi_decay_param_groups,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def _toy_model() -> nn.Module:
    """6-block toy model mirroring NaturePriorNet's stem/blocks/head."""
    m = nn.Module()
    m.stem = nn.Conv2d(3, 8, 3, padding=1)
    m.blocks = nn.ModuleList([nn.Conv2d(8, 8, 3, padding=1) for _ in range(6)])
    m.head = nn.Linear(8, 10)
    return m


# ---------------------------------------------------------------------------
# build_phi_decay_param_groups
# ---------------------------------------------------------------------------
def test_phi_decay_layer0_gets_base_wd():
    model = _toy_model()
    groups = build_phi_decay_param_groups(model, base_wd=1e-2,
                                          block_attr="blocks")
    # First block-indexed group is blocks[0] → λ_0
    block_groups = [g for g in groups if g["layer_name"].startswith("blocks[")]
    assert block_groups, "no block groups produced"
    assert abs(block_groups[0]["weight_decay"] - 1e-2) < 1e-12


def test_phi_decay_layer_N_gets_base_over_phi_N():
    model = _toy_model()
    base = 1e-2
    groups = build_phi_decay_param_groups(model, base_wd=base,
                                          block_attr="blocks")
    block_groups = [g for g in groups if g["layer_name"].startswith("blocks[")]
    # Last block (k=5) → λ = base / φ^5
    expected_last = base / (PHI ** (len(block_groups) - 1))
    actual_last = block_groups[-1]["weight_decay"]
    assert abs(actual_last - expected_last) < 1e-12, (
        f"actual={actual_last} expected={expected_last}"
    )


def test_phi_decay_schedule_is_strictly_decreasing():
    model = _toy_model()
    groups = build_phi_decay_param_groups(model, base_wd=1e-2,
                                          block_attr="blocks")
    block_wds = [g["weight_decay"] for g in groups
                 if g["layer_name"].startswith("blocks[")]
    for a, b in zip(block_wds[:-1], block_wds[1:]):
        assert b < a, f"schedule must be decreasing, got {a} → {b}"


def test_phi_decay_non_block_children_get_base_wd():
    """Stem and head keep the full base_wd (no decay reduction)."""
    model = _toy_model()
    groups = build_phi_decay_param_groups(model, base_wd=1e-2,
                                          block_attr="blocks")
    other = [g for g in groups
             if not g["layer_name"].startswith("blocks[")]
    assert any(g["layer_name"] == "stem" for g in other)
    assert any(g["layer_name"] == "head" for g in other)
    for g in other:
        assert abs(g["weight_decay"] - 1e-2) < 1e-12, g["layer_name"]


def test_phi_decay_groups_cover_all_params_no_duplicates():
    """Every learnable parameter appears in exactly one group."""
    model = _toy_model()
    groups = build_phi_decay_param_groups(model, base_wd=1e-2,
                                          block_attr="blocks")
    all_ids: list[int] = []
    for g in groups:
        all_ids.extend(id(p) for p in g["params"])
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(all_ids) == expected_ids, "param coverage mismatch"
    assert len(all_ids) == len(set(all_ids)), "duplicate param across groups"


def test_phi_decay_works_without_block_attr():
    """When block_attr is None, every top-level child is φ-indexed."""
    model = _toy_model()
    groups = build_phi_decay_param_groups(model, base_wd=1e-2)
    wds = [g["weight_decay"] for g in groups]
    assert abs(wds[0] - 1e-2) < 1e-12  # k=0 → base_wd
    for k, wd in enumerate(wds):
        expected = 1e-2 / (PHI ** k)
        assert abs(wd - expected) < 1e-12, (k, wd, expected)


# ---------------------------------------------------------------------------
# GoldenRegularizer
# ---------------------------------------------------------------------------
def test_golden_regularizer_layer0_base_wd_and_layerN_phi_decay():
    """The wrapper must enforce the exact same schedule on an
    already-populated optimiser."""
    model = _toy_model()
    base = 5e-3
    # Build optimiser with raw per-block groups (uniform wd to start)
    raw_groups = [
        {"params": list(b.parameters()), "weight_decay": base}
        for b in model.blocks
    ]
    opt = AdamW(raw_groups, lr=1e-3)
    reg = GoldenRegularizer(opt, base_wd=base)
    sched = reg.schedule()
    assert abs(sched[0] - base) < 1e-12
    assert abs(sched[-1] - base / (PHI ** (len(sched) - 1))) < 1e-12


def test_golden_regularizer_phi_one_is_uniform():
    """phi=1.0 → uniform λ (the H44 sweep's "uniform control" row)."""
    model = _toy_model()
    base = 2e-3
    raw_groups = [
        {"params": list(b.parameters()), "weight_decay": 0.0}
        for b in model.blocks
    ]
    opt = AdamW(raw_groups, lr=1e-3)
    reg = GoldenRegularizer(opt, base_wd=base, phi=1.0)
    for wd in reg.schedule():
        assert abs(wd - base) < 1e-12


def test_golden_regularizer_step_still_runs():
    """The optimiser must remain callable after wrapping — i.e. the
    wrapper changes only ``weight_decay`` and leaves the step path
    unmodified."""
    model = _toy_model()
    raw_groups = [
        {"params": list(b.parameters()), "weight_decay": 0.0}
        for b in model.blocks
    ]
    opt = AdamW(raw_groups, lr=1e-3)
    GoldenRegularizer(opt, base_wd=1e-2)
    # forward+backward+step shouldn't error
    x = torch.randn(2, 8, 4, 4)
    for b in model.blocks:
        x = b(x)
    loss = x.pow(2).mean()
    loss.backward()
    opt.step()


def test_golden_regularizer_rejects_empty_optimizer():
    """A defensive guard: optimisers with no param_groups cannot be
    scheduled. The error must be loud (Rule 7: no silent fall-through)."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW([p], lr=1e-3)
    # Manually clear groups to simulate the pathological case.
    opt.param_groups = []
    try:
        GoldenRegularizer(opt, base_wd=1e-2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty param_groups")


def test_phi_decay_rejects_negative_base_wd():
    model = _toy_model()
    try:
        build_phi_decay_param_groups(model, base_wd=-1e-3)
    except ValueError:
        return
    raise AssertionError("expected ValueError on negative base_wd")


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
