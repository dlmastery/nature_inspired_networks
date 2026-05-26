"""Tests for the autoresearch reasoning gates."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sacgeo.reasoning import ReasoningEntry, validate_entry  # noqa: E402


def _good_entry(**over):
    base = dict(
        experiment_id="exp01",
        title="t",
        diagnosis=("Establishing the literature anchor against which every sacred "
                   "geometry variant in this campaign will be measured; this is the "
                   "very first experiment with no prior runs to diagnose. We need to "
                   "verify the training loop, FLOP counter, GPU latency timer, and "
                   "rotation equivariance error probe all wire up before any single "
                   "prior ablation is interpretable. The composite metric formula "
                   "must also be SHA-256 fingerprinted exactly once at this run so "
                   "every subsequent experiment fails loudly if the formula is "
                   "edited."),
        citations=[
            ("He, Zhang, Ren, Sun 2015 CVPR 'Deep Residual Learning for Image "
             "Recognition' (arXiv:1512.03385) -- Source of the canonical ResNet-20 "
             "CIFAR-10 variant that we use as the literature anchor in this "
             "experiment campaign; the 91.25 percent number from their Table 6 is "
             "the target asymptote for our 12-epoch quick budget.")
        ],
        hypothesis=("Because the model is the canonical 0.27 M-parameter ResNet-20 "
                    "with no sacred priors switched on, mechanism-wise it is just "
                    "stacked 3 by 3 conv plus BN plus ReLU residual blocks with the "
                    "He 2015 widths 16 32 64; we expect a 12-epoch top-1 in the low "
                    "to mid 80s on CIFAR-10 per the He 2015 recipe, which is enough "
                    "head-room to detect sacred-prior deltas of one to three points."),
        prediction=("composite in the range 0.78 to 0.83; top-1 in the range 0.80 "
                    "to 0.85 after 12 epochs; parameter count exactly 272474; GPU "
                    "batch-1 latency in the 5 to 12 ms band; rotation equivariance "
                    "error around 0.5 to 0.9."),
    )
    base.update(over)
    return ReasoningEntry(**base)


def test_passes_for_good_entry():
    assert validate_entry(_good_entry()) == []


def test_rejects_short_diagnosis():
    e = _good_entry(diagnosis="short.")
    errs = validate_entry(e)
    assert any("diagnosis" in s for s in errs)


def test_rejects_bad_citation_format():
    e = _good_entry(citations=["(He2016) said something."])
    errs = validate_entry(e)
    assert any("citation" in s for s in errs)


def test_rejects_no_citations():
    e = _good_entry(citations=[])
    errs = validate_entry(e)
    assert any("citation" in s for s in errs)


def test_post_run_requires_verdict_and_learning():
    e = _good_entry()
    errs = validate_entry(e, require_post=True)
    assert any("verdict" in s for s in errs)
    assert any("learning" in s for s in errs)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
