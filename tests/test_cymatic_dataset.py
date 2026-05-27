"""Unit tests for H56 — Cymatic Pattern Synthetic Dataset.

Run as a script:
    python tests/test_cymatic_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.cymatic_dataset import (  # noqa: E402
    CymaticDataset,
    generate_cymatic_pattern,
    generate_dataset,
)


def test_pattern_shape_and_dtype():
    pat = generate_cymatic_pattern(h=16, w=24, mode=(2, 3))
    assert pat.shape == (16, 24), pat.shape
    assert pat.dtype == torch.float32, pat.dtype
    assert torch.isfinite(pat).all()


def test_pattern_value_range():
    """Pattern values must lie in [-1, 1] for the analytical sin*sin formula."""
    for mode in [(2, 2), (3, 4), (5, 5), (1, 7)]:
        pat = generate_cymatic_pattern(h=32, w=32, mode=mode)
        assert pat.min().item() >= -1.0 - 1e-6, (mode, pat.min().item())
        assert pat.max().item() <= +1.0 + 1e-6, (mode, pat.max().item())


def test_pattern_mode_distinguishability():
    """Distinct (m, n) modes must produce visibly different images."""
    a = generate_cymatic_pattern(h=32, w=32, mode=(2, 3))
    b = generate_cymatic_pattern(h=32, w=32, mode=(4, 5))
    diff = (a - b).abs().mean().item()
    assert diff > 0.1, f"two distinct modes too similar (mean abs diff={diff})"


def test_dataset_length_and_item_type():
    n = 32
    ds = generate_dataset(n_samples=n, h=16, w=16, seed=0)
    assert isinstance(ds, TensorDataset), type(ds)
    assert len(ds) == n, len(ds)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor), type(img)
    assert img.shape == (1, 16, 16), img.shape
    # label is a 0-d tensor in a TensorDataset, but extracting the int value
    assert int(label.item()) >= 0 and int(label.item()) < 16


def test_cymatic_dataset_class_len_and_item():
    ds = CymaticDataset(n_samples=20, h=16, w=16, seed=42)
    assert len(ds) == 20
    img, cls = ds[5]
    assert img.shape == (1, 16, 16), img.shape
    assert isinstance(cls, int), type(cls)
    assert 0 <= cls < 16, cls


def test_cymatic_dataset_deterministic():
    """Two CymaticDataset instances with the same seed yield the same item."""
    a = CymaticDataset(n_samples=10, h=16, w=16, seed=7)
    b = CymaticDataset(n_samples=10, h=16, w=16, seed=7)
    for i in [0, 3, 9]:
        ia, ca = a[i]
        ib, cb = b[i]
        assert ca == cb
        assert torch.allclose(ia, ib)


def test_generate_dataset_label_coverage():
    """Over many draws the dataset's labels must cover most of the 16 classes."""
    ds = generate_dataset(n_samples=400, h=12, w=12, seed=0)
    labels = ds.tensors[1]
    n_unique = int(torch.unique(labels).numel())
    # 400 draws over 16 classes -> overwhelmingly likely to see >= 10 distinct
    assert n_unique >= 10, f"only {n_unique} distinct labels in 400 samples"


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items() if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
