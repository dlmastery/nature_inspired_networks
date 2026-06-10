"""Unit tests for the Imagenette v2-160 loader (Wave-0 / SYNTHESIS_100 B1).

The loader downloads the ``imagenette2-160.tgz`` tarball on first call.
Per CLAUDE.md Rule 12, tests must SKIP gracefully when the dataset is
absent so a clean machine without the ~98 MB tarball can still pass the
unit-test gate. Network I/O is NEVER initiated from these tests --
download is a side-effect of the production loader, exercised by the
Wave-0 launch script, not the unit-test suite.

Covered:
  1. ``test_imagenette_train_loader_yields_correct_shape`` -- a batch
     pulled from the train loader has shape (B, 3, 160, 160).
  2. ``test_imagenette_classes_are_10`` -- the loader's class count
     is the published 10-class Imagenette taxonomy.
  3. ``test_imagenette_headline_mode_seeded_workers`` -- two
     headline_mode-seeded loads of the train loader at the same seed
     produce bit-identical first-batch augmentation tensors. Mirrors
     the contract exercised by ``tests/test_headline_mode.py`` for
     CIFAR.

Run via:
    .venv/Scripts/python -m pytest tests/test_imagenette_loader.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nature_inspired_networks.data import (  # noqa: E402
    IMAGENETTE_DIRNAME,
    imagenette_loaders,
)


# ---------------------------------------------------------------------------
# Shared skip helper -- production-loader is download-aware but the unit
# tests must NEVER touch the network. If the dataset isn't already on
# disk, skip with a clear message.
# ---------------------------------------------------------------------------
def _imagenette_dir() -> Path:
    return ROOT / "data" / IMAGENETTE_DIRNAME


def _skip_if_imagenette_missing() -> None:
    ds_dir = _imagenette_dir()
    if not (ds_dir / "train").is_dir() or not (ds_dir / "val").is_dir():
        pytest.skip(
            f"Imagenette not present at {ds_dir}; run the Wave-0 "
            f"launch script (or `python -c 'from "
            f"nature_inspired_networks.data import imagenette_loaders; "
            f"imagenette_loaders()'`) to download the ~98 MB tarball "
            f"and re-run these tests."
        )


# ---------------------------------------------------------------------------
# 1. Shape of a train batch.
# ---------------------------------------------------------------------------
def test_imagenette_train_loader_yields_correct_shape():
    """A batch pulled from the Imagenette train loader has the
    expected (B, 3, 160, 160) shape and float dtype."""
    _skip_if_imagenette_missing()
    tr_loader, _te_loader, n_cls, c_in = imagenette_loaders(
        root=str(ROOT / "data"), batch_size=4, num_workers=0,
        image_size=160,
    )
    assert n_cls == 10, n_cls
    assert c_in == 3
    x, y = next(iter(tr_loader))
    assert x.shape == (4, 3, 160, 160), x.shape
    assert x.dtype == torch.float32
    # Labels are class indices in [0, 10).
    assert int(y.min().item()) >= 0
    assert int(y.max().item()) < 10


# ---------------------------------------------------------------------------
# 2. Class count.
# ---------------------------------------------------------------------------
def test_imagenette_classes_are_10():
    """Imagenette is a 10-class ImageNet subset. The loader must
    report ``num_classes == 10`` (and the underlying ImageFolder must
    expose 10 class directories)."""
    _skip_if_imagenette_missing()
    tr_loader, te_loader, n_cls, _ = imagenette_loaders(
        root=str(ROOT / "data"), batch_size=2, num_workers=0,
    )
    assert n_cls == 10, n_cls
    assert len(tr_loader.dataset.classes) == 10
    assert len(te_loader.dataset.classes) == 10
    # Train + val must share the same class taxonomy (otherwise
    # ImageFolder would emit mis-aligned label indices).
    assert tr_loader.dataset.classes == te_loader.dataset.classes


# ---------------------------------------------------------------------------
# 3. Headline-mode determinism for the augmentation stream.
# ---------------------------------------------------------------------------
def test_imagenette_headline_mode_seeded_workers():
    """Two headline_mode-seeded loads at the same seed must produce
    bit-identical first-batch augmentation tensors. Mirrors the
    contract enforced by tests/test_headline_mode.py for CIFAR.

    The contract is: in ``headline_mode=True`` the DataLoader wires
    a per-loader ``torch.Generator`` + ``worker_init_fn`` so that
    RandomResizedCrop / HFlip / RandAugment / Normalize all draw from
    a deterministic RNG seeded by the per-run seed. The same seed
    therefore produces byte-identical augmentations.

    We exercise this with RandAugment enabled so the augmentation
    stream is non-trivial (without RandAugment the augmentation is
    only RandomResizedCrop + HFlip which still consume an RNG, but
    the bit-identical contract holds for all combinations).
    """
    _skip_if_imagenette_missing()
    from nature_inspired_networks.runner import set_seed

    def _first_batch(seed: int):
        # Seed the main-process RNG before constructing the loader so
        # the worker_init_fn (which derives its per-worker seed from
        # torch.initial_seed()) is itself deterministic.
        set_seed(seed, headline_mode=True)
        tr_loader, _, _, _ = imagenette_loaders(
            root=str(ROOT / "data"), batch_size=2, num_workers=0,
            image_size=160,
            randaugment_n=1, randaugment_m=9,
            headline_mode=True, seed=seed,
        )
        x, y = next(iter(tr_loader))
        # Restore default fast-mode for any subsequent tests.
        set_seed(0, headline_mode=False)
        return x.clone(), y.clone()

    x1, y1 = _first_batch(seed=0)
    x2, y2 = _first_batch(seed=0)
    assert x1.shape == x2.shape == (2, 3, 160, 160)
    # Equal, not allclose -- the headline-mode contract is
    # bit-identical (test_headline_mode.py / Rule 35 sibling).
    assert torch.equal(x1, x2), (
        f"headline_mode determinism broken at first batch: max abs "
        f"diff {(x1 - x2).abs().max().item():.6e}"
    )
    assert torch.equal(y1, y2)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
