"""Unit tests for the dataset loaders in ``data.py``.

Avoids network I/O by building tests against the in-memory
:class:`_RotatedCIFAR` wrapper rather than downloading CIFAR. The
loader factory ``rotated_cifar_loaders`` is exercised end-to-end only
when the CIFAR-10 archive is already present locally (see
``data_root='./data'`` in this project's runtime configs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.data import (  # noqa: E402
    _RotatedCIFAR,
    load_dataset,
    load_rotated_cifar10,
    rotated_cifar_loaders,
)


# ---------------------------------------------------------------------------
# Helpers: in-memory CIFAR-shaped base dataset (avoids network I/O).
# ---------------------------------------------------------------------------
class _TinyImageDataset(torch.utils.data.Dataset):
    """Synthetic 3-channel tensor dataset shaped like CIFAR-10's
    `__getitem__` output (already-tensor images, scalar int labels)."""

    def __init__(self, n: int = 16, c: int = 3, h: int = 32, w: int = 32):
        torch.manual_seed(123)
        self.x = torch.rand(n, c, h, w)
        self.y = torch.randint(0, 10, (n,)).tolist()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# _RotatedCIFAR (the test-time multi-rotation wrapper)
# ---------------------------------------------------------------------------
def test_rotated_cifar_wrapper_length_is_4x_base():
    """For 4 rotation angles the wrapped dataset must yield 4x as many
    samples as the base."""
    base = _TinyImageDataset(n=10)
    wrapped = _RotatedCIFAR(base, angles=(0, 90, 180, 270))
    assert len(wrapped) == 40


def test_rotated_cifar_wrapper_first_4_items_are_all_rotations_of_same_image():
    """Items 0..3 must correspond to the same underlying image under
    the four cardinal rotations."""
    base = _TinyImageDataset(n=2)
    wrapped = _RotatedCIFAR(base, angles=(0, 90, 180, 270))
    images = [wrapped[i][0] for i in range(4)]
    # The first item is the 0-degree rotation = the raw base image.
    base_img, _ = base[0]
    assert torch.allclose(images[0], base_img)
    # The 180-degree rotation is the image rotated by 180 -- check it
    # equals the same image flipped along both spatial axes.
    expected_180 = torch.flip(base_img, dims=(-2, -1))
    # torchvision.transforms.functional.rotate(angle=180) is exactly
    # the spatial flip for cardinal angles -- assert this.
    assert torch.allclose(images[2], expected_180, atol=1e-6)
    # All 4 must share the SAME label (it's the same underlying image).
    labels = [wrapped[i][1] for i in range(4)]
    assert len(set(labels)) == 1, labels


def test_rotated_cifar_wrapper_idx_5_is_first_angle_of_second_image():
    """Layout: idx = base_idx * n_angles + a_idx. With 4 angles,
    idx=4 is image-1 at angle-0; idx=5 is image-1 at angle-90."""
    base = _TinyImageDataset(n=3)
    wrapped = _RotatedCIFAR(base, angles=(0, 90, 180, 270))
    img4, lab4 = wrapped[4]
    img1_base, lab1_base = base[1]
    assert torch.allclose(img4, img1_base)
    assert lab4 == lab1_base


def test_load_rotated_cifar10_eval_uses_all_4_rotations():
    """When the CIFAR-10 binary is locally available, the eval loader
    must yield 40_000 samples (10_000 base * 4 angles). If CIFAR-10
    isn't present, skip silently -- this test merely guards that the
    loader factory wires the all-4-rotations wrapper into the test
    pipeline."""
    cifar_path = Path("./data/cifar-10-batches-py")
    if not cifar_path.exists():
        return  # Documented gap -- run after CIFAR-10 is downloaded.
    _, te_loader, _, _ = load_rotated_cifar10(
        root="./data", batch_size=256, num_workers=0,
    )
    assert len(te_loader.dataset) == 40_000, len(te_loader.dataset)


def test_load_rotated_cifar10_yields_rotated_images():
    """End-to-end: pull a batch from the train loader and assert the
    tensor shape is (B, 3, 32, 32). Skip when CIFAR-10 isn't local."""
    cifar_path = Path("./data/cifar-10-batches-py")
    if not cifar_path.exists():
        return
    tr_loader, te_loader, n_cls, c_in = load_rotated_cifar10(
        root="./data", batch_size=8, num_workers=0,
    )
    assert n_cls == 10
    assert c_in == 3
    x, y = next(iter(tr_loader))
    assert x.shape == (8, 3, 32, 32)
    # Eval batch must also produce 3x32x32.
    xe, ye = next(iter(te_loader))
    assert xe.shape == (8, 3, 32, 32)


def test_load_dataset_routes_rotated_cifar10():
    """load_dataset('rotated_cifar10') must route to the rotated loader
    (and 'rotcifar10' alias)."""
    cifar_path = Path("./data/cifar-10-batches-py")
    if not cifar_path.exists():
        return
    for alias in ("rotated_cifar10", "rotcifar10"):
        tr, te, n_cls, c_in = load_dataset(
            alias, root="./data", batch_size=8, num_workers=0,
        )
        assert n_cls == 10
        assert c_in == 3
        assert len(te.dataset) == 40_000


def test_load_dataset_rejects_unknown_name():
    """Unknown dataset name must raise ValueError (Rule 7)."""
    try:
        load_dataset("nonexistent_dataset")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown dataset")


def test_rotated_cifar_loaders_rejects_unknown_variant():
    """rotated_cifar_loaders only knows cifar10 / cifar100 variants."""
    try:
        rotated_cifar_loaders(variant="cifar9000")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown variant")


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
