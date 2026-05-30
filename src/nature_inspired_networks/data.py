"""Dataset loaders: CIFAR-10/100 + MedMNIST 2D wrappers.

All loaders return (train_loader, test_loader, num_classes, in_channels).
Mean/std normalization values are dataset-standard.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def _cifar_tfs(mean, std, train: bool):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def cifar_loaders(root: str = "./data", batch_size: int = 256, num_workers: int = 4,
                  variant: str = "cifar10"):
    root = Path(root)
    if variant == "cifar10":
        Ds = CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        n_cls = 10
    elif variant == "cifar100":
        Ds = CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        n_cls = 100
    else:
        raise ValueError(variant)
    tr = Ds(root=str(root), train=True, download=True,
            transform=_cifar_tfs(mean, std, train=True))
    te = Ds(root=str(root), train=False, download=True,
            transform=_cifar_tfs(mean, std, train=False))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           drop_last=True, persistent_workers=num_workers > 0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0)
    return tr_loader, te_loader, n_cls, 3


def medmnist_loaders(root: str = "./data/medmnist", flag: str = "pathmnist",
                     batch_size: int = 256, num_workers: int = 4, size: int = 28):
    """Load a MedMNIST 2D dataset. flag in {pathmnist, organamnist, octmnist, ...}."""
    import medmnist  # type: ignore
    from medmnist import INFO

    info = INFO[flag]
    n_cls = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    Path(root).mkdir(parents=True, exist_ok=True)
    mean = (0.5,) * (3 if size > 28 else 3)
    std = (0.5,) * 3
    tfs = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(mean, std),
    ])
    tr = DataClass(split="train", transform=tfs, download=True, root=root, size=size)
    te = DataClass(split="test", transform=tfs, download=True, root=root, size=size)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader, n_cls, 3


class _RotatedCIFAR(torch.utils.data.Dataset):
    """Wraps a CIFAR-10/100 dataset with deterministic per-image rotation
    chosen from a fixed set of angles.

    Used by :func:`load_rotated_cifar10` to build the eval pipeline
    (where rotations must be deterministic so the metric is stable
    across runs) and the all-4-rotations TTA pipeline.
    """

    def __init__(self, base, angles: tuple[int, ...] = (0, 90, 180, 270),
                 fill: int = 0):
        self.base = base
        self.angles = tuple(int(a) for a in angles)
        self.fill = float(fill)

    def __len__(self) -> int:
        return len(self.base) * len(self.angles)

    def __getitem__(self, idx: int):
        n = len(self.angles)
        base_idx = idx // n
        a_idx = idx % n
        img, label = self.base[base_idx]
        angle = self.angles[a_idx]
        if angle != 0:
            img = T.functional.rotate(img, angle, fill=[self.fill] * 3)
        return img, label


def rotated_cifar_loaders(root: str = "./data", batch_size: int = 256,
                          num_workers: int = 0, variant: str = "cifar10",
                          rotation_degrees: tuple[int, ...] = (0, 90, 180, 270),
                          fill: int = 0):
    """Load CIFAR-10/100 with rotation augmentation.

    Train pipeline applies a stochastic :class:`torchvision.transforms.
    RandomRotation` over ``rotation_degrees`` (treated as a discrete set
    rather than a continuous range -- the union of the four cardinal
    rotations is the canonical rotation-equivariance benchmark for
    H71's CIFAR proxy).

    Eval pipeline applies ALL four rotations as test-time augmentation
    via :class:`_RotatedCIFAR`, so the test set is 4x the size of the
    base CIFAR test split. Each underlying image therefore appears
    exactly once at each of the rotations in ``rotation_degrees``;
    accuracy reported by the runner is the average top-1 across the
    four rotated copies (rotation-equivariance-aware accuracy).

    Returns
    -------
    (train_loader, test_loader, num_classes, in_channels)
        Standard quadruple compatible with the runner.
    """
    root = Path(root)
    if variant == "cifar10":
        Ds = CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        n_cls = 10
    elif variant == "cifar100":
        Ds = CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        n_cls = 100
    else:
        raise ValueError(variant)

    # Train: standard CIFAR augmentations + stochastic rotation over the
    # angle set. RandomRotation accepts a (min, max) range; for a
    # discrete-cardinal-angle set we sample uniformly via a Lambda.
    angles = tuple(int(a) for a in rotation_degrees)
    fill_list = [float(fill)] * 3

    def _random_cardinal_rotate(img):
        """Pick one of the discrete angles uniformly at random."""
        a = int(angles[torch.randint(0, len(angles), (1,)).item()])
        if a == 0:
            return img
        return T.functional.rotate(img, a, fill=fill_list)

    train_tfs = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Lambda(_random_cardinal_rotate),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tfs = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    tr = Ds(root=str(root), train=True, download=True, transform=train_tfs)
    te_base = Ds(root=str(root), train=False, download=True, transform=eval_tfs)
    te = _RotatedCIFAR(te_base, angles=angles, fill=fill)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           drop_last=True,
                           persistent_workers=num_workers > 0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0)
    return tr_loader, te_loader, n_cls, 3


def load_rotated_cifar10(root: str = "./data", batch_size: int = 256,
                         num_workers: int = 0,
                         rotation_degrees: tuple[int, ...] = (0, 90, 180, 270),
                         fill: int = 0):
    """Backwards-compatible alias used by tests and the cfg routing.

    Forwards directly to :func:`rotated_cifar_loaders` with
    ``variant='cifar10'``.
    """
    return rotated_cifar_loaders(
        root=root, batch_size=batch_size, num_workers=num_workers,
        variant="cifar10", rotation_degrees=rotation_degrees, fill=fill,
    )


def load_dataset(name: str, root: str = "./data", batch_size: int = 256,
                 num_workers: int = 4):
    name = name.lower()
    if name in {"cifar10", "cifar100"}:
        return cifar_loaders(root=root, batch_size=batch_size,
                             num_workers=num_workers, variant=name)
    if name in {"rotated_cifar10", "rotcifar10"}:
        return rotated_cifar_loaders(
            root=root, batch_size=batch_size,
            num_workers=num_workers, variant="cifar10",
        )
    if name in {"rotated_cifar100", "rotcifar100"}:
        return rotated_cifar_loaders(
            root=root, batch_size=batch_size,
            num_workers=num_workers, variant="cifar100",
        )
    if name.startswith("medmnist:"):
        flag = name.split(":", 1)[1]
        return medmnist_loaders(root=f"{root}/medmnist", flag=flag,
                                batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"unknown dataset '{name}'")
