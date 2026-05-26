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


def load_dataset(name: str, root: str = "./data", batch_size: int = 256,
                 num_workers: int = 4):
    name = name.lower()
    if name in {"cifar10", "cifar100"}:
        return cifar_loaders(root=root, batch_size=batch_size,
                             num_workers=num_workers, variant=name)
    if name.startswith("medmnist:"):
        flag = name.split(":", 1)[1]
        return medmnist_loaders(root=f"{root}/medmnist", flag=flag,
                                batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"unknown dataset '{name}'")
