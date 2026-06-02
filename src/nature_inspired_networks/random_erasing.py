"""Random Erasing (Zhong et al., 2020) — torchvision wrapper at p=0.25.

Random Erasing zeroes (or random-fills) a random rectangle of each
training image. It is applied AFTER ``ToTensor`` + ``Normalize`` so the
zeros are in the *normalised* space. The convention in the modern
recipe (Bello 2021) is ``p=0.25``, ``scale=(0.02, 0.33)``,
``ratio=(0.3, 3.3)``, ``value=0`` (constant zero fill in normalised
space — equivalent to the per-channel mean of CIFAR after normalisation
which is exactly 0).

This module is a thin factory around :class:`torchvision.transforms.
RandomErasing` so the rest of the codebase imports a single neutral
name and the recipe defaults live in one place.

References (Citation Rigor)::

    Zhong, Zhun and Zheng, Liang and Kang, Guoliang and Li, Shaozi and
    Yang, Yi 2020 AAAI 'Random Erasing Data Augmentation'
    (arXiv:1708.04896) -- defines the erasing transform used here.
"""
from __future__ import annotations

import torch
import torchvision.transforms as T

__all__ = ["build_random_erasing"]


def build_random_erasing(
    p: float = 0.25,
    scale: tuple[float, float] = (0.02, 0.33),
    ratio: tuple[float, float] = (0.3, 3.3),
    value: float = 0.0,
) -> T.RandomErasing:
    """Return a configured :class:`torchvision.transforms.RandomErasing`.

    The torchvision implementation operates in-place on the tensor; the
    caller is responsible for placing this transform AFTER ``ToTensor``
    and ``Normalize`` in the train pipeline.

    Defaults reproduce the Bello-2021 modern recipe (p=0.25, scale=(0.02,
    0.33), ratio=(0.3, 3.3), value=0 constant zero fill).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"RandomErasing p must be in [0, 1]; got {p}")
    return T.RandomErasing(p=float(p), scale=tuple(scale), ratio=tuple(ratio),
                           value=float(value), inplace=True)


def apply_random_erasing_to_batch(
    x: torch.Tensor,
    erasing: T.RandomErasing,
) -> torch.Tensor:
    """Apply a configured :class:`RandomErasing` to every image in a batch.

    Mostly a convenience helper for the test suite; the production path
    integrates RandomErasing into the train pipeline via
    :class:`torchvision.transforms.Compose` so the operation is applied
    per-image inside the DataLoader.
    """
    out = x.clone()
    for i in range(out.size(0)):
        out[i] = erasing(out[i])
    return out
