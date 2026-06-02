"""RandAugment (Cubuk et al., 2020) — torchvision wrapper at N=2, M=14.

RandAugment applies ``num_ops=N`` randomly-sampled operations from a
fixed augmentation pool (e.g. AutoContrast, Brightness, Color, Contrast,
Equalize, Posterize, Rotate, Sharpness, ShearX/Y, TranslateX/Y, Solarize)
each with magnitude ``M`` (out of 30 in torchvision's convention). The
Bello-2021 modern recipe uses ``(N, M) = (2, 14)`` for the CIFAR ResNet
family.

torchvision's :class:`RandAugment` operates on a ``PIL.Image`` (or a
``uint8`` Tensor). It must therefore be placed BEFORE ``ToTensor`` in
the train pipeline, with the usual ``RandomCrop`` / ``RandomHorizontalFlip``
preceding it.

References (Citation Rigor)::

    Cubuk, Ekin D. and Zoph, Barret and Shlens, Jonathon and Le, Quoc V.
    2020 CVPRW 'RandAugment: Practical automated data augmentation with
    a reduced search space' (arXiv:1909.13719) -- the parametrisation
    (N, M) and uniform-sample-then-uniform-magnitude policy used here.
"""
from __future__ import annotations

import torchvision.transforms as T

__all__ = ["build_randaugment"]


def build_randaugment(num_ops: int = 2, magnitude: int = 14) -> T.RandAugment:
    """Return a configured :class:`torchvision.transforms.RandAugment`.

    Defaults reproduce the Bello-2021 modern recipe ``(N, M) = (2, 14)``.
    The transform produces 3-channel ``uint8`` Tensors (or PIL images);
    callers should always follow it with ``ToTensor`` + ``Normalize``.
    """
    if num_ops < 0:
        raise ValueError(f"num_ops must be >= 0; got {num_ops}")
    if not 0 <= magnitude <= 30:
        raise ValueError(f"magnitude must be in [0, 30]; got {magnitude}")
    return T.RandAugment(num_ops=int(num_ops), magnitude=int(magnitude))
