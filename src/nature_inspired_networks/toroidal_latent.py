"""H78 — Toroidal Latent Embedding.

Project a latent vector onto a 2-torus ``T^2``. A linear map sends the
input ``(B, D)`` to two angles ``(theta1, theta2)``; each angle is
embedded as a unit-circle pair ``(cos t, sin t)``, giving a point on
``T^2`` embedded in ``R^4`` (two orthogonal circles). A second linear map
returns to ``out_dim``. The intermediate representation is periodic and
bounded, which is useful when the underlying factor of variation is
cyclic (orientation, phase, time-of-day) or when a smooth, wrap-around
continuous code is wanted.

Esoteric origin (acknowledged in one sentence): the "torus field" energy
motif motivates the toroidal latent; the implementation is a standard
angle-to-(cos,sin) embedding onto the product of two circles.

Refs (Citation Rigor):
    Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T., Tomczak, J. M.
    2018 UAI 'Hyperspherical Variational Auto-Encoders'
    (arXiv:1804.00891) - establishes that embedding latents on compact
    manifolds (circles/spheres, products of which give the torus) yields
    bounded, periodic codes with better behaviour for cyclic factors.

Public surface
--------------
- :class:`ToroidalLatent`     project a vector onto T^2 and back
- :func:`toroidal_distance`   geodesic distance on T^2 (wrapped angles)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .priors import PHI  # noqa: F401  (reused convention across primitives)

__all__ = ["ToroidalLatent", "toroidal_distance"]


def toroidal_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on the 2-torus between two angle tensors.

    ``z1`` and ``z2`` are angle tensors of shape ``(..., 2)`` holding
    ``(theta1, theta2)`` in radians. On each circle the wrapped angular
    distance is ``min(|d|, 2*pi - |d|)`` (so the distance between ``0.1``
    and ``2*pi - 0.1`` is ``0.2``, not ``2*pi - 0.2``). The torus
    distance is the Euclidean combination of the two per-circle wrapped
    distances:

        ``d = sqrt(d1^2 + d2^2)``  with  ``di = wrap(theta1_i - theta2_i)``.

    Returns a tensor of shape ``(...)`` (the trailing pair axis is
    reduced).
    """
    if z1.shape[-1] != 2 or z2.shape[-1] != 2:
        raise ValueError(
            f"expected trailing dim 2 (theta1, theta2); got "
            f"{tuple(z1.shape)} and {tuple(z2.shape)}"
        )
    two_pi = 2.0 * math.pi
    diff = z1 - z2
    # Wrap each angular difference into [-pi, pi], then take |.|; this is
    # exactly min(|d|, 2*pi - |d|).
    wrapped = torch.remainder(diff + math.pi, two_pi) - math.pi
    return torch.linalg.vector_norm(wrapped, dim=-1)


class ToroidalLatent(nn.Module):
    """Project a latent vector onto a 2-torus and back.

    Forward pipeline for input ``x: (B, in_dim)``:

    1. ``angles = Linear(in_dim -> 2)`` giving ``(theta1, theta2)``.
    2. ``embed = (cos t1, sin t1, cos t2, sin t2)`` -> ``(B, 4)``; this is
       a point on ``T^2`` (each ``(cos, sin)`` pair has unit norm).
    3. ``out = Linear(4 -> out_dim)``.

    The intermediate ``(B, 4)`` tensor always lies on the product of two
    unit circles, so the latent is bounded and periodic by construction.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    bias : bool, default True
        Whether the two linear maps carry biases.

    The forward also exposes :meth:`angles` and :meth:`embed_angles` so
    callers can recover the ``T^2`` coordinates for use with
    :func:`toroidal_distance`.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        assert in_dim > 0 and out_dim > 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.to_angles = nn.Linear(in_dim, 2, bias=bias)
        self.from_torus = nn.Linear(4, out_dim, bias=bias)

    def angles(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, 2)`` ``(theta1, theta2)`` angles for ``x``."""
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected (B, {self.in_dim}), got {tuple(x.shape)}"
            )
        return self.to_angles(x)

    @staticmethod
    def embed_angles(angles: torch.Tensor) -> torch.Tensor:
        """Embed ``(..., 2)`` angles as ``(..., 4)`` points on ``T^2``.

        Layout: ``(cos t1, sin t1, cos t2, sin t2)``.
        """
        t1 = angles[..., 0]
        t2 = angles[..., 1]
        return torch.stack(
            [torch.cos(t1), torch.sin(t1), torch.cos(t2), torch.sin(t2)],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.angles(x)               # (B, 2)
        embed = self.embed_angles(angles)     # (B, 4) on T^2
        return self.from_torus(embed)         # (B, out_dim)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


# TODO runner wiring:
#   - models.py: add an optional `toroidal_latent=True` config branch that
#     inserts a ToroidalLatent bottleneck between the global-avg-pool
#     feature and the classifier head (compresses to T^2 then expands).
#   - configs/cifar10_quick.yaml: add a `toroidal_latent_dim` flag so the
#     ablation row carries a distinct tag. This is a latent module, not a
#     CNN-droppable conv block, so no sweep row is expected by default.
#   - run_sweep.py: gate the row on a positive SOTA-smoke pre-flight (Rule 13).
