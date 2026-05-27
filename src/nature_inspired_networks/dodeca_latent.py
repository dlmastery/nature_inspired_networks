"""H25 — Dodecahedral Latent.

The dodecahedron has 20 vertices with golden-ratio coordinates

    (±1, ±1, ±1),                       # 8 cube vertices
    (0, ±1/φ, ±φ),                      # 4 "rectangle in y-z plane"
    (±1/φ, ±φ, 0),                      # 4 "rectangle in x-y plane"
    (±φ, 0, ±1/φ).                      # 4 "rectangle in x-z plane"

This module:

- :func:`dodecahedron_vertices` — return the 20 vertices as ``(20, 3)``.
- :class:`DodecaLatentProjector` — softmax-projects an input feature
  vector onto the 20-vertex codebook, producing both the soft-assigned
  3-D latent and a small linear head back to ``out_dim``.
- :func:`vertex_distance_loss` — MSE between the projected 3-D latent
  and a target dodeca vertex (selected by integer index per sample).

References
----------
van den Oord 2017 'Neural Discrete Representation Learning'
(arXiv:1711.00937); Snell 2017 'Prototypical Networks'
(arXiv:1703.05175); Huh 2024 'The Platonic Representation Hypothesis'
(arXiv:2405.07987). See
``hypotheses/g3_topologies_graphs/H25_dodecahedral_latent.md``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.priors import PHI


_N_VERTICES = 20


def dodecahedron_vertices() -> torch.Tensor:
    """Return the 20 dodecahedron vertices as a ``(20, 3)`` float32 tensor.

    Coordinates use the standard golden-ratio parameterisation:

    - 8 cube vertices ``(±1, ±1, ±1)``
    - 4 vertices ``(0, ±1/φ, ±φ)``
    - 4 vertices ``(±1/φ, ±φ, 0)``
    - 4 vertices ``(±φ, 0, ±1/φ)``

    Each vertex has the same Euclidean distance from the origin —
    ``sqrt(3)`` — so the vertex set lies on a common sphere.
    """
    inv_phi = 1.0 / PHI
    verts: list[list[float]] = []
    # 8 cube vertices (±1, ±1, ±1)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                verts.append([sx, sy, sz])
    # (0, ±1/φ, ±φ)
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            verts.append([0.0, sy * inv_phi, sz * PHI])
    # (±1/φ, ±φ, 0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            verts.append([sx * inv_phi, sy * PHI, 0.0])
    # (±φ, 0, ±1/φ)
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            verts.append([sx * PHI, 0.0, sz * inv_phi])

    V = torch.tensor(verts, dtype=torch.float32)
    assert V.shape == (_N_VERTICES, 3), V.shape
    return V


class DodecaLatentProjector(nn.Module):
    """Soft-assignment bottleneck that projects an input feature onto
    20 dodecahedron-vertex prototypes.

    Forward: ``(B, in_dim) → (z_3d, soft_assign, out)`` where

    - ``z_3d ∈ (B, 3)`` is the convex combination of the 20 dodeca
      vertices weighted by ``softmax(W·x)``.
    - ``soft_assign ∈ (B, 20)`` is the soft assignment over vertices
      (useful as classification logits in a prototype-network style).
    - ``out ∈ (B, out_dim)`` is the linear head back to ``out_dim``.

    The 20 vertex coordinates are registered as a non-trainable buffer
    named ``vertices`` so they move with ``.to(device)``.
    """

    def __init__(self, in_dim: int, out_dim: int, normalize_vertices: bool = False) -> None:
        super().__init__()
        assert in_dim > 0 and out_dim > 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize_vertices = normalize_vertices

        V = dodecahedron_vertices()  # (20, 3)
        if normalize_vertices:
            V = V / V.norm(dim=-1, keepdim=True)
        self.register_buffer("vertices", V)

        self.to_codebook = nn.Linear(in_dim, _N_VERTICES, bias=True)
        self.head = nn.Linear(3, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.dim() == 2 and x.shape[1] == self.in_dim, (
            f"DodecaLatentProjector expects (B, {self.in_dim}); got {tuple(x.shape)}"
        )
        logits = self.to_codebook(x)              # (B, 20)
        soft = F.softmax(logits, dim=-1)          # (B, 20)
        z_3d = soft @ self.vertices               # (B, 3)
        out = self.head(z_3d)                     # (B, out_dim)
        return z_3d, soft, out


def vertex_distance_loss(z: torch.Tensor, target_vertex_idx: torch.Tensor) -> torch.Tensor:
    """MSE between projected 3-D latents and the addressed dodeca vertex.

    Parameters
    ----------
    z : ``(B, 3)`` tensor — the soft-assigned 3-D latent.
    target_vertex_idx : ``(B,)`` integer tensor in ``[0, 20)`` —
        the target vertex per sample.

    Returns the per-batch MSE (scalar). If ``z`` lies exactly on the
    target vertex for every sample, the loss is zero.
    """
    assert z.dim() == 2 and z.shape[-1] == 3, z.shape
    assert target_vertex_idx.dim() == 1 and target_vertex_idx.shape[0] == z.shape[0], (
        target_vertex_idx.shape, z.shape
    )
    V = dodecahedron_vertices().to(z.device).to(z.dtype)  # (20, 3)
    target = V[target_vertex_idx]                          # (B, 3)
    return F.mse_loss(z, target)


# TODO runner wiring:
#   - models.py: insert a DodecaLatentProjector between the pooled
#     feature vector and the final linear classifier. Choose
#     `in_dim = pool_dim, out_dim = num_classes`. The (B, 3) latent
#     can be exposed for OOD-AUC evaluation per H25 hypothesis.
#   - configs/cifar10_quick.yaml: gate behind a `dodeca_head: true`
#     ablation row.
#   - For VQ-VAE-style training, add an auxiliary `vertex_distance_loss`
#     term with a hard-assignment target (argmax over the 20-soft).
