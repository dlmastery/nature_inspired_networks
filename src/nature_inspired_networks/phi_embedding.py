"""H15 — phi-Initialised Embedding (golden-spiral lattice projected to d_model).

Design doc: ``hypotheses/g2_layer_channel_neuron/H15_phi_initialized_embedding.md``.

Initialises an :class:`nn.Embedding` weight matrix from a 2-D golden-
spiral lattice -- each token sits at radius ``sqrt(k+1)`` and angle
``k * 137.508 deg`` for k in ``range(num_embeddings)``. The 2-D lattice
is then projected up to ``embedding_dim`` via a random orthonormal
matrix sampled from the Haar measure on O(d). The construction is
deterministic given a seed and preserves the phyllotactic non-overlap
property of the original lattice in high dimensions.

Per the H15 design doc (sec. 2) the hypothesis is that this structured
init pre-equips the embedding space with angular separation, reducing
WikiText-103 perplexity by 0.3-0.8 over Xavier init at 1 epoch.

Public surface
--------------
- :func:`golden_spiral_embedding_init_` in-place initialiser (mirrors
                                        :func:`nn.init.xavier_uniform_`).
- :class:`PhiEmbedding`                  drop-in :class:`nn.Embedding`
                                        whose ``__init__`` applies the
                                        spiral-lattice init.

References (Citation Rigor):
    Mu, Viswanath 2018 ICLR 'All-but-the-Top: Simple and Effective
    Postprocessing for Word Representations' (arXiv:1702.01417) --
    motivates structured embedding inits.
    Mikolov, Chen, Corrado, Dean 2013 ICLR 'Efficient Estimation of
    Word Representations in Vector Space' (arXiv:1301.3781) -- the
    canonical embedding-quality baseline.
    Vogel 1979 Math Biosciences 'A better way to construct the
    sunflower head' -- 137.5 deg golden-angle construction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .priors import PHI


# Vogel's golden angle in radians: 2*pi * (1 - 1/phi) ~= 2.39996 rad
# (= 137.50776 deg)
GOLDEN_ANGLE_RAD: float = 2.0 * math.pi * (1.0 - 1.0 / PHI)


# ---------------------------------------------------------------------------
# In-place initialiser
# ---------------------------------------------------------------------------
def golden_spiral_embedding_init_(
    emb_layer: nn.Embedding,
    seed: int | None = None,
    scale: float | None = None,
) -> nn.Embedding:
    """Overwrite ``emb_layer.weight.data`` with golden-spiral-lattice init.

    Algorithm:

      1. Build the 2-D lattice
         ``pts[k] = (sqrt(k+1) * cos(k * golden_angle),
                      sqrt(k+1) * sin(k * golden_angle))``
         for ``k = 0, 1, ..., num_embeddings - 1``. The square-root
         radius keeps the points on a sunflower-style spiral with uniform
         angular density and ``O(sqrt(k))`` growth in radius.
      2. Sample an orthonormal matrix ``Q in R^{d x 2}`` from the Haar
         measure on the Stiefel manifold V_2(R^d) via QR decomposition
         of a random ``d x 2`` Gaussian.
      3. Project: ``W[k, :] = scale * pts[k] @ Q.t()`` -- shape
         ``(num_embeddings, embedding_dim)``.

    The default ``scale`` reproduces approximately the Gaussian-init
    variance ``1 / sqrt(d)`` so the embedding magnitudes are compatible
    with the downstream attention / softmax normalisation (the rationale
    in the H15 sec. 5.2 mechanism).

    Parameters
    ----------
    emb_layer : nn.Embedding
        Will be mutated in place; returned for chaining.
    seed : int, optional
        RNG seed used both for the lattice and for the projection. When
        omitted, uses the ambient ``torch.default_generator``.
    scale : float, optional
        Multiplicative scale on the final projected coordinates. Default
        ``1 / sqrt(2 * mean_radius_sq)`` -- chosen so the per-row L2
        norm has expected value 1.0.
    """
    if not isinstance(emb_layer, nn.Embedding):
        raise TypeError(
            f"golden_spiral_embedding_init_ expects nn.Embedding; "
            f"got {type(emb_layer).__name__}"
        )
    n = int(emb_layer.num_embeddings)
    d = int(emb_layer.embedding_dim)
    if n < 1 or d < 2:
        raise ValueError(
            f"need num_embeddings >= 1 and embedding_dim >= 2; got n={n}, d={d}"
        )
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    # Step 1: 2-D lattice (sqrt-radius sunflower spiral).
    k = torch.arange(n, dtype=torch.float64)
    angles = k * GOLDEN_ANGLE_RAD
    radii = torch.sqrt(k + 1.0)  # avoid zero-radius for k=0
    pts = torch.stack([radii * torch.cos(angles),
                       radii * torch.sin(angles)], dim=1)  # (n, 2)
    # Step 2: orthonormal projection matrix Q (d, 2) sampled via QR
    a = torch.randn(d, 2, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")  # (d, 2), Q^T Q = I_2
    # Step 3: project to d_model. Default scale -> unit per-row L2 norm.
    proj = (pts @ q.t()).to(dtype=emb_layer.weight.dtype)  # (n, d)
    if scale is None:
        # E[||row||^2] = E[radius^2] = mean over k of (k+1) = (n+1)/2.
        # So scale = 1 / sqrt(mean_radius_sq) gives unit-mean row norm.
        mean_r2 = float((radii * radii).mean().item())
        scale = float(1.0 / math.sqrt(mean_r2 + 1e-12))
    with torch.no_grad():
        emb_layer.weight.copy_(proj * scale)
    return emb_layer


# ---------------------------------------------------------------------------
# PhiEmbedding drop-in
# ---------------------------------------------------------------------------
class PhiEmbedding(nn.Embedding):
    """``nn.Embedding`` whose ``__init__`` runs the spiral-lattice init.

    Forward / state_dict / parameter contracts are identical to
    :class:`nn.Embedding`; only the post-construction weight values
    differ. Pass ``seed`` for reproducibility.

    Example::

        emb = PhiEmbedding(num_embeddings=50_257, embedding_dim=192,
                           seed=0)
        x = torch.tensor([1, 5, 50_000])
        y = emb(x)  # (3, 192)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        seed: int | None = None,
        scale: float | None = None,
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        golden_spiral_embedding_init_(self, seed=seed, scale=scale)


# ---------------------------------------------------------------------------
# Runner wiring (TODO; left for the integration pass)
# ---------------------------------------------------------------------------
# TODO runner wiring:
#   - PhiEmbedding is LLM-track; the CIFAR runner does not currently host
#     a token-embedding step. To enable an H15 sweep row, integrate with
#     the 124M decoder reference (whichever exists once the LLM track is
#     wired up) via a build_llm_model dispatcher.
#   - The default scale heuristic targets unit per-row L2 norm; for
#     fp16/bf16 numerical safety the caller may pass ``scale=1/sqrt(d)``
#     to match Xavier-uniform's effective variance.
#   - Per Rule 1 the H15 sweep row is atomic: only the embedding init is
#     swapped, leaving every other LM component on its default init.
