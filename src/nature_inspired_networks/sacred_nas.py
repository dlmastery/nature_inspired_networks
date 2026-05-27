"""H45 — Sacred / Nature-Inspired Neural Architecture Search (stub).

Restricts a DARTS-style differentiable architecture search to a cell
library composed entirely of nature-inspired-prior operators:

- channel widths drawn from Fibonacci / φ schedules (H01 / H04 / H12);
- block primitives ∈ {NaturePriorBlock, _FractalPath, HexConv2d,
  GroupConv2d} (H05 / H21 / H24);
- connectivity restricted to {Metatron-Cube adjacency, φ-small-world}
  (H23 / H29).

The full DARTS supernet is out-of-scope for this stub. We provide:

- :func:`sacred_search_space` — descriptor of the restricted library
  (channel choices, block types, connectivity options).
- :func:`random_arch_sample` — deterministic random sample from the
  space given a seed (used by the smoke tests + the future search
  driver).
- :class:`SacredNASController` — minimal differentiable-arch
  controller: per-cell learnable logits α produce softmax mixture
  weights over the 4 candidate block types. ``forward(x)`` returns
  the softmax-weighted sum of block outputs.

This is a deliberately thin scaffold so the design space restriction
itself can be unit-tested before the full bi-level optimisation is
plumbed into ``scripts/run_sweep.py``.

References
----------
Liu, Hanxiao and Simonyan, Karen and Yang, Yiming 2019 ICLR 'DARTS:
Differentiable Architecture Search' (arXiv:1806.09055); Pham, Hieu
and Guan, Melody Y. and others 2018 ICML 'Efficient Neural
Architecture Search via Parameter Sharing' (arXiv:1802.03268). See
``hypotheses/g5_optimization_init_reg_nas/H45_sacred_nas.md``.
"""
from __future__ import annotations

import random as _random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nature_inspired_networks.blocks import NaturePriorBlock, _FractalPath, NaturePriorFlags
from nature_inspired_networks.platonic_graph import metatron_cube_adjacency
from nature_inspired_networks.priors import (
    PHI,
    GroupConv2d,
    HexConv2d,
    fibonacci_channels,
)


# ---------------------------------------------------------------------------
# Connectivity primitives
# ---------------------------------------------------------------------------
def platonic_graph_adjacency() -> torch.Tensor:
    """Return the canonical 13-node Metatron-Cube adjacency (alias).

    Provided here so the Sacred-NAS search-space descriptor names a
    single connectivity primitive without leaking the H23 module
    location. The returned tensor is the **same buffer** as
    :func:`metatron_cube_adjacency` and is symmetric with zero
    diagonal.
    """
    return metatron_cube_adjacency()


def phi_small_world(n: int, seed: int = 0) -> torch.Tensor:
    """Return a ``(n, n)`` φ-small-world adjacency matrix.

    Construction (Watts-Strogatz-like, φ-parameterised):

    1. Start from a ring lattice where every node connects to its
       ``k = max(2, round(n / PHI**2))`` nearest neighbours on each side.
    2. With probability ``1 / PHI`` (≈ 0.618), rewire each edge to a
       uniformly random non-self target. Self-loops are skipped; the
       resulting matrix is symmetrised by ``A = max(A, A.T)``.

    The output is binary (0/1) and symmetric with zero diagonal.
    Seed-deterministic so smoke tests reproduce.
    """
    assert n >= 3, f"phi_small_world requires n>=3; got {n}"
    rng = _random.Random(seed)
    k = max(2, int(round(n / (PHI * PHI))))
    A = torch.zeros(n, n, dtype=torch.float32)

    # 1. Ring lattice: connect each node to its ±k neighbours mod n.
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            A[i, j] = 1.0
            A[j, i] = 1.0

    # 2. φ-rewiring: each existing edge has probability 1/φ of being
    # replaced with a fresh random edge.
    p_rewire = 1.0 / PHI
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]
    for (i, j) in edges:
        if rng.random() < p_rewire:
            # remove old edge
            A[i, j] = 0.0
            A[j, i] = 0.0
            # add a fresh edge to a random non-self target
            attempts = 0
            while True:
                new_j = rng.randrange(n)
                attempts += 1
                if new_j == i or A[i, new_j] > 0:
                    if attempts > 4 * n:
                        # graph saturated; restore original edge and bail
                        A[i, j] = 1.0
                        A[j, i] = 1.0
                        break
                    continue
                A[i, new_j] = 1.0
                A[new_j, i] = 1.0
                break

    # zero diagonal as a sanity guarantee
    for i in range(n):
        A[i, i] = 0.0
    return A


# ---------------------------------------------------------------------------
# Search-space descriptor
# ---------------------------------------------------------------------------
def sacred_search_space(
    c0: int = 16,
    n_stages: int = 5,
) -> dict[str, Any]:
    """Return a dictionary describing the restricted Sacred-NAS cell
    library.

    Keys
    ----
    ``channel_choices``
        ``dict`` with two channel schedules:
          - ``"fib"`` — Fibonacci-spaced widths (H04);
          - ``"phi"`` — φ-geometric widths (H01).
        Each is a list[int] of length ``n_stages``.
    ``block_types``
        Tuple of 4 candidate block constructors (the cell ops the
        controller mixes over):
            ``(NaturePriorBlock, _FractalPath, HexConv2d, GroupConv2d)``.
    ``connectivity``
        List of two connectivity-primitive callables:
            ``[platonic_graph_adjacency, phi_small_world]``.

    The dictionary is **plain data**: no torch.nn.Module is instantiated
    by calling this function. Constructors / adjacency callables are
    stored as references so they can be wired up by the supernet later.
    """
    return {
        "channel_choices": {
            "fib": fibonacci_channels(c0, n_stages, mode="fib"),
            "phi": fibonacci_channels(c0, n_stages, mode="phi"),
        },
        "block_types": (NaturePriorBlock, _FractalPath, HexConv2d, GroupConv2d),
        "connectivity": [platonic_graph_adjacency, phi_small_world],
    }


def random_arch_sample(search_space: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Sample one architecture from the Sacred-NAS search space.

    The sample is fully deterministic given ``seed`` so smoke tests
    can pin behaviour. Each stage independently picks:

    - a channel-schedule mode ∈ {"fib", "phi"};
    - one of the 4 ``block_types``;
    - one of the connectivity primitives (by index).

    Returns
    -------
    dict
        ``{"channel_mode": str, "channel_widths": list[int],
          "block_picks": list[int], "connectivity_idx": int}``.
    """
    rng = _random.Random(seed)
    chan_mode = rng.choice(list(search_space["channel_choices"].keys()))
    widths = list(search_space["channel_choices"][chan_mode])
    n_blocks = len(search_space["block_types"])
    block_picks = [rng.randrange(n_blocks) for _ in widths]
    conn_idx = rng.randrange(len(search_space["connectivity"]))
    return {
        "channel_mode": chan_mode,
        "channel_widths": widths,
        "block_picks": block_picks,
        "connectivity_idx": conn_idx,
    }


# ---------------------------------------------------------------------------
# Minimal DARTS-style controller
# ---------------------------------------------------------------------------
class SacredNASController(nn.Module):
    """Minimal differentiable architecture-search controller.

    A single cell with 4 candidate operators (one per ``block_types``
    entry in the search space). Learnable logits ``α ∈ R^4`` produce a
    softmax mixture; ``forward(x)`` returns the weighted sum.

    This is a *stub* of full DARTS — sufficient to demonstrate the
    design-space restriction (only the 4 sacred ops are reachable) and
    to unit-test α normalisation + shape preservation. Bi-level
    optimisation, edge-level mixtures, and discrete derivation are
    deferred to the full search driver.

    Parameters
    ----------
    c_in : int
        Input channels. Output channels are forced equal to ``c_in``
        so the controller can be stacked into a chain.
    c_out : int | None
        If supplied, output channels (default = ``c_in``).
    """

    def __init__(self, c_in: int, c_out: int | None = None) -> None:
        super().__init__()
        c_out = c_out if c_out is not None else c_in
        self.c_in = c_in
        self.c_out = c_out
        # 4 candidate ops: align with sacred_search_space()["block_types"].
        flags = NaturePriorFlags(
            # keep the candidate ops cheap: disable expensive sub-flags by default
            hex=False, group=False, fractal=False, toroidal=False,
            cymatic_init=False, golden_modulate=False,
        )
        self.op_nature_prior = NaturePriorBlock(c_in, c_out, stride=1, flags=flags)
        self.op_fractal = _FractalPath(c_in, c_out, stride=1, depth=2, flags=flags)
        self.op_hex = HexConv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.op_group = GroupConv2d(c_in, c_out, kernel_size=3, padding=1,
                                    group="c4", bias=False)
        # learnable logits → softmax weights α_k
        self.logits = nn.Parameter(torch.zeros(4))

    @property
    def alpha(self) -> torch.Tensor:
        """Return the softmax-normalised architecture weights."""
        return F.softmax(self.logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        ops = [self.op_nature_prior, self.op_fractal, self.op_hex, self.op_group]
        out = a[0] * ops[0](x)
        for k in range(1, 4):
            out = out + a[k] * ops[k](x)
        return out
