"""H52 — Drop-Path Regularization & Anytime Evaluation (FractalNet).

Design doc: ``hypotheses/g6_topological_bridging/H52_drop_path_anytime.md``.

The existing ``_FractalPath`` in :mod:`nature_inspired_networks.blocks` is a
depth-2 fractal recursion *without* drop-path. This module supplies the
additive FractalNet-style drop-path regularizer plus an "anytime" eval
helper, both as standalone primitives that the future trainer wiring
can pull in without modifying ``blocks.py``.

Three public classes / functions:

  * :class:`DropPath` — per-sample residual drop (timm convention). At
    eval time it is the identity; at train time it zeroes a random
    fraction of samples and rescales by ``1/(1-p)`` so the expectation
    is unchanged.
  * :class:`FractalDropPath` — wraps an arbitrary ``nn.ModuleList`` of
    blocks and applies per-block drop-path with probabilities increasing
    linearly from 0 to ``p_max`` along depth (this is the canonical
    FractalNet / "stochastic depth" schedule from Huang 2016).
  * :func:`anytime_forward` — eval-time helper that queries a model at
    multiple truncation depths via a ``set_max_depth`` hook the host
    model is expected to expose. Documented as *additive*: the helper
    does not patch the model.

References (Citation Rigor)::

    Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory 2017
    ICLR 'FractalNet: Ultra-Deep Neural Networks without Residuals'
    (arXiv:1605.07648) -- defines drop-path and the anytime-evaluation
    property at full fractal depth.

    Huang, Gao and Sun, Yu and Liu, Zhuang and Sedra, Daniel and
    Weinberger, Kilian Q. 2016 ECCV 'Deep Networks with Stochastic
    Depth' (arXiv:1603.09382) -- linear schedule that increases drop
    probability with depth; the basis for FractalDropPath.

The implementation deliberately *avoids* touching ``blocks.py`` /
``models.py`` so the rule "do not modify the training loop or models
mid-stream" is respected.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

__all__ = ["DropPath", "FractalDropPath", "anytime_forward"]


# ---------------------------------------------------------------------------
# DropPath (single-block, timm-style)
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Drop the residual path with probability ``p`` per sample.

    Per the ``timm`` convention this is *sample-wise* stochastic depth: a
    Bernoulli(1 - p) mask of shape ``(B, 1, 1, ..., 1)`` multiplies the
    input and the survivor is rescaled by ``1 / (1 - p)`` so the
    expectation is the identity. At eval time the layer is a no-op so
    the deterministic-eval invariant from the design doc holds bit-for-
    bit.

    Parameters
    ----------
    p : float
        Drop probability, in ``[0, 1)``. ``p == 0`` is exactly the
        identity in train mode (no stochasticity).
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"DropPath p must be in [0, 1); got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        # Per-sample mask: shape (B, 1, 1, ..., 1) broadcastable with x.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        # Rescale survivor so E[output] == x.
        return x * (mask / keep_prob)

    def extra_repr(self) -> str:
        return f"p={self.p}"


# ---------------------------------------------------------------------------
# FractalDropPath — depth-scheduled per-block drop
# ---------------------------------------------------------------------------
class FractalDropPath(nn.Module):
    """Apply per-block drop-path to an arbitrary block list, depth-scheduled.

    Wraps the supplied ``blocks`` (any iterable of ``nn.Module``) inside an
    ``nn.ModuleList`` and pairs each block with its own :class:`DropPath`
    instance. Drop probabilities increase linearly from ``0.0`` at the
    first block to ``p_max`` at the last — the canonical "stochastic
    depth" schedule from Huang 2016. This makes shallow features
    deterministic (they need to be reliable for the rest of the network)
    and deepens the regularisation gradually.

    Parameters
    ----------
    blocks : Iterable[nn.Module]
        Residual blocks to wrap. Each block is called sequentially; the
        output of block ``k`` is gated by ``DropPath(schedule[k])`` and
        added residually to the input of block ``k+1`` (i.e. the wrapper
        emulates ResNet-style residual addition with stochastic depth).
    p_max : float
        Drop probability of the deepest block. Default ``0.15``
        matches the FractalNet recommendation for CIFAR-scale stacks.

    Notes
    -----
    The schedule ``self.drop_probs`` is exposed as a plain Python list so
    upstream test code can verify monotonicity. The wrapper does *not*
    require the blocks to all have the same channel count — it leaves
    the residual addition to the block itself if the block already
    implements one; we apply DropPath on the block's *output* before
    summing into the running activation. If shapes mismatch (e.g. a
    downsampling block), users should keep the FractalDropPath stack
    within one stage.
    """

    def __init__(self, blocks: Iterable[nn.Module], p_max: float = 0.15) -> None:
        super().__init__()
        block_list = list(blocks)
        if not block_list:
            raise ValueError("FractalDropPath requires at least one block")
        if not 0.0 <= p_max < 1.0:
            raise ValueError(f"p_max must be in [0, 1); got {p_max}")
        self.blocks = nn.ModuleList(block_list)
        n = len(block_list)
        # Linear schedule: 0 at the first block, p_max at the last.
        if n == 1:
            self.drop_probs: List[float] = [0.0]
        else:
            self.drop_probs = [p_max * k / (n - 1) for k in range(n)]
        self.drops = nn.ModuleList([DropPath(p) for p in self.drop_probs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, drop in zip(self.blocks, self.drops):
            y = block(x)
            # Residual addition if shapes match; otherwise fall through.
            if y.shape == x.shape:
                x = x + drop(y)
            else:
                x = drop(y)
        return x


# ---------------------------------------------------------------------------
# anytime_forward helper
# ---------------------------------------------------------------------------
def anytime_forward(
    model: nn.Module,
    x: torch.Tensor,
    depths: Sequence[int] = (1, 2, 3),
) -> Dict[int, torch.Tensor]:
    """Run ``model`` at several truncation depths and collect the outputs.

    The host model is expected to expose a ``set_max_depth(d)`` method
    that mutates an internal cap (e.g. the number of fractal recursion
    levels or transformer layers used in the forward pass). The
    integration is **additive**: this helper does not patch the model;
    if ``set_max_depth`` is not present we raise a clear ``AttributeError``
    so the trainer-side wiring can detect the missing hook at start-up.

    After the sweep we restore the original ``_max_depth`` attribute
    (if any) so subsequent training is unaffected. The helper is
    decorated with ``torch.no_grad`` because anytime evaluation is
    eval-time only — gradient flow at multiple depths simultaneously
    would require independent forward passes anyway.

    Parameters
    ----------
    model : nn.Module
        Anytime-capable model exposing ``set_max_depth(d)``.
    x : Tensor
        Single input batch.
    depths : Sequence[int]
        Truncation depths to evaluate. Each entry is passed to
        ``model.set_max_depth``.

    Returns
    -------
    Dict[int, Tensor]
        ``{depth: output}`` mapping; each value is the model output
        evaluated at the corresponding depth, ordered by ``depths``.
    """
    if not hasattr(model, "set_max_depth"):
        raise AttributeError(
            "anytime_forward requires the model to implement "
            "`set_max_depth(d)`; integration is additive — see the H52 "
            "design doc."
        )
    prev = getattr(model, "_max_depth", None)
    was_training = model.training
    model.eval()
    out: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for d in depths:
            model.set_max_depth(d)
            out[int(d)] = model(x)
    # Restore prior state.
    if prev is not None:
        model.set_max_depth(prev)
    if was_training:
        model.train()
    return out
