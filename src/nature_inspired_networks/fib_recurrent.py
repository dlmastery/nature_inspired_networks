"""H14 — Fibonacci Recurrent (Fib-sized GRU stack with phi-gated updates).

Design doc: ``hypotheses/g2_layer_channel_neuron/H14_fibonacci_recurrent.md``.

A stack of :class:`nn.GRUCell` layers whose hidden states grow by
consecutive Fibonacci numbers (default ``[8, 13, 21, 34]``). The
hypothesis is that this matches the prefrontal working-memory recurrence
rule -- each layer's hidden state size is the sum of the two prior
layers' sizes -- and that biasing the update gate toward ``1/phi``
(via the ``phi_gate`` flag) preserves the biological memory-decay
half-life. The expected benefit (per H14 sec. 2) is +2-5 pp accuracy on
synthetic long-range copy tasks at -25 pct parameters vs constant-hidden
GRU.

Public surface
--------------
- :func:`default_fib_hidden`  canonical [8, 13, 21, 34] schedule.
- :class:`FibGRU`             stacked GRUCell stack with optional phi-gating.

References (Citation Rigor):
    Cho, van Merrienboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio
    2014 EMNLP 'Learning Phrase Representations using RNN Encoder-Decoder'
    (arXiv:1406.1078) -- GRU introduction; H14 modifies the update
    equation toward 1/phi.
    Hochreiter, Schmidhuber 1997 Neural Computation 'Long Short-Term
    Memory' (no arXiv) -- foundational LSTM; H14 inherits the gating
    motivation.
    Miller 1956 Psychological Review 'The Magical Number Seven, Plus or
    Minus Two' -- biological precedent for bounded working memory with
    Fibonacci decay.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .priors import PHI


PHI_RECIPROCAL: float = 1.0 / PHI  # 0.6180339...
# Logit of 1/phi: log(p/(1-p)) where p = 1/phi ≈ 0.618. With p = 1/phi we
# have 1 - p = 1/phi^2 = 2 - phi (because 1/phi + 1/phi^2 = 1), so
# logit(1/phi) = log((1/phi) / (1/phi^2)) = log(phi) ≈ +0.4812.
# This is the pre-sigmoid shift that makes sigmoid(b_z) average around 1/phi
# at init — the canonical mechanism per H14 design doc sec. 5.1'.
LOGIT_PHI_RECIPROCAL: float = math.log(PHI_RECIPROCAL / (1.0 - PHI_RECIPROCAL))


# ---------------------------------------------------------------------------
# Schedule helper
# ---------------------------------------------------------------------------
def default_fib_hidden() -> list[int]:
    """Return the canonical H14 stacked-GRU hidden schedule.

    ``[8, 13, 21, 34]`` -- four consecutive Fibonacci terms (F_6..F_9).
    The terminal hidden size 34 matches the design doc's predicted
    sequence-copy ceiling at -25 pct params vs a constant-64 GRU stack.
    """
    return [8, 13, 21, 34]


# ---------------------------------------------------------------------------
# FibGRU -- stacked GRUCells with optional phi-gated update
# ---------------------------------------------------------------------------
class FibGRU(nn.Module):
    """Stack of :class:`nn.GRUCell` with Fibonacci-spaced hidden sizes.

    Architecture::

        x : (B, T, input_dim)
          -> GRUCell(in=input_dim, hid=h0)   -> per-step hidden h^(0)_t
          -> GRUCell(in=h0,        hid=h1)   -> h^(1)_t
          -> ...
          -> GRUCell(in=h_{L-2},  hid=h_{L-1})
          -> Linear(h_{L-1}, output_dim)     -> (B, T, output_dim)

    When ``phi_gate=True`` the update-gate's *pre-sigmoid bias* is
    initialised to ``logit(1/phi) = log(phi) ≈ +0.4812``. Concretely,
    each cell's ``bias_ih[hidden:2*hidden]`` slice (the slot
    corresponding to the update-gate ``z`` in PyTorch's
    ``[r | z | n]`` flattening) is filled with ``logit(1/phi)``. At
    init, with zero inputs/hidden, the sigmoid of that bias yields
    update probability ``≈ 1/phi``, encouraging longer memory retention
    (the biological half-life motivation, H14 sec. 1).

    **Logit-vs-multiplicative distinction.** A *bias init* (this code)
    shifts the pre-sigmoid logit so the gate's average update
    probability at init is ``1/phi``; the sigmoid is unbounded above
    after training drifts. A *multiplicative rescale* (``z * 1/phi``,
    rejected here) caps the post-sigmoid output at ``1/phi`` forever.
    These are NOT equivalent: the bias-init can be undone by gradient
    descent (it is just an init), whereas the multiplicative cap
    permanently distorts the gate. The H14 design doc sec. 5.1'
    prescribes the bias-init recipe, which is what this implementation
    now does.

    Parameters
    ----------
    input_dim : int
        Feature dimension of the input ``(B, T, input_dim)``.
    hidden_sizes : sequence of int, optional
        Stacked hidden sizes. Defaults to :func:`default_fib_hidden`
        (``[8, 13, 21, 34]``).
    output_dim : int
        Output feature dim per time step (e.g. vocabulary size for
        synthetic copy tasks).
    phi_gate : bool
        When True, divide the GRU update probability by phi. Default
        False -- the constant-recurrence baseline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] | None = None,
        output_dim: int = 2,
        phi_gate: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1; got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1; got {output_dim}")
        if hidden_sizes is None:
            hidden_sizes = default_fib_hidden()
        sizes = [int(h) for h in hidden_sizes]
        if any(h < 1 for h in sizes):
            raise ValueError(f"hidden_sizes must be >= 1; got {sizes}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = sizes
        self.phi_gate = bool(phi_gate)

        # Build stacked GRUCells: GRUCell(input_dim -> h0), (h0 -> h1), ...
        cells: list[nn.GRUCell] = []
        prev = input_dim
        for h in sizes:
            cells.append(nn.GRUCell(prev, h))
            prev = h
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(sizes[-1], output_dim)

        # phi-gate bias init: per H14 sec. 5.1', the update-gate
        # bias slice (bias_ih[hidden:2*hidden]) is initialised to
        # logit(1/phi) so the sigmoid output averages around 1/phi at
        # init. PyTorch's GRUCell flattens the gate biases as
        # [b_ir | b_iz | b_in], hence the [h:2h] slice IS the
        # update-gate input-to-hidden bias.
        if self.phi_gate:
            with torch.no_grad():
                for cell in self.cells:
                    h_size = cell.hidden_size
                    cell.bias_ih.data[h_size:2 * h_size].fill_(
                        LOGIT_PHI_RECIPROCAL
                    )

    def _step(self, cell: nn.GRUCell, x_t: torch.Tensor,
              h_prev: torch.Tensor) -> torch.Tensor:
        """One GRU step (delegates to the standard PyTorch GRUCell).

        The phi-gate prior is realised by a one-shot **bias init** in
        :meth:`__init__` (see :data:`LOGIT_PHI_RECIPROCAL`), not by a
        forward-time rescale. Both phi_gate=True and phi_gate=False
        therefore share the exact same forward path; the only
        difference is that phi_gate=True wrote ``logit(1/phi)`` into
        each cell's ``bias_ih[hidden:2*hidden]`` slot at construction
        time so the update gate's sigmoid output averages ``1/phi``
        at init.
        """
        return cell(x_t, h_prev)

    def forward(self, x: torch.Tensor,
                h0: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Forward an (B, T, input_dim) sequence and project per-step.

        Returns
        -------
        torch.Tensor
            ``(B, T, output_dim)`` -- per-step logits produced by the
            terminal linear head fed from the topmost cell's hidden
            state.
        """
        if x.ndim != 3:
            raise ValueError(
                f"FibGRU expects (B, T, input_dim); got {tuple(x.shape)}"
            )
        B, T, F = x.shape
        if F != self.input_dim:
            raise ValueError(
                f"last dim {F} != input_dim {self.input_dim}"
            )
        device = x.device
        if h0 is None:
            h = [torch.zeros(B, hsize, device=device) for hsize in self.hidden_sizes]
        else:
            if len(h0) != len(self.hidden_sizes):
                raise ValueError(
                    f"len(h0)={len(h0)} != n_cells={len(self.hidden_sizes)}"
                )
            h = list(h0)
        outs: list[torch.Tensor] = []
        for t in range(T):
            inp = x[:, t]
            for k, cell in enumerate(self.cells):
                h[k] = self._step(cell, inp, h[k])
                inp = h[k]
            outs.append(self.head(h[-1]))
        return torch.stack(outs, dim=1)  # (B, T, output_dim)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Runner wiring (TODO; left for the integration pass)
# ---------------------------------------------------------------------------
# TODO runner wiring:
#   - FibGRU is a sequence-model primitive; the CIFAR runner does not
#     currently host a sequence-data path. To enable an H14 sweep row,
#     a build_seq_model dispatcher (analogous to build_phi_model) is
#     needed in models.py and a dataset adapter for the synthetic copy
#     task (or WikiText-103 char-level) in data.py.
#   - The H14 sweep row would be a 4-way ablation: {constant-64 GRU,
#     Fib GRU phi_gate=False, Fib GRU phi_gate=True, Fib GRU phi_gate
#     trainable scalar}. Rule 1 keeps the variants atomic.
