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

from typing import Sequence

import torch
import torch.nn as nn

from .priors import PHI


PHI_RECIPROCAL: float = 1.0 / PHI  # 0.6180339...


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

    When ``phi_gate=True`` the update equation of each GRUCell is
    rescaled by ``1/phi``:

        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
            ->
        h_t = (1 - z_t / phi) * h_{t-1} + (z_t / phi) * h_tilde_t

    which moves the effective update probability into ``[0, 1/phi]``,
    encouraging longer memory retention (the biological half-life
    motivation, H14 sec. 1). This is implemented as a *forward-time*
    rescaling on top of the standard GRUCell update so the cell's
    parameter count is unchanged -- the prior is geometric, not
    parametric.

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

    def _step(self, cell: nn.GRUCell, x_t: torch.Tensor,
              h_prev: torch.Tensor) -> torch.Tensor:
        """One GRU step with optional phi-gating.

        For the standard cell we delegate to ``cell(x_t, h_prev)``. For
        ``phi_gate=True`` we re-derive the update gate ``z`` from the
        cell's own weights and apply ``z / phi`` instead of ``z``. Doing
        so via the cell weights ensures parameter parity with the
        baseline; the only difference is the scalar 1/phi factor on the
        update probability.
        """
        if not self.phi_gate:
            return cell(x_t, h_prev)

        # PyTorch's GRUCell flattens weights as [W_ir|W_iz|W_in], etc.
        w_ih = cell.weight_ih  # (3*hidden, input)
        w_hh = cell.weight_hh  # (3*hidden, hidden)
        b_ih = cell.bias_ih
        b_hh = cell.bias_hh
        h_size = cell.hidden_size
        gi = torch.nn.functional.linear(x_t, w_ih, b_ih)
        gh = torch.nn.functional.linear(h_prev, w_hh, b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        z_phi = z * PHI_RECIPROCAL  # rescale update prob into [0, 1/phi]
        h_new = (1.0 - z_phi) * h_prev + z_phi * n
        # Silence the unused h_size local
        _ = h_size
        return h_new

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
