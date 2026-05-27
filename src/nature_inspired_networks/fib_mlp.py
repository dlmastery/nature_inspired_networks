"""H11 — Pure Fibonacci MLP (hidden sizes 8-13-21-34-... for tabular data).

Design doc: ``hypotheses/g2_layer_channel_neuron/H11_pure_fibonacci_mlp.md``.

Multi-layer perceptron whose hidden-layer widths follow consecutive
Fibonacci numbers. Default schedule for tabular benchmarks (e.g. the
Higgs UCI baseline) is ``[8, 13, 21, 34, 21, 13, 8]`` -- a
"diamond" pyramid that climbs to F_8=34 and steps back down. The
hypothesis (per the H11 design doc) is that the phi-paced growth rate
is the unique sweet spot for tabular data where features have
intermediate-cardinality interactions: pairs and triples matter, four-
plus rarely.

Public surface
--------------
- :class:`FibMLP`             nn.Module with Fib hidden sizes.
- :func:`default_fib_hidden`  the canonical diamond schedule helper.

References (Citation Rigor):
    Arik, Pfister 2021 AAAI 'TabNet: Attentive Interpretable Tabular
    Learning' (arXiv:1908.07442) -- tabular baseline.
    Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting Deep
    Learning Models for Tabular Data' (arXiv:2106.11189) -- the modern
    reference for tabular MLP design; default width=192 is the power-of-
    twos baseline H11 replaces.
    Baldi, Sadowski, Whiteson 2014 Nature Communications 'Searching for
    exotic particles in high-energy physics with deep learning'
    (arXiv:1402.4735) -- the Higgs UCI benchmark origin paper.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .activations import PhiGELU
from .priors import PHI  # noqa: F401  (re-exported for callers / docs)


# ---------------------------------------------------------------------------
# Schedule helper
# ---------------------------------------------------------------------------
def default_fib_hidden() -> list[int]:
    """Return the canonical H11 diamond hidden-size schedule.

    ``[8, 13, 21, 34, 21, 13, 8]`` -- four consecutive Fibonacci terms
    (F_6..F_9) on the ascent followed by the symmetric descent. The
    "peak at 34" matches the design doc's predicted parameter budget
    against the 64x4 linear-doubling baseline at -25 pct params.
    """
    return [8, 13, 21, 34, 21, 13, 8]


# ---------------------------------------------------------------------------
# FibMLP -- pure Python feed-forward stack with PhiGELU activation
# ---------------------------------------------------------------------------
class FibMLP(nn.Module):
    """Fully-connected feed-forward stack with Fibonacci hidden widths.

    Architecture::

        x : (B, input_dim)
          -> Linear(input_dim, h0)   -> activation
          -> Linear(h0, h1)          -> activation
          -> ...
          -> Linear(h_last, output_dim)

    The activation between hidden layers defaults to :class:`PhiGELU`
    (the H39 phi-Swish), keeping the activation choice composable with
    the rest of the nature-inspired family. The final linear projection
    has NO activation -- the output is logit-shaped for classification
    or regression-ready.

    Parameters
    ----------
    input_dim : int
        Feature dimension of the input ``(B, input_dim)``.
    hidden_sizes : sequence of int, optional
        Hidden widths. Defaults to :func:`default_fib_hidden`
        (``[8, 13, 21, 34, 21, 13, 8]``). May be any positive integer
        sequence -- a "uniform" baseline can be obtained by passing
        ``[64, 64, 64]`` (used by the drop-in regression test).
    output_dim : int
        Output feature dimension.
    activation : nn.Module factory, optional
        Callable that returns a fresh ``nn.Module`` for the inter-layer
        activation. Defaults to :class:`PhiGELU` so the MLP is
        "Fibonacci + phi-GELU" -- the canonical H11 build.
    bias : bool
        Whether to include bias in each Linear. Default True.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] | None = None,
        output_dim: int = 2,
        activation: type[nn.Module] | None = None,
        bias: bool = True,
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
        self.activation_cls = activation or PhiGELU

        # All dims in the dense chain (input -> hidden... -> output)
        all_dims = [self.input_dim] + sizes + [self.output_dim]
        layers: list[nn.Module] = []
        for k in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[k], all_dims[k + 1], bias=bias))
            # No activation after the final linear (logit head)
            if k < len(all_dims) - 2:
                layers.append(self.activation_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_sizes={self.hidden_sizes}, "
            f"output_dim={self.output_dim}"
        )


# ---------------------------------------------------------------------------
# Runner wiring (TODO; left for the integration pass)
# ---------------------------------------------------------------------------
# TODO runner wiring:
#   - FibMLP is tabular-first; the CIFAR runner does not currently host a
#     tabular path. Wiring this in needs a build_tabular_model dispatcher
#     in models.py (analogous to build_phi_model) that takes a config
#     entry ``model: fib_mlp`` plus ``input_dim`` / ``hidden_sizes`` /
#     ``output_dim``.
#   - The H39 PhiGELU activation is already wired via swap_relu_with_phigelu
#     so the activation flag can be controlled at the config level.
#   - Higgs UCI benchmark: 28 input features, 2-class output, target
#     accuracy >= 79.0% (per Baldi 2014). The default hidden schedule
#     [8,13,21,34,21,13,8] has 28->8 (=232) ... ->2 (=18) params; total
#     ~= 2.8k params -- a strong -25% reduction over the 64x4 linear-
#     doubling baseline (~12k params).
