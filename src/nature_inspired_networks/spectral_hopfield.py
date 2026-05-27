"""H84 — Spectral Hopfield Associative Memory.

A modern continuous Hopfield network (Ramsauer et al. 2020) stores a set
of patterns ``X_stored`` and retrieves the pattern closest to a query
``q`` via a single softmax-weighted readout:

    ``retrieved = softmax(beta * X_stored @ q) @ X_stored``.

This is exactly the attention update rule, and with large ``beta`` it
converges to (near) exact recall of the matching stored pattern.

The "spectral / crystal" twist: patterns are matched in a real-FFT
(``rfft``) eigenmode basis rather than in the raw signal basis, so
retrieval keys on each pattern's vibrational-mode (frequency) structure.
Because ``rfft`` is a unitary-up-to-scale linear map, the energy /
softmax landscape is preserved, and we can store the (complex) spectra,
match in spectral space, then ``irfft`` the retrieved spectrum back to
the signal domain. The FFT round-trip ``irfft(rfft(x)) == x`` is lossless
for real inputs, so spectral matching never destroys information.

Esoteric origin (acknowledged in one sentence): the "crystal vibration
memory" motif motivates matching on vibrational modes; the implementation
is a modern Hopfield layer with an exact-invertible spectral basis.

Refs (Citation Rigor):
    Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich, M. 2020
    'Hopfield Networks is All You Need' (arXiv:2008.02217) - the modern
    continuous Hopfield update (softmax retrieval, exponential capacity)
    this module implements; the spectral basis is an exact linear
    reparameterisation of its energy landscape.

Public surface
--------------
- :class:`SpectralHopfieldMemory`  store(patterns) / retrieve(query)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .priors import PHI  # noqa: F401  (shared convention across primitives)

__all__ = ["SpectralHopfieldMemory"]


class SpectralHopfieldMemory(nn.Module):
    """Modern continuous Hopfield memory with spectral-basis matching.

    Patterns are real vectors of dimension ``dim``. :meth:`store` records
    a bank of patterns (kept as a buffer, so it persists with the module
    and moves with ``.to(device)``). :meth:`retrieve` returns, for each
    query, the softmax-weighted combination of stored patterns; at high
    ``beta`` this is (near) exact associative recall.

    Spectral twist: matching is done on the ``rfft`` spectra of patterns
    and query. We compare real spectra by stacking real and imaginary
    parts (an isometry of the complex inner product up to the usual rfft
    scaling), so the similarity score respects vibrational-mode structure.
    Retrieval mixes the *signal-domain* stored patterns with the spectral
    similarity weights, returning a real vector of shape ``(B, dim)``.

    Parameters
    ----------
    dim : int
        Pattern dimensionality.
    beta : float, default 8.0
        Inverse-temperature of the retrieval softmax. Larger ``beta`` →
        sharper (more nearest-neighbour-like) retrieval.

    Notes
    -----
    ``store`` may be called repeatedly; each call *replaces* the bank.
    Retrieval before any ``store`` raises a ``RuntimeError``.
    """

    def __init__(self, dim: int, beta: float = 8.0) -> None:
        super().__init__()
        assert dim > 0
        self.dim = dim
        self.beta = float(beta)
        # Stored patterns in the signal domain: (M, dim). Empty until store().
        self.register_buffer("patterns", torch.empty(0, dim))

    # -- spectral helpers ---------------------------------------------------
    @staticmethod
    def _to_spectral(x: torch.Tensor) -> torch.Tensor:
        """Map real signal ``(..., dim)`` to a real spectral feature
        ``(..., 2 * (dim//2 + 1))`` by stacking rfft real & imag parts.
        """
        spec = torch.fft.rfft(x, dim=-1)  # complex (..., dim//2 + 1)
        return torch.cat([spec.real, spec.imag], dim=-1)

    @staticmethod
    def spectral_roundtrip(x: torch.Tensor) -> torch.Tensor:
        """``irfft(rfft(x))`` — the lossless FFT round-trip (exposed for
        tests and for callers that want a no-op spectral pass)."""
        n = x.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1), n=n, dim=-1)

    # -- memory API ---------------------------------------------------------
    def store(self, patterns: torch.Tensor) -> None:
        """Store a bank of patterns ``(M, dim)`` (replaces any existing)."""
        if patterns.ndim != 2 or patterns.shape[-1] != self.dim:
            raise ValueError(
                f"expected patterns (M, {self.dim}), got {tuple(patterns.shape)}"
            )
        self.patterns = patterns.detach().clone()

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve the softmax-weighted stored pattern(s) for ``query``.

        ``query`` is ``(B, dim)`` (or ``(dim,)``, treated as a single
        query). Returns ``(B, dim)`` (or ``(dim,)`` to match the input
        rank). Similarity is computed in the spectral basis; the readout
        mixes the signal-domain stored patterns.
        """
        if self.patterns.numel() == 0:
            raise RuntimeError("retrieve() called before store()")
        squeeze = False
        if query.ndim == 1:
            query = query.unsqueeze(0)
            squeeze = True
        if query.ndim != 2 or query.shape[-1] != self.dim:
            raise ValueError(
                f"expected query (B, {self.dim}), got {tuple(query.shape)}"
            )
        patt = self.patterns.to(device=query.device, dtype=query.dtype)
        # Spectral features for stored patterns and queries.
        s_patt = self._to_spectral(patt)        # (M, F)
        s_query = self._to_spectral(query)       # (B, F)
        scores = self.beta * (s_query @ s_patt.t())  # (B, M)
        weights = torch.softmax(scores, dim=-1)       # (B, M)
        out = weights @ patt                          # (B, dim) signal domain
        if squeeze:
            out = out.squeeze(0)
        return out

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`retrieve`."""
        return self.retrieve(query)

    def extra_repr(self) -> str:
        m = self.patterns.shape[0] if self.patterns.numel() else 0
        return f"dim={self.dim}, beta={self.beta}, stored={m}"


# TODO runner wiring:
#   - models.py: add an optional `spectral_hopfield_head=True` config
#     branch that stores class-prototype feature vectors and retrieves a
#     denoised feature for the classifier head (Hopfield as a learned
#     associative memory layer over pooled features).
#   - configs/cifar10_quick.yaml: add a `spectral_hopfield_beta` flag so
#     the ablation row carries a distinct tag. This is a memory module,
#     not CNN-droppable into ResNet-20, so no sweep row is expected by
#     default.
#   - run_sweep.py: gate the row on a positive SOTA-smoke pre-flight (Rule 13).
