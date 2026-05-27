"""H35 - Cymatic Wavelet Kernels - implementation module.

Per Rule 14 (shared primitives live in ``src/``), the orthonormal
banded Chladni init originally drafted here was **promoted** to
``src/nature_inspired_networks/priors.py`` as the
``orthonormalize=True, band=(2, 5)`` branch of ``cymatic_init_`` and
the standalone ``chladni_modes_banded`` helper. This module now
re-exports those names plus a thin ``cymatic_init_ortho_`` wrapper for
backward compatibility with any caller still using the H35-local API.

The corrected init draws (m, n) from a configurable band, builds N
distinct modes for the N output channels, Gram-Schmidt orthonormalises
across channels, and rescales by the He standard deviation so
BatchNorm sees the same variance it would after Kaiming init.
"""
from __future__ import annotations

import torch.nn as nn

from nature_inspired_networks.priors import (  # noqa: F401
    chladni_modes,
    chladni_modes_banded,
    cymatic_init_,
)


def cymatic_init_ortho_(
    conv: nn.Conv2d,
    band: tuple[int, int] = (2, 5),
    seed: int = 0,
) -> None:
    """Thin wrapper kept for backward compatibility; delegates to the
    promoted ``cymatic_init_(orthonormalize=True, band=band)`` in
    ``nature_inspired_networks.priors``.
    """
    cymatic_init_(conv, orthonormalize=True, band=band, seed=seed)


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H35",
        short="cymatic_wavelet",
        primitives_touched=[
            "chladni_modes", "cymatic_init_",
            "chladni_modes_banded", "cymatic_init_ortho_",
        ],
        flags_touched=["cymatic_init"],
        legacy_t17_top1=0.7744,
        legacy_t17_composite=0.7883,
    )
