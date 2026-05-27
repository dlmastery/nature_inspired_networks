"""H59 — Trained-Feature Betti Curves.

Design doc: ``hypotheses/g6_topological_bridging/H59_trained_feature_betti.md``.

Post-hoc analysis pipeline: load a ``best.pt`` checkpoint, run
``n_samples`` test examples through the model, and compute the per-
stage Betti curves on the resulting trained features. The existing
``scripts/compute_topology.py`` (and the helpers in
``nature_inspired_networks.topology``) compute Betti curves on *fresh-
init* features, never on trained ones. This module is the trained-
feature counterpart -- the diagnostic enabler for every downstream
topology claim (H51 / H54 / H65 / Naitzat 2020 -style class-cohesion
analysis).

Key difference from ``topology.betti_curve``: the result is keyed by
*stage name* rather than indexed by position, so callers can correlate
each stage's β₀ / β₁ with the named ``model.stagewise_features``
sub-modules (``stem``, ``stage1``, ``stage2``, ``stage3``). The
underlying persistent-homology computation reuses
``topology.betti_curve`` unchanged.

Public surface
--------------
- :func:`compute_trained_betti` -- load a checkpoint, extract stage
                                    features, return per-stage Betti
                                    curves keyed by stage name.

References (Citation Rigor)
---------------------------
    Naitzat, Gregory, Zhitnikov, Andrey, Lim, Lek-Heng 2020 JMLR
    'Topology of deep neural networks' (arXiv:2004.06093) -- trained
    networks topologically simplify while fresh-init networks do not;
    the empirical motivation for trained-feature analysis.
    Hofer, Christoph and others 2017 NeurIPS 'Deep Learning with
    Topological Signatures' (arXiv:1707.04041) -- foundational PH-
    based representation analysis.
    Bauer, Ulrich 2021 JACM 'Ripser: efficient computation of
    Vietoris-Rips persistence barcodes' (arXiv:1908.02518) -- the
    implementation library used for PH computation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology import betti_curve, collect_features


_DEFAULT_STAGE_NAMES = ("stem", "stage1", "stage2", "stage3")


def _stage_names_for(n_stages: int) -> tuple[str, ...]:
    """Return per-stage names of length ``n_stages``.

    For the canonical ``stem + stage1 + stage2 + stage3`` (4-stage)
    NaturePriorNet / ResNet-20 layout, return the canonical names.
    For other counts, return ``"stage_{i}"`` placeholders so the
    function is robust to architectures with fewer or more stages.
    """
    if n_stages == len(_DEFAULT_STAGE_NAMES):
        return _DEFAULT_STAGE_NAMES
    return tuple(f"stage_{i}" for i in range(n_stages))


def compute_trained_betti(
    checkpoint_path: str | Path,
    model: nn.Module,
    dataloader: Any,
    n_samples: int = 256,
    device: str = "cuda",
    rel_thresh: float = 0.20,
) -> dict[str, dict[str, list[int] | float]]:
    """Compute per-stage Betti curves on a *trained* checkpoint.

    The pipeline is:

    1. Load ``state_dict`` from ``checkpoint_path`` into ``model``.
       The checkpoint can either be the raw ``state_dict`` or a dict
       containing a ``"state_dict"`` / ``"model"`` / ``"model_state"``
       key (the runner saves under ``"model_state"``).
    2. Move ``model`` to ``device`` and switch to eval mode.
    3. Use :func:`topology.collect_features` to run ``n_samples`` test
       examples through ``model.stagewise_features``.
    4. Call :func:`topology.betti_curve` on the resulting per-stage
       feature matrices.
    5. Re-key the result by stage name so callers can map each
       (β₀, β₁) back to a named stage.

    Parameters
    ----------
    checkpoint_path : str or Path
        path to ``best.pt`` (or any ``.pt`` produced by the runner's
        ``save_run``).
    model : nn.Module
        architecture matching the checkpoint. Must expose
        ``stagewise_features(x) -> list[Tensor]`` or be wrapped so it
        falls back to a single-stage output (handled by
        ``collect_features``).
    dataloader : iterable yielding (x, y) batches
        test loader supplying the examples.
    n_samples : int, default 256
        number of examples to collect features for.
    device : str, default 'cuda'
        device on which to run inference. Falls back to ``'cpu'`` if
        CUDA is unavailable.
    rel_thresh : float, default 0.20
        relative persistence threshold (passed through to
        :func:`betti_curve`).

    Returns
    -------
    dict[str, dict]
        mapping ``stage_name -> {"b0": int, "b1": int}``. Every value
        is NaN-free and bounded above by ``n_samples``.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # Device fallback so the function is testable without a GPU.
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Step 1: load state dict (tolerate runner / lightning / vanilla
    # checkpoint key conventions).
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for k in ("model_state", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                state = raw[k]
                break
        else:
            state = raw
    else:
        state = raw
    model.load_state_dict(state, strict=False)

    # Step 2 + 3: collect per-stage features.
    feats = collect_features(model, dataloader, device=device, n_points=n_points_for(n_samples))

    # Step 4: compute betti curve (returns dict with parallel-indexed lists)
    curve = betti_curve(feats, rel_thresh=rel_thresh)
    b0_list = curve["b0"]
    b1_list = curve["b1"]
    threshold = float(curve.get("threshold", 0.0))

    # Step 5: re-key by stage name.
    stage_names = _stage_names_for(len(b0_list))
    out: dict[str, dict[str, list[int] | float]] = {}
    for name, b0, b1 in zip(stage_names, b0_list, b1_list):
        out[name] = {"b0": int(b0), "b1": int(b1), "threshold": threshold}
    return out


def n_points_for(n_samples: int) -> int:
    """Return the ``n_points`` argument forwarded to ``collect_features``.

    Trivial helper kept as a named function so a future change (e.g.,
    over-sampling to compensate for class imbalance) is centralised.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    return int(n_samples)
