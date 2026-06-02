"""Exponential Moving Average (EMA) of model parameters.

The Bello-2021 modern recipe uses an EMA decay of 0.9999 applied once
per optimisation step. The EMA shadow weights are evaluated on the test
set in addition to (or instead of) the raw weights; on CIFAR-scale this
typically adds 0.3-0.8 pp to top-1 at convergence.

This module provides :class:`ModelEMA` — a minimal wrapper that holds a
deep-copy of the model on the same device and updates it via
``ema.update(model)`` after each ``opt.step()``. The wrapper exposes
``ema.module`` for eval-time forward calls.

References (Citation Rigor)::

    Tarvainen, Antti and Valpola, Harri 2017 NeurIPS 'Mean teachers are
    better role models: Weight-averaged consistency targets improve
    semi-supervised deep learning results' (arXiv:1703.01780) -- the
    Mean-Teacher formulation of step-wise EMA used here.

    Bello, Irwan and Fedus, William and Du, Xianzhi and Cubuk, Ekin D.
    and Srinivas, Aravind and Lin, Tsung-Yi and Shlens, Jonathon and
    Zoph, Barret 2021 NeurIPS 'Revisiting ResNets: Improved Training
    and Scaling Strategies' (arXiv:2103.07579) -- specifies the
    decay=0.9999 default used in the convergence-regime recipe.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

__all__ = ["ModelEMA"]


class ModelEMA:
    """Exponential moving average of model parameters and buffers.

    The shadow model is a deep-copy held on the same device as the live
    model. ``update(model)`` is called after each optimiser step; the
    shadow's parameters are blended as

        p_ema <- decay * p_ema + (1 - decay) * p_live.

    Buffers (running BatchNorm statistics, etc.) are COPIED rather than
    blended — this is the convention used by ``timm`` and matches the
    Mean-Teacher reference implementation, since buffers are not
    learnable and a blended buffer can drift away from the live
    statistics it is supposed to track.

    Parameters
    ----------
    model : nn.Module
        The live model whose parameters to track.
    decay : float
        EMA decay coefficient. Standard values: 0.999 (fast), 0.9999
        (slow, Bello 2021), 0.99999 (very slow, large-batch). Must be
        in ``[0, 1)``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema decay must be in [0, 1); got {decay}")
        self.decay = float(decay)
        # Snapshot the live model. eval() so any dropout/batchnorm-in-train
        # quirks in the shadow are deterministic; the caller switches the
        # SHADOW into eval() at evaluation time anyway.
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend ``model``'s parameters into the EMA shadow.

        Buffers are COPIED, not blended (see class docstring).
        """
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if not torch.is_floating_point(v):
                # Integer buffers (num_batches_tracked, etc.) — straight copy.
                v.copy_(msd[k])
                continue
            # Float param / buffer.
            new = msd[k].detach()
            if k in dict(model.named_parameters()):
                # Learnable parameter — blend.
                v.mul_(d).add_(new, alpha=1.0 - d)
            else:
                # Non-learnable float buffer (BN running stats) — copy.
                v.copy_(new)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)
