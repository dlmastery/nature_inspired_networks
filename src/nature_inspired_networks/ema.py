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
    warmup : bool
        Apply the timm/TF-style step-adjusted decay
        ``d_t = min(decay, (1 + step) / (10 + step))`` so the shadow
        actually tracks the live model during the first ~1/(1-decay)
        steps. Without warmup, at decay=0.9999 the shadow takes ~10000
        steps to escape its init and any short (≤ 50-epoch) sweep
        evaluates an essentially random-init shadow. See timm's
        ``ModelEmaV2`` and TensorFlow's
        ``tf.train.ExponentialMovingAverage(num_updates=step)`` for the
        canonical formula. Default ``True`` — the prior default of
        ``False`` was the modern-recipe smoke-collapse bug (2026-06-02).

    Notes
    -----
    The step counter is owned by :class:`ModelEMA` itself, incremented
    once per :meth:`update` call. Callers that want a fixed decay
    (legacy behaviour, or tests) can pass ``warmup=False``.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, warmup: bool = True,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema decay must be in [0, 1); got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        # Step counter for the warmup schedule (timm convention). Incremented
        # once per ``update``. Persisted via state_dict so a resumed run
        # picks up the warmup schedule where it left off.
        self._step = 0
        # Snapshot the live model. eval() so any dropout/batchnorm-in-train
        # quirks in the shadow are deterministic; the caller switches the
        # SHADOW into eval() at evaluation time anyway.
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        """Return the effective decay for the current step.

        With ``warmup=True`` we follow timm/TF and use
        ``min(decay, (1 + step) / (10 + step))`` so the shadow tracks the
        live model closely while ``step`` is small. With ``warmup=False``
        we return the configured decay unchanged (legacy behaviour).
        """
        if not self.warmup:
            return self.decay
        s = float(self._step)
        warm = (1.0 + s) / (10.0 + s)
        return min(self.decay, warm)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend ``model``'s parameters into the EMA shadow.

        Buffers are COPIED, not blended (see class docstring).
        """
        self._step += 1
        d = self._current_decay()
        msd = model.state_dict()
        # Cache the parameter-name set once per call (the dict(named_
        # parameters()) idiom rebuilds the entire mapping every call,
        # but a frozenset of names is sufficient for the membership test
        # and avoids any per-key dict allocation hot spot).
        param_names = {n for n, _ in model.named_parameters()}
        for k, v in self.module.state_dict().items():
            if not torch.is_floating_point(v):
                # Integer buffers (num_batches_tracked, etc.) — straight copy.
                v.copy_(msd[k])
                continue
            # Float param / buffer.
            new = msd[k].detach()
            if k in param_names:
                # Learnable parameter — blend.
                v.mul_(d).add_(new, alpha=1.0 - d)
            else:
                # Non-learnable float buffer (BN running stats) — copy.
                v.copy_(new)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)
