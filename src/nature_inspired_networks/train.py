"""Single-experiment training loop with bf16 AMP, cosine LR, label-smoothing CE."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .eval import (
    RunMetrics,
    composite_score,
    count_flops,
    count_params,
    gpu_latency_ms,
    rotation_equivariance_error,
    topk_accuracy,
)
from .schedulers import PhiDecayLR


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    target_top1: float = 0.85
    use_bf16: bool = True
    log_every: int = 50
    # H10 — Scheduler dispatch. 'cosine' preserves the legacy behaviour;
    # 'phi_decay' selects schedulers.PhiDecayLR with T_max=epochs.
    scheduler: str = "cosine"
    phi_lr_floor: float = 1e-6
    # Phase C — optimizer + per-layer weight decay routing.
    # ``optimizer`` is 'adamw' (default) or 'golden_adam' (H41).
    # ``phi_decay_wd`` enables per-block weight decay = base / phi^k (H44).
    optimizer: str = "adamw"
    phi_decay_wd: bool = False
    phi_decay_base: float = 5e-4
    # Phase D — epoch callbacks. ``prune_schedule='fibonacci'`` triggers
    # H43 fibonacci_prune at Fib-indexed epochs; ``momentum_schedule=
    # 'golden'`` wraps the optimizer with H48 GoldenMomentumScheduler;
    # ``fib_ensemble`` (dict or None) enables H20 FibEnsemble averaging.
    prune_schedule: str = ""
    prune_length: int = 5
    momentum_schedule: str = ""
    fib_ensemble: object = None  # dict {enabled: bool, K: int} or None
    # H64 — Dynamic Growth + Pruning schedule. When True, the Trainer
    # invokes ``GrowthPruningSchedule.step(epoch, model)`` at the end of
    # each epoch. The pre-constructed schedule object is attached via
    # ``growth_pruning_schedule_obj`` so the caller controls the
    # ``model_factory``; the Trainer just owns the per-epoch step.
    growth_pruning_schedule: bool = False
    growth_pruning_schedule_obj: object = None  # GrowthPruningSchedule or None


def _build_scheduler(opt: torch.optim.Optimizer, cfg: "TrainConfig"):
    """Dispatch on ``cfg.scheduler`` — default 'cosine' is the legacy path.

    Adding new modes here is the canonical wiring point so the Trainer
    body stays scheduler-agnostic. PhiDecayLR uses ``T_max=epochs`` so
    the LR shrinks by exactly 1/phi over the run (matching H10's
    cosine-comparable mid-train LR).
    """
    name = getattr(cfg, "scheduler", "cosine").lower()
    if name == "cosine":
        return CosineAnnealingLR(opt, T_max=cfg.epochs)
    if name in ("phi_decay", "phidecay", "phi"):
        return PhiDecayLR(opt, T_max=cfg.epochs,
                          lr_floor=getattr(cfg, "phi_lr_floor", 1e-6))
    raise ValueError(f"unknown scheduler '{name}'")


class Trainer:
    def __init__(self, model: nn.Module, train_loader, test_loader,
                 num_classes: int, cfg: TrainConfig | None = None,
                 device: str = "cuda", on_epoch: Callable[[dict], None] | None = None):
        self.cfg = cfg or TrainConfig()
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.on_epoch = on_epoch
        self.opt = self._build_optimizer(self.model, self.cfg)
        self.sched = _build_scheduler(self.opt, self.cfg)
        # H48 — optional golden-momentum scheduler that decays beta1 once
        # per epoch. Constructed AFTER the optimizer so the override
        # respects the optimizer family detected by GoldenMomentumScheduler.
        self.momentum_sched = None
        if (getattr(self.cfg, "momentum_schedule", "") or "").lower() in (
            "golden", "phi", "golden_momentum",
        ):
            from .schedulers import GoldenMomentumScheduler
            # PAPER_GAP_G5: use the closed-form exponential decay with
            # ``total_epochs=epochs`` so τ = epochs / 3 yields a smooth
            # curve over the full training horizon. ``T_max=epochs`` is
            # also forwarded so the legacy multiplicative path remains
            # tunable from the same config.
            self.momentum_sched = GoldenMomentumScheduler(
                self.opt,
                T_max=int(self.cfg.epochs),
                mode="exponential",
                total_epochs=int(self.cfg.epochs),
            )
        # H47 — collect PhiDropout modules so fit() can advance their
        # epoch-keyed curriculum once per epoch.
        from .regularizers import PhiDropout as _PhiDropout
        self._phi_dropouts = [
            m for m in self.model.modules() if isinstance(m, _PhiDropout)
        ]
        # H20 — optional FibEnsemble averager. Off by default; when on,
        # state-dict snapshots are pushed once per epoch and the final
        # averaged weights are loaded into ``self.model`` at the end of
        # ``fit()``.
        self.fib_ensemble = None
        fe_cfg = getattr(self.cfg, "fib_ensemble", None)
        if isinstance(fe_cfg, dict) and bool(fe_cfg.get("enabled", False)):
            from .ensemble import FibEnsemble
            self.fib_ensemble = FibEnsemble(K=int(fe_cfg.get("K", 8)))
        self.history: list[dict] = []
        # H43 — Fibonacci pruning. The flag triggers fibonacci_prune at
        # epoch boundaries from FIB_SCHEDULE[:prune_length].
        self._prune_enabled = (
            (getattr(self.cfg, "prune_schedule", "") or "").lower()
            in ("fibonacci", "fib")
        )
        # H64 — Dynamic Growth + Pruning schedule wiring. The schedule
        # object is constructed by the caller (so the model_factory is
        # caller-controlled) and the Trainer owns the per-epoch step.
        self.growth_pruning_schedule = None
        if bool(getattr(self.cfg, "growth_pruning_schedule", False)):
            obj = getattr(self.cfg, "growth_pruning_schedule_obj", None)
            if obj is None:
                # Build a default schedule using the current model as the
                # factory output. The factory returns the SAME model
                # instance each call -- the DynamicGrowthCallback will
                # store this reference but step() mutates the model in
                # place so it stays in sync.
                from .hybrid_growth_pruning import GrowthPruningSchedule
                obj = GrowthPruningSchedule(
                    model_factory=lambda m=self.model: m,
                )
            self.growth_pruning_schedule = obj
            self._growth_pruning_step_calls = 0

    @staticmethod
    def _build_optimizer(model: nn.Module, cfg: "TrainConfig") -> torch.optim.Optimizer:
        """Dispatch the optimizer based on cfg.optimizer + cfg.phi_decay_wd.

        Priority:
          1. ``phi_decay_wd=True`` -> build per-layer param groups via
             :func:`phi_decay.phi_decay_param_groups` and pass to
             :class:`torch.optim.AdamW` (or GoldenRatioAdamW when also
             optimizer='golden_adam').
          2. ``optimizer='golden_adam'`` -> H41 GoldenRatioAdamW.
          3. Default -> stock AdamW (the legacy behaviour).
        """
        opt_name = (getattr(cfg, "optimizer", "adamw") or "adamw").lower()
        if getattr(cfg, "phi_decay_wd", False):
            from .phi_decay import phi_decay_param_groups
            groups = phi_decay_param_groups(
                model, base_wd=float(getattr(cfg, "phi_decay_base", 5e-4)),
            )
            if opt_name in ("golden_adam", "golden_adamw", "phi_adam", "phi_adamw"):
                from .optimizers import GoldenRatioAdamW
                return GoldenRatioAdamW(groups, lr=cfg.lr,
                                        weight_decay=cfg.weight_decay)
            return AdamW(groups, lr=cfg.lr,
                         weight_decay=cfg.weight_decay)
        if opt_name in ("golden_adam", "golden_adamw", "phi_adam", "phi_adamw"):
            from .optimizers import build_optimizer
            return build_optimizer("golden_adam", model.parameters(),
                                   lr=cfg.lr, weight_decay=cfg.weight_decay)
        return AdamW(model.parameters(), lr=cfg.lr,
                     weight_decay=cfg.weight_decay)

    def _step(self, x, y) -> tuple[float, float]:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        if self.cfg.use_bf16 and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x)
                loss = F.cross_entropy(logits, y,
                                       label_smoothing=self.cfg.label_smoothing)
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits, y,
                                   label_smoothing=self.cfg.label_smoothing)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean().item()
        return float(loss.item()), float(acc)

    def fit(self) -> dict:
        epochs_to_target = -1
        t0 = time.perf_counter()
        train_top1_final = 0.0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            # H47 — advance every PhiDropout's curriculum to this epoch
            # BEFORE the first forward pass so the rate is correct for
            # the entire epoch interval. ``set_epoch`` pins the index
            # explicitly; this is equivalent to (but more robust than)
            # the duck-typed ``step_epoch`` hook used for arbitrary
            # epoch-keyed modules.
            for _pd in self._phi_dropouts:
                _pd.set_epoch(epoch)
            # Generic duck-typed epoch hook: any module exposing
            # ``step_epoch`` (other than the PhiDropouts already handled
            # above via the explicit pin) gets called once per epoch.
            for _m in self.model.modules():
                if _m in self._phi_dropouts:
                    continue
                hook = getattr(_m, "step_epoch", None)
                if callable(hook):
                    hook()
            losses, accs = [], []
            for it, (x, y) in enumerate(self.train_loader):
                lo, ac = self._step(x, y)
                losses.append(lo); accs.append(ac)
            self.sched.step()
            # H48 — decay beta1 once per epoch. The momentum scheduler
            # exposes ``set_epoch`` (closed-form exponential mode); fall
            # back to ``step`` for the legacy multiplicative path.
            if self.momentum_sched is not None:
                hook = getattr(self.momentum_sched, "set_epoch", None)
                if callable(hook):
                    hook(epoch + 1)
                else:
                    self.momentum_sched.step()
            # H43 — fibonacci prune at Fib-indexed epochs.
            if self._prune_enabled:
                from .pruning import fibonacci_prune
                fibonacci_prune(
                    self.model, epoch=epoch,
                    schedule_length=int(getattr(self.cfg, "prune_length", 5)),
                    make_permanent=False,
                )
            # H64 — Dynamic Growth + Pruning schedule step. Invoked at
            # the end of every epoch; the schedule's internal logic
            # decides whether ``epoch`` is a grow / prune / none event.
            # When a grow event fires, the optimizer is rebuilt from
            # scratch so the newly added parameters are tracked.
            if self.growth_pruning_schedule is not None:
                new_model, event = self.growth_pruning_schedule.step(
                    epoch=epoch, model=self.model,
                )
                self._growth_pruning_step_calls += 1
                if event == "grow":
                    # The model may have been mutated in place or replaced.
                    self.model = new_model.to(self.device)
                    self.opt = self._build_optimizer(self.model, self.cfg)
                    # Rebuild the LR scheduler too so it tracks the new opt.
                    self.sched = _build_scheduler(self.opt, self.cfg)
            tr_loss = sum(losses) / len(losses)
            tr_acc = sum(accs) / len(accs)
            train_top1_final = tr_acc

            # eval
            te = topk_accuracy(self.model, self.test_loader, device=self.device)
            row = dict(epoch=epoch, train_loss=tr_loss, train_top1=tr_acc,
                       test_top1=te["top1"], test_top5=te["top5"],
                       lr=self.sched.get_last_lr()[0])
            self.history.append(row)
            if self.on_epoch:
                self.on_epoch(row)
            # H20 — push current weights into the FibEnsemble buffer.
            if self.fib_ensemble is not None:
                self.fib_ensemble.update(self.model.state_dict())
            if epochs_to_target < 0 and te["top1"] >= self.cfg.target_top1:
                epochs_to_target = epoch + 1

        # H20 finalisation -- load Fib-weighted averaged weights so the
        # post-fit evaluation runs through the ensemble. Pruning leaves
        # behind ``weight_orig`` / ``weight_mask`` parameterisations that
        # are not strict-compatible with the FibEnsemble snapshots
        # captured pre-prune; we skip the load in that case to keep the
        # raw final weights.
        if self.fib_ensemble is not None and len(self.fib_ensemble) > 0:
            try:
                self.fib_ensemble.load_into(self.model)
            except RuntimeError:  # state-dict shape mismatch — keep raw
                pass

        train_seconds = time.perf_counter() - t0
        return dict(
            history=self.history,
            train_seconds=train_seconds,
            epochs_to_target=epochs_to_target,
            train_top1_final=train_top1_final,
        )


def evaluate_full(model: nn.Module, test_loader, dataset: str, tag: str,
                  seed: int, epochs: int, fit_info: dict,
                  input_size: tuple[int, ...] = (1, 3, 32, 32),
                  device: str = "cuda") -> RunMetrics:
    te = topk_accuracy(model, test_loader, device=device)
    params = count_params(model)
    flops = count_flops(model, input_size=input_size)
    lat = gpu_latency_ms(model, input_size=input_size)
    eq = rotation_equivariance_error(model, test_loader, device=device)
    comp = composite_score(te["top1"], params, lat)
    gap = max(0.0, fit_info["train_top1_final"] - te["top1"])
    return RunMetrics(
        tag=tag, dataset=dataset, seed=seed, epochs=epochs,
        top1=te["top1"], top5=te["top5"],
        params=params, flops=flops, latency_ms=lat,
        rot_eq_err=eq, composite=comp,
        epochs_to_target=fit_info["epochs_to_target"],
        train_seconds=fit_info["train_seconds"],
        train_top1=fit_info["train_top1_final"],
        generalization_gap=gap,
    )


def save_run(out_dir: str, metrics: RunMetrics, fit_info: dict,
             model: nn.Module | None = None) -> Path:
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    (p / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    (p / "history.json").write_text(json.dumps(fit_info["history"], indent=2))
    if model is not None:
        # Save the trained state_dict so post-hoc topology / CKA can use
        # *trained* features, not fresh-init ones.
        torch.save(model.state_dict(), p / "best.pt")
    return p
