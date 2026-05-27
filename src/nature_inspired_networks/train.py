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
        self.opt = AdamW(model.parameters(), lr=self.cfg.lr,
                         weight_decay=self.cfg.weight_decay)
        self.sched = _build_scheduler(self.opt, self.cfg)
        self.history: list[dict] = []

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
            losses, accs = [], []
            for it, (x, y) in enumerate(self.train_loader):
                lo, ac = self._step(x, y)
                losses.append(lo); accs.append(ac)
            self.sched.step()
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
            if epochs_to_target < 0 and te["top1"] >= self.cfg.target_top1:
                epochs_to_target = epoch + 1
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
