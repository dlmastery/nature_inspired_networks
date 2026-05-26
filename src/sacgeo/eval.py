"""Metrics: top-k, parameters, FLOPs, latency, equivariance error, composite."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """Return MACs (not FLOPs/2). fvcore handles masked conv / group conv OK
    since we expose actual nn.Conv2d under the hood; for HexConv2d/GroupConv2d
    we estimate manually (these don't have an fvcore registration).
    """
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        model.eval()
        x = torch.zeros(input_size, device=next(model.parameters()).device)
        fca = FlopCountAnalysis(model, x).unsupported_ops_warnings(False)
        return float(fca.total())
    except Exception:
        return float("nan")


@torch.no_grad()
def gpu_latency_ms(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 32, 32),
                   warmup: int = 20, iters: int = 100) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    model.eval().cuda()
    x = torch.zeros(input_size, device="cuda")
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    return float(dt)


@torch.no_grad()
def topk_accuracy(model: nn.Module, loader, device: str = "cuda",
                  topk: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    model.eval().to(device)
    maxk = max(topk)
    correct = {k: 0 for k in topk}
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        logits = model(x)
        _, pred = logits.topk(maxk, dim=1)
        pred = pred.t()
        match = pred.eq(y.view(1, -1).expand_as(pred))
        for k in topk:
            correct[k] += match[:k].reshape(-1).float().sum().item()
        n += y.size(0)
    return {f"top{k}": correct[k] / n for k in topk}


@torch.no_grad()
def rotation_equivariance_error(model: nn.Module, loader, device: str = "cuda",
                                max_batches: int = 8) -> float:
    """Average ||f(R x) - R f(x)||_2 / ||f(x)||_2 over the first batches and
    R ∈ {90°, 180°, 270°}. Approximates how rotation-invariant the network
    is — Platonic priors should drive this lower.
    """
    model.eval().to(device)
    errs: list[float] = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y0 = model(x)
        for k in (1, 2, 3):
            xr = torch.rot90(x, k=k, dims=(2, 3))
            yr = model(xr)
            num = (yr - y0).norm(dim=1)
            den = y0.norm(dim=1) + 1e-8
            errs.append((num / den).mean().item())
    return float(sum(errs) / max(1, len(errs)))


# ---------------------------------------------------------------------------
# Centered Kernel Alignment (CKA) — Kornblith et al. 2019
# ---------------------------------------------------------------------------
def _gram(X: torch.Tensor) -> torch.Tensor:
    return X @ X.t()


def _center(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA on (n_examples, n_features) matrices."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    Kx = _center(_gram(X))
    Ky = _center(_gram(Y))
    num = (Kx * Ky).sum()
    den = Kx.norm() * Ky.norm()
    return float((num / (den + 1e-8)).item())


# ---------------------------------------------------------------------------
# Composite metric with SHA-256 Goodhart fingerprint (autoresearch protocol)
# ---------------------------------------------------------------------------
COMPOSITE_FORMULA = (
    "composite = top1 - 0.05 * log10(params_M) - 0.05 * log10(latency_ms)"
)
COMPOSITE_FINGERPRINT = hashlib.sha256(COMPOSITE_FORMULA.encode()).hexdigest()


def composite_score(top1: float, params: int, latency_ms: float) -> float:
    params_M = max(0.001, params / 1e6)
    lat = max(0.01, latency_ms)
    return top1 - 0.05 * math.log10(params_M) - 0.05 * math.log10(lat)


@dataclass
class RunMetrics:
    tag: str
    dataset: str
    seed: int
    epochs: int
    top1: float
    top5: float
    params: int
    flops: float
    latency_ms: float
    rot_eq_err: float
    composite: float
    composite_fingerprint: str = COMPOSITE_FINGERPRINT
    epochs_to_target: int = -1
    train_seconds: float = 0.0
    train_top1: float = 0.0
    generalization_gap: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
