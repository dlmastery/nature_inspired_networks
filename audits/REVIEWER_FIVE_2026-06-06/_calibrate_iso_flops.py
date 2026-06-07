"""A1 ISO-FLOPS calibration for SYNTHESIS_100 (R5 BLOCKER 1).

Pure CPU sweep of (phi_budget_total, n_stages) to find a configuration
whose measured FLOPs land in the band [37.1M, 45.3M] on 32x32 CIFAR-100
input (target 41,224,448 +/- 10%).

Reads `src/nature_inspired_networks/phi_scaling.py` only; never imports
the runner or touches the GPU. fvcore is a CPU-side analyser.
"""
from __future__ import annotations

import math
import sys
import os
from pathlib import Path

# Ensure CPU only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from fvcore.nn import FlopCountAnalysis  # noqa: E402

from nature_inspired_networks.phi_scaling import (  # noqa: E402
    PhiBudgetNet,
    phi_budget_widths,
)
from nature_inspired_networks.priors import PHI  # noqa: E402


def count_flops(model: torch.nn.Module, input_size=(1, 3, 32, 32)) -> float:
    model.eval()
    x = torch.zeros(input_size)
    fca = FlopCountAnalysis(model, x).unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return float(fca.total())


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stage_param_costs(model: PhiBudgetNet) -> list[int]:
    """Per-stage parameter counts (matches PhiBudgetNet stages)."""
    costs = []
    for s in model.stages:
        costs.append(sum(p.numel() for p in s.parameters()))
    return costs


def phi_ratios(stage_params: list[int]) -> list[float]:
    base = stage_params[0]
    return [c / base for c in stage_params]


def adj_ratios(stage_params: list[int]) -> list[float]:
    return [stage_params[k + 1] / stage_params[k]
            for k in range(len(stage_params) - 1)]


def evaluate(B: int, n_stages: int, blocks_per_stage: int = 2,
             num_classes: int = 100) -> dict:
    model = PhiBudgetNet(
        num_classes=num_classes,
        B_total=B,
        n_stages=n_stages,
        blocks_per_stage=blocks_per_stage,
        budget_mode="phi",
    )
    flops = count_flops(model)
    params = count_params(model)
    stage_p = stage_param_costs(model)
    return {
        "B_total": B,
        "n_stages": n_stages,
        "blocks": blocks_per_stage,
        "widths": list(model.widths),
        "stage_params": stage_p,
        "phi_ratios": phi_ratios(stage_p),
        "adj_ratios": adj_ratios(stage_p),
        "params": params,
        "flops": flops,
        "flops_M": flops / 1e6,
    }


def main() -> None:
    TARGET = 41_224_448
    LO, HI = TARGET * 0.9, TARGET * 1.1
    print(f"Target FLOPs band: [{LO/1e6:.2f}M, {HI/1e6:.2f}M] "
          f"(41.22M +/- 10%)")
    print()

    # Reference: the CURRENT broken config (270k, 3 stages, 2 blocks).
    print("=" * 78)
    print("CURRENT (broken) config:")
    res = evaluate(270_000, 3)
    print(f"  B={res['B_total']:>7}, n_stages={res['n_stages']}, "
          f"blocks/stage={res['blocks']}")
    print(f"  widths            = {res['widths']}")
    print(f"  stage_params      = {res['stage_params']}")
    print(f"  phi-ratios (1:x:y) = "
          f"{[f'{r:.3f}' for r in res['phi_ratios']]}")
    print(f"  adj ratios (stage k+1/k) = "
          f"{[f'{r:.3f}' for r in res['adj_ratios']]} "
          f"(target phi={PHI:.4f})")
    print(f"  params            = {res['params']:,}")
    print(f"  FLOPs             = {res['flops_M']:.2f} M")
    print(f"  In band [37.1, 45.3] MFLOPs? "
          f"{'YES' if LO <= res['flops'] <= HI else 'NO'}")
    print()

    # Sweep.
    print("=" * 78)
    print("SWEEP — searching for (B_total, n_stages, blocks_per_stage):")
    print()

    sweep_grid = []
    # Primary: vary B downwards at n_stages=3, blocks_per_stage=2.
    for B in [60_000, 70_000, 80_000, 90_000, 100_000, 110_000,
              120_000, 130_000, 140_000, 150_000, 160_000, 170_000,
              180_000, 200_000, 220_000, 250_000, 270_000]:
        sweep_grid.append((B, 3, 2))
    # Also test n_stages=4 (loses one phi-step? no, ADDS one)
    for B in [60_000, 80_000, 100_000, 120_000, 150_000, 200_000]:
        sweep_grid.append((B, 4, 2))
    # And n_stages=2 (only 1 phi-step ratio)
    for B in [100_000, 150_000, 200_000, 250_000, 300_000]:
        sweep_grid.append((B, 2, 2))
    # blocks_per_stage=3 sensitivity (less common but possible).
    for B in [80_000, 100_000, 120_000, 150_000]:
        sweep_grid.append((B, 3, 3))

    results = []
    for B, n, bps in sweep_grid:
        try:
            r = evaluate(B, n, blocks_per_stage=bps)
        except Exception as e:
            print(f"  [skip] B={B}, n={n}, bps={bps}: {e}")
            continue
        in_band = LO <= r["flops"] <= HI
        r["in_band"] = in_band
        results.append(r)
        marker = " <== IN BAND" if in_band else ""
        adj = ", ".join(f"{x:.3f}" for x in r["adj_ratios"])
        print(f"  B={B:>7}, n={n}, bps={bps}: "
              f"widths={r['widths']}, "
              f"params={r['params']:>7,}, "
              f"FLOPs={r['flops_M']:>6.2f}M, "
              f"adj={adj}{marker}")

    print()
    print("=" * 78)
    print("BEST CANDIDATES (in band, sorted by FLOPs closest to target):")
    in_band = [r for r in results if r["in_band"]]
    in_band.sort(key=lambda r: abs(r["flops"] - TARGET))
    for r in in_band[:10]:
        target_ratios = [PHI ** k for k in range(r["n_stages"])]
        ratio_err_pct = max(
            abs(a - b) / b * 100
            for a, b in zip(r["adj_ratios"], [PHI] * len(r["adj_ratios"]))
        ) if r["adj_ratios"] else 0.0
        within_1pct = ratio_err_pct <= 1.0
        print(f"  B={r['B_total']:>7}, n={r['n_stages']}, "
              f"bps={r['blocks']}, "
              f"widths={r['widths']}, "
              f"FLOPs={r['flops_M']:.2f}M, "
              f"max-adj-ratio-err={ratio_err_pct:.2f}% "
              f"(phi within 1%? {'YES' if within_1pct else 'NO'})")


if __name__ == "__main__":
    main()
