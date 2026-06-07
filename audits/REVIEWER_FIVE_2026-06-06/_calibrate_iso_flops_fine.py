"""Fine-grained A1 sweep: probe edges + look at force-width to escape
the integer-quantization plateau."""
from __future__ import annotations

import os
import sys
from pathlib import Path

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


def stage_param_costs(model: PhiBudgetNet) -> list[int]:
    return [sum(p.numel() for p in s.parameters()) for s in model.stages]


def evaluate(B, n_stages, blocks_per_stage=2, num_classes=100):
    model = PhiBudgetNet(
        num_classes=num_classes,
        B_total=B,
        n_stages=n_stages,
        blocks_per_stage=blocks_per_stage,
        budget_mode="phi",
    )
    flops = count_flops(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stage_p = stage_param_costs(model)
    adj = [stage_p[k + 1] / stage_p[k] for k in range(len(stage_p) - 1)]
    return {
        "B": B, "n": n_stages, "bps": blocks_per_stage,
        "widths": list(model.widths),
        "params": params,
        "stage_params": stage_p,
        "adj_ratios": adj,
        "flops": flops,
        "flops_M": flops / 1e6,
    }


def main() -> None:
    TARGET = 41_224_448
    LO, HI = TARGET * 0.9, TARGET * 1.1
    print(f"Target band: [{LO/1e6:.2f}M, {HI/1e6:.2f}M]")
    print()

    # The bps=2 grid jumps from 31.69M @ B=120k to 46.35M @ B=130k because
    # the integer-width search relocates from widths=[23,30,38] to
    # [28,36,46]. Both miss the 41M target. Let's probe many B's in [120k, 135k]
    # at bps=2 to make sure no integer config lands.
    print("FINE PROBE 1: bps=2, n=3, B in [115k, 135k]:")
    for B in range(115_000, 136_000, 1000):
        r = evaluate(B, 3, 2)
        in_band = LO <= r["flops"] <= HI
        marker = " <== IN BAND" if in_band else ""
        adj = ", ".join(f"{x:.3f}" for x in r["adj_ratios"])
        print(f"  B={B:>7}, widths={str(r['widths']):<18}, "
              f"params={r['params']:>6,}, "
              f"FLOPs={r['flops_M']:>6.2f}M, adj={adj}{marker}")
    print()

    # Compare bps=3 fine sweep near 120k.
    print("FINE PROBE 2: bps=3, n=3, B in [105k, 145k]:")
    for B in range(105_000, 146_000, 5_000):
        r = evaluate(B, 3, 3)
        in_band = LO <= r["flops"] <= HI
        marker = " <== IN BAND" if in_band else ""
        adj = ", ".join(f"{x:.3f}" for x in r["adj_ratios"])
        print(f"  B={B:>7}, widths={str(r['widths']):<18}, "
              f"params={r['params']:>6,}, "
              f"FLOPs={r['flops_M']:>6.2f}M, adj={adj}{marker}")
    print()

    # bps=4 (high depth, tightens the (B->widths) inversion).
    print("FINE PROBE 3: bps=4, n=3, B in [80k, 130k]:")
    for B in range(80_000, 131_000, 5_000):
        r = evaluate(B, 3, 4)
        in_band = LO <= r["flops"] <= HI
        marker = " <== IN BAND" if in_band else ""
        adj = ", ".join(f"{x:.3f}" for x in r["adj_ratios"])
        print(f"  B={B:>7}, widths={str(r['widths']):<18}, "
              f"params={r['params']:>6,}, "
              f"FLOPs={r['flops_M']:>6.2f}M, adj={adj}{marker}")
    print()


if __name__ == "__main__":
    main()
