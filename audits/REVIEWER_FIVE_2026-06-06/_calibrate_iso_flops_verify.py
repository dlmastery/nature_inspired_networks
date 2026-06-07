"""Verify the chosen config FLOPs are invariant under (a) momentum_schedule
+ phi_decay_wd (optimizer-only knobs) and (b) sine_activation swap."""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from fvcore.nn import FlopCountAnalysis  # noqa: E402
from nature_inspired_networks.phi_scaling import PhiBudgetNet  # noqa: E402
from nature_inspired_networks.sinusoidal_activation import swap_relu_with_sine  # noqa: E402
from nature_inspired_networks.priors import PHI  # noqa: E402


def count_flops(model, input_size=(1, 3, 32, 32)):
    model.eval()
    x = torch.zeros(input_size)
    fca = FlopCountAnalysis(model, x).unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return float(fca.total())


def stage_param_costs(model):
    return [sum(p.numel() for p in s.parameters()) for s in model.stages]


def report(model, label):
    flops = count_flops(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sp = stage_param_costs(model)
    adj = [sp[k + 1] / sp[k] for k in range(len(sp) - 1)]
    err = [abs(r - PHI) / PHI * 100 for r in adj]
    print(f"  [{label}]")
    print(f"    widths       = {model.widths}")
    print(f"    stage_params = {sp}")
    print(f"    adj_ratios   = {[f'{r:.4f}' for r in adj]}")
    print(f"    phi-err pct  = {[f'{e:.2f}%' for e in err]} "
          f"(max {max(err):.2f}%)")
    print(f"    params       = {params:,}")
    print(f"    FLOPs        = {flops/1e6:.4f} M")
    return flops


def main():
    print("=" * 78)
    print("PROPOSED CONFIG #A: B=125_000, n=3, bps=2 -> widths [27,35,44]")
    print("=" * 78)
    m1 = PhiBudgetNet(num_classes=100, B_total=125_000, n_stages=3,
                      blocks_per_stage=2, budget_mode="phi")
    f1_relu = report(m1, "ReLU baseline (sg_only_phi_budget)")

    # Apply sine swap (slot_act_sine variant).
    m1s = PhiBudgetNet(num_classes=100, B_total=125_000, n_stages=3,
                       blocks_per_stage=2, budget_mode="phi")
    swap_relu_with_sine(m1s, omega_init=1.0)
    f1_sine = report(m1s, "After swap_relu_with_sine (slot_act_sine)")
    print(f"    sine-vs-ReLU FLOP delta = "
          f"{(f1_sine - f1_relu) / 1e6:+.4f} M "
          f"({(f1_sine / f1_relu - 1) * 100:+.3f} %)")
    print()

    print("=" * 78)
    print("PROPOSED CONFIG #B: B=110_000, n=3, bps=3 -> widths [21,27,34]")
    print("=" * 78)
    m2 = PhiBudgetNet(num_classes=100, B_total=110_000, n_stages=3,
                      blocks_per_stage=3, budget_mode="phi")
    report(m2, "ReLU baseline")
    print()

    # Print model architecture summary for the preferred config.
    print("=" * 78)
    print("Architecture print: B=125_000, n=3, bps=2")
    print("=" * 78)
    print(m1)


if __name__ == "__main__":
    main()
