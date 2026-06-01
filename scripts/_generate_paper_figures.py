"""Generate the six headline publication-quality figures for the paper.

Outputs (paper/figures/):
  fig1_pareto.{pdf,png}              — 3-panel Pareto frontier (top1 vs params/FLOPs/latency)
  fig2_ablation_groups.{pdf,png}     — 4x2 small-multiples Δtop1 per G1..G8
  fig3_three_regimes.{pdf,png}       — Forest-plot, 3 winners x 3 regimes, 95% bootstrap CIs
  fig4_calibration_n62.{pdf,png}     — Stacked PASS/MINOR/MAJOR/BROKEN bar + Wilson 95% CI bar
  fig5_cross_family_concordance.{pdf,png}  — 10-row original vs re-audit verdict heatmap
  fig6_orthogonality_matrix.{pdf,png} — 20x20 axis-pair orthogonality heatmap

Data sources:
  experiments/cifar100/<tag>_seed<N>/metrics.json   — Pareto
  experiments/cifar10/<tag>_seed<N>/metrics.json    — Ablation (CIFAR-10 12-ep screen)
  paper/STATISTICAL_TESTS.md (Sections 1, 7, 10)     — 3-regime CIs
  audits/AUDIT_CALIBRATION_THIRD_PARTY.md Appendix A — n=62 calibration data
  audits/CROSS_FAMILY_HONEST_REAUDIT.md §3          — concordance verdicts
  audits/COMBINATIONS_RESEARCH/B_theoretical_orthogonality.md §3 — orthogonality matrix

Run:
  .venv/Scripts/python scripts/_generate_paper_figures.py

CLAUDE.md compliance:
  Rule 2 — composite fingerprint is read (not edited) per row.
  Rule 3 — append-only; no metrics.json edits.
  Rule 11 — caller is expected to commit+push after this script runs.
  Matplotlib only (no seaborn).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

# ICML-style serif typography; fall back to default serif if Source Serif 4 missing.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Source Serif 4",
        "Source Serif Pro",
        "Times New Roman",
        "DejaVu Serif",
    ],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.4,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Tag → hypothesis-group mapping (derived from hypotheses/IDEA_TABLE.md)
# ----------------------------------------------------------------------------

# Map experiment-tag stems → group label.
# Groups: G1..G8 + Baseline + Combo. The mapping covers the 35-row CIFAR-10
# screening sweep and the 4-tag CIFAR-100 family (pair_gm_pdw, slot_act_sine,
# sg_only_phi_budget, baseline_resnet20).
TAG_GROUP: Dict[str, str] = {
    # Baseline
    "baseline_resnet20": "Baseline",
    "baseline_sg_vanilla": "Baseline",
    # G1 Scaling & Growth
    "sg_only_phi_compound": "G1",
    "sg_only_fib_depth": "G1",
    "sg_only_golden_resize": "G1",
    "sg_only_fractal": "G1",
    "sg_only_golden_bottleneck": "G1",
    "sg_only_phi_multiscale": "G1",
    "sg_only_phi_budget": "G1",
    "sg_only_phi_lr": "G1",
    # G2 Layer / Channel / Neuron
    "sg_chan_fib": "G2",
    "sg_chan_phi": "G2",
    "sg_only_phi_sparse": "G2",
    "sg_only_golden_skip": "G2",
    "sg_only_fib_stride": "G2",
    "sg_only_phi_relu": "G2",
    "sg_only_phi_activation": "G2",
    "sg_only_fib_ensemble": "G2",
    # G3 Topologies & Graphs
    "sg_only_hex": "G3",
    "sg_only_toroidal": "G3",
    "sg_only_group": "G3",
    "sg_only_group_avg": "G3",
    # G4 Kernels / Attention / Filters
    "sg_only_golden_modulate": "G4",
    "sg_only_constant_width": "G4",
    # G5 Optimization / Init / Reg / NAS
    "sg_only_phi_decay": "G5",
    "sg_only_phi_dropout": "G5",
    "sg_only_phi_init": "G5",
    "sg_only_golden_adam": "G5",
    "sg_only_golden_momentum": "G5",
    "sg_only_golden_spiral_init": "G5",
    "sg_only_fib_prune": "G5",
    "slot_init_phi": "G5",
    "slot_init_cymatic": "G5",
    "slot_init_spiral": "G5",
    # G6 Topological Bridging
    "sg_only_cymatic_init": "G6",
    # G7 Cross-Paradigm Hybrids
    "sg_full_fib": "G7",
    "sg_full_fib_avg": "G7",
    # G8 Esoteric Extensions
    "sg_only_sine_act": "G8",
    "slot_act_sine": "G8",
    "slot_act_phi": "G8",
    # Combos
    "combo2_pb_gm": "Combo",
    "combo3_pb_gm_pd": "Combo",
    "combo4_pb_gm_pd_pdw": "Combo",
    "combo5_pb_gm_pd_pdw_plr": "Combo",
    "combo6_pb_gm_pd_pdw_plr_fe": "Combo",
    "combo7_pb_gm_pd_pdw_plr_fe_sa": "Combo",
    "combo8_pb_gm_pd_pdw_plr_fe_sa_fp": "Combo",
    "pair_gm_pdw": "Combo",
    "pair_gm_plr": "Combo",
    "pair_pd_pdw": "Combo",
    "pair_pd_plr": "Combo",
    "pair_pdw_plr": "Combo",
    "loo_no_fe": "Combo",
    "loo_no_fp": "Combo",
    "loo_no_gm": "Combo",
    "loo_no_pd": "Combo",
    "loo_no_pdw": "Combo",
    "loo_no_plr": "Combo",
    "loo_no_sa": "Combo",
}

# Consistent colour palette per group (colour-blind friendly).
GROUP_COLOURS: Dict[str, str] = {
    "Baseline": "#4a4a4a",
    "G1": "#1f77b4",
    "G2": "#ff7f0e",
    "G3": "#2ca02c",
    "G4": "#d62728",
    "G5": "#9467bd",
    "G6": "#8c564b",
    "G7": "#e377c2",
    "G8": "#17becf",
    "Combo": "#bcbd22",
}

PHASE8_WINNERS = ("pair_gm_pdw", "slot_act_sine", "sg_only_phi_budget")


def _tag_stem_from_dir(dirname: str) -> Tuple[str, Optional[int]]:
    """Return (tag, seed) parsed from a run-directory name."""
    m = re.match(r"^(.*?)_seed(\d+)$", dirname)
    if not m:
        return dirname, None
    return m.group(1), int(m.group(2))


def _group_for_tag(tag: str) -> str:
    # Strip hill-climb tuning suffix `__hc_lr...` if present.
    base = tag.split("__hc_")[0]
    return TAG_GROUP.get(base, "Other")


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------

def load_metrics(dataset: str) -> List[dict]:
    """Read every metrics.json under experiments/<dataset>/*_seed*/."""
    rows: List[dict] = []
    root = REPO_ROOT / "experiments" / dataset
    if not root.exists():
        return rows
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        mfile = run_dir / "metrics.json"
        if not mfile.exists():
            continue
        try:
            with mfile.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        tag, seed = _tag_stem_from_dir(run_dir.name)
        m.setdefault("tag", tag)
        m.setdefault("seed", seed if seed is not None else 0)
        m["_run_dir"] = run_dir.name
        # Strip __hc_ suffix to canonical tag for grouping.
        m["_base_tag"] = m["tag"].split("__hc_")[0]
        m["_group"] = _group_for_tag(m["tag"])
        rows.append(m)
    return rows


# ----------------------------------------------------------------------------
# Fig 1 — Pareto frontier (top1 vs params, FLOPs, latency)
# ----------------------------------------------------------------------------

def fig1_pareto() -> None:
    rows = load_metrics("cifar100")
    if not rows:
        raise RuntimeError("No CIFAR-100 metrics found")

    # Keep only rows with the canonical 30 epochs and base-tags
    # (drop hill-climb sub-cells where epochs differ).
    rows = [r for r in rows if int(r.get("epochs", 0)) >= 30]

    # Aggregate to median per (base_tag), keeping default-config seeds only
    # (i.e., directory name has no "__hc_" suffix).
    canonical = [r for r in rows if "__hc_" not in r["tag"]]
    by_tag: Dict[str, List[dict]] = {}
    for r in canonical:
        by_tag.setdefault(r["_base_tag"], []).append(r)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), constrained_layout=True)
    metric_panels = [
        ("params", "Parameters", lambda v: v),
        ("flops", "FLOPs", lambda v: v),
        ("latency_ms", "Latency (ms)", lambda v: v),
    ]

    # Group-collect aggregated points.
    agg: List[dict] = []
    for tag, rs in by_tag.items():
        top1s = [r["top1"] for r in rs]
        agg.append({
            "tag": tag,
            "group": _group_for_tag(tag),
            "top1_med": float(np.median(top1s)),
            "params": float(rs[0].get("params", np.nan)),
            "flops": float(rs[0].get("flops", np.nan)),
            "latency_ms": float(rs[0].get("latency_ms", np.nan)),
            "n_seeds": len(rs),
        })

    # Group order for legend
    group_order = ["Baseline", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "Combo", "Other"]
    seen_groups: List[str] = []

    for ax, (key, label, _) in zip(axes, metric_panels):
        ax.set_xscale("log")
        for grp in group_order:
            pts = [p for p in agg if p["group"] == grp]
            if not pts:
                continue
            xs = [p[key] for p in pts]
            ys = [100.0 * p["top1_med"] for p in pts]
            colour = GROUP_COLOURS.get(grp, "#777777")
            ax.scatter(
                xs, ys,
                s=22, c=colour, edgecolors="black", linewidths=0.4,
                alpha=0.85, zorder=2, label=grp if grp not in seen_groups else None,
            )
            if grp not in seen_groups:
                seen_groups.append(grp)
            # Star markers on Phase-8 winners
            for p in pts:
                if p["tag"] in PHASE8_WINNERS:
                    ax.scatter(
                        [p[key]], [100.0 * p["top1_med"]],
                        s=130, marker="*", c="gold",
                        edgecolors="black", linewidths=0.6, zorder=4,
                    )

        ax.set_xlabel(label)
        if ax is axes[0]:
            ax.set_ylabel("CIFAR-100 30-ep top-1 (%)")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
        ax.set_title(f"Top-1 vs {label.lower()}")

    # Combined legend below
    handles, labels = axes[0].get_legend_handles_labels()
    # Add a star handle for winners
    handles.append(plt.scatter([], [], marker="*", c="gold",
                               edgecolors="black", s=130, linewidths=0.6))
    labels.append("Phase-8 winner")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Figure 1 — Pareto frontier on CIFAR-100 (30 epochs, default-config seeds)",
                 y=1.02, fontsize=10)
    out_pdf = FIG_DIR / "fig1_pareto.pdf"
    out_png = FIG_DIR / "fig1_pareto.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 2 — Ablation matrix, 4x2 small-multiples of Δtop1 vs baseline per group
# ----------------------------------------------------------------------------

def fig2_ablation_groups() -> None:
    """CIFAR-10 12-epoch screening Δtop1 vs baseline_resnet20 seed0, grouped by G1..G8."""
    rows = load_metrics("cifar10")
    if not rows:
        raise RuntimeError("No CIFAR-10 metrics found")

    # Use seed-0 baseline as anchor (from Section 5 of STATISTICAL_TESTS.md).
    baseline_seed0 = None
    for r in rows:
        if r["_base_tag"] == "baseline_resnet20" and r.get("seed") == 0 and "__hc_" not in r["tag"]:
            baseline_seed0 = r["top1"]
            break
    if baseline_seed0 is None:
        raise RuntimeError("baseline_resnet20 seed0 not found in CIFAR-10")

    # Collect Δ per tag for seed 0
    delta_rows: List[dict] = []
    for r in rows:
        if r.get("seed") != 0 or "__hc_" in r["tag"]:
            continue
        if r["_base_tag"] == "baseline_resnet20":
            continue
        grp = _group_for_tag(r["_base_tag"])
        if grp not in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
            continue
        delta_rows.append({
            "tag": r["_base_tag"],
            "group": grp,
            "delta_pp": 100.0 * (r["top1"] - baseline_seed0),
        })

    # 2σ_pooled = 1.21 pp per STATISTICAL_TESTS.md §4
    NOISE_BAND = 1.21
    groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    group_titles = {
        "G1": "G1 — Scaling & growth",
        "G2": "G2 — Layer/channel/neuron",
        "G3": "G3 — Topologies & graphs",
        "G4": "G4 — Kernels/attention",
        "G5": "G5 — Opt/init/reg/NAS",
        "G6": "G6 — Topological bridging",
        "G7": "G7 — Cross-paradigm",
        "G8": "G8 — Esoteric extensions",
    }

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.0), constrained_layout=True)
    for idx, grp in enumerate(groups):
        ax = axes[idx // 2, idx % 2]
        gpts = [d for d in delta_rows if d["group"] == grp]
        gpts.sort(key=lambda d: d["delta_pp"])
        tags = [d["tag"] for d in gpts]
        deltas = [d["delta_pp"] for d in gpts]
        if not deltas:
            ax.set_title(group_titles[grp] + " (no data)")
            ax.axis("off")
            continue
        colours = []
        for v in deltas:
            if v > NOISE_BAND:
                colours.append("#2ca02c")
            elif v < -NOISE_BAND:
                colours.append("#d62728")
            else:
                colours.append("#bbbbbb")
        ypos = np.arange(len(deltas))
        ax.barh(ypos, deltas, color=colours,
                edgecolor="black", linewidth=0.3, height=0.7)
        ax.axvspan(-NOISE_BAND, NOISE_BAND, color="#dddddd", alpha=0.45, zorder=0)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(tags, fontsize=7)
        ax.set_title(group_titles[grp])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Δ top-1 vs baseline (pp)")
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)

    # Legend
    legend_handles = [
        Patch(color="#2ca02c", label="Δ > +2σ (1.21 pp)"),
        Patch(color="#bbbbbb", label="within 2σ noise band"),
        Patch(color="#d62728", label="Δ < −2σ (−1.21 pp)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 2 — CIFAR-10 12-ep ablation per group (Δtop-1 vs baseline_resnet20 seed-0)",
                 fontsize=10, y=1.02)

    fig.savefig(FIG_DIR / "fig2_ablation_groups.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_ablation_groups.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 3 — Three regimes (n=7 default, hill-climbed n=3, iso-tuned n=3)
# ----------------------------------------------------------------------------

def fig3_three_regimes() -> None:
    """Forest plot: 3 winners x 3 regimes, with 95% bootstrap CI on Δmean."""
    # Values lifted directly from paper/STATISTICAL_TESTS.md Sections 1, 7, 10.
    # Each row: (winner, regime, dmean_pp, ci_lo_pp, ci_hi_pp, n)
    data = [
        # Section 1 — default-config n=7
        ("pair_gm_pdw",        "default n=7",     +1.74, +1.42, +2.09, 7),
        ("slot_act_sine",      "default n=7",     +1.78, +1.38, +2.18, 7),
        ("sg_only_phi_budget", "default n=7",     +1.24, +0.84, +1.67, 7),
        # Section 7 — hill-climbed n=3
        ("pair_gm_pdw",        "hill-climbed n=3", +1.22, +0.15, +1.99, 3),
        ("slot_act_sine",      "hill-climbed n=3", +1.31, +0.20, +2.23, 3),
        ("sg_only_phi_budget", "hill-climbed n=3", +0.79, -0.32, +1.76, 3),
        # Section 10 — iso-tuned bs=128 lr=3e-3 n=3
        ("pair_gm_pdw",        "iso-tuned n=3",   +1.59, +0.43, +2.62, 3),
        ("slot_act_sine",      "iso-tuned n=3",   +1.68, +0.49, +2.77, 3),
        ("sg_only_phi_budget", "iso-tuned n=3",   +1.16, -0.04, +2.30, 3),
    ]
    winners = ["pair_gm_pdw", "slot_act_sine", "sg_only_phi_budget"]
    regimes = ["default n=7", "hill-climbed n=3", "iso-tuned n=3"]
    regime_colour = {
        "default n=7":      "#1f77b4",
        "hill-climbed n=3": "#ff7f0e",
        "iso-tuned n=3":    "#2ca02c",
    }
    regime_marker = {
        "default n=7":      "o",
        "hill-climbed n=3": "s",
        "iso-tuned n=3":    "D",
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    y_base = np.arange(len(winners))[::-1] * 1.6  # space rows
    offset = {regimes[0]: +0.45, regimes[1]: 0.0, regimes[2]: -0.45}

    for w_idx, w in enumerate(winners):
        for rg in regimes:
            for row in data:
                if row[0] != w or row[1] != rg:
                    continue
                _, _, dmean, lo, hi, n = row
                y = y_base[w_idx] + offset[rg]
                err_lo = dmean - lo
                err_hi = hi - dmean
                ax.errorbar(
                    [dmean], [y],
                    xerr=[[err_lo], [err_hi]],
                    fmt=regime_marker[rg], color=regime_colour[rg],
                    ecolor=regime_colour[rg],
                    elinewidth=1.3, capsize=3.5, capthick=1.0,
                    markerfacecolor=regime_colour[rg],
                    markeredgecolor="black", markeredgewidth=0.5,
                    markersize=7,
                    label=f"{rg}" if w_idx == 0 else None,
                )
                # Annotate Δmean to the right of the highest CI
                ax.text(hi + 0.08, y, f"+{dmean:.2f} pp (n={n})",
                        fontsize=6.5, va="center")

    # H0 line
    ax.axvline(0.0, color="black", linewidth=0.7)
    # Holm-Bonferroni α'=0.0167 threshold marker — represented as note since
    # the p-value clearance is shown via CIs not crossing 0. We mark with text.
    ax.text(0.02, 1.02, "Holm-Bonferroni α′=0.0167 cleared (default n=7): pair_gm_pdw, slot_act_sine, sg_only_phi_budget",
            transform=ax.transAxes, fontsize=7, style="italic")

    ax.set_yticks(y_base)
    ax.set_yticklabels(winners, fontsize=8)
    ax.set_xlabel("Δmean top-1 vs baseline (pp, 95% bootstrap CI)")
    ax.set_xlim(-1.2, 3.3)
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Figure 3 — Three robustness regimes for Phase-8 winners (CIFAR-100, 30 epochs)",
                 fontsize=10)

    fig.savefig(FIG_DIR / "fig3_three_regimes.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_three_regimes.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 4 — Audit calibration at n=62
# ----------------------------------------------------------------------------

def fig4_calibration_n62() -> None:
    """Two panels: (a) per-repo stacked PASS/MINOR/MAJOR/BROKEN; (b) Wilson CIs."""
    # From AUDIT_CALIBRATION_THIRD_PARTY.md Appendix A.4 (n=62 extension).
    # Per-repo distribution. The §2 n=15 calibration is folded into pytorch/vision.
    # The n=15 baseline (10 PASS, 5 MINOR, 0 MAJOR, 0 BROKEN) is the pytorch/vision
    # core block. Phase-9b adds 5 repos (timm 19, HF 15, Bolts/fastai 6, torch.optim 4,
    # mamba 3) — but the prompt enumerates 6 named repos: pytorch/vision, timm, HF,
    # Lightning Bolts (combined with fastai for the row), torch.optim, Mamba.
    repos = [
        ("pytorch/vision", 10, 5, 0, 0),
        ("timm",           15, 4, 0, 0),
        ("HF Transformers",14, 1, 0, 0),
        ("Lightning Bolts /\nfastai", 3, 3, 0, 0),
        ("torch.optim",     3, 1, 0, 0),
        ("Mamba (SSM)",     3, 0, 0, 0),
    ]
    cats = ["PASS", "MINOR", "MAJOR", "BROKEN"]
    cat_colours = {"PASS": "#2ca02c", "MINOR": "#ffbf00",
                   "MAJOR": "#d62728", "BROKEN": "#7b1f1f"}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 4.2),
                                     gridspec_kw={"width_ratios": [1.4, 1.0]},
                                     constrained_layout=True)

    # Panel (a): stacked bars per repo
    names = [r[0] for r in repos]
    pass_  = np.array([r[1] for r in repos], dtype=float)
    minor_ = np.array([r[2] for r in repos], dtype=float)
    major_ = np.array([r[3] for r in repos], dtype=float)
    broken_= np.array([r[4] for r in repos], dtype=float)
    x = np.arange(len(repos))
    ax_a.bar(x, pass_, color=cat_colours["PASS"], edgecolor="black",
             linewidth=0.4, label="PASS")
    ax_a.bar(x, minor_, bottom=pass_, color=cat_colours["MINOR"],
             edgecolor="black", linewidth=0.4, label="MINOR")
    ax_a.bar(x, major_, bottom=pass_+minor_, color=cat_colours["MAJOR"],
             edgecolor="black", linewidth=0.4, label="MAJOR")
    ax_a.bar(x, broken_, bottom=pass_+minor_+major_, color=cat_colours["BROKEN"],
             edgecolor="black", linewidth=0.4, label="BROKEN")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax_a.set_ylabel("Audit findings (count)")
    ax_a.set_title("(a) Per-repo verdict distribution (n = 62)")
    ax_a.legend(loc="upper right", frameon=False, fontsize=7)
    # Annotate counts at top of each bar
    totals = pass_ + minor_ + major_ + broken_
    for i, t in enumerate(totals):
        ax_a.text(i, t + 0.4, f"n={int(t)}", ha="center", fontsize=6.5)

    # Panel (b): Wilson 95% CI on MAJOR/BROKEN rate
    # Project: 18/83 (21.7%) — Wilson [14.2%, 31.7%]
    # Calibration: 0/62 (0.0%) — Wilson [0.0%, 5.8%]
    rates = [21.7, 0.0]
    lo    = [14.2, 0.0]
    hi    = [31.7, 5.8]
    labels = [f"Project\n(18/83)", f"Calibration\n(0/62)"]
    xc = np.arange(2)
    errs_lo = [rates[i] - lo[i] for i in range(2)]
    errs_hi = [hi[i] - rates[i] for i in range(2)]
    bar_colors = ["#d62728", "#2ca02c"]
    ax_b.bar(xc, rates, color=bar_colors, edgecolor="black", linewidth=0.5,
             yerr=[errs_lo, errs_hi], capsize=6, width=0.55,
             error_kw={"linewidth": 1.2})
    ax_b.set_xticks(xc)
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.set_ylabel("MAJOR/BROKEN rate (%)")
    ax_b.set_ylim(0, 36)
    ax_b.set_title("(b) Wilson 95% CIs on MAJOR/BROKEN rate")
    # Mark separation
    ax_b.axhline(lo[0], color="black", linewidth=0.4, linestyle=":")
    ax_b.axhline(hi[1], color="black", linewidth=0.4, linestyle=":")
    ax_b.annotate(
        "CIs no longer overlap\n(8.3-pp separation)",
        xy=(0.5, (lo[0] + hi[1]) / 2),
        xytext=(0.5, 18),
        ha="center", fontsize=7,
        arrowprops=dict(arrowstyle="->", linewidth=0.6),
    )
    ax_b.text(0.98, 0.97,
              "Fisher exact (two-sided)\np = 1.94 × 10⁻⁵",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=7,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black",
                        linewidth=0.4))
    ax_b.grid(True, axis="y", linewidth=0.3, alpha=0.4)

    fig.suptitle("Figure 4 — Audit calibration at n = 62 (third-party trusted repos)",
                 fontsize=10)

    fig.savefig(FIG_DIR / "fig4_calibration_n62.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_calibration_n62.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5 — Cross-Family Re-Audit Concordance Heatmap (10 hypotheses)
# ----------------------------------------------------------------------------

def fig5_cross_family_concordance() -> None:
    # Data from audits/CROSS_FAMILY_HONEST_REAUDIT.md §3
    # (id, original-tier, reaudit-tier, concordant)
    rows = [
        ("H06", "MAJOR",  "MAJOR-RES", True),
        ("H14", "MAJOR",  "MAJOR-RES", True),
        ("H21", "MAJOR",  "MAJOR-RES", True),
        ("H22", "MAJOR",  "MAJOR-RES", True),
        ("H31", "MAJOR",  "MAJOR-RES", True),
        ("H47", "MAJOR",  "MINOR",     False),  # discordant downgrade
        ("H48", "MAJOR",  "MAJOR-RES", True),
        ("H55", "BROKEN", "BROKEN-RES",True),
        ("H67", "BROKEN", "MAJOR",     False),  # discordant residual
        ("H74", "BROKEN", "BROKEN-RES",True),
    ]
    # Build a 10x2 grid where each cell is coloured by tier-class.
    # Tier colour scheme — RES-suffixed tiers mean "resolved" (post-fix verified).
    tier_to_val: Dict[str, int] = {
        "PASS": 0, "MINOR": 1, "MAJOR": 2, "BROKEN": 3,
        "MAJOR-RES": 2, "BROKEN-RES": 3,
    }
    tier_colours = {
        0: "#2ca02c",  # PASS
        1: "#ffbf00",  # MINOR
        2: "#d62728",  # MAJOR
        3: "#7b1f1f",  # BROKEN
    }

    # Compact tier labels (text drawn inside cells) — strip the "-RES" suffix
    # in the text because the hatch already encodes "resolved post-fix".
    def _short(tier: str) -> str:
        return tier.replace("-RES", "")

    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    cell_w, cell_h = 2.0, 0.85
    for i, (hid, orig, reau, conc) in enumerate(rows):
        for j, tier in enumerate([orig, reau]):
            v = tier_to_val[tier]
            colour = tier_colours[v]
            hatch = "////" if tier.endswith("-RES") else None
            ax.add_patch(plt.Rectangle(
                (j * cell_w, (n - 1 - i) * cell_h),
                cell_w, cell_h,
                facecolor=colour, edgecolor="black", linewidth=0.7,
                hatch=hatch,
            ))
            txt_colour = "white" if v >= 2 else "black"
            ax.text(j * cell_w + cell_w / 2,
                    (n - 1 - i) * cell_h + cell_h / 2,
                    _short(tier), ha="center", va="center",
                    fontsize=8, color=txt_colour, weight="bold")
        # Mark discordant rows with red ring around their cells
        if not conc:
            ax.add_patch(plt.Rectangle(
                (0, (n - 1 - i) * cell_h),
                2 * cell_w, cell_h,
                facecolor="none", edgecolor="red", linewidth=2.0,
                linestyle="--",
            ))
            ax.text(2 * cell_w + 0.15, (n - 1 - i) * cell_h + cell_h / 2,
                    "DISCORDANT", ha="left", va="center",
                    color="red", fontsize=8, weight="bold")

    ax.set_xlim(-0.1, 2 * cell_w + 2.2)
    ax.set_ylim(-0.05, n * cell_h + 0.05)
    ax.set_xticks([cell_w / 2, 1.5 * cell_w])
    ax.set_xticklabels(["Original\n(Track-A Claude)", "Re-audit\n(method-diverse)"],
                       fontsize=8)
    ax.set_yticks([(n - 1 - i) * cell_h + cell_h / 2 for i in range(n)])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Legend
    legend_handles = [
        Patch(facecolor=tier_colours[2], edgecolor="black", label="MAJOR"),
        Patch(facecolor=tier_colours[3], edgecolor="black", label="BROKEN"),
        Patch(facecolor=tier_colours[1], edgecolor="black", label="MINOR"),
        Patch(facecolor="white", edgecolor="black", hatch="////",
              label="post-fix RESOLVED"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, -0.05), frameon=False, fontsize=7,
              ncol=4)
    ax.set_title("Figure 5 — Cross-family re-audit concordance\n"
                 "(8/10 CONCORDANT, 2/10 DISCORDANT)", fontsize=10)

    fig.savefig(FIG_DIR / "fig5_cross_family_concordance.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_cross_family_concordance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 6 — Axis-pair orthogonality matrix (8 hypothesis-group axes block-summary)
# ----------------------------------------------------------------------------

def fig6_orthogonality_matrix() -> None:
    """The 8 hypothesis groups (G1..G8) × 8 axis-pair orthogonality matrix.

    The 20-axis matrix in B_theoretical_orthogonality.md §3 is collapsed to
    the 8 hypothesis-group axes (one row per group). Each cell = fraction of
    cross-group axis pairs marked ORTHOGONAL (O) per §3's pair classification.

    Mapping (groups → primary axis blocks from B):
        G1 (scaling/growth)       → A1, A7
        G2 (layer/channel/neuron) → A2, A7, A11
        G3 (topologies/graphs)    → A3, A18
        G4 (kernels/attn/filters) → A3, A4, A5, A6
        G5 (opt/init/reg/NAS)     → A9, A12, A13, A14, A15
        G6 (topological bridging) → A16, A19
        G7 (cross-paradigm)       → A20
        G8 (esoteric extensions)  → A8, A10
    """
    # The B_theoretical_orthogonality.md §3 matrix, encoded.
    # cell value: 'O' (orthogonal), 'F' (forward-path conflict),
    #             'S' (same-axis), 'U' (uncertain)
    AXES = list(range(1, 21))  # 20 axes
    M = {
        # A1 row (upper triangle as listed in doc); fill symmetric below.
        (1,1):'S',(1,2):'F',(1,3):'F',(1,4):'O',(1,5):'O',(1,6):'O',
        (1,7):'F',(1,8):'F',(1,9):'O',(1,10):'F',(1,11):'F',(1,12):'O',
        (1,13):'O',(1,14):'O',(1,15):'O',(1,16):'O',(1,17):'O',(1,18):'F',
        (1,19):'O',(1,20):'F',
        (2,2):'S',(2,3):'F',(2,4):'O',(2,5):'O',(2,6):'O',(2,7):'F',
        (2,8):'F',(2,9):'O',(2,10):'F',(2,11):'F',(2,12):'O',(2,13):'O',
        (2,14):'O',(2,15):'O',(2,16):'O',(2,17):'O',(2,18):'F',(2,19):'O',
        (2,20):'F',
        (3,3):'S',(3,4):'O',(3,5):'O',(3,6):'O',(3,7):'F',(3,8):'F',
        (3,9):'O',(3,10):'F',(3,11):'F',(3,12):'O',(3,13):'O',(3,14):'O',
        (3,15):'O',(3,16):'O',(3,17):'O',(3,18):'F',(3,19):'O',(3,20):'F',
        (4,4):'S',(4,5):'U',(4,6):'U',(4,7):'O',(4,8):'O',(4,9):'O',
        (4,10):'O',(4,11):'O',(4,12):'O',(4,13):'O',(4,14):'O',(4,15):'O',
        (4,16):'O',(4,17):'O',(4,18):'O',(4,19):'O',(4,20):'F',
        (5,5):'S',(5,6):'U',(5,7):'O',(5,8):'O',(5,9):'O',(5,10):'O',
        (5,11):'O',(5,12):'O',(5,13):'O',(5,14):'O',(5,15):'O',(5,16):'O',
        (5,17):'O',(5,18):'O',(5,19):'O',(5,20):'F',
        (6,6):'S',(6,7):'O',(6,8):'O',(6,9):'O',(6,10):'O',(6,11):'O',
        (6,12):'O',(6,13):'O',(6,14):'O',(6,15):'O',(6,16):'O',(6,17):'O',
        (6,18):'O',(6,19):'O',(6,20):'F',
        (7,7):'S',(7,8):'F',(7,9):'O',(7,10):'F',(7,11):'F',(7,12):'O',
        (7,13):'O',(7,14):'O',(7,15):'O',(7,16):'O',(7,17):'O',(7,18):'F',
        (7,19):'O',(7,20):'F',
        (8,8):'S',(8,9):'O',(8,10):'F',(8,11):'F',(8,12):'O',(8,13):'O',
        (8,14):'O',(8,15):'O',(8,16):'O',(8,17):'O',(8,18):'F',(8,19):'O',
        (8,20):'F',
        (9,9):'S',(9,10):'O',(9,11):'O',(9,12):'O',(9,13):'O',(9,14):'O',
        (9,15):'O',(9,16):'O',(9,17):'O',(9,18):'O',(9,19):'O',(9,20):'O',
        (10,10):'S',(10,11):'F',(10,12):'O',(10,13):'O',(10,14):'O',
        (10,15):'O',(10,16):'O',(10,17):'O',(10,18):'F',(10,19):'O',
        (10,20):'F',
        (11,11):'S',(11,12):'O',(11,13):'O',(11,14):'O',(11,15):'O',
        (11,16):'O',(11,17):'O',(11,18):'F',(11,19):'U',(11,20):'F',
        (12,12):'S',(12,13):'U',(12,14):'O',(12,15):'O',(12,16):'O',
        (12,17):'O',(12,18):'O',(12,19):'O',(12,20):'O',
        (13,13):'S',(13,14):'U',(13,15):'U',(13,16):'O',(13,17):'O',
        (13,18):'O',(13,19):'O',(13,20):'O',
        (14,14):'S',(14,15):'U',(14,16):'O',(14,17):'O',(14,18):'O',
        (14,19):'O',(14,20):'O',
        (15,15):'S',(15,16):'O',(15,17):'O',(15,18):'O',(15,19):'O',
        (15,20):'O',
        (16,16):'S',(16,17):'O',(16,18):'O',(16,19):'O',(16,20):'U',
        (17,17):'S',(17,18):'O',(17,19):'U',(17,20):'O',
        (18,18):'S',(18,19):'O',(18,20):'F',
        (19,19):'S',(19,20):'O',
        (20,20):'S',
    }
    # Symmetrize
    full = {}
    for (i, j), v in M.items():
        full[(i, j)] = v
        full[(j, i)] = v

    GROUP_TO_AXES = {
        "G1": [1, 7],
        "G2": [2, 7, 11],
        "G3": [3, 18],
        "G4": [3, 4, 5, 6],
        "G5": [9, 12, 13, 14, 15],
        "G6": [16, 19],
        "G7": [20],
        "G8": [8, 10],
    }
    groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]

    # For each (group_a, group_b) compute % of axis-pairs (drawn from
    # their respective axis-sets) marked O.
    pct = np.zeros((len(groups), len(groups)))
    for a, ga in enumerate(groups):
        for b, gb in enumerate(groups):
            pairs = []
            for i in GROUP_TO_AXES[ga]:
                for j in GROUP_TO_AXES[gb]:
                    if i == j:
                        # same axis = same-axis-conflict (not orthogonal)
                        pairs.append('S')
                    else:
                        pairs.append(full.get((i, j), 'O'))
            if not pairs:
                pct[a, b] = np.nan
            else:
                pct[a, b] = 100.0 * sum(1 for p in pairs if p == 'O') / len(pairs)

    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    for i in range(len(groups)):
        for j in range(len(groups)):
            v = pct[i, j]
            txt_colour = "black" if 30 <= v <= 75 else "white"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=7.5, color=txt_colour)
    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel("Group B")
    ax.set_ylabel("Group A")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("% ORTHOGONAL axis pairs", fontsize=8)
    ax.set_title("Figure 6 — Hypothesis-group orthogonality matrix\n"
                 "(8 groups; cells = % of cross-group axis pairs marked ORTHOGONAL\n"
                 "in B_theoretical_orthogonality.md §3 20-axis matrix)",
                 fontsize=9.5)

    fig.savefig(FIG_DIR / "fig6_orthogonality_matrix.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_orthogonality_matrix.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

def main() -> int:
    print(f"Writing figures into {FIG_DIR}")
    fig1_pareto()
    print("  fig1_pareto.{pdf,png}                done")
    fig2_ablation_groups()
    print("  fig2_ablation_groups.{pdf,png}       done")
    fig3_three_regimes()
    print("  fig3_three_regimes.{pdf,png}         done")
    fig4_calibration_n62()
    print("  fig4_calibration_n62.{pdf,png}       done")
    fig5_cross_family_concordance()
    print("  fig5_cross_family_concordance.{pdf,png}  done")
    fig6_orthogonality_matrix()
    print("  fig6_orthogonality_matrix.{pdf,png}  done")
    # Final sanity sweep — every file present.
    expected = [
        "fig1_pareto.pdf", "fig1_pareto.png",
        "fig2_ablation_groups.pdf", "fig2_ablation_groups.png",
        "fig3_three_regimes.pdf", "fig3_three_regimes.png",
        "fig4_calibration_n62.pdf", "fig4_calibration_n62.png",
        "fig5_cross_family_concordance.pdf", "fig5_cross_family_concordance.png",
        "fig6_orthogonality_matrix.pdf", "fig6_orthogonality_matrix.png",
    ]
    missing = [f for f in expected if not (FIG_DIR / f).exists()]
    if missing:
        print(f"MISSING: {missing}")
        return 1
    sizes = {f: (FIG_DIR / f).stat().st_size for f in expected}
    print("\nByte sizes:")
    for f, s in sizes.items():
        print(f"  {f:50s} {s:>10,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
