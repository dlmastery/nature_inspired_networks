"""Autoresearch-style static dashboard generator.

Reads experiment_log.jsonl and per-experiment metrics.json files; emits
PNG plots + a single self-contained sortable-table HTML dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(results_dir: str | Path) -> pd.DataFrame:
    p = Path(results_dir)
    rows: list[dict] = []
    for mj in p.glob("**/metrics.json"):
        try:
            row = json.loads(mj.read_text())
            row["_run_dir"] = str(mj.parent.relative_to(p))
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_pareto(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, x_field, x_label, x_unit in [
        (axes[0], "params", "Params", "M"),
        (axes[1], "flops", "FLOPs (MACs)", "G"),
        (axes[2], "latency_ms", "Latency", "ms"),
    ]:
        x = df[x_field].astype(float)
        if x_unit == "M":
            x = x / 1e6
        elif x_unit == "G":
            x = x / 1e9
        y = df["top1"].astype(float) * 100
        is_base = df["tag"].str.startswith("baseline_")
        ax.scatter(x[~is_base], y[~is_base], c="C1", s=80, label="SacredGeo", edgecolor="k")
        ax.scatter(x[is_base], y[is_base], c="C0", s=120, marker="*",
                   label="Baseline", edgecolor="k")
        for _, r in df.iterrows():
            ax.annotate(
                r["tag"][:18],
                (x.loc[r.name], y.loc[r.name]),
                fontsize=7, alpha=0.7,
                xytext=(4, 4), textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_xlabel(f"{x_label} ({x_unit})")
        ax.set_ylabel("Top-1 acc (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Pareto fronts — accuracy vs. compute", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    grp = df.groupby("tag").agg(
        top1_mean=("top1", "mean"),
        top1_std=("top1", "std"),
        params=("params", "mean"),
        composite_mean=("composite", "mean"),
    ).reset_index().sort_values("composite_mean", ascending=True)
    grp["top1_std"] = grp["top1_std"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(grp))))
    bars = ax.barh(grp["tag"], grp["top1_mean"] * 100,
                   xerr=grp["top1_std"] * 100, color="C1", alpha=0.85, edgecolor="k")
    for b, p in zip(bars, grp["params"]):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                f"{p/1e6:.2f}M", va="center", fontsize=8)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title("Ablation matrix — top-1 ± seed-std (label = #params)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(results_dir: str | Path, out_png: Path) -> None:
    p = Path(results_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for hj in sorted(p.glob("**/history.json")):
        tag = hj.parent.name
        try:
            hist = json.loads(hj.read_text())
            xs = [r["epoch"] for r in hist]
            ys = [r["test_top1"] for r in hist]
            ax.plot(xs, [100 * y for y in ys], label=tag, lw=1.5)
        except Exception:
            continue
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test top-1 (%)")
    ax.set_title("Training curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_betti(betti_rows: list[dict], out_png: Path) -> None:
    if not betti_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for row in betti_rows:
        axes[0].plot(row["b0"], "-o", label=row["tag"])
        axes[1].plot(row["b1"], "-o", label=row["tag"])
    for ax, t in zip(axes, ("Betti-0 (connected components)",
                            "Betti-1 (1-D holes)")):
        ax.set_xlabel("Stage (deeper →)")
        ax.set_ylabel("β")
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Persistent-homology Betti collapse across layers", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


HTML_HEAD = """<!doctype html>
<html><head><meta charset='utf-8'>
<title>SacredGeoBlock — autoresearch dashboard</title>
<style>
 body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#222;}
 h1{margin-bottom:0;} .sub{color:#666;margin-top:4px;margin-bottom:18px;}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
 .card{border:1px solid #ddd;border-radius:8px;padding:14px;background:#fff;
       box-shadow:0 1px 3px rgba(0,0,0,0.04);}
 img{max-width:100%;height:auto;border-radius:4px;}
 table{border-collapse:collapse;width:100%;font-size:13px;}
 th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:right;}
 th:first-child,td:first-child{text-align:left;}
 th{position:sticky;top:0;background:#fafafa;cursor:pointer;}
 tr:hover{background:#f8f8f8;}
 .filter{padding:6px;margin-bottom:8px;border:1px solid #ccc;border-radius:4px;width:240px;}
 .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#f0f0f0;
       font-size:11px;margin-right:4px;}
 .best{background:#d6f5e0;}
 a{color:#1857b8;text-decoration:none;}
 a:hover{text-decoration:underline;}
 .meta{font-size:12px;color:#888;}
</style>
<script>
 function sortTable(n){var t=document.getElementById('runs'),rs=Array.from(t.tBodies[0].rows);
  var d=t.dataset.dir==='asc'?-1:1;t.dataset.dir=d===1?'asc':'desc';
  rs.sort(function(a,b){var x=a.cells[n].dataset.v||a.cells[n].textContent,
                       y=b.cells[n].dataset.v||b.cells[n].textContent;
    var nx=parseFloat(x),ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny))return d*(nx-ny);
    return d*x.localeCompare(y);});
  rs.forEach(function(r){t.tBodies[0].appendChild(r);});}
 function filterTable(){var q=document.getElementById('q').value.toLowerCase();
  Array.from(document.querySelectorAll('#runs tbody tr')).forEach(function(r){
   r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';});}
</script>
</head><body>
"""


def render_dashboard(results_dir: str | Path, out_html: Path,
                     extra_sections: list[tuple[str, str]] | None = None,
                     title: str = "SacredGeoBlock — autoresearch dashboard",
                     subtitle: str = "Sacred-geometry ablations on CIFAR-10 + topology / CKA / Pareto") -> None:
    df = load_runs(results_dir)
    extra_sections = extra_sections or []

    p = Path(results_dir)
    pareto_png = p / "plot_pareto.png"
    ablate_png = p / "plot_ablation.png"
    curves_png = p / "plot_curves.png"
    plot_pareto(df, pareto_png)
    plot_ablation(df, ablate_png)
    plot_training_curves(results_dir, curves_png)

    # rank rows by composite, mark best per dataset
    if not df.empty:
        df = df.sort_values(["dataset", "composite"], ascending=[True, False])
        best_idx = df.groupby("dataset")["composite"].idxmax().tolist()
    else:
        best_idx = []

    html = [HTML_HEAD]
    html.append(f"<h1>{title}</h1>")
    html.append(f"<div class='sub'>{subtitle}</div>")
    html.append("<div class='grid'>")
    for img, cap in [
        (pareto_png.name, "Pareto: accuracy vs. params / FLOPs / latency"),
        (ablate_png.name, "Ablation matrix: each sacred prior toggled independently"),
    ]:
        html.append(f"<div class='card'><h3>{cap}</h3><img src='{img}'/></div>")
    html.append(f"<div class='card' style='grid-column:1/3'><h3>Training curves</h3>"
                f"<img src='{curves_png.name}'/></div>")
    for title2, body in extra_sections:
        html.append(f"<div class='card' style='grid-column:1/3'><h3>{title2}</h3>{body}</div>")
    html.append("</div>")

    html.append("<h2 style='margin-top:32px'>Runs (click headers to sort)</h2>")
    html.append("<input id='q' class='filter' placeholder='filter…' oninput='filterTable()'/>")
    cols = [
        ("tag", "Tag", False),
        ("dataset", "Dataset", False),
        ("seed", "Seed", True),
        ("epochs", "Epochs", True),
        ("top1", "Top-1", True),
        ("top5", "Top-5", True),
        ("params", "Params", True),
        ("flops", "FLOPs", True),
        ("latency_ms", "Latency (ms)", True),
        ("rot_eq_err", "Rot-eq err", True),
        ("composite", "Composite", True),
        ("epochs_to_target", "ET₍target₎", True),
        ("train_seconds", "Train (s)", True),
    ]
    html.append("<table id='runs' data-dir='asc'><thead><tr>")
    for i, (_, label, _num) in enumerate(cols):
        html.append(f"<th onclick='sortTable({i})'>{label}</th>")
    html.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        klass = "best" if r.name in best_idx else ""
        html.append(f"<tr class='{klass}'>")
        for k, _l, _n in cols:
            v = r.get(k, "")
            disp = v
            if k == "params":
                disp = f"{v/1e6:.3f}M"
            elif k == "flops":
                try:
                    disp = f"{float(v)/1e6:.1f}M"
                except Exception:
                    disp = str(v)
            elif k == "top1" or k == "top5":
                try:
                    disp = f"{float(v)*100:.2f}%"
                except Exception:
                    disp = str(v)
            elif isinstance(v, float):
                disp = f"{v:.4f}"
            html.append(f"<td data-v='{v}'>{disp}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    html.append("<div class='meta'>Generated by sacgeo.dashboard.</div>")
    html.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html), encoding="utf-8")
