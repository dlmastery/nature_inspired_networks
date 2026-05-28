"""Autoresearch-style static dashboard generator (autoresearchspy-rich edition).

Reads experiment_log.jsonl, per-experiment metrics.json + history.json,
reasoning_annotations.json, betti.json, IDEA_TABLE.md, EXPERIMENT_LOG.md,
hypotheses/INDEX.md + hypotheses/H<NN>_*.md, and FINDINGS.md to emit a
single self-contained dark-theme HTML dashboard with rich detail panels.

Style is modelled after dlmastery/autoresearchspy/docs/spy_dashboard:
- dark GitHub-flavoured palette (#0d1117 / #161b22 / #58a6ff / #3fb950 / #f85149 / #d29922)
- summary KPI cards
- composite-formula chip with SHA-256 fingerprint
- 71-cell hypothesis status grid (G1..G7) with click-to-load .md side panel
- T0..T6 tier filter chips for the runs table
- runs-table rows -> click pops a modal with the matching reasoning entry
- per-row <details> expansion with training-curve sparkline + latency breakdown
- findings ribbon under the title
- existing PNG plot panels: pareto / ablation / curves / betti
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_runs(results_dir: str | Path) -> pd.DataFrame:
    p = Path(results_dir)
    rows: list[dict] = []
    for mj in p.glob("**/metrics.json"):
        try:
            row = json.loads(mj.read_text())
            row["_run_dir"] = str(mj.parent.relative_to(p)).replace("\\", "/")
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_history(results_dir: str | Path) -> dict[str, list[dict]]:
    p = Path(results_dir)
    out: dict[str, list[dict]] = {}
    for hj in p.glob("**/history.json"):
        tag = hj.parent.name
        try:
            out[tag] = json.loads(hj.read_text())
        except Exception:
            continue
    return out


def load_reasoning(reasoning_path: str | Path) -> list[dict]:
    p = Path(reasoning_path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def load_betti(betti_path: str | Path) -> list[dict]:
    p = Path(betti_path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Markdown helpers (extract hypothesis status + tier maps from EXPERIMENT_LOG.md)
# ---------------------------------------------------------------------------

_STATUS_GLYPH_MAP = {
    "✓": "done", "▶": "running", "⏸": "queued",
    "○": "planned", "✗": "failed", "♻": "superseded",
}


def parse_hypothesis_index(index_md: str | Path) -> list[dict]:
    """Parse hypotheses/INDEX.md into [{id, group, idx, file, idea}, ...]."""
    p = Path(index_md)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict] = []
    current_group: str | None = None
    for line in text.splitlines():
        m = re.match(r"##\s+Group\s+(G\d+)\s+", line)
        if m:
            current_group = m.group(1)
            continue
        m = re.match(r"\|\s*(H\d{2})\s*\|\s*`([^`]+)`\s*\|\s*(.+?)\s*\|", line)
        if m and current_group:
            hid = m.group(1)
            fname = m.group(2)
            idea = m.group(3).strip()
            try:
                idx_in_group = int(hid[1:]) - (int(current_group[1:]) - 1) * 10
            except Exception:
                idx_in_group = 0
            rows.append({
                "id": hid,
                "group": current_group,
                "idx": idx_in_group,
                "file": fname,
                "idea": idea,
            })
    return rows


def parse_experiment_log_tiers(log_md: str | Path) -> dict[str, dict]:
    """Parse EXPERIMENT_LOG.md and return {tag: {tier, status, idea}}.

    Also detects pure 'planned-row' tags (`*_cifar100` etc.) which we treat as
    placeholder rows in their tier.
    """
    p = Path(log_md)
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="ignore")
    current_tier: str | None = None
    tier_pattern = re.compile(r"##\s+Tier\s+(\d)")
    for line in text.splitlines():
        tm = tier_pattern.search(line)
        if tm:
            current_tier = f"T{tm.group(1)}"
            continue
        if current_tier and line.lstrip().startswith("|") and "`" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 4:
                continue
            # Find first backtick-wrapped token in any cell -- that's the tag
            tag = None
            for c in cells:
                bm = re.search(r"`([^`]+)`", c)
                if bm:
                    tag = bm.group(1)
                    break
            if not tag:
                continue
            # Status glyph search across cells
            status = "planned"
            for c in cells:
                for g, s in _STATUS_GLYPH_MAP.items():
                    if g in c:
                        status = s
                        break
                if status != "planned":
                    break
            idea_cell = cells[1] if len(cells) > 1 else ""
            out[tag] = {"tier": current_tier, "status": status, "idea": idea_cell}
    return out


def parse_findings_headline(findings_md: str | Path, max_chars: int = 600) -> str:
    """Extract the headline blockquote (the negative finding) from FINDINGS.md."""
    p = Path(findings_md)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(">"):
            cleaned = re.sub(r"^>+\s*", "", s)
            if cleaned:
                lines.append(cleaned)
        elif lines:
            break
    return " ".join(lines)[:max_chars]


def parse_findings_metrics(findings_md: str | Path) -> list[tuple[str, str, str]]:
    """Pull a handful of headline numeric metrics from FINDINGS.md.

    Returns list of (label, value, mood) where mood in {"neutral", "positive", "negative"}.
    """
    p = Path(findings_md)
    out: list[tuple[str, str, str]] = []
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="ignore")

    def find(pattern: str, group: int = 1) -> str | None:
        m = re.search(pattern, text)
        return m.group(group) if m else None

    base_top1 = find(r"`baseline_resnet20`\s*\|\s*([\d.]+)\s*%")
    full_top1 = find(r"`sg_full_fib`\s*\|\s*([\d.]+)\s*%")
    full_lat = find(r"`sg_full_fib`[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|")
    fractal_d = find(r"`fractal`\s*\|\s*\*\*([+\-][\d.]+\s*pp)")
    group_d = find(r"`group`[^|]*\|\s*\*\*([+\-][\d.]+\s*pp)")

    if base_top1:
        out.append(("Baseline ResNet-20 top-1", f"{base_top1}%", "neutral"))
    if full_top1:
        out.append(("Full-hybrid top-1", f"{full_top1}%", "negative"))
    if full_lat:
        out.append(("Full-hybrid latency", f"{full_lat} ms", "negative"))
    if fractal_d:
        out.append(("Fractal Δ top-1", fractal_d.strip(), "positive"))
    if group_d:
        out.append(("C4-group Δ top-1", group_d.strip(), "negative"))
    return out


# ---------------------------------------------------------------------------
# PNG plots (existing four, kept as-is so PNG remains PNG per constraint)
# ---------------------------------------------------------------------------

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
        ax.scatter(x[~is_base], y[~is_base], c="C1", s=80, label="NaturePrior", edgecolor="k")
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


# ---------------------------------------------------------------------------
# Inline SVG sparkline (per-row training curve)
# ---------------------------------------------------------------------------

def _sparkline_svg(history: list[dict] | None, w: int = 220, h: int = 36) -> str:
    if not history:
        return ""
    pts = [(r.get("epoch"), r.get("test_top1")) for r in history
           if r.get("test_top1") is not None]
    if len(pts) < 2:
        return ""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1e-6
    pad = 2
    def sx(x): return pad + (w - 2 * pad) * (x - xmin) / (xmax - xmin)
    def sy(y): return h - pad - (h - 2 * pad) * (y - ymin) / (ymax - ymin)
    pts_str = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    last_x, last_y = sx(xs[-1]), sy(ys[-1])
    label = f"{100 * ys[-1]:.1f}%"
    return (
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
        f"xmlns='http://www.w3.org/2000/svg' style='vertical-align:middle'>"
        f"<polyline points='{pts_str}' fill='none' stroke='#58a6ff' "
        f"stroke-width='1.6'/>"
        f"<circle cx='{last_x:.1f}' cy='{last_y:.1f}' r='2.4' fill='#3fb950'/>"
        f"<text x='{w-2}' y='{h-4}' fill='#8b949e' font-size='10' "
        f"text-anchor='end'>{label}</text>"
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# Per-experiment page (independent, self-contained, inline SVG line charts)
# ---------------------------------------------------------------------------

def run_page_filename(run_dir_name: str) -> str:
    """Map a run-directory basename (``<tag>_seed<N>``) to its HTML filename."""
    return f"{run_dir_name}.html"


def _esc(s: object) -> str:
    """HTML-escape an arbitrary value for safe text insertion."""
    out = str(s)
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 ('"', "&quot;"), ("'", "&#39;")):
        out = out.replace(a, b)
    return out


def _line_chart_svg(series: list[tuple[str, list[float], list[float], str]],
                    title: str, y_label: str,
                    w: int = 460, h: int = 240,
                    y_as_pct: bool = False) -> str:
    """Render multiple (label, xs, ys, color) series as one inline SVG line chart.

    Axes, gridlines, a legend, and end-point value labels are drawn purely with
    SVG primitives — no JS, no external assets — so the file is self-contained
    and renders identically from ``file://`` or GitHub Pages.
    """
    series = [s for s in series if s[1] and s[2] and len(s[1]) == len(s[2])]
    if not series:
        return f"<div style='color:#8b949e'><i>No data for {_esc(title)}.</i></div>"
    ml, mr, mt, mb = 46, 14, 30, 28
    pw, ph = w - ml - mr, h - mt - mb
    all_x = [x for _, xs, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _ in series for y in ys]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + (1e-6 if ymax == 0 else abs(ymax) * 0.1)
    yspan = ymax - ymin
    ymin -= yspan * 0.06
    ymax += yspan * 0.06

    def sx(x: float) -> float:
        return ml + pw * (x - xmin) / (xmax - xmin)

    def sy(y: float) -> float:
        return mt + ph * (1 - (y - ymin) / (ymax - ymin))

    parts: list[str] = [
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
        f"xmlns='http://www.w3.org/2000/svg' "
        f"style='background:#0d1117;border:1px solid #30363d;border-radius:6px'>",
        f"<text x='{ml}' y='16' fill='#c9d1d9' font-size='12' "
        f"font-weight='600'>{_esc(title)}</text>",
    ]
    # Horizontal gridlines + y tick labels (5 ticks)
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        yy = sy(yv)
        parts.append(
            f"<line x1='{ml}' y1='{yy:.1f}' x2='{ml + pw}' y2='{yy:.1f}' "
            f"stroke='#21262d' stroke-width='1'/>"
        )
        lbl = f"{yv * 100:.1f}%" if y_as_pct else f"{yv:.3f}"
        parts.append(
            f"<text x='{ml - 6}' y='{yy + 3:.1f}' fill='#8b949e' font-size='9' "
            f"text-anchor='end'>{lbl}</text>"
        )
    # X axis ticks (first / last epoch)
    parts.append(
        f"<text x='{sx(xmin):.1f}' y='{h - 8}' fill='#8b949e' font-size='9' "
        f"text-anchor='middle'>{int(xmin)}</text>"
    )
    parts.append(
        f"<text x='{sx(xmax):.1f}' y='{h - 8}' fill='#8b949e' font-size='9' "
        f"text-anchor='middle'>{int(xmax)}</text>"
    )
    parts.append(
        f"<text x='{ml + pw / 2:.1f}' y='{h - 8}' fill='#8b949e' font-size='9' "
        f"text-anchor='middle'>epoch</text>"
    )
    parts.append(
        f"<text x='12' y='{mt + ph / 2:.1f}' fill='#8b949e' font-size='9' "
        f"text-anchor='middle' transform='rotate(-90 12 {mt + ph / 2:.1f})'>"
        f"{_esc(y_label)}</text>"
    )
    # Series polylines + legend
    legend_x = ml + 6
    for li, (label, xs, ys, color) in enumerate(series):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        parts.append(
            f"<polyline points='{pts}' fill='none' stroke='{color}' "
            f"stroke-width='1.8'/>"
        )
        lx, ly = sx(xs[-1]), sy(ys[-1])
        parts.append(f"<circle cx='{lx:.1f}' cy='{ly:.1f}' r='2.6' fill='{color}'/>")
        ly_txt = mt + 12 + li * 14
        parts.append(
            f"<rect x='{legend_x}' y='{ly_txt - 8}' width='10' height='10' "
            f"fill='{color}' rx='2'/>"
        )
        parts.append(
            f"<text x='{legend_x + 15}' y='{ly_txt}' fill='#c9d1d9' "
            f"font-size='10'>{_esc(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


_EXP_PAGE_CSS = """
 *{margin:0;padding:0;box-sizing:border-box;}
 body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;
      color:#c9d1d9;padding:24px 28px;line-height:1.5;}
 a{color:#58a6ff;text-decoration:none;} a:hover{text-decoration:underline;}
 h1{color:#58a6ff;font-size:1.5em;margin-bottom:4px;}
 .sub{color:#8b949e;font-size:0.9em;margin-bottom:18px;}
 .back{display:inline-block;margin-bottom:14px;font-size:0.9em;}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;
       padding:16px 18px;margin-bottom:18px;}
 .card h2{color:#58a6ff;font-size:1.05em;margin-bottom:12px;font-weight:600;}
 table{width:100%;border-collapse:collapse;font-size:0.86em;}
 th{background:#0d1117;color:#8b949e;text-align:left;padding:7px 10px;
    border-bottom:2px solid #30363d;font-size:0.74em;text-transform:uppercase;
    letter-spacing:0.4px;}
 td{padding:7px 10px;border-bottom:1px solid #21262d;}
 td.k{color:#8b949e;width:42%;}
 td.v{color:#c9d1d9;font-family:Consolas,monospace;}
 .formula-chip{background:#0d1117;border:1px solid #30363d;border-radius:6px;
               padding:10px 12px;font-family:Consolas,monospace;font-size:0.82em;
               margin-bottom:12px;}
 .breakdown td.term{font-family:Consolas,monospace;}
 .pos{color:#3fb950;} .neg{color:#f85149;} .mut{color:#8b949e;}
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
 @media(max-width:980px){.charts{grid-template-columns:1fr;}}
 .meta{font-size:0.78em;color:#484f58;margin-top:18px;}
 code{background:#0d1117;padding:1px 5px;border-radius:3px;font-size:0.92em;}
"""


def render_experiment_page(metrics: dict, history: list[dict] | None,
                           run_dir_name: str, out_html: Path) -> None:
    """Write a single self-contained per-experiment HTML page.

    Sections: metrics table, composite-formula term breakdown, and inline-SVG
    per-epoch training curves (loss + top-1). Falls back gracefully when
    history.json is missing. ``out_html`` is assumed to live one directory
    below the central dashboard (``dashboard/experiments/``) so the back link
    is ``../dashboard.html``.
    """
    tag = metrics.get("tag", run_dir_name)
    seed = metrics.get("seed", "")
    dataset = metrics.get("dataset", "")
    title = f"{tag} · seed {seed}"

    # ---- metrics table -------------------------------------------------
    def _fmt(key: str) -> str:
        v = metrics.get(key)
        if v is None:
            return "<span class='mut'>—</span>"
        if key in ("top1", "top5", "train_top1", "generalization_gap"):
            try:
                return f"{float(v) * 100:.2f}%"
            except Exception:
                return _esc(v)
        if key == "params":
            try:
                return f"{int(v):,} ({float(v) / 1e6:.3f} M)"
            except Exception:
                return _esc(v)
        if key == "flops":
            try:
                return f"{float(v) / 1e6:.1f} M MACs"
            except Exception:
                return _esc(v)
        if key == "latency_ms":
            try:
                return f"{float(v):.3f} ms"
            except Exception:
                return _esc(v)
        if key == "train_seconds":
            try:
                return f"{float(v):.1f} s"
            except Exception:
                return _esc(v)
        if isinstance(v, float):
            return f"{v:.6f}"
        return _esc(v)

    metric_keys = [
        ("dataset", "Dataset"), ("seed", "Seed"), ("epochs", "Epochs"),
        ("top1", "Top-1 accuracy"), ("top5", "Top-5 accuracy"),
        ("train_top1", "Train top-1"),
        ("generalization_gap", "Generalization gap (train−test)"),
        ("params", "Parameters"), ("flops", "FLOPs"),
        ("latency_ms", "Latency (batch=1)"), ("rot_eq_err", "Rot-eq error"),
        ("epochs_to_target", "Epochs-to-target"),
        ("train_seconds", "Train wall-clock"),
        ("composite", "Composite score"),
        ("composite_fingerprint", "Composite fingerprint (SHA-256)"),
    ]
    rows = "".join(
        f"<tr><td class='k'>{_esc(lbl)}</td><td class='v'>{_fmt(k)}</td></tr>"
        for k, lbl in metric_keys if k in metrics
    )
    metrics_table = f"<table><tbody>{rows}</tbody></table>"

    # ---- composite-formula breakdown -----------------------------------
    import math
    formula = metrics.get(
        "composite_formula",
        "top1 - 0.05*log10(params_M) - 0.05*log10(latency_ms)",
    )
    breakdown_html = ""
    try:
        top1 = float(metrics["top1"])
        params_m = float(metrics["params"]) / 1e6
        lat = float(metrics["latency_ms"])
        t_params = -0.05 * math.log10(params_m) if params_m > 0 else 0.0
        t_lat = -0.05 * math.log10(lat) if lat > 0 else 0.0
        total = top1 + t_params + t_lat
        reported = float(metrics.get("composite", total))

        def _term(name: str, expr: str, val: float) -> str:
            cls = "pos" if val >= 0 else "neg"
            sign = "+" if val >= 0 else "−"
            return (
                f"<tr><td class='term'>{_esc(name)}</td>"
                f"<td class='term mut'>{_esc(expr)}</td>"
                f"<td class='term {cls}' style='text-align:right'>"
                f"{sign}{abs(val):.5f}</td></tr>"
            )

        breakdown_html = (
            f"<div class='formula-chip'>composite ≔ <code>{_esc(formula)}</code></div>"
            "<table class='breakdown'><thead><tr><th>Term</th><th>Expression</th>"
            "<th style='text-align:right'>Contribution</th></tr></thead><tbody>"
            + _term("top-1", f"{top1:.5f}", top1)
            + _term("params penalty", f"-0.05·log10({params_m:.4f} M)", t_params)
            + _term("latency penalty", f"-0.05·log10({lat:.3f} ms)", t_lat)
            + "<tr style='border-top:2px solid #30363d'>"
            f"<td class='term' colspan='2'><b>Σ (recomputed)</b></td>"
            f"<td class='term' style='text-align:right'><b>{total:.5f}</b></td></tr>"
            f"<tr><td class='term mut' colspan='2'>reported composite</td>"
            f"<td class='term mut' style='text-align:right'>{reported:.5f}</td></tr>"
            "</tbody></table>"
        )
    except Exception:
        breakdown_html = (
            f"<div class='formula-chip'>composite ≔ <code>{_esc(formula)}</code></div>"
            "<div style='color:#8b949e'><i>Insufficient fields to break the "
            "composite down term-by-term.</i></div>"
        )

    # ---- training curves (inline SVG) ----------------------------------
    if history:
        epochs = [r.get("epoch") for r in history]
        train_loss = [r.get("train_loss") for r in history]
        train_top1 = [r.get("train_top1") for r in history]
        test_top1 = [r.get("test_top1") for r in history]
        test_top5 = [r.get("test_top5") for r in history]

        def _clean(xs, ys):
            px, py = [], []
            for x, y in zip(xs, ys):
                if x is not None and y is not None:
                    px.append(x)
                    py.append(y)
            return px, py

        lx, ly = _clean(epochs, train_loss)
        loss_chart = _line_chart_svg(
            [("train loss", lx, ly, "#f85149")],
            "Loss", "loss", y_as_pct=False,
        )
        acc_series = []
        ax, ay = _clean(epochs, train_top1)
        if ay:
            acc_series.append(("train top-1", ax, ay, "#d29922"))
        bx, by = _clean(epochs, test_top1)
        if by:
            acc_series.append(("test top-1", bx, by, "#58a6ff"))
        cx, cy = _clean(epochs, test_top5)
        if cy:
            acc_series.append(("test top-5", cx, cy, "#3fb950"))
        acc_chart = _line_chart_svg(acc_series, "Accuracy", "accuracy", y_as_pct=True)
        charts_html = (
            f"<div class='charts'><div>{loss_chart}</div>"
            f"<div>{acc_chart}</div></div>"
        )
    else:
        charts_html = (
            "<div style='color:#8b949e'><i>history.json absent for this run — "
            "no per-epoch curves available.</i></div>"
        )

    page = (
        "<!doctype html>\n"
        "<html lang='en'><head><meta charset='utf-8'>\n"
        f"<title>{_esc(title)} — experiment page</title>\n"
        f"<style>{_EXP_PAGE_CSS}</style></head><body>\n"
        "<a class='back' href='../dashboard.html'>&larr; back to dashboard</a>\n"
        f"<h1>{_esc(tag)}</h1>\n"
        f"<div class='sub'>seed {_esc(seed)} · {_esc(dataset)} · "
        f"run directory <code>{_esc(run_dir_name)}</code></div>\n"
        f"<div class='card'><h2>Metrics</h2>{metrics_table}</div>\n"
        f"<div class='card'><h2>Composite-score breakdown</h2>{breakdown_html}</div>\n"
        f"<div class='card'><h2>Per-epoch training curves</h2>{charts_html}</div>\n"
        "<div class='meta'>Generated by "
        "<code>nature_inspired_networks.dashboard.render_experiment_page</code> · "
        "self-contained inline SVG; no external assets.</div>\n"
        "</body></html>\n"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")


def render_all_experiment_pages(results_dir: str | Path,
                                out_dir: str | Path) -> list[str]:
    """Render one per-experiment page per run dir holding a ``metrics.json``.

    Returns the sorted list of HTML filenames written under ``out_dir`` (which
    should be ``<dashboard>/experiments``). Idempotent — rewrites stable output.
    """
    rp = Path(results_dir)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for mj in sorted(rp.glob("**/metrics.json")):
        run_dir = mj.parent
        try:
            metrics = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        hist: list[dict] | None = None
        hj = run_dir / "history.json"
        if hj.exists():
            try:
                hist = json.loads(hj.read_text(encoding="utf-8"))
            except Exception:
                hist = None
        fname = run_page_filename(run_dir.name)
        render_experiment_page(metrics, hist, run_dir.name, od / fname)
        written.append(fname)
    return sorted(written)


# ---------------------------------------------------------------------------
# Static HTML head (dark theme, all CSS+JS inline)
# ---------------------------------------------------------------------------

HTML_HEAD = """<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>
<title>nature_inspired_networks — autoresearch dashboard</title>
<style>
 *{margin:0;padding:0;box-sizing:border-box;}
 body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;
      color:#c9d1d9;padding:20px;line-height:1.45;}
 a{color:#58a6ff;text-decoration:none;} a:hover{text-decoration:underline;}
 h1{color:#58a6ff;margin-bottom:4px;font-size:1.7em;font-weight:700;}
 h2{color:#58a6ff;font-size:1.15em;margin:24px 0 10px 0;font-weight:600;
    letter-spacing:0.3px;}
 h3{color:#c9d1d9;font-size:0.95em;margin-bottom:10px;font-weight:600;}
 .sub{color:#8b949e;margin-bottom:14px;font-size:0.92em;}
 .ribbon{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
         gap:10px;margin:10px 0 14px 0;}
 .kpi{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 14px;}
 .kpi .label{color:#8b949e;font-size:0.7em;text-transform:uppercase;
             letter-spacing:0.6px;}
 .kpi .value{font-size:1.55em;font-weight:700;margin-top:3px;color:#c9d1d9;}
 .kpi.positive .value{color:#3fb950;}
 .kpi.negative .value{color:#f85149;}
 .kpi.neutral  .value{color:#58a6ff;}
 .formula-chip{display:inline-block;background:#161b22;border:1px solid #30363d;
               border-radius:6px;padding:6px 10px;margin:4px 0 12px 0;
               font-family:Consolas,monospace;font-size:0.78em;color:#c9d1d9;}
 .formula-chip .fp{color:#8b949e;}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px;}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;}
 .card img{max-width:100%;height:auto;border-radius:4px;
           background:#fff;padding:4px;}
 .panel-2col{grid-column:1 / 3;}
 table{width:100%;border-collapse:collapse;font-size:0.83em;}
 th{background:#161b22;color:#8b949e;text-align:right;padding:8px 10px;
    border-bottom:2px solid #30363d;font-weight:600;text-transform:uppercase;
    font-size:0.72em;letter-spacing:0.4px;cursor:pointer;position:sticky;top:0;
    z-index:6;}
 th:first-child{text-align:left;}
 td{padding:7px 10px;border-bottom:1px solid #21262d;text-align:right;
    color:#c9d1d9;}
 td:first-child{text-align:left;}
 tr:hover{background:#1c2128;}
 tr.clickable{cursor:pointer;}
 tr.best-row{background:#0d2818!important;border-left:4px solid #00d26a;}
 tr.best-row td{font-weight:600;}
 .tag-pill{display:inline-block;background:#21262d;border:1px solid #30363d;
           border-radius:10px;padding:1px 8px;font-size:0.75em;color:#c9d1d9;
           font-family:Consolas,monospace;}
 .tier-chip,.chip{display:inline-block;background:#21262d;border:1px solid #30363d;
                  border-radius:14px;padding:3px 11px;margin:2px 4px 2px 0;
                  font-size:0.78em;cursor:pointer;color:#c9d1d9;
                  user-select:none;}
 .tier-chip:hover,.chip:hover{background:#30363d;border-color:#58a6ff;}
 .tier-chip.active{background:#1f6feb;border-color:#58a6ff;color:#fff;
                   font-weight:600;}
 .filter{padding:6px 10px;margin:6px 0;border:1px solid #30363d;border-radius:6px;
         width:280px;background:#0d1117;color:#c9d1d9;font-size:0.85em;}
 .detail-row td{background:#0d1117;padding:14px 16px;border-bottom:2px solid #21262d;}
 .detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px;}
 .detail-block{background:#161b22;border:1px solid #30363d;border-radius:6px;
               padding:10px 12px;font-size:0.83em;}
 .detail-block .lbl{color:#8b949e;font-size:0.72em;text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:4px;}
 .meta{font-size:0.78em;color:#484f58;margin-top:18px;}
 .status-done{background:#3fb950;}
 .status-running{background:#d29922;animation:pulse 1.5s infinite;}
 .status-queued{background:#1f6feb;}
 .status-planned{background:#484f58;}
 .status-failed{background:#f85149;}
 .status-superseded{background:#6e7681;}
 @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.45;}}
 /* hypothesis grid */
 .hyp-grid-row{display:flex;align-items:center;margin-bottom:4px;
               font-family:Consolas,monospace;font-size:0.78em;}
 .hyp-grid-row .gid{width:34px;color:#8b949e;}
 .hyp-grid-row .cell{width:26px;height:22px;margin-right:3px;border-radius:3px;
                     display:inline-flex;align-items:center;justify-content:center;
                     font-size:0.7em;color:#fff;cursor:pointer;
                     border:1px solid #21262d;}
 .hyp-grid-row .cell:hover{outline:2px solid #58a6ff;}
 .hyp-grid-row .cell.empty{background:#21262d;color:#484f58;cursor:default;
                            border-color:transparent;}
 .legend-row{margin:8px 0 12px 0;font-size:0.78em;color:#8b949e;}
 .legend-row .swatch{display:inline-block;width:12px;height:12px;border-radius:2px;
                     vertical-align:middle;margin:0 4px 0 10px;}
 /* side panel + modal */
 #side-panel{position:fixed;top:0;right:-560px;width:540px;height:100vh;
             background:#0d1117;border-left:1px solid #30363d;
             box-shadow:-4px 0 24px rgba(0,0,0,0.6);
             padding:20px 22px;overflow-y:auto;transition:right 0.25s ease;
             z-index:50;}
 #side-panel.open{right:0;}
 #side-panel h3{color:#58a6ff;font-size:1.05em;margin-bottom:10px;}
 #side-panel pre{background:#161b22;border:1px solid #30363d;border-radius:6px;
                 padding:12px;white-space:pre-wrap;color:#c9d1d9;font-size:0.82em;
                 font-family:Consolas,monospace;line-height:1.5;}
 .close-btn{background:#21262d;border:1px solid #30363d;color:#8b949e;
            padding:4px 12px;border-radius:4px;cursor:pointer;font-size:0.82em;
            float:right;}
 .close-btn:hover{background:#30363d;color:#c9d1d9;}
 #modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,0.6);display:none;
                 z-index:40;}
 #modal-backdrop.open{display:block;}
 #modal{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
        width:min(820px,92vw);max-height:84vh;overflow-y:auto;
        background:#0d1117;border:1px solid #30363d;border-radius:10px;
        padding:24px 26px;z-index:60;display:none;
        box-shadow:0 12px 40px rgba(0,0,0,0.7);}
 #modal.open{display:block;}
 #modal h3{color:#58a6ff;font-size:1.1em;margin-bottom:6px;}
 #modal .meta{color:#8b949e;font-size:0.8em;margin-bottom:14px;}
 #modal section{margin-bottom:14px;}
 #modal section .lbl{color:#3fb950;font-size:0.75em;text-transform:uppercase;
                     letter-spacing:0.6px;margin-bottom:4px;font-weight:600;}
 #modal section .body{font-size:0.88em;line-height:1.5;color:#c9d1d9;
                      white-space:pre-wrap;}
 #modal section ul{margin-left:18px;font-size:0.85em;}
 details{margin-top:8px;}
 details > summary{cursor:pointer;color:#58a6ff;font-size:0.8em;font-weight:600;
                   padding:2px 0;}
 details > summary:hover{text-decoration:underline;}
</style>
</head><body>
"""


HTML_JS = r"""
<script>
function sortTable(n){
  var t=document.getElementById('runs');
  var rs=Array.from(t.tBodies[0].rows).filter(function(r){return !r.classList.contains('detail-row');});
  var pairs=rs.map(function(r){
    var d=r.nextElementSibling && r.nextElementSibling.classList.contains('detail-row')?r.nextElementSibling:null;
    return [r,d];
  });
  var d=t.dataset.dir==='asc'?-1:1;t.dataset.dir=d===1?'asc':'desc';
  pairs.sort(function(a,b){
    var x=a[0].cells[n].dataset.v||a[0].cells[n].textContent;
    var y=b[0].cells[n].dataset.v||b[0].cells[n].textContent;
    var nx=parseFloat(x),ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny)) return d*(nx-ny);
    return d*x.localeCompare(y);
  });
  pairs.forEach(function(p){t.tBodies[0].appendChild(p[0]);if(p[1])t.tBodies[0].appendChild(p[1]);});
}

var activeTier='ALL';
function setTier(name,btn){
  activeTier=name;
  document.querySelectorAll('.tier-chip').forEach(function(c){c.classList.remove('active');});
  btn.classList.add('active');
  applyFilters();
}
function applyFilters(){
  var q=(document.getElementById('q').value||'').toLowerCase();
  Array.from(document.querySelectorAll('#runs tbody tr')).forEach(function(r){
    if(r.classList.contains('detail-row')) return;
    var tier=r.dataset.tier||'';
    var matchTier=(activeTier==='ALL')||(tier===activeTier);
    var matchQ=r.textContent.toLowerCase().includes(q);
    var show=matchTier && matchQ;
    r.style.display=show?'':'none';
    var det=r.nextElementSibling;
    if(det && det.classList.contains('detail-row')) det.style.display='none';
  });
}

/* row click -> open modal with reasoning entry */
function openReasoning(tag){
  var entry=window._REASONING_BY_TAG[tag];
  var m=document.getElementById('modal'),b=document.getElementById('modal-backdrop');
  if(!entry){
    document.getElementById('modal-body').innerHTML=
     '<section><div class="body">No reasoning entry found for <code>'+tag+'</code> in reasoning_annotations.json.</div></section>';
    document.getElementById('modal-title').textContent=tag;
    document.getElementById('modal-meta').textContent='';
  } else {
    document.getElementById('modal-title').textContent=entry.title||tag;
    document.getElementById('modal-meta').textContent=
      'experiment_id: '+(entry.experiment_id||'(n/a)')+
      '  ·  composite_fp: '+(entry.composite_fingerprint||'').slice(0,12);
    var sections=[
      ['diagnosis','Diagnosis'],['hypothesis','Hypothesis'],
      ['prediction','Prediction'],['verdict','Verdict'],['learning','Learning'],
    ];
    var html='';
    sections.forEach(function(p){
      var txt=entry[p[0]];
      if(txt){
        html+='<section><div class="lbl">'+p[1]+'</div><div class="body">'+
              escapeHtml(txt)+'</div></section>';
      }
    });
    if(entry.citations && entry.citations.length){
      html+='<section><div class="lbl">Citations</div><ul>';
      entry.citations.forEach(function(c){html+='<li>'+escapeHtml(c)+'</li>';});
      html+='</ul></section>';
    }
    document.getElementById('modal-body').innerHTML=html;
  }
  m.classList.add('open');b.classList.add('open');
}
function closeModal(){
  document.getElementById('modal').classList.remove('open');
  document.getElementById('modal-backdrop').classList.remove('open');
}

/* hypothesis-cell click -> open side panel with hypotheses/H<NN>_*.md */
function openHypothesis(hid,fname){
  var sp=document.getElementById('side-panel');
  document.getElementById('side-panel-title').textContent=hid+' — '+fname;
  document.getElementById('side-panel-body').textContent='Loading…';
  fetch('../hypotheses/'+fname).then(function(r){
    if(!r.ok) throw new Error('HTTP '+r.status);
    return r.text();
  }).then(function(t){
    document.getElementById('side-panel-body').textContent=t;
  }).catch(function(e){
    document.getElementById('side-panel-body').textContent=
      'Could not fetch hypotheses/'+fname+' ('+e.message+').\n'+
      'When opened via file:// many browsers block fetch; serve via\n'+
      '`python -m http.server` from the repo root and reload, or open the file directly\n'+
      'at hypotheses/'+fname+'.';
  });
  sp.classList.add('open');
}
function closeSide(){document.getElementById('side-panel').classList.remove('open');}

/* per-row <details> expansion — populates the lazy panel on first open */
function toggleDetail(btn,tag){
  var det=btn.closest('tr').nextElementSibling;
  if(!det || !det.classList.contains('detail-row')) return;
  det.style.display=det.style.display==='none' || det.style.display===''? 'table-row':'none';
}

function escapeHtml(s){
  if(s===null || s===undefined) return '';
  return String(s).replace(/[&<>"']/g,function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c];
  });
}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){closeModal();closeSide();}
});
</script>
"""


# ---------------------------------------------------------------------------
# HTML fragment builders
# ---------------------------------------------------------------------------

def _hypothesis_grid_html(index_rows: list[dict],
                          tag_status: dict[str, dict],
                          reasoning_by_tag: dict[str, dict]) -> str:
    """Build the 71-cell heatmap (7 rows × ~10 cols).

    Status precedence for a hypothesis cell:
      done if any tag tied to this hypothesis has status done in EXPERIMENT_LOG.md,
      else running/queued/failed as recorded, else 'planned'.
    Since tag<->hypothesis mapping is sparse, we use a couple of heuristic
    bindings driven by EXPERIMENT_LOG.md cells that contain 'H<NN>'.
    """
    if not index_rows:
        return "<i style='color:#8b949e'>hypotheses/INDEX.md not found.</i>"

    # Build hypothesis status map. Default: 'planned' until evidence says
    # otherwise.
    status_for: dict[str, str] = {h["id"]: "planned" for h in index_rows}
    # Scan tag_status for ideas containing H<NN> references.
    for tag, meta in tag_status.items():
        idea = meta.get("idea", "")
        for m in re.finditer(r"H(\d{2})", idea):
            hid = "H" + m.group(1)
            if hid in status_for:
                # Done > running > queued > failed > superseded > planned.
                order = ["done", "running", "queued", "failed", "superseded", "planned"]
                cur = status_for[hid]
                new = meta["status"]
                if order.index(new) < order.index(cur):
                    status_for[hid] = new

    # H05/H17/H21/H22/H24/H34/H35/H50/H58 are exercised by the canonical
    # Tier-1 sweep tags — promote those to 'done' if at least one matching
    # tag has status done.
    direct = {
        "H04": ["sg_chan_fib", "sg_chan_phi"],
        "H05": ["sg_only_fractal"],
        "H17": ["sg_only_golden_modulate"],
        "H21": ["sg_only_hex"],
        "H22": ["sg_only_toroidal"],
        "H24": ["sg_only_group"],
        "H34": ["sg_only_golden_modulate"],
        "H35": ["sg_only_cymatic_init"],
        "H50": ["sg_full_fib"],
        "H58": ["sg_only_group_avg"],
    }
    for hid, tags in direct.items():
        if hid not in status_for:
            continue
        for t in tags:
            meta = tag_status.get(t)
            if not meta:
                continue
            if meta["status"] == "done" and status_for[hid] != "done":
                status_for[hid] = "done"
            elif meta["status"] == "running" and status_for[hid] == "planned":
                status_for[hid] = "running"

    # Group rows
    by_group: dict[str, list[dict]] = {}
    for h in index_rows:
        by_group.setdefault(h["group"], []).append(h)

    out = ["<div class='legend-row'>Legend:"]
    for s in ("done", "running", "queued", "planned", "failed", "superseded"):
        out.append(f"<span class='swatch status-{s}'></span>{s}")
    out.append("</div>")
    out.append("<div id='hyp-grid'>")
    for g in sorted(by_group.keys()):
        cells = sorted(by_group[g], key=lambda r: r["idx"])
        max_idx = max((c["idx"] for c in cells), default=10)
        row_html = [f"<div class='hyp-grid-row'><span class='gid'>{g}</span>"]
        cells_by_idx = {c["idx"]: c for c in cells}
        for i in range(1, max(max_idx, 10) + 1):
            c = cells_by_idx.get(i)
            if not c:
                row_html.append("<span class='cell empty'>·</span>")
                continue
            st = status_for.get(c["id"], "planned")
            hid = c["id"]
            fname = c["file"]
            idea = c["idea"].replace("'", "&#39;").replace('"', "&quot;")
            row_html.append(
                "<span class='cell status-" + st + "' title='" + hid + " — "
                + idea + " (status: " + st + ")' "
                + "onclick=\"openHypothesis('" + hid + "','" + fname + "')\">"
                + hid[1:] + "</span>"
            )
        row_html.append("</div>")
        out.append("".join(row_html))
    out.append("</div>")
    return "\n".join(out)


def _tier_chips_html(tiers_present: list[str]) -> str:
    chips = ["<span class='tier-chip active' onclick=\"setTier('ALL',this)\">ALL</span>"]
    for t in sorted(tiers_present):
        chips.append(
            f"<span class='tier-chip' onclick=\"setTier('{t}',this)\">{t}</span>"
        )
    return "".join(chips)


def _ribbon_html(metrics: list[tuple[str, str, str]]) -> str:
    if not metrics:
        return ""
    cards = []
    for label, value, mood in metrics:
        cards.append(
            f"<div class='kpi {mood}'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div></div>"
        )
    return "<div class='ribbon'>" + "".join(cards) + "</div>"


def _composite_formula_chip(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    formula = ""
    fp = ""
    if "composite_formula" in df.columns:
        formula = next((v for v in df["composite_formula"].dropna().tolist() if v), "")
    if not formula:
        formula = "composite = top1 - 0.05 * log10(params_M) - 0.05 * log10(latency_ms)"
    if "composite_fingerprint" in df.columns:
        fp = next((v for v in df["composite_fingerprint"].dropna().tolist() if v), "")
    if not fp:
        fp = hashlib.sha256(formula.encode("utf-8")).hexdigest()
    return (
        f"<div class='formula-chip'><b>composite</b> ≔ <code>{formula}</code>"
        f"<br><span class='fp'>SHA-256 · {fp}</span></div>"
    )


# ---------------------------------------------------------------------------
# Render entry point
# ---------------------------------------------------------------------------

def render_dashboard(results_dir: str | Path,
                     out_html: Path,
                     extra_sections: list[tuple[str, str]] | None = None,
                     title: str = "NaturePriorBlock — autoresearch dashboard",
                     subtitle: str = "Nature-inspired ablations on CIFAR-10 + topology / Pareto / hypothesis ledger",
                     repo_root: str | Path | None = None) -> None:
    """Render the rich dark-theme dashboard.

    `repo_root` defaults to the parent of `results_dir`; it is used to locate
    EXPERIMENT_LOG.md, hypotheses/INDEX.md, FINDINGS.md.
    """
    df = load_runs(results_dir)
    extra_sections = extra_sections or []

    p = Path(results_dir)
    root = Path(repo_root) if repo_root else p.parent
    pareto_png = p / "plot_pareto.png"
    ablate_png = p / "plot_ablation.png"
    curves_png = p / "plot_curves.png"
    betti_png = p / "plot_betti.png"
    plot_pareto(df, pareto_png)
    plot_ablation(df, ablate_png)
    plot_training_curves(results_dir, curves_png)
    bj = p / "betti.json"
    betti_rows: list[dict] = []
    if bj.exists():
        betti_rows = load_betti(bj)
        plot_betti(betti_rows, betti_png)

    reasoning = load_reasoning(p / "reasoning_annotations.json")
    history = load_history(results_dir)
    tag_to_tier = parse_experiment_log_tiers(root / "EXPERIMENT_LOG.md")
    index_rows = parse_hypothesis_index(root / "hypotheses" / "INDEX.md")
    findings_blurb = parse_findings_headline(root / "FINDINGS.md")
    findings_metrics = parse_findings_metrics(root / "FINDINGS.md")

    # Map reasoning by tag (use experiment_id suffix or title match)
    reasoning_by_tag: dict[str, dict] = {}
    for entry in reasoning:
        eid = entry.get("experiment_id", "")
        # experiment_id form: exp01_baseline_resnet20 -> tag baseline_resnet20
        m = re.match(r"exp\d+_(.+)$", eid)
        if m:
            reasoning_by_tag[m.group(1)] = entry

    # rank rows by composite, mark best per dataset
    if not df.empty:
        df = df.sort_values(["dataset", "composite"], ascending=[True, False])
        best_idx = df.groupby("dataset")["composite"].idxmax().tolist()
    else:
        best_idx = []

    html = [HTML_HEAD]
    html.append(f"<h1>{title}</h1>")
    html.append(f"<div class='sub'>{subtitle} · "
                f"<a href='https://github.com/dlmastery/nature_inspired_networks'>"
                f"GitHub</a> · "
                f"<a href='../FINDINGS.md'>FINDINGS.md</a> · "
                f"<a href='../EXPERIMENT_LOG.md'>EXPERIMENT_LOG</a></div>")
    if findings_blurb:
        html.append(
            "<div style='background:#161b22;border-left:4px solid #d29922;"
            "border:1px solid #30363d;border-radius:6px;padding:12px 16px;"
            "margin:6px 0 10px 0;font-size:0.92em;color:#c9d1d9'>"
            f"<b style='color:#d29922'>Headline negative finding · </b>"
            f"{findings_blurb}</div>"
        )
    html.append(_composite_formula_chip(df))
    html.append(_ribbon_html(findings_metrics))

    # PNG plot grid
    html.append("<div class='grid'>")
    html.append(
        f"<div class='card'><h3>Pareto: accuracy vs. params / FLOPs / latency</h3>"
        f"<img src='{pareto_png.name}'/></div>"
    )
    html.append(
        f"<div class='card'><h3>Ablation matrix: each nature-inspired prior toggled "
        f"independently</h3><img src='{ablate_png.name}'/></div>"
    )
    html.append(
        f"<div class='card panel-2col'><h3>Training curves</h3>"
        f"<img src='{curves_png.name}'/></div>"
    )
    if betti_png.exists():
        html.append(
            f"<div class='card panel-2col'><h3>Persistent-homology Betti collapse</h3>"
            f"<img src='{betti_png.name}'/></div>"
        )
    for sec_title, body in extra_sections:
        html.append(
            f"<div class='card panel-2col'><h3>{sec_title}</h3>{body}</div>"
        )

    # Hypothesis status grid (71-cell heatmap)
    html.append(
        "<div class='card panel-2col'><h3>Hypothesis ledger — 71-cell status grid "
        "(G1..G7 × hypothesis index; click to view the spec)</h3>"
        + _hypothesis_grid_html(index_rows, tag_to_tier, reasoning_by_tag)
        + "</div>"
    )
    html.append("</div>")  # end .grid

    # ------------------------------------------------------------------
    # Runs table with tier chips, filter, click-modal, <details> rows
    # ------------------------------------------------------------------
    html.append("<h2 style='margin-top:30px'>Runs — click a row for reasoning ledger; "
                "▸ for per-run training-curve + latency breakdown</h2>")

    tiers_present = sorted({m["tier"] for m in tag_to_tier.values()})
    html.append("<div>" + _tier_chips_html(tiers_present) + "</div>")
    html.append(
        "<input id='q' class='filter' placeholder='filter runs (tag/dataset/seed)…' "
        "oninput='applyFilters()'/>"
    )

    cols = [
        ("tag", "Tag", False),
        ("dataset", "Dataset", False),
        ("seed", "Seed", True),
        ("epochs", "Epochs", True),
        ("top1", "Top-1", True),
        ("top5", "Top-5", True),
        ("params", "Params", True),
        ("flops", "FLOPs", True),
        ("latency_ms", "Latency", True),
        ("rot_eq_err", "Rot-eq err", True),
        ("composite", "Composite", True),
        ("epochs_to_target", "ET(target)", True),
        ("train_seconds", "Train (s)", True),
    ]
    html.append("<table id='runs' data-dir='asc'><thead><tr>")
    html.append("<th style='width:24px'></th>")
    for i, (_k, label, _n) in enumerate(cols):
        html.append(f"<th onclick='sortTable({i + 1})'>{label}</th>")
    html.append("<th>Tier</th>")
    html.append("</tr></thead><tbody>")

    for _, r in df.iterrows():
        tag = r.get("tag", "")
        klass = "best-row" if r.name in best_idx else ""
        tier_meta = tag_to_tier.get(tag, {})
        tier = tier_meta.get("tier", "")
        status = tier_meta.get("status", "")
        html.append(
            f"<tr class='clickable {klass}' data-tier='{tier}' "
            f"onclick=\"openReasoning('{tag}')\">"
        )
        html.append(
            f"<td onclick='event.stopPropagation();toggleDetail(this,\"{tag}\")' "
            f"style='cursor:pointer;color:#58a6ff;font-weight:600;text-align:center'>▸</td>"
        )
        run_dir_name = str(r.get("_run_dir", "")).split("/")[-1]
        page_href = f"experiments/{run_page_filename(run_dir_name)}" if run_dir_name else ""
        for k, _l, _n in cols:
            v = r.get(k, "")
            if k == "tag":
                pill = f"<span class='tag-pill'>{v}</span>"
                if page_href:
                    pill = (
                        f"<a href='{page_href}' title='open per-experiment page' "
                        f"onclick='event.stopPropagation()'>{pill}</a>"
                    )
                disp = pill
                if status:
                    disp = (
                        f"<span class='swatch status-{status}' "
                        f"style='display:inline-block;width:8px;height:8px;"
                        f"border-radius:50%;margin-right:6px;vertical-align:middle'>"
                        f"</span>" + disp
                    )
                html.append(f"<td data-v='{v}'>{disp}</td>")
                continue
            if k == "params":
                try:
                    disp = f"{float(v)/1e6:.3f}M"
                except Exception:
                    disp = str(v)
            elif k == "flops":
                try:
                    disp = f"{float(v)/1e6:.1f}M"
                except Exception:
                    disp = str(v)
            elif k in ("top1", "top5"):
                try:
                    disp = f"{float(v)*100:.2f}%"
                except Exception:
                    disp = str(v)
            elif k == "latency_ms":
                try:
                    disp = f"{float(v):.2f} ms"
                except Exception:
                    disp = str(v)
            elif isinstance(v, float):
                disp = f"{v:.4f}"
            else:
                disp = str(v)
            html.append(f"<td data-v='{v}'>{disp}</td>")
        html.append(
            f"<td data-v='{tier}'><span class='chip'>{tier or '—'}</span></td>"
        )
        html.append("</tr>")

        # Detail row (lazy-on-DOM; hidden by default)
        hist = history.get(tag) or history.get(r.get("_run_dir", "").split("/")[-1])
        spark = _sparkline_svg(hist)
        flags = r.get("flags") or {}
        flag_chips = ""
        if isinstance(flags, dict):
            for fk, fv in flags.items():
                color = "#3fb950" if fv else "#484f58"
                flag_chips += (
                    f"<span class='chip' style='border-color:{color};color:{color}'>"
                    f"{fk}={fv}</span>"
                )
        try:
            lat = float(r.get("latency_ms", 0))
            train_s = float(r.get("train_seconds", 0))
        except Exception:
            lat, train_s = 0.0, 0.0
        gen_gap = r.get("generalization_gap", "")
        try:
            gen_gap_disp = f"{float(gen_gap)*100:.2f} pp"
        except Exception:
            gen_gap_disp = str(gen_gap)

        html.append(
            "<tr class='detail-row' style='display:none'>"
            f"<td colspan='{len(cols) + 2}'>"
            "<div class='detail-grid'>"
            "<div class='detail-block'>"
            "<div class='lbl'>Training curve (epoch → test top-1)</div>"
            f"<div>{spark or '<i style=\"color:#8b949e\">no history.json</i>'}</div>"
            "</div>"
            "<div class='detail-block'>"
            "<div class='lbl'>Latency / compute breakdown</div>"
            f"<div>latency (batch=1): <b>{lat:.3f} ms</b><br>"
            f"train wall-clock: <b>{train_s:.1f} s</b><br>"
            f"generalisation gap (train−test top1): <b>{gen_gap_disp}</b><br>"
            f"model: <b>{r.get('model','?')}</b> · channel_mode: <b>{r.get('channel_mode','?')}</b></div>"
            "</div>"
            "<div class='detail-block panel-2col' style='grid-column:1/3'>"
            "<div class='lbl'>Prior flags</div>"
            f"<div>{flag_chips or '<i style=\"color:#8b949e\">no flags recorded</i>'}</div>"
            "</div>"
            "</div>"
            "</td></tr>"
        )
    html.append("</tbody></table>")

    html.append(
        "<div class='meta'>Generated by "
        "<code>nature_inspired_networks.dashboard.render_dashboard</code> · "
        "dark theme adapted from dlmastery/autoresearchspy/docs/spy_dashboard · "
        "all CSS+JS inline; no external CDN; PNG plots kept as PNG.</div>"
    )

    # Modal + side panel HTML
    html.append("""
<div id='modal-backdrop' onclick='closeModal()'></div>
<div id='modal'>
  <button class='close-btn' onclick='closeModal()'>close · Esc</button>
  <h3 id='modal-title'>—</h3>
  <div class='meta' id='modal-meta'></div>
  <div id='modal-body'></div>
</div>
<div id='side-panel'>
  <button class='close-btn' onclick='closeSide()'>close · Esc</button>
  <h3 id='side-panel-title'>Hypothesis spec</h3>
  <pre id='side-panel-body'>—</pre>
</div>
""")

    # Embed reasoning_by_tag as a JS literal so click-handler is offline-safe
    reasoning_js = json.dumps(reasoning_by_tag, ensure_ascii=False)
    html.append(
        f"<script>window._REASONING_BY_TAG = {reasoning_js};</script>"
    )
    html.append(HTML_JS)
    html.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html), encoding="utf-8")
