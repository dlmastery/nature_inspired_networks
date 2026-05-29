"""Autoresearch-style static dashboard generator (group-organised edition).

Reads experiment_log.jsonl, per-experiment metrics.json + history.json,
reasoning_annotations.json, betti.json, IDEA_TABLE.md, EXPERIMENT_LOG.md,
hypotheses/INDEX.md + hypotheses/g<N>_*/H<NN>_*.md, and FINDINGS.md to emit a
single self-contained dark-theme HTML dashboard plus rich per-experiment
pages.

The 2026-05-27 rewrite (issue feedback) removes the modal entirely, sections
the leaderboard by hypothesis group, and expands each per-experiment page
with hypothesis digest, FINDINGS verdict, reasoning blob, configuration,
metrics, composite breakdown, training curves, cross-references, and footer.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Baseline reference accuracies for per-group SVG visualisations
# ---------------------------------------------------------------------------
BASELINE_TOP1 = {
    "cifar10": 0.8478,   # baseline_resnet20 seed=0 (FINDINGS.md)
    "cifar100": 0.5652,  # baseline_resnet20 3-seed median (FINDINGS.md)
}


# ---------------------------------------------------------------------------
# Tag -> (hypothesis ID, group letter) mapping
#
# Canonical, exhaustive map covering every tag that has produced a sweep row
# to date (see ``experiments/cifar10/*`` and ``experiments/cifar100/*``).
# Unknown tags fall back to ("Uncategorised", "Uncategorized").
# ---------------------------------------------------------------------------

TAG_TO_HYP: dict[str, tuple[str | None, str]] = {
    # Baselines (no hypothesis doc)
    "baseline_resnet20": (None, "Baseline"),
    "baseline_sg_vanilla": (None, "Baseline"),
    # G1 Scaling & Growth
    "sg_chan_fib": ("H04", "G1"),
    "sg_chan_phi": ("H04", "G1"),
    "sg_only_phi_compound": ("H01", "G1"),
    "sg_only_fib_depth": ("H02", "G1"),
    "sg_only_golden_resize": ("H03", "G1"),
    "sg_only_fractal": ("H05", "G1"),
    "sg_only_golden_bottleneck": ("H06", "G1"),
    "sg_only_phi_multiscale": ("H07", "G1"),
    "sg_only_phi_budget": ("H09", "G1"),
    "sg_only_phi_lr": ("H10", "G1"),
    # G2 Layer / Channel / Neuron
    "sg_only_phi_sparse": ("H13", "G2"),
    "sg_only_golden_skip": ("H17", "G2"),
    "sg_only_fib_stride": ("H18", "G2"),
    "sg_only_phi_relu": ("H19", "G2"),
    "sg_only_fib_ensemble": ("H20", "G2"),
    "sg_only_golden_modulate": ("H17", "G2"),
    # G3 Topologies / Graphs
    "sg_only_hex": ("H21", "G3"),
    "sg_only_toroidal": ("H22", "G3"),
    # G4 Kernels / Attention
    "sg_only_golden_spiral_init": ("H31", "G4"),
    "sg_only_cymatic_init": ("H35", "G4"),
    "sg_only_phi_activation": ("H39", "G4"),
    # G5 Optimisation / Init / Regularisation
    "sg_only_golden_adam": ("H41", "G5"),
    "sg_only_phi_init": ("H42", "G5"),
    "sg_only_fib_prune": ("H43", "G5"),
    "sg_only_phi_decay": ("H44", "G5"),
    "sg_only_phi_dropout": ("H47", "G5"),
    "sg_only_golden_momentum": ("H48", "G5"),
    "sg_full_fib": ("H50", "G5"),
    "sg_full_fib_avg": ("H50", "G5"),
    # G6 Topological & Bridging
    "sg_only_group": ("H58", "G6"),
    "sg_only_group_avg": ("H58", "G6"),
    # G8 Esoteric Extensions
    "sg_only_constant_width": ("H80", "G8"),
    "sg_only_sine_act": ("H81", "G8"),
}


GROUP_ORDER: list[str] = [
    "Baseline", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "Uncategorized",
]

GROUP_HEADERS: dict[str, tuple[str, str]] = {
    "Baseline": (
        "Baselines",
        "Stock ResNet-20 plus the all-priors-off NaturePriorNet scaffold — "
        "the controls every hypothesis is judged against.",
    ),
    "G1": (
        "G1 — Scaling & Growth",
        "φ / Fibonacci growth laws applied to depth, width, resolution, "
        "parameter budgets, and learning-rate schedulers (H01–H10).",
    ),
    "G2": (
        "G2 — Layer / Channel / Neuron",
        "Fibonacci-sized MLPs, channel counts, neuron connectivity, "
        "skip scaling, head diversity, activation thresholds, ensembles "
        "(H11–H20).",
    ),
    "G3": (
        "G3 — Topologies & Graphs",
        "Hexagonal lattices, toroidal closures, Platonic / icosahedral / "
        "dodecahedral equivariance, fractal toroidal, cymatic-hex resonance "
        "(H21–H30).",
    ),
    "G4": (
        "G4 — Kernels / Attention / Filters",
        "Golden-spiral kernels, Fibonacci dilation, vesica-piscis filters, "
        "golden-angle rotary, cymatic wavelets, harmonic activations "
        "(H31–H40).",
    ),
    "G5": (
        "G5 — Optimisation / Init / Regularisation / NAS",
        "Golden-ratio AdamW, φ-weight init, Fibonacci pruning, φ-dropout, "
        "golden-momentum schedulers, full-hybrid stacks (H41–H50).",
    ),
    "G6": (
        "G6 — Topological & Bridging",
        "Persistent-homology losses, drop-path anytime nets, icosa-unfold "
        "bridges, C4-group avg-pool, trained-feature Betti probes "
        "(H51–H60).",
    ),
    "G7": (
        "G7 — Cross-Paradigm Hybrids",
        "Liquid / JEPA / KAN / Transformer / GNN cross-paradigm hybrids "
        "(H61–H75). No CIFAR sweep rows yet — placeholder section.",
    ),
    "G8": (
        "G8 — Esoteric Extensions",
        "Reuleaux constant-width kernels, SIREN sinusoidal activations, "
        "tetrahedral dual paths, radial-12 attention, spectral Hopfield "
        "(H76–H84).",
    ),
    "Uncategorized": (
        "Uncategorised",
        "Tags whose hypothesis mapping is not yet wired into "
        "TAG_TO_HYP — please update dashboard.py.",
    ),
}


def hyp_file_for(hid: str) -> str | None:
    """Resolve ``H<NN>`` to its ``g<N>_*/H<NN>_*.md`` filename (relative).

    Returns ``None`` if the docs cannot be located.
    """
    if not hid:
        return None
    try:
        n = int(hid[1:])
    except Exception:
        return None
    group_idx = (n - 1) // 10 + 1
    if n >= 76:
        group_idx = 8
    subdirs = {
        1: "g1_scaling_growth",
        2: "g2_layer_channel_neuron",
        3: "g3_topologies_graphs",
        4: "g4_kernels_attention_filters",
        5: "g5_optimization_init_reg_nas",
        6: "g6_topological_bridging",
        7: "g7_cross_paradigm_hybrids",
        8: "g8_esoteric_extensions",
    }
    return f"{subdirs.get(group_idx, '')}/{hid}_*.md"


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
# Markdown helpers — hypothesis index + experiment-log tiers + findings
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
        m = re.match(r"\|\s*(H\d{2})\s*\|\s*`([^`]+)`\s*\|", line)
        if m and current_group:
            hid = m.group(1)
            fname = m.group(2)
            try:
                idx_in_group = int(hid[1:]) - (int(current_group[1:]) - 1) * 10
            except Exception:
                idx_in_group = 0
            rows.append({
                "id": hid,
                "group": current_group,
                "idx": idx_in_group,
                "file": fname,
                "idea": "",
            })
    return rows


def parse_experiment_log_tiers(log_md: str | Path) -> dict[str, dict]:
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
            tag = None
            for c in cells:
                bm = re.search(r"`([^`]+)`", c)
                if bm:
                    tag = bm.group(1)
                    break
            if not tag:
                continue
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


def parse_hypothesis_doc(md_path: Path) -> dict[str, str]:
    """Extract a digest from a committee-grade hypothesis Markdown file.

    Returns a dict with keys: ``title``, ``oneline``, ``motivation``,
    ``formal``, ``falsifier``, ``mechanism``, ``citation``, ``predicted``.
    Each value is best-effort; missing sections come back as empty strings.
    """
    out = {
        "title": "", "oneline": "", "motivation": "", "formal": "",
        "falsifier": "", "mechanism": "", "citation": "", "predicted": "",
    }
    if not md_path or not md_path.exists():
        return out
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Title (first H1)
    for ln in lines:
        m = re.match(r"#\s+(.+)", ln)
        if m:
            out["title"] = m.group(1).strip()
            break

    # One-line claim: first blockquote with "One-line claim:"
    m = re.search(
        r">\s*\*\*One-line claim:\*\*\s*(.+?)(?:\n>\s*\n|\n\n)",
        text, re.DOTALL,
    )
    if m:
        oneline = re.sub(r"\n>\s*", " ", m.group(1)).strip()
        out["oneline"] = re.sub(r"\s+", " ", oneline)

    # Section helper: from a heading regex to the next ## heading.
    def section(pattern: str) -> str:
        m = re.search(
            rf"##\s+\d*\.?\s*{pattern}.*?\n(.+?)(?=\n##\s|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return ""
        body = m.group(1).strip()
        # Strip code fences
        body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        return body.strip()

    motivation = section(r"Motivation")
    if motivation:
        # First paragraph only
        first_para = motivation.split("\n\n", 1)[0]
        out["motivation"] = re.sub(r"\s+", " ", first_para).strip()

    formal = section(r"Formal hypothesis")
    if formal:
        first_para = formal.split("\n\n", 1)[0]
        out["formal"] = re.sub(r"\s+", " ", first_para).strip()

    falsifier = section(r"Falsifier")
    if falsifier:
        first_para = falsifier.split("\n\n", 1)[0]
        out["falsifier"] = re.sub(r"\s+", " ", first_para).strip()

    citations = section(r"Citations")
    if citations:
        # Take the first arXiv reference
        m = re.search(
            r"([A-Z][^.\n]+?\(arXiv:\d{4}\.\d{4,5}\)[^.\n]*(?:\.|--)[^.\n]*)",
            citations,
        )
        if m:
            out["citation"] = re.sub(r"\s+", " ", m.group(1)).strip()
        else:
            # Fallback: first non-empty line
            for ln in citations.splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("```"):
                    out["citation"] = ln
                    break

    mechanism = section(r"Mechanism")
    if mechanism:
        # First paragraph
        for para in mechanism.split("\n\n"):
            cleaned = re.sub(r"\s+", " ", para).strip()
            if cleaned and not cleaned.startswith("###"):
                out["mechanism"] = cleaned[:600]
                break

    # Predicted Δ section can be named various things — try a few.
    for pat in (r"Predicted\s*Δ", r"Predicted Delta", r"Predictions?"):
        body = section(pat)
        if body:
            first_para = body.split("\n\n", 1)[0]
            out["predicted"] = re.sub(r"\s+", " ", first_para).strip()
            break

    return out


def find_hypothesis_path(repo_root: Path, hid: str) -> Path | None:
    """Locate ``hypotheses/g<N>_*/H<NN>_*.md`` for a given hypothesis ID."""
    if not hid:
        return None
    base = repo_root / "hypotheses"
    for sub in sorted(base.glob("g*_*/")):
        for f in sub.glob(f"{hid}_*.md"):
            return f
    return None


def parse_audit_verdict(repo_root: Path, hid: str, group: str) -> dict[str, str]:
    """Parse ``audits/G<N>_audit.md`` for a specific hypothesis verdict.

    Returns a dict with keys: ``verdict`` (PASS/MINOR/MAJOR/BROKEN/""),
    ``summary`` (first sentence of audit body), ``url`` (GitHub link),
    ``local`` (relative local link). All empty if missing.
    """
    out = {"verdict": "", "summary": "", "url": "", "local": ""}
    if not hid or not group or not group.startswith("G"):
        return out
    audit_md = repo_root / "audits" / f"{group}_audit.md"
    if not audit_md.exists():
        return out
    try:
        text = audit_md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    # Find the hypothesis subsection block: ### H09 — ... up to next ### or end.
    m = re.search(
        rf"^###\s+{re.escape(hid)}\b.*?\n(.+?)(?=^###\s+H\d{{2}}\b|\Z)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not m:
        return out
    block = m.group(1)
    vm = re.search(r"\*\*Verdict:\*\*\s*([A-Z]+(?:\s+\([^)]+\))?)", block)
    if vm:
        # Keep only the first all-caps word as the canonical verdict.
        raw = vm.group(1).strip()
        kw = re.match(r"([A-Z]+)", raw)
        out["verdict"] = kw.group(1) if kw else raw
    # First non-empty paragraph after Verdict line as a 1-sentence summary.
    sm = re.search(r"\*\*Mechanism check:\*\*\s*(.+?)(?:\n\n|\Z)",
                   block, re.DOTALL)
    if sm:
        summary = re.sub(r"\s+", " ", sm.group(1)).strip()
        out["summary"] = summary[:400]
    rel = audit_md.relative_to(repo_root).as_posix()
    out["local"] = rel
    out["url"] = (
        f"https://github.com/dlmastery/nature_inspired_networks/blob/main/"
        f"{rel}#{hid.lower()}-{group.lower()}"
    )
    return out


def parse_scicritic_verdict(md_path: Path | None) -> dict[str, str]:
    """Parse the trailing 'Addendum: Research-Scientist Critique' verdict.

    Returns dict with keys: ``verdict`` (NOVEL/DERIVATIVE/NUMEROLOGY/
    FALSIFIED/UNFALSIFIABLE/INFRASTRUCTURE/""), ``raw`` (first sentence of
    the verdict block), ``found`` (bool).
    """
    out = {"verdict": "", "raw": "", "found": False}
    if md_path is None or not md_path.exists():
        return out
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    # Match "### Verdict" inside the addendum, capture body to next "##" or EOF
    m = re.search(
        r"##\s+Addendum:\s*Research-Scientist.*?###\s+Verdict\s*\n(.+?)(?=\n##\s|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return out
    body = m.group(1).strip()
    out["found"] = True
    # First all-caps token, possibly with "+TESTABLE" / "+CONFIRMED" suffix.
    vm = re.search(
        r"\*?\*?(NOVEL|DERIVATIVE|NUMEROLOGY|FALSIFIED|UNFALSIFIABLE|"
        r"INFRASTRUCTURE)\b(?:\+[A-Z]+)?",
        body,
    )
    if vm:
        out["verdict"] = vm.group(0).replace("*", "").strip()
    # First sentence-ish chunk for tooltip / inline display.
    cleaned = re.sub(r"\s+", " ", body).strip()
    out["raw"] = cleaned[:420]
    return out


def _git_short_sha(repo_root: Path, target: Path | None) -> str:
    """Return the short SHA of the last commit touching ``target``.

    Falls back to repo HEAD if the target is missing from git history,
    or empty string if git is not available.
    """
    try:
        if target is not None and target.exists():
            try:
                rel = target.relative_to(repo_root).as_posix()
            except Exception:
                rel = str(target)
            r = subprocess.run(
                ["git", "log", "-1", "--format=%h", "--", rel],
                cwd=str(repo_root), capture_output=True, text=True, timeout=5,
            )
            sha = r.stdout.strip()
            if sha:
                return sha
        # Fallback: HEAD
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def parse_findings_for_tag(findings_md: Path, tag: str) -> str:
    """Extract a short verdict blurb for the given tag from FINDINGS.md."""
    if not findings_md.exists() or not tag:
        return ""
    text = findings_md.read_text(encoding="utf-8", errors="ignore")
    # Strategy: find each paragraph (separated by blank lines) that mentions
    # the tag, then return the first one that includes verdict-y language.
    # Wrap-around: also accept the tag without the leading ``sg_only_`` prefix.
    short = tag.replace("sg_only_", "").replace("sg_", "")
    paragraphs = re.split(r"\n\s*\n", text)
    candidates: list[str] = []
    for para in paragraphs:
        if f"`{tag}`" in para or short in para:
            cleaned = re.sub(r"\s+", " ", para).strip()
            if len(cleaned) > 30:
                candidates.append(cleaned)
    if not candidates:
        return ""
    # Prefer ones with verdict markers.
    verdict_terms = (
        "verdict", "DISCARD", "SURVIVES", "DEMOTED", "graduate", "neutral",
        "falsified", "winner", "lead", "negative", "positive",
    )
    for c in candidates:
        if any(t.lower() in c.lower() for t in verdict_terms):
            return c[:1200]
    return candidates[0][:1200]


# ---------------------------------------------------------------------------
# PNG plots (unchanged from previous edition)
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
# Inline SVG sparkline
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


def _esc(s: object) -> str:
    out = str(s)
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 ('"', "&quot;"), ("'", "&#39;")):
        out = out.replace(a, b)
    return out


def _line_chart_svg(series: list[tuple[str, list[float], list[float], str]],
                    title: str, y_label: str,
                    w: int = 460, h: int = 240,
                    y_as_pct: bool = False) -> str:
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


# ---------------------------------------------------------------------------
# Per-experiment page
# ---------------------------------------------------------------------------

def run_page_filename(run_dir_name: str, dataset: str | None = None) -> str:
    """Return ``<dataset>__<run_dir_name>.html`` so cifar10 + cifar100 don't collide.

    Older callers that only know ``run_dir_name`` get a plain ``<name>.html``;
    that path is only used for cross-references discovered via globbing the
    experiments directory.
    """
    if dataset:
        return f"{dataset}__{run_dir_name}.html"
    return f"{run_dir_name}.html"


_EXP_PAGE_CSS = """
 *{margin:0;padding:0;box-sizing:border-box;}
 body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;
      color:#c9d1d9;padding:24px 28px;line-height:1.55;max-width:1180px;margin:0 auto;}
 a{color:#58a6ff;text-decoration:none;} a:hover{text-decoration:underline;}
 h1{color:#58a6ff;font-size:1.6em;margin-bottom:4px;}
 h2{color:#58a6ff;font-size:1.05em;margin-bottom:12px;font-weight:600;
    letter-spacing:0.3px;}
 .sub{color:#8b949e;font-size:0.95em;margin-bottom:18px;}
 .pill{display:inline-block;background:#21262d;border:1px solid #30363d;
       border-radius:10px;padding:2px 10px;font-size:0.78em;color:#c9d1d9;
       font-family:Consolas,monospace;margin-right:6px;}
 .pill.hyp{background:#1f3a5f;border-color:#58a6ff;color:#cfe6ff;}
 .pill.grp{background:#2c1e3a;border-color:#a371f7;color:#dbc9f7;}
 .back{display:inline-block;margin-bottom:14px;font-size:0.9em;}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;
       padding:18px 22px;margin-bottom:18px;}
 .card h2{color:#58a6ff;font-size:1.05em;margin-bottom:12px;font-weight:600;}
 .card h3{color:#c9d1d9;font-size:0.92em;margin:14px 0 6px 0;font-weight:600;
          text-transform:uppercase;letter-spacing:0.5px;}
 .card p{margin-bottom:10px;font-size:0.92em;}
 .card ul{margin-left:22px;font-size:0.9em;}
 table{width:100%;border-collapse:collapse;font-size:0.86em;}
 th{background:#0d1117;color:#8b949e;text-align:left;padding:7px 10px;
    border-bottom:2px solid #30363d;font-size:0.74em;text-transform:uppercase;
    letter-spacing:0.4px;}
 td{padding:7px 10px;border-bottom:1px solid #21262d;}
 td.k{color:#8b949e;width:42%;}
 td.v{color:#c9d1d9;font-family:Consolas,monospace;}
 .formula-chip{background:#0d1117;border:1px solid #30363d;border-radius:6px;
               padding:10px 12px;font-family:Consolas,monospace;font-size:0.82em;
               margin-bottom:12px;color:#c9d1d9;word-break:break-all;}
 .breakdown td.term{font-family:Consolas,monospace;}
 .pos{color:#3fb950;} .neg{color:#f85149;} .mut{color:#8b949e;}
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
 @media(max-width:980px){.charts{grid-template-columns:1fr;}}
 .meta{font-size:0.78em;color:#484f58;margin-top:18px;line-height:1.6;}
 code{background:#0d1117;padding:1px 5px;border-radius:3px;font-size:0.92em;
      font-family:Consolas,monospace;}
 pre{background:#0d1117;border:1px solid #30363d;border-radius:6px;
     padding:12px;overflow-x:auto;font-family:Consolas,monospace;
     font-size:0.84em;color:#c9d1d9;white-space:pre-wrap;line-height:1.5;}
 .reason-section{margin-bottom:14px;}
 .reason-section .lbl{color:#3fb950;font-size:0.72em;text-transform:uppercase;
                      letter-spacing:0.6px;margin-bottom:4px;font-weight:600;}
 .reason-section .body{font-size:0.9em;color:#c9d1d9;white-space:pre-wrap;}
 .quote{border-left:3px solid #d29922;padding:8px 14px;background:#0d1117;
        border-radius:0 6px 6px 0;font-size:0.9em;color:#c9d1d9;
        margin:8px 0;}
 .verdict-label{color:#d29922;font-weight:600;}
 .empty{color:#8b949e;font-style:italic;font-size:0.88em;}
 .xrefs li{margin:4px 0;font-size:0.9em;font-family:Consolas,monospace;}
"""


def _load_experiment_log_row(experiment_log: Path,
                             tag: str, seed, dataset: str) -> dict | None:
    """Return the latest matching jsonl row for (tag, seed, dataset)."""
    if not experiment_log.exists():
        return None
    try:
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = None
        match: dict | None = None
        with experiment_log.open("r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or not ln.startswith("{"):
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if row.get("tag") != tag:
                    continue
                if dataset and row.get("dataset") != dataset:
                    continue
                if seed_int is not None and row.get("seed") != seed_int:
                    continue
                match = row  # keep last (newest) match
        return match
    except Exception:
        return None


# Two-tone palette for hypothesis verdict badges.
_VERDICT_COLORS = {
    "PASS":      ("#3fb950", "#0d2818"),
    "MINOR":     ("#d29922", "#2c220a"),
    "MAJOR":     ("#f0883e", "#3a1f0c"),
    "BROKEN":    ("#f85149", "#3a0e0e"),
    "NOVEL":     ("#3fb950", "#0d2818"),
    "DERIVATIVE": ("#58a6ff", "#0f1d33"),
    "NUMEROLOGY": ("#f85149", "#3a0e0e"),
    "FALSIFIED":  ("#f85149", "#3a0e0e"),
    "UNFALSIFIABLE": ("#a371f7", "#1c142e"),
    "INFRASTRUCTURE": ("#8b949e", "#21262d"),
}


def _verdict_badge(label: str, verdict: str, url: str = "",
                   tooltip: str = "") -> str:
    if not verdict:
        return ""
    base = verdict.split("+")[0].strip()
    fg, bg = _VERDICT_COLORS.get(base, ("#c9d1d9", "#21262d"))
    title = _esc(tooltip) if tooltip else _esc(verdict)
    inner = (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:10px;"
        f"background:{bg};border:1px solid {fg};color:{fg};font-weight:600;"
        f"font-size:0.78em;font-family:Consolas,monospace' "
        f"title='{title}'>{_esc(label)}: {_esc(verdict)}</span>"
    )
    if url:
        return (
            f"<a href='{_esc(url)}' style='text-decoration:none;"
            f"margin-right:6px'>{inner}</a>"
        )
    return f"<span style='margin-right:6px'>{inner}</span>"


# ---------------------------------------------------------------------------
# Per-group SVG visualisation for the aggregate dashboard
# ---------------------------------------------------------------------------

def _group_visualisation_svg(rows: list[pd.Series], group_letter: str,
                             chart_id: str, width: int = 960,
                             height: int = 280) -> str:
    """Render a compact horizontal-bar SVG chart for a hypothesis group.

    Each row becomes one bar (top-1 %). Baseline reference is overlaid as a
    vertical dashed line per dataset.
    """
    if not rows:
        return ""
    bars: list[dict] = []
    for r in rows:
        try:
            top1 = float(r.get("top1"))
        except Exception:
            continue
        bars.append({
            "tag": str(r.get("tag", ""))[:30],
            "seed": r.get("seed", ""),
            "dataset": str(r.get("dataset", "")),
            "top1": top1,
            "params": float(r.get("params", 0) or 0),
            "composite": float(r.get("composite", 0) or 0),
            "run_dir": str(r.get("_run_dir", "")),
        })
    if not bars:
        return ""
    # Sort descending by top-1 so the best is at the top.
    bars.sort(key=lambda d: d["top1"], reverse=True)
    # Cap to 20 bars to keep chart compact.
    bars = bars[:20]

    # Layout
    ml, mr, mt, mb = 200, 60, 28, 28
    pw = width - ml - mr
    bh = 14
    gap = 6
    ph = len(bars) * (bh + gap)
    h = mt + mb + ph + 12  # +12 for baseline legend
    # x-axis: from 0 to max(top1, baseline) * 1.05
    datasets_in_group = {b["dataset"] for b in bars}
    baselines = [BASELINE_TOP1[d] for d in datasets_in_group if d in BASELINE_TOP1]
    xmax = max([b["top1"] for b in bars] + baselines + [0.5]) * 1.06
    xmin = 0.0

    def sx(x: float) -> float:
        return ml + pw * (x - xmin) / (xmax - xmin)

    parts: list[str] = [
        f"<svg viewBox='0 0 {width} {h}' preserveAspectRatio='xMinYMin meet' "
        f"xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%;height:auto;background:#0d1117;border:1px solid "
        f"#30363d;border-radius:6px;display:block;margin-bottom:8px' "
        f"id='{chart_id}'>",
        f"<text x='{ml}' y='18' fill='#c9d1d9' font-size='12' "
        f"font-weight='600'>Top-1 by tag · {_esc(group_letter)} "
        f"(n={len(bars)})</text>",
    ]
    # Grid + x-axis ticks at 25 %, 50 %, 75 % of xmax
    for frac in (0.25, 0.5, 0.75, 1.0):
        xv = xmin + (xmax - xmin) * frac
        xx = sx(xv)
        parts.append(
            f"<line x1='{xx:.1f}' y1='{mt}' x2='{xx:.1f}' "
            f"y2='{mt + ph}' stroke='#21262d' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{xx:.1f}' y='{mt + ph + 14}' fill='#8b949e' "
            f"font-size='9' text-anchor='middle'>{xv*100:.0f}%</text>"
        )
    # Baseline reference line(s)
    bl_colors = {"cifar10": "#d29922", "cifar100": "#f0883e"}
    legend_x = ml
    for ds in sorted(datasets_in_group):
        if ds not in BASELINE_TOP1:
            continue
        bx = sx(BASELINE_TOP1[ds])
        col = bl_colors.get(ds, "#d29922")
        parts.append(
            f"<line x1='{bx:.1f}' y1='{mt - 4}' x2='{bx:.1f}' "
            f"y2='{mt + ph + 4}' stroke='{col}' stroke-width='1.6' "
            f"stroke-dasharray='4,3' opacity='0.85'/>"
        )
        parts.append(
            f"<text x='{bx + 4:.1f}' y='{mt - 4}' fill='{col}' "
            f"font-size='9'>{_esc(ds)} baseline {BASELINE_TOP1[ds]*100:.1f}%</text>"
        )
    # Bars + labels
    palette = {
        "cifar10": "#58a6ff",
        "cifar100": "#a371f7",
    }
    for i, b in enumerate(bars):
        y = mt + i * (bh + gap)
        bx0 = ml
        bx1 = sx(b["top1"])
        col = palette.get(b["dataset"], "#58a6ff")
        ds_tag = f"{b['tag']} · {b['dataset']} · seed{b['seed']}"
        parts.append(
            f"<rect x='{bx0:.1f}' y='{y}' width='{bx1 - bx0:.1f}' "
            f"height='{bh}' fill='{col}' opacity='0.85' rx='2'>"
            f"<title>{_esc(ds_tag)} · top-1 {b['top1']*100:.2f}% · "
            f"params {b['params']/1e6:.3f}M · composite {b['composite']:.4f}</title>"
            f"</rect>"
        )
        parts.append(
            f"<text x='{ml - 6}' y='{y + bh - 3}' fill='#c9d1d9' "
            f"font-size='10' text-anchor='end' "
            f"font-family='Consolas,monospace'>{_esc(b['tag'][:24])}</text>"
        )
        parts.append(
            f"<text x='{bx1 + 4:.1f}' y='{y + bh - 3}' fill='#8b949e' "
            f"font-size='9'>{b['top1']*100:.2f}%</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _nearest_in_group(df: pd.DataFrame, group_letter: str, this_tag: str,
                      this_dataset: str, this_composite: float,
                      k: int = 3) -> list[dict]:
    """Return up to k nearest-by-composite tags inside the same group/dataset."""
    if df is None or df.empty:
        return []
    sub = df[df["__group"] == group_letter] if "__group" in df.columns else None
    if sub is None or sub.empty:
        return []
    sub = sub[sub["dataset"] == this_dataset]
    sub = sub[sub["tag"] != this_tag]
    if sub.empty:
        return []
    sub = sub.copy()
    try:
        sub["__dist"] = (sub["composite"].astype(float) - float(this_composite)).abs()
    except Exception:
        return []
    sub = sub.sort_values("__dist").head(k)
    out: list[dict] = []
    for _, r in sub.iterrows():
        out.append({
            "tag": r.get("tag", ""),
            "seed": r.get("seed", ""),
            "dataset": r.get("dataset", ""),
            "composite": float(r.get("composite", 0) or 0),
            "top1": float(r.get("top1", 0) or 0),
            "run_dir": str(r.get("_run_dir", "")),
        })
    return out


def _render_hypothesis_section(hid: str | None, group: str,
                               repo_root: Path) -> str:
    """Build the inline hypothesis digest block for a per-experiment page."""
    if not hid:
        return (
            "<div class='card'><h2>Hypothesis</h2>"
            "<p class='empty'>No matching hypothesis document — this row "
            "is a baseline reference.</p></div>"
        )
    md = find_hypothesis_path(repo_root, hid)
    if md is None:
        return (
            f"<div class='card'><h2>Hypothesis {_esc(hid)}</h2>"
            f"<p class='empty'>(hypothesis doc not located on disk for "
            f"{_esc(hid)})</p></div>"
        )
    digest = parse_hypothesis_doc(md)
    rel = md.relative_to(repo_root).as_posix()
    gh_url = f"https://github.com/dlmastery/nature_inspired_networks/blob/main/{rel}"
    sci = parse_scicritic_verdict(md)
    parts: list[str] = [
        f"<div class='card'><h2>Hypothesis {_esc(hid)} — {_esc(digest['title'] or hid)}</h2>"
    ]
    # Prominent GitHub design-doc CTA up front + sci-critic badge inline
    cta_bits = [
        f"<a href='{_esc(gh_url)}' style='display:inline-block;background:"
        f"#1f6feb;color:#fff;padding:6px 14px;border-radius:6px;font-weight:600;"
        f"font-size:0.86em;border:1px solid #58a6ff;text-decoration:none;"
        f"margin-right:8px'>&#128196; Hypothesis design doc &rarr;</a>"
    ]
    if sci["verdict"]:
        cta_bits.append(_verdict_badge(
            "sci-critic", sci["verdict"], url=gh_url, tooltip=sci["raw"],
        ))
    parts.append(
        f"<div style='margin:4px 0 14px 0'>{''.join(cta_bits)}</div>"
    )
    if digest["oneline"]:
        parts.append(
            f"<p><b style='color:#d29922'>One-line claim:</b> "
            f"<i>{_esc(digest['oneline'])}</i></p>"
        )
    if digest["motivation"]:
        parts.append(
            f"<h3>Motivation</h3><p>{_esc(digest['motivation'])}</p>"
        )
    if digest["formal"]:
        parts.append(
            f"<h3>Formal hypothesis</h3><p>{_esc(digest['formal'])}</p>"
        )
    if digest["mechanism"]:
        parts.append(
            f"<h3>Mechanism (because…)</h3><p>{_esc(digest['mechanism'])}</p>"
        )
    if digest["falsifier"]:
        parts.append(
            f"<h3>Numeric falsifier</h3><p>{_esc(digest['falsifier'])}</p>"
        )
    if digest["predicted"]:
        parts.append(
            f"<h3>Predicted Δ</h3><p>{_esc(digest['predicted'])}</p>"
        )
    if digest["citation"]:
        parts.append(
            f"<h3>Primary citation</h3><p style='font-family:Consolas,"
            f"monospace;font-size:0.85em'>{_esc(digest['citation'])}</p>"
        )
    parts.append(
        f"<p style='margin-top:14px'>"
        f"<a href='{_esc(gh_url)}'>Full design doc on GitHub →</a> · "
        f"<a href='../../{_esc(rel)}'>local: {_esc(rel)}</a></p>"
    )
    parts.append("</div>")
    return "".join(parts)


def _render_verdict_section(repo_root: Path, tag: str) -> str:
    findings_md = repo_root / "FINDINGS.md"
    blurb = parse_findings_for_tag(findings_md, tag)
    gh_url = (
        "https://github.com/dlmastery/nature_inspired_networks/blob/main/"
        "FINDINGS.md"
    )
    if not blurb:
        return (
            "<div class='card'><h2>Verdict (FINDINGS.md)</h2>"
            "<p class='empty'>No paragraph in FINDINGS.md mentions this tag "
            "yet.</p>"
            f"<p><a href='{gh_url}'>Browse FINDINGS.md →</a></p></div>"
        )
    return (
        "<div class='card'><h2>Verdict (FINDINGS.md)</h2>"
        f"<div class='quote'>{_esc(blurb)}</div>"
        f"<p style='margin-top:10px'><a href='{gh_url}'>Full FINDINGS.md →</a> · "
        f"<a href='../../FINDINGS.md'>local: FINDINGS.md</a></p></div>"
    )


def _render_reasoning_section(reasoning_path: Path | None) -> str:
    if reasoning_path is None or not reasoning_path.exists():
        return (
            "<div class='card'><h2>Reasoning blob</h2>"
            "<p class='empty'>No <code>reasoning.json</code> for this run "
            "directory.</p></div>"
        )
    try:
        data = json.loads(reasoning_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            "<div class='card'><h2>Reasoning blob</h2>"
            f"<p class='empty'>Failed to parse: {_esc(exc)}.</p></div>"
        )
    if isinstance(data, list):
        # Could be a list of entries; take the first.
        data = data[0] if data else {}
    parts = ["<div class='card'><h2>Reasoning blob</h2>"]
    fields = [
        ("diagnosis", "Diagnosis"),
        ("citations", "Citations"),
        ("hypothesis", "Hypothesis"),
        ("prediction", "Prediction"),
        ("verdict", "Verdict"),
        ("learning", "Learning"),
    ]
    for key, label in fields:
        v = data.get(key)
        if not v:
            continue
        if isinstance(v, list):
            body = "<ul>" + "".join(f"<li>{_esc(x)}</li>" for x in v) + "</ul>"
        else:
            body = f"<div class='body'>{_esc(v)}</div>"
        parts.append(
            f"<div class='reason-section'><div class='lbl'>{label}</div>{body}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_configuration_section(run_dir: Path, metrics: dict,
                                  results_dir: Path | None = None) -> str:
    cfg_yaml = run_dir / "config.yaml"
    if cfg_yaml.exists():
        try:
            text = cfg_yaml.read_text(encoding="utf-8")
        except Exception:
            text = "(failed to read config.yaml)"
        return (
            "<div class='card'><h2>Configuration</h2>"
            f"<pre>{_esc(text)}</pre></div>"
        )
    # Fallback 1: enrich with the matching experiments/experiment_log.jsonl row.
    log_row: dict | None = None
    if results_dir is not None:
        log_row = _load_experiment_log_row(
            results_dir / "experiment_log.jsonl",
            metrics.get("tag", ""), metrics.get("seed"),
            metrics.get("dataset", ""),
        )
    # Build the inferred dict: prefer log_row keys when both present
    keys_of_interest = (
        "tag", "model", "dataset", "channel_mode", "n_stages", "n_blocks",
        "epochs", "batch_size", "lr", "optimizer", "weight_decay",
        "scheduler", "seed", "flags", "composite_formula",
    )
    inferred: dict = {}
    for k in keys_of_interest:
        v = (log_row or {}).get(k)
        if v is None:
            v = metrics.get(k)
        if v is not None:
            inferred[k] = v
    if not inferred:
        return (
            "<div class='card'><h2>Configuration</h2>"
            "<p class='empty'>No <code>config.yaml</code> in run directory "
            "and neither <code>metrics.json</code> nor "
            "<code>experiment_log.jsonl</code> carries recoverable config "
            "overrides.</p></div>"
        )
    src = "experiment_log.jsonl + metrics.json" if log_row else "metrics.json"
    return (
        f"<div class='card'><h2>Configuration (inferred from {src})</h2>"
        f"<pre>{_esc(json.dumps(inferred, indent=2))}</pre></div>"
    )


def _render_audit_verdict_section(hid: str | None, group: str,
                                  repo_root: Path) -> str:
    """Implementation-critic verdict card (parsed from audits/G<X>_audit.md)."""
    if not hid:
        return ""
    audit = parse_audit_verdict(repo_root, hid, group)
    if not audit["verdict"]:
        return ""
    badge = _verdict_badge("impl-critic", audit["verdict"],
                           url=audit["url"], tooltip=audit["summary"])
    body = audit["summary"] or "(see full audit on GitHub)"
    local_link = (
        f"<a href='../../{_esc(audit['local'])}'>"
        f"local: {_esc(audit['local'])}</a>"
    ) if audit["local"] else ""
    return (
        f"<div class='card'><h2>Implementation-critic verdict "
        f"(audits/{group}_audit.md)</h2>"
        f"<div style='margin-bottom:10px'>{badge}</div>"
        f"<div class='quote'>{_esc(body)}</div>"
        f"<p style='margin-top:10px'>"
        f"<a href='{_esc(audit['url'])}'>Full audit on GitHub &rarr;</a>"
        f"{' &middot; ' + local_link if local_link else ''}</p></div>"
    )


def _render_cross_refs(run_dir_name: str, tag: str, dataset: str,
                       results_dir: Path,
                       df: pd.DataFrame | None = None,
                       group_letter: str | None = None,
                       composite: float | None = None) -> str:
    """Discover other seeds / datasets / nearest-in-group neighbours."""
    items_seed: list[str] = []
    items_dataset: list[str] = []
    items_group: list[str] = []
    # Other seeds, same dataset
    same_ds_dir = results_dir / dataset
    if same_ds_dir.exists():
        for sib in sorted(same_ds_dir.iterdir()):
            if not sib.is_dir():
                continue
            if not sib.name.startswith(f"{tag}_seed"):
                continue
            if sib.name == run_dir_name:
                continue
            href = f"../experiments/{run_page_filename(sib.name, dataset=dataset)}"
            items_seed.append(
                f"<li><a href='{_esc(href)}'>{_esc(sib.name)}</a></li>"
            )
    # Same tag on other dataset(s)
    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == dataset:
            continue
        for sib in sorted(ds_dir.iterdir()):
            if not sib.is_dir():
                continue
            if not sib.name.startswith(f"{tag}_seed"):
                continue
            href = f"../experiments/{run_page_filename(sib.name, dataset=ds_dir.name)}"
            items_dataset.append(
                f"<li><a href='{_esc(href)}'>{_esc(sib.name)}</a> "
                f"<span class='mut'>("
                f"<code>{_esc(ds_dir.name)}</code>)</span></li>"
            )
    # Nearest neighbours in the same hypothesis group
    if (df is not None and group_letter is not None
            and composite is not None and not df.empty):
        neighbours = _nearest_in_group(
            df, group_letter, tag, dataset, composite, k=3,
        )
        for n in neighbours:
            run_name = n["run_dir"].split("/")[-1]
            href = (
                f"../experiments/{run_page_filename(run_name, dataset=n['dataset'])}"
            )
            delta = (n["composite"] - composite)
            sign = "+" if delta >= 0 else "−"
            items_group.append(
                f"<li><a href='{_esc(href)}'>{_esc(n['tag'])}</a> "
                f"<span class='mut'>(seed {_esc(n['seed'])}, top-1 "
                f"{n['top1']*100:.2f}%, composite Δ {sign}"
                f"{abs(delta):.4f})</span></li>"
            )

    blocks: list[str] = []
    if items_seed:
        blocks.append(
            "<h3>Other seeds (same tag, same dataset)</h3>"
            "<ul class='xrefs'>" + "".join(items_seed) + "</ul>"
        )
    if items_dataset:
        blocks.append(
            "<h3>Same tag on other dataset(s)</h3>"
            "<ul class='xrefs'>" + "".join(items_dataset) + "</ul>"
        )
    if items_group:
        blocks.append(
            "<h3>Nearest neighbours in this hypothesis group "
            "(by composite distance)</h3>"
            "<ul class='xrefs'>" + "".join(items_group) + "</ul>"
        )
    if not blocks:
        body = "<p class='empty'>No companion runs found.</p>"
    else:
        body = "".join(blocks)
    return f"<div class='card'><h2>Cross-references</h2>{body}</div>"


def _render_footer(metrics: dict, run_dir: Path,
                   repo_root: Path | None = None) -> str:
    fp = metrics.get("composite_fingerprint", "")
    epochs = metrics.get("epochs", "?")
    train_s = metrics.get("train_seconds", "?")
    gen_gap = metrics.get("generalization_gap", "")
    try:
        gen_gap_disp = f"{float(gen_gap)*100:.2f} pp"
    except Exception:
        gen_gap_disp = str(gen_gap) if gen_gap != "" else "—"
    try:
        train_s_disp = f"{float(train_s):.1f} s"
    except Exception:
        train_s_disp = str(train_s)
    sha = ""
    if repo_root is not None:
        try:
            sha = _git_short_sha(repo_root, run_dir / "metrics.json")
        except Exception:
            sha = ""
    sha_html = ""
    if sha:
        sha_url = (
            f"https://github.com/dlmastery/nature_inspired_networks/commit/{sha}"
        )
        sha_html = (
            f"&nbsp;·&nbsp; metrics.json git commit: "
            f"<a href='{_esc(sha_url)}' style='color:#58a6ff'>"
            f"<code>{_esc(sha)}</code></a>"
        )
    try:
        run_path_rel = str(run_dir.relative_to(repo_root)) if repo_root else run_dir.name
    except Exception:
        run_path_rel = run_dir.name
    return (
        "<div class='meta'>"
        f"composite formula SHA-256: <code>{_esc(fp)}</code> &nbsp;·&nbsp; "
        f"epochs run: <b>{_esc(epochs)}</b> &nbsp;·&nbsp; "
        f"train wall-clock: <b>{train_s_disp}</b> &nbsp;·&nbsp; "
        f"generalisation gap (train − test top-1): <b>{gen_gap_disp}</b>"
        f"{sha_html}"
        f"<br>run directory: <code>{_esc(run_path_rel)}</code>"
        "<br>Generated by "
        "<code>nature_inspired_networks.dashboard.render_experiment_page</code>"
        " · self-contained inline SVG; no external assets."
        "</div>"
    )


def render_experiment_page(metrics: dict, history: list[dict] | None,
                           run_dir_name: str, out_html: Path,
                           run_dir: Path | None = None,
                           results_dir: Path | None = None,
                           repo_root: Path | None = None,
                           runs_df: pd.DataFrame | None = None) -> dict[str, bool]:
    """Write a single self-contained per-experiment HTML page.

    Returns a dict of section presence flags so callers can build coverage
    statistics (which sections rendered with real content vs. fallback).
    """
    tag = metrics.get("tag", run_dir_name)
    seed = metrics.get("seed", "")
    dataset = metrics.get("dataset", "")
    title = f"{tag} · seed {seed} · {dataset}"

    hid, group = TAG_TO_HYP.get(tag, (None, "Uncategorized"))
    grp_header, grp_desc = GROUP_HEADERS.get(group, (group, ""))

    if repo_root is None:
        repo_root = Path.cwd()
    if results_dir is None:
        results_dir = repo_root / "experiments"
    if run_dir is None:
        run_dir = results_dir / dataset / run_dir_name

    flags = {
        "hypothesis": False,
        "verdict": False,
        "reasoning": False,
        "config": False,
        "history": history is not None and len(history) > 0,
        "cross_refs": False,
        "audit": False,
        "scicritic": False,
    }

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
        ("generalization_gap", "Generalisation gap (train − test)"),
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
            "<div class='empty'>history.json absent for this run — no "
            "per-epoch curves available.</div>"
        )

    # ---- hypothesis / verdict / reasoning / config / cross-refs --------
    hyp_html = _render_hypothesis_section(hid, group, repo_root)
    md_path = find_hypothesis_path(repo_root, hid) if hid else None
    if hid is not None and md_path is not None:
        flags["hypothesis"] = True
        if parse_scicritic_verdict(md_path)["verdict"]:
            flags["scicritic"] = True

    # Implementation-critic verdict from audits/G<X>_audit.md
    audit_html = _render_audit_verdict_section(hid, group, repo_root)
    if audit_html:
        flags["audit"] = True

    verdict_html = _render_verdict_section(repo_root, tag)
    if parse_findings_for_tag(repo_root / "FINDINGS.md", tag):
        flags["verdict"] = True

    reasoning_path = run_dir / "reasoning.json"
    reasoning_html = _render_reasoning_section(
        reasoning_path if reasoning_path.exists() else None
    )
    flags["reasoning"] = reasoning_path.exists()

    config_html = _render_configuration_section(
        run_dir, metrics, results_dir=results_dir,
    )
    # "config" flag counts ANY successfully rendered config block — exact
    # config.yaml OR an experiment_log.jsonl row OR sufficient metrics fields.
    if (run_dir / "config.yaml").exists() or "Configuration (inferred" in config_html:
        flags["config"] = True

    # Composite for nearest-neighbour distance ranking
    try:
        comp = float(metrics.get("composite"))
    except Exception:
        comp = None
    xrefs_html = _render_cross_refs(
        run_dir_name, tag, dataset, results_dir,
        df=runs_df, group_letter=group, composite=comp,
    )
    if "<li>" in xrefs_html:
        flags["cross_refs"] = True

    footer_html = _render_footer(metrics, run_dir, repo_root=repo_root)

    # ---- header pills --------------------------------------------------
    hyp_pill = (
        f"<span class='pill hyp'>{_esc(hid)}</span>"
        if hid else "<span class='pill'>baseline</span>"
    )
    grp_pill = f"<span class='pill grp'>{_esc(grp_header)}</span>"
    ds_pill = f"<span class='pill'>{_esc(dataset)}</span>"
    seed_pill = f"<span class='pill'>seed {_esc(seed)}</span>"

    # Hypothesis-doc GitHub link for the header strip
    hyp_url = ""
    if hid:
        mdp = find_hypothesis_path(repo_root, hid)
        if mdp is not None:
            rel = mdp.relative_to(repo_root).as_posix()
            hyp_url = (
                "https://github.com/dlmastery/nature_inspired_networks/blob/"
                f"main/{rel}"
            )
    hyp_link_html = (
        f"&nbsp;·&nbsp; <a href='{_esc(hyp_url)}' "
        f"style='color:#58a6ff;font-weight:600'>"
        f"&#128196; Hypothesis design doc &rarr;</a>"
        if hyp_url else ""
    )

    page = (
        "<!doctype html>\n"
        "<html lang='en'><head><meta charset='utf-8'>\n"
        f"<title>{_esc(title)} — experiment page</title>\n"
        f"<style>{_EXP_PAGE_CSS}</style></head><body>\n"
        "<a class='back' href='../dashboard.html'>&larr; back to dashboard</a>\n"
        f"<h1>{_esc(tag)}</h1>\n"
        f"<div class='sub'>{hyp_pill}{grp_pill}{ds_pill}{seed_pill}"
        f" &nbsp;·&nbsp; run directory <code>{_esc(run_dir_name)}</code>"
        f"{hyp_link_html}</div>\n"
        f"{hyp_html}\n"
        f"{audit_html}\n"
        f"{verdict_html}\n"
        f"{reasoning_html}\n"
        f"{config_html}\n"
        f"<div class='card'><h2>Metrics</h2>{metrics_table}</div>\n"
        f"<div class='card'><h2>Composite-score breakdown</h2>{breakdown_html}</div>\n"
        f"<div class='card'><h2>Per-epoch training curves</h2>{charts_html}</div>\n"
        f"{xrefs_html}\n"
        f"{footer_html}\n"
        "</body></html>\n"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return flags


def render_all_experiment_pages(results_dir: str | Path,
                                out_dir: str | Path,
                                repo_root: str | Path | None = None,
                                ) -> tuple[list[str], dict[str, int]]:
    """Render one per-experiment page per run dir holding a ``metrics.json``.

    Returns (sorted filename list, section-coverage counts) so the build
    script can report how many of the 60+ pages got each section filled.
    """
    rp = Path(results_dir)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root else rp.parent
    written: list[str] = []
    coverage = {
        "hypothesis": 0, "verdict": 0, "reasoning": 0, "config": 0,
        "history": 0, "cross_refs": 0, "audit": 0, "scicritic": 0,
    }
    # Build the runs dataframe once so we can drive per-group nearest-neighbour
    # cross-references without re-globbing.
    runs_df = load_runs(rp)
    if not runs_df.empty:
        runs_df["__group"] = runs_df["tag"].map(
            lambda t: TAG_TO_HYP.get(t, (None, "Uncategorized"))[1]
        )
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
        dataset = metrics.get("dataset") or run_dir.parent.name
        fname = run_page_filename(run_dir.name, dataset=dataset)
        flags = render_experiment_page(
            metrics, hist, run_dir.name, od / fname,
            run_dir=run_dir, results_dir=rp, repo_root=root,
            runs_df=runs_df,
        )
        for k, present in flags.items():
            if present:
                coverage[k] = coverage.get(k, 0) + 1
        written.append(fname)
    return sorted(written), coverage


# ---------------------------------------------------------------------------
# Main dashboard HTML head + sortable JS (NO MODAL anywhere)
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
 h2{color:#58a6ff;font-size:1.15em;margin:30px 0 6px 0;font-weight:600;
    letter-spacing:0.3px;}
 h3{color:#c9d1d9;font-size:0.95em;margin-bottom:10px;font-weight:600;}
 .sub{color:#8b949e;margin-bottom:14px;font-size:0.92em;}
 .group-desc{color:#8b949e;font-size:0.88em;margin:2px 0 10px 0;
             max-width:980px;line-height:1.55;}
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
 table.runs{width:100%;border-collapse:collapse;font-size:0.83em;
            background:#161b22;border:1px solid #30363d;border-radius:8px;
            overflow:hidden;}
 table.runs th{background:#0d1117;color:#8b949e;text-align:right;padding:8px 10px;
    border-bottom:2px solid #30363d;font-weight:600;text-transform:uppercase;
    font-size:0.72em;letter-spacing:0.4px;cursor:pointer;}
 table.runs th:first-child{text-align:left;}
 table.runs td{padding:7px 10px;border-bottom:1px solid #21262d;text-align:right;
    color:#c9d1d9;}
 table.runs td:first-child{text-align:left;}
 table.runs tr.row-link{cursor:pointer;}
 table.runs tr.row-link:hover{background:#1c2128;}
 table.runs tr.best-row{background:#0d2818;border-left:4px solid #00d26a;}
 table.runs tr.best-row td{font-weight:600;}
 .tag-pill{display:inline-block;background:#21262d;border:1px solid #30363d;
           border-radius:10px;padding:1px 8px;font-size:0.75em;color:#c9d1d9;
           font-family:Consolas,monospace;}
 .chip{display:inline-block;background:#21262d;border:1px solid #30363d;
       border-radius:14px;padding:3px 11px;margin:2px 4px 2px 0;
       font-size:0.78em;color:#c9d1d9;}
 .meta{font-size:0.78em;color:#484f58;margin-top:18px;}
 .status-done{background:#3fb950;}
 .status-running{background:#d29922;animation:pulse 1.5s infinite;}
 .status-queued{background:#1f6feb;}
 .status-planned{background:#484f58;}
 .status-failed{background:#f85149;}
 .status-superseded{background:#6e7681;}
 @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.45;}}
 .group-section{margin-top:26px;background:#161b22;border:1px solid #30363d;
                border-radius:10px;padding:14px 18px;}
 .group-section h2{margin-top:0;color:#58a6ff;font-size:1.1em;}
 .group-empty{color:#8b949e;font-style:italic;font-size:0.88em;
              padding:6px 0;}
 .group-chart{margin:8px 0 10px 0;}
 table.runs.compact{font-size:0.78em;}
 table.runs.compact th{padding:5px 7px;font-size:0.66em;}
 table.runs.compact td{padding:4px 7px;}
 .hyp-link{color:#58a6ff;font-family:Consolas,monospace;font-size:0.78em;
           white-space:nowrap;}
 /* side panel (hypothesis quick-open) */
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
 .legend-row{margin:8px 0 12px 0;font-size:0.78em;color:#8b949e;}
 .legend-row .swatch{display:inline-block;width:12px;height:12px;border-radius:2px;
                     vertical-align:middle;margin:0 4px 0 10px;}
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
</style>
</head><body>
"""


HTML_JS = r"""
<script>
function sortTable(tableId, n){
  var t=document.getElementById(tableId);
  if(!t) return;
  var rs=Array.from(t.tBodies[0].rows);
  var d=t.dataset.dir==='asc'?-1:1;t.dataset.dir=d===1?'asc':'desc';
  rs.sort(function(a,b){
    var x=a.cells[n].dataset.v||a.cells[n].textContent;
    var y=b.cells[n].dataset.v||b.cells[n].textContent;
    var nx=parseFloat(x),ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny)) return d*(nx-ny);
    return d*x.localeCompare(y);
  });
  rs.forEach(function(r){t.tBodies[0].appendChild(r);});
}

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
      '`python -m http.server` from the repo root and reload.';
  });
  sp.classList.add('open');
}
function closeSide(){document.getElementById('side-panel').classList.remove('open');}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){closeSide();}
});
</script>
"""


# ---------------------------------------------------------------------------
# HTML fragment builders (hypothesis grid + ribbons + group sections)
# ---------------------------------------------------------------------------

def _hypothesis_grid_html(index_rows: list[dict],
                          tag_status: dict[str, dict]) -> str:
    if not index_rows:
        return "<i style='color:#8b949e'>hypotheses/INDEX.md not found.</i>"

    status_for: dict[str, str] = {h["id"]: "planned" for h in index_rows}
    order = ["done", "running", "queued", "failed", "superseded", "planned"]
    for tag, meta in tag_status.items():
        idea = meta.get("idea", "")
        for m in re.finditer(r"H(\d{2})", idea):
            hid = "H" + m.group(1)
            if hid in status_for:
                cur = status_for[hid]
                new = meta["status"]
                if order.index(new) < order.index(cur):
                    status_for[hid] = new

    direct = {
        "H04": ["sg_chan_fib", "sg_chan_phi"],
        "H05": ["sg_only_fractal"],
        "H17": ["sg_only_golden_modulate"],
        "H21": ["sg_only_hex"],
        "H22": ["sg_only_toroidal"],
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
            row_html.append(
                "<span class='cell status-" + st + "' title='" + hid
                + " (status: " + st + ")' "
                + "onclick=\"openHypothesis('" + hid + "','" + fname + "')\">"
                + hid[1:] + "</span>"
            )
        row_html.append("</div>")
        out.append("".join(row_html))
    out.append("</div>")
    return "\n".join(out)


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


# Compact table: only the essentials.  Heavier columns (FLOPs, latency,
# rot-eq, top-5) live on the per-experiment page now.
_RUNS_COLS = [
    ("tag", "Tag"),
    ("dataset", "DS"),
    ("seed", "Seed"),
    ("top1", "Top-1"),
    ("composite", "Composite"),
    ("params", "Params"),
]


def _format_cell(k: str, v) -> tuple[str, str]:
    """Return (data-v string, display string) for a runs-table cell."""
    if v is None or v == "":
        return "", "—"
    try:
        if k == "params":
            return str(v), f"{float(v)/1e6:.3f}M"
        if k == "flops":
            return str(v), f"{float(v)/1e6:.1f}M"
        if k in ("top1", "top5"):
            return str(v), f"{float(v)*100:.2f}%"
        if k == "latency_ms":
            return str(v), f"{float(v):.2f} ms"
        if k == "composite":
            return str(v), f"{float(v):.4f}"
        if isinstance(v, float):
            return str(v), f"{v:.4f}"
        return str(v), str(v)
    except Exception:
        return str(v), str(v)


def _group_section_html(group_letter: str,
                        rows: list[pd.Series],
                        best_idx: set,
                        table_id: str) -> str:
    """Render one ``<section>`` for a hypothesis-group block."""
    header, desc = GROUP_HEADERS.get(group_letter, (group_letter, ""))
    n = len(rows)
    parts = [
        f"<section class='group-section'>",
        f"<h2>{_esc(header)} <span style='color:#8b949e;"
        f"font-weight:400;font-size:0.78em'>({n} run{'s' if n != 1 else ''})"
        f"</span></h2>",
        f"<div class='group-desc'>{_esc(desc)}</div>",
    ]
    if not rows:
        parts.append(
            "<div class='group-empty'>No sweep rows yet in this group.</div>"
        )
        parts.append("</section>")
        return "".join(parts)
    # Compact per-group SVG visualisation BEFORE the table.
    chart = _group_visualisation_svg(
        rows, group_letter,
        chart_id=f"svg-{table_id}",
    )
    if chart:
        parts.append("<div class='group-chart'>" + chart + "</div>")
    parts.append(
        f"<table class='runs compact' id='{table_id}' data-dir='asc'>"
        "<thead><tr>"
    )
    for i, (_k, label) in enumerate(_RUNS_COLS):
        parts.append(
            f"<th onclick=\"sortTable('{table_id}', {i})\">{label}</th>"
        )
    parts.append("<th>Hyp</th>")
    parts.append("</tr></thead><tbody>")

    for r in rows:
        run_dir_path = str(r.get("_run_dir", ""))
        run_dir_name = run_dir_path.split("/")[-1]
        ds = r.get("dataset")
        if not ds and "/" in run_dir_path:
            ds = run_dir_path.split("/")[0]
        page_href = (
            f"experiments/{run_page_filename(run_dir_name, dataset=ds)}"
            if run_dir_name else ""
        )
        klass = "row-link" + (" best-row" if r.name in best_idx else "")
        if page_href:
            row_attrs = (
                f"class='{klass}' "
                f"onclick=\"window.location.href='{page_href}'\""
            )
        else:
            row_attrs = f"class='{klass}'"
        parts.append(f"<tr {row_attrs}>")
        for k, _label in _RUNS_COLS:
            v = r.get(k, "")
            dv, disp = _format_cell(k, v)
            if k == "tag":
                tag_html = (
                    f"<a href='{page_href}' class='tag-pill' "
                    f"onclick='event.stopPropagation()' "
                    f"style='color:#58a6ff'>{_esc(disp)}</a>"
                    if page_href else
                    f"<span class='tag-pill'>{_esc(disp)}</span>"
                )
                parts.append(
                    f"<td data-v='{_esc(dv)}' style='text-align:left'>{tag_html}</td>"
                )
            else:
                parts.append(
                    f"<td data-v='{_esc(dv)}'>{_esc(disp)}</td>"
                )
        # Hypothesis link cell
        tag = r.get("tag", "")
        hid, _grp = TAG_TO_HYP.get(tag, (None, "Uncategorized"))
        if hid:
            parts.append(
                f"<td style='text-align:left'>"
                f"<span class='hyp-link'>{_esc(hid)}</span></td>"
            )
        else:
            parts.append(
                "<td style='text-align:left'><span class='mut'>—</span></td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    parts.append("</section>")
    return "".join(parts)


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

    Now organised by hypothesis group: instead of one mega-leaderboard, every
    group (Baseline + G1..G8 + Uncategorised) gets its own sortable section.
    Rows link to the per-experiment page; no modal.
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

    tag_to_tier = parse_experiment_log_tiers(root / "EXPERIMENT_LOG.md")
    index_rows = parse_hypothesis_index(root / "hypotheses" / "INDEX.md")
    findings_blurb = parse_findings_headline(root / "FINDINGS.md")
    findings_metrics = parse_findings_metrics(root / "FINDINGS.md")

    if not df.empty:
        df = df.sort_values(["dataset", "composite"], ascending=[True, False])
        df["__group"] = df["tag"].map(
            lambda t: TAG_TO_HYP.get(t, (None, "Uncategorized"))[1]
        )
        best_idx = set(df.groupby("dataset")["composite"].idxmax().tolist())
    else:
        df["__group"] = pd.Series(dtype=str)
        best_idx = set()

    html = [HTML_HEAD]
    html.append(f"<h1>{title}</h1>")
    html.append(
        f"<div class='sub'>{subtitle} · "
        f"<a href='https://github.com/dlmastery/nature_inspired_networks'>"
        f"GitHub</a> · "
        f"<a href='../FINDINGS.md'>FINDINGS.md</a> · "
        f"<a href='../EXPERIMENT_LOG.md'>EXPERIMENT_LOG</a></div>"
    )
    if findings_blurb:
        html.append(
            "<div style='background:#161b22;border-left:4px solid #d29922;"
            "border:1px solid #30363d;border-radius:6px;padding:12px 16px;"
            "margin:6px 0 10px 0;font-size:0.92em;color:#c9d1d9'>"
            f"<b style='color:#d29922'>Headline finding · </b>"
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

    html.append(
        "<div class='card panel-2col'><h3>Hypothesis ledger — status grid "
        "(G1..G8 × hypothesis index; click a cell to view the spec)</h3>"
        + _hypothesis_grid_html(index_rows, tag_to_tier)
        + "</div>"
    )
    html.append("</div>")  # end .grid

    # ------------------------------------------------------------------
    # Runs sections by hypothesis group (NO modal anywhere)
    # ------------------------------------------------------------------
    html.append(
        "<h2 style='margin-top:34px'>Runs by hypothesis group "
        "<span style='color:#8b949e;font-weight:400;font-size:0.78em'>"
        "— click any row for the full per-experiment page</span></h2>"
    )

    rows_by_group: dict[str, list[pd.Series]] = {g: [] for g in GROUP_ORDER}
    if not df.empty:
        for _, r in df.iterrows():
            g = r.get("__group", "Uncategorized")
            rows_by_group.setdefault(g, []).append(r)
        for g in rows_by_group:
            rows_by_group[g].sort(
                key=lambda r: float(r.get("composite", 0) or 0), reverse=True,
            )

    for g in GROUP_ORDER:
        rows = rows_by_group.get(g, [])
        html.append(
            _group_section_html(
                g, rows, best_idx,
                table_id=f"runs-{g.lower()}",
            )
        )

    html.append(
        "<div class='meta'>Generated by "
        "<code>nature_inspired_networks.dashboard.render_dashboard</code> · "
        "dark theme adapted from dlmastery/autoresearchspy/docs/spy_dashboard · "
        "all CSS+JS inline; no external CDN; PNG plots kept as PNG; "
        "leaderboard organised by hypothesis group; rows link to per-experiment pages.</div>"
    )

    # Side panel (kept — hypothesis-cell quick-open; NOT a modal)
    html.append("""
<div id='side-panel'>
  <button class='close-btn' onclick='closeSide()'>close · Esc</button>
  <h3 id='side-panel-title'>Hypothesis spec</h3>
  <pre id='side-panel-body'>—</pre>
</div>
""")

    html.append(HTML_JS)
    html.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html), encoding="utf-8")
