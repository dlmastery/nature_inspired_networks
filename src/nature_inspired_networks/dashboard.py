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
    # Combo ladder (additive stacks on phi_budget H09 base — G1)
    "combo2_pb_gm": ("H09", "G1"),
    "combo3_pb_gm_pd": ("H09", "G1"),
    "combo4_pb_gm_pd_pdw": ("H09", "G1"),
    "combo5_pb_gm_pd_pdw_plr": ("H09", "G1"),
    "combo6_pb_gm_pd_pdw_plr_fe": ("H09", "G1"),
    "combo7_pb_gm_pd_pdw_plr_fe_sa": ("H09", "G1"),
    "combo8_pb_gm_pd_pdw_plr_fe_sa_fp": ("H09", "G1"),
    # LOO subtractive from combo8 (phi_budget H09 base)
    "loo_no_gm": ("H09", "G1"),
    "loo_no_pd": ("H09", "G1"),
    "loo_no_pdw": ("H09", "G1"),
    "loo_no_plr": ("H09", "G1"),
    "loo_no_fe": ("H09", "G1"),
    "loo_no_sa": ("H09", "G1"),
    "loo_no_fp": ("H09", "G1"),
    # Two-at-a-time PAIR interaction matrix (phi_budget H09 base)
    "pair_gm_pdw": ("H09", "G1"),
    "pair_gm_plr": ("H09", "G1"),
    "pair_pd_pdw": ("H09", "G1"),
    "pair_pd_plr": ("H09", "G1"),
    "pair_pdw_plr": ("H09", "G1"),
    # Mutually-exclusive SLOT ablation (phi_budget H09 base × one axis)
    "slot_act_sine": ("H81", "G8"),
    "slot_act_phi": ("H39", "G4"),
    "slot_init_spiral": ("H31", "G4"),
    "slot_init_phi": ("H42", "G5"),
    "slot_init_cymatic": ("H35", "G4"),
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


def parse_idea_table_status(idea_table_md: str | Path) -> dict[str, str]:
    """Parse IDEA_TABLE.md status column into {H<NN>: status}.

    Returns one of ``"done"`` (✓ done — implemented + at least one CIFAR row),
    ``"impl"`` (✓ impl — module + tests, no CIFAR row), ``"partial"``
    (~ partial), ``"planned"`` (○ not started), or ``"deferred"`` (× deferred).

    The IDEA_TABLE.md table rows look like::

        | 1 | **H01** | φ-Compound Scaling — ... | ✓ done (`phi_compound` .8042) | ...
        | 2 | **H02** | Fibonacci Depth Progression — ... | ✓ impl | ...

    Missing hypotheses default to ``"planned"`` upstream — this function only
    returns the rows it can parse.
    """
    p = Path(idea_table_md)
    out: dict[str, str] = {}
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="ignore")
    hid_re = re.compile(r"\*\*H(\d{2})\*\*")
    for line in text.splitlines():
        if "|" not in line:
            continue
        m = hid_re.search(line)
        if not m:
            continue
        hid = "H" + m.group(1)
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        # Status is conventionally the 4th column; scan all cells defensively
        # for the legend glyphs so off-by-one schema drift doesn't lose status.
        status = "planned"
        for c in cells:
            low = c.lower()
            if "✓ done" in c or "done" == low or low.startswith("done ") \
                    or "✓done" in c:
                status = "done"; break
            if "✓ impl" in c or low.startswith("impl"):
                status = "impl"; break
            if "~ partial" in c or "partial" in low:
                status = "partial"; break
            if "× deferred" in c or "deferred" in low:
                status = "deferred"; break
            if "○ not started" in c:
                status = "planned"; break
        out[hid] = status
    return out


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


def parse_findings_headline(findings_md: str | Path,
                            max_chars: int = 4000) -> str:
    """Return the first contiguous ``>``-blockquoted headline block.

    Preserves the original line breaks (NEW: 2026-05-29 fix). The
    previous implementation joined the lines with a single space, which
    destroyed the GFM-table row separators and made the markdown
    converter unable to parse the embedded headline table. The
    downstream ``_md_to_html`` call strips the leading ``>`` markers
    and parses the inner Markdown faithfully — preserving the table.
    """
    p = Path(findings_md)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines: list[str] = []
    for ln in text.splitlines():
        # Match any line that starts (possibly after spaces) with ``>``.
        if ln.lstrip().startswith(">"):
            lines.append(ln)
        elif lines:
            break
    return "\n".join(lines)[:max_chars]


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


# ---------------------------------------------------------------------------
# Phase-8 winners (Rule 28 — n=3 evaluation tier)
#
# The three hypotheses that cleared the worst-leader-seed > best-baseline-seed
# Phase-5 gate on CIFAR-100 (3 seeds × 30 epochs). All other rows are
# n=1-seed SCREENING by default.
# ---------------------------------------------------------------------------
PHASE8_EVALUATION_TAGS: dict[str, set[str]] = {
    "cifar100": {"pair_gm_pdw", "slot_act_sine", "sg_only_phi_budget"},
}


def _seed_count_for_tag(results_dir: Path, tag: str, dataset: str) -> int:
    """Return the number of seed run-dirs on disk for (tag, dataset)."""
    if not tag or not dataset:
        return 0
    ds_dir = Path(results_dir) / dataset
    if not ds_dir.exists():
        return 0
    count = 0
    for sib in ds_dir.iterdir():
        if not sib.is_dir():
            continue
        if not sib.name.startswith(f"{tag}_seed"):
            continue
        if (sib / "metrics.json").exists():
            count += 1
    return count


def _evaluation_tier(tag: str, dataset: str, seed_count: int) -> str:
    """Return ``"EVALUATION"`` for Phase-8 winners with n>=3, else ``"SCREENING"``.

    The Phase-5 gate (best-baseline-seed < worst-leader-seed) is verified
    in FINDINGS.md for the three CIFAR-100 winners listed in
    ``PHASE8_EVALUATION_TAGS`` — so for any other (tag, dataset) we
    classify as SCREENING regardless of seed_count.
    """
    if seed_count < 3:
        return "SCREENING"
    eval_tags = PHASE8_EVALUATION_TAGS.get(dataset, set())
    if tag in eval_tags:
        return "EVALUATION"
    return "SCREENING"


def _build_stamp(repo_root: Path) -> dict[str, str]:
    """Return ``{sha, sha_short, iso_date, url}`` for the HEAD commit."""
    out = {"sha": "", "sha_short": "", "iso_date": "", "url": ""}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        sha = r.stdout.strip()
        out["sha"] = sha
        out["sha_short"] = sha[:7] if sha else ""
        r2 = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        out["iso_date"] = r2.stdout.strip()
        if sha:
            out["url"] = (
                f"https://github.com/dlmastery/nature_inspired_networks/"
                f"commit/{sha}"
            )
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Multi-hypothesis pill expansion for combo / loo / pair / slot ladders
# ---------------------------------------------------------------------------
#
# A run tag like ``combo3_pb_gm_pd`` stacks three priors on top of the
# H09 phi_budget base. The aggregate-dashboard / per-experiment-page
# H-pill display previously showed only the leading H ID. Each suffix
# token maps to its hypothesis ID:
_COMBO_TOKEN_TO_HYP: dict[str, str] = {
    "pb": "H09",   # phi_budget base
    "gm": "H48",   # golden_momentum
    "pd": "H47",   # phi_dropout
    "pdw": "H44",  # phi_decay (weight decay)
    "plr": "H10",  # phi_lr
    "fe": "H20",   # fib_ensemble
    "sa": "H81",   # sine_act / SIREN
    "fp": "H43",   # fib_prune
}


def hypotheses_for_tag(tag: str) -> list[str]:
    """Return ALL hypothesis IDs contributing to a tag (combo / loo / pair / slot).

    - For combo/loo/pair/slot tags: parses the underscore-separated tokens
      after the prefix and maps each to its H-ID via ``_COMBO_TOKEN_TO_HYP``.
      Combo always includes the H09 base.
    - For everything else: returns the single TAG_TO_HYP[tag] HID (or []).
    """
    if not tag:
        return []
    # Combo: combo<N>_<tok1>_<tok2>_... — always includes phi_budget base
    if tag.startswith("combo"):
        m = re.match(r"combo\d+_(.+)", tag)
        if m:
            toks = m.group(1).split("_")
            ids: list[str] = ["H09"]  # phi_budget base
            for t in toks:
                hid = _COMBO_TOKEN_TO_HYP.get(t)
                if hid and hid not in ids:
                    ids.append(hid)
            return ids
    # LOO: loo_no_<tok> — base combo8 minus one prior (still on phi_budget base)
    if tag.startswith("loo_no_"):
        tok = tag[len("loo_no_"):]
        ids = ["H09"]
        # combo8 base = pb + gm + pd + pdw + plr + fe + sa + fp; LOO removes one
        for t in ("gm", "pd", "pdw", "plr", "fe", "sa", "fp"):
            if t == tok:
                continue
            hid = _COMBO_TOKEN_TO_HYP.get(t)
            if hid and hid not in ids:
                ids.append(hid)
        return ids
    # PAIR: pair_<tok1>_<tok2> — phi_budget base + two priors
    if tag.startswith("pair_"):
        toks = tag[len("pair_"):].split("_")
        ids = ["H09"]
        for t in toks:
            hid = _COMBO_TOKEN_TO_HYP.get(t)
            if hid and hid not in ids:
                ids.append(hid)
        return ids
    # SLOT: slot_<axis>_<value> — one prior swapped onto phi_budget base.
    # The TAG_TO_HYP map already records the swapped-in hypothesis, so
    # we add the H09 base in front.
    if tag.startswith("slot_"):
        hid_pair = TAG_TO_HYP.get(tag)
        if hid_pair and hid_pair[0]:
            return ["H09", hid_pair[0]]
        return ["H09"]
    hid_pair = TAG_TO_HYP.get(tag)
    if hid_pair and hid_pair[0]:
        return [hid_pair[0]]
    return []


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
    """Extract a verdict blurb for the given tag from FINDINGS.md.

    NEW (2026-05-29 fix): preserve newlines so embedded GFM tables /
    bullet lists inside the matched paragraph survive into the markdown
    converter (the previous implementation collapsed all whitespace to
    a single space, which destroyed the table structure).
    """
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
            # Preserve newlines — only trim trailing/leading whitespace
            # so the markdown converter sees the original block structure.
            cleaned = para.rstrip()
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
            return c[:2400]
    return candidates[0][:2400]


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
                    w: int = 460, h: int = 320,
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
        f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='xMinYMin meet' "
        f"width='100%' height='auto' "
        f"xmlns='http://www.w3.org/2000/svg' "
        f"style='background:#0a0a0d;border:1px solid #1c1c20;display:block;"
        f"font-family:\"IBM Plex Mono\",monospace'>",
        f"<text x='{ml}' y='18' fill='#e6e1d6' font-size='14' "
        f"font-family='Source Serif 4,Georgia,serif' "
        f"font-weight='600'>{_esc(title)}</text>",
    ]
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        yy = sy(yv)
        parts.append(
            f"<line x1='{ml}' y1='{yy:.1f}' x2='{ml + pw}' y2='{yy:.1f}' "
            f"stroke='#1c1c20' stroke-width='1'/>"
        )
        lbl = f"{yv * 100:.1f}%" if y_as_pct else f"{yv:.3f}"
        parts.append(
            f"<text x='{ml - 6}' y='{yy + 3:.1f}' fill='#a89e8c' font-size='9' "
            f"text-anchor='end'>{lbl}</text>"
        )
    parts.append(
        f"<text x='{sx(xmin):.1f}' y='{h - 10}' fill='#a89e8c' font-size='9' "
        f"text-anchor='middle'>{int(xmin)}</text>"
    )
    parts.append(
        f"<text x='{sx(xmax):.1f}' y='{h - 10}' fill='#a89e8c' font-size='9' "
        f"text-anchor='middle'>{int(xmax)}</text>"
    )
    parts.append(
        f"<text x='{ml + pw / 2:.1f}' y='{h - 10}' fill='#a89e8c' "
        f"font-size='10' font-family='Source Serif 4,Georgia,serif' "
        f"text-anchor='middle'>epoch →</text>"
    )
    parts.append(
        f"<text x='14' y='{mt + ph / 2:.1f}' fill='#a89e8c' font-size='10' "
        f"font-family='Source Serif 4,Georgia,serif' "
        f"text-anchor='middle' transform='rotate(-90 14 {mt + ph / 2:.1f})'>"
        f"{_esc(y_label)} →</text>"
    )
    legend_x = ml + 6
    for li, (label, xs, ys, color) in enumerate(series):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        parts.append(
            f"<polyline points='{pts}' fill='none' stroke='{color}' "
            f"stroke-width='1.8'/>"
        )
        lx, ly = sx(xs[-1]), sy(ys[-1])
        parts.append(
            f"<circle cx='{lx:.1f}' cy='{ly:.1f}' r='2.6' fill='{color}'/>"
        )
        ly_txt = mt + 14 + li * 16
        parts.append(
            f"<rect x='{legend_x}' y='{ly_txt - 9}' width='10' height='10' "
            f"fill='{color}'/>"
        )
        parts.append(
            f"<text x='{legend_x + 16}' y='{ly_txt}' fill='#e6e1d6' "
            f"font-size='10' font-family='IBM Plex Mono,monospace'>"
            f"{_esc(label)}</text>"
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


_EXP_FONT_LINK = (
    # Academic-restrained palette: Source Serif 4 (Adobe's open serif designed
    # for academic long-form reading — Frank Grießhammer / Adobe Design); IBM
    # Plex Mono for technical text. No italic-as-emphasis display flourishes.
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link href='https://fonts.googleapis.com/css2?"
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&"
    "family=IBM+Plex+Mono:wght@400;500;600&display=swap' rel='stylesheet'>"
)


def _strip_blockquote_markers(text: str) -> str:
    """Strip leading ``>`` blockquote markers (one or more, possibly nested).

    The FINDINGS / impl-critic / sci-critic excerpts embedded in the
    dashboard are quoted FROM their source .md files using the
    project's "we're embedding this section" convention (leading ``> ``
    per line). The ``markdown`` library treats those leading markers as
    blockquotes containing the inner GFM table, but its parser does NOT
    recognise GFM tables nested inside a blockquote — so the pipes
    survive into the HTML output as literal ``|`` characters.

    The right fix is to strip the leading blockquote markers BEFORE
    handing the body to the markdown parser; the caller re-wraps the
    rendered HTML in a styled ``<div class='findings-quote md-body'>``
    so the visual blockquote affordance is retained.

    This handles arbitrary nesting depth (e.g., ``> > | tag | ...``):
    we strip ALL leading ``>`` markers at the start of each line.
    """
    lines: list[str] = []
    for line in text.splitlines():
        # Strip arbitrarily many leading "> " sequences (allowing spaces
        # between them), then a single trailing space if present.
        s = line
        while True:
            t = s.lstrip()
            if t.startswith("> "):
                s = t[2:]
                continue
            if t.startswith(">"):
                s = t[1:]
                continue
            break
        lines.append(s)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pages-link Rule 27 enforcement helpers
# ---------------------------------------------------------------------------
# GitHub Pages publishes ONLY ``docs/``. Any link in generated dashboard
# HTML pointing to a repo file outside ``docs/`` must use the absolute
# GitHub-blob URL or it 404s on Pages. Source markdown files
# (``paper/FINDINGS.md``, ``PAPER.md``, ``paper/AUDIT_SUMMARY.md`` …)
# legitimately use relative paths like ``../CLAUDE.md#rule-28`` because
# they live inside ``paper/`` and the relative path resolves correctly
# on the GitHub blob view. But once we render those markdown bodies INTO
# ``docs/dashboard/experiments/*.html`` the same relative path becomes
# Pages-rooted and 404s. This rewriter converts them to absolute URLs
# before the markdown library sees them.
_GITHUB_BLOB_PREFIX = (
    "https://github.com/dlmastery/nature_inspired_networks/blob/main/"
)
# Top-level repo files / directories that are NOT served from docs/.
# Any markdown link ``](../<NAME>...)`` where NAME matches one of these
# stems gets rewritten to the absolute GitHub blob URL.
_NON_DOCS_REPO_STEMS = (
    "CLAUDE.md", "README.md", "PAPER.md", "MANIFESTO.md", "MINDMAP.md",
    "ARCHITECTURE.md", "AUTORESEARCH_PROCESS.md", "IDEA_TABLE.md",
    "EXPERIMENT_LOG.md", "EXPERIMENT_LEDGER.md", "PARADIGM_COMPARISON.md",
    "NATURE_INSPIRED_NETWORKS.md", "FINDINGS.md", "RESULTS.md",
    "SOTA_COMPARISON.md", "MEDIUM.md", "SETUP.md",
    "RESTRUCTURE_PLAN.md", "RESTRUCTURE_PLAN_v2.md",
    "sota_catalog.yaml", "pyproject.toml",
    "paper/", "hypotheses/", "audits/", "scripts/", "ideas/",
    "src/", "tests/", "configs/", "experiments/", "memory/", "skills/",
)


def _rewrite_rule27_links(text: str) -> str:
    """Rewrite ``](../X)`` markdown links to absolute GitHub-blob URLs.

    Per CLAUDE.md Rule 27, only ``docs/`` is published to GitHub Pages.
    Any relative ``../<repo-root-file>`` link rendered into a
    ``docs/dashboard/`` HTML page 404s on Pages. This helper converts
    every such reference to the absolute GitHub-blob URL so the rendered
    HTML is safe to mirror under ``docs/dashboard/``.

    Also rewrites direct ``](paper/...)``, ``](audits/...)``, …
    relative-from-repo-root links the same way, for any non-``docs/``
    target stem listed in ``_NON_DOCS_REPO_STEMS``.
    """
    if not text or "](" not in text:
        return text

    def _is_non_docs(target: str) -> bool:
        # Strip any leading ../ chain.
        stripped = target
        while stripped.startswith("../"):
            stripped = stripped[3:]
        # Strip leading ./
        if stripped.startswith("./"):
            stripped = stripped[2:]
        # Match against non-docs repo stems.
        for stem in _NON_DOCS_REPO_STEMS:
            if stem.endswith("/") and stripped.startswith(stem):
                return True
            if stripped == stem or stripped.startswith(stem + "#"):
                return True
        return False

    def _abs_path(target: str) -> str:
        stripped = target
        while stripped.startswith("../"):
            stripped = stripped[3:]
        if stripped.startswith("./"):
            stripped = stripped[2:]
        return _GITHUB_BLOB_PREFIX + stripped

    def _sub(m: re.Match) -> str:
        target = m.group(1)
        # Skip absolute URLs, anchors, mailto, etc.
        if target.startswith(("http://", "https://", "#", "mailto:",
                              "tel:", "data:")):
            return m.group(0)
        # Skip links to the same docs/ tree (relative HTML siblings).
        if not target.startswith(("../", "./", "paper/", "audits/",
                                  "hypotheses/", "scripts/", "ideas/",
                                  "src/", "tests/", "configs/",
                                  "experiments/", "memory/", "skills/",
                                  "CLAUDE.md", "README.md", "PAPER.md",
                                  "MANIFESTO.md", "MINDMAP.md",
                                  "ARCHITECTURE.md", "FINDINGS.md")):
            return m.group(0)
        if _is_non_docs(target):
            return f"]({_abs_path(target)})"
        return m.group(0)

    return re.sub(r"\]\(([^)]+)\)", _sub, text)


def _md_to_html(text: str, *, inline_only: bool = False,
                strip_blockquote: bool = True) -> str:
    """Render a chunk of markdown text to safe HTML for inline display.

    Uses the ``markdown`` library (3.10+) with conservative extensions:
    ``tables``, ``fenced_code``, ``smarty``. Returns the empty string for
    empty input. When ``inline_only=True``, strips the wrapping ``<p>`` so
    the result fits inline with surrounding text. Markdown is trusted-input
    here (it comes from repo .md files the agent has read), so we do not
    pass through bleach.

    When ``strip_blockquote=True`` (default), leading ``>`` markdown
    blockquote markers are removed before parsing — this lets embedded
    GFM tables inside an excerpted blockquote render properly (the
    ``markdown`` library's blockquote nesting does not recurse into the
    tables extension).

    Per CLAUDE.md Rule 27, relative ``../<non-docs-target>`` markdown
    links are rewritten to absolute GitHub-blob URLs BEFORE markdown
    parsing so the resulting HTML is safe to mirror under
    ``docs/dashboard/`` (which is the GitHub Pages root) without 404s.
    """
    if not text or not text.strip():
        return ""
    # Rule 27 link discipline: rewrite non-docs relative paths to
    # absolute GitHub-blob URLs.
    text = _rewrite_rule27_links(text)
    if strip_blockquote:
        text = _strip_blockquote_markers(text)
    # Normalise stray un-paired bold/italic delimiters left over from
    # source-doc typos. The H09 sci-critic preamble has
    # ``*A...**B**.***`` (italic-with-bold-inside followed by THREE stars
    # at end) — that is a single-letter typo over ``*A...**B**.*``.
    # Markdown parses the malformed pattern as two adjacent italics +
    # one literal ``*`` left over. Strategy:
    #   (a) collapse ``****+`` to ``**`` so we never end up with empty
    #       bold tags,
    #   (b) rewrite ``...word.***\s|$`` to ``...word.*`` (collapsing the
    #       extra two stars that were a typo, preserving the intended
    #       italic-closer),
    #   (c) balance odd parity of ``**`` per paragraph (drop the last
    #       unmatched).
    def _fix_stars(line: str) -> str:
        line = re.sub(r"\*{4,}", "**", line)
        # ``word.***`` at end-of-line → ``word.*`` (preserve italic-close)
        line = re.sub(r"\*\*\*(\s|$)", r"*\1", line)
        # ``***word`` at start → ``*word`` (preserve italic-open)
        line = re.sub(r"(^|\s)\*\*\*", r"\1*", line)
        return line
    text = "\n".join(_fix_stars(ln) for ln in text.splitlines())
    # Drop unbalanced ``**`` at end of a paragraph (single stranded token).
    cleaned_paras: list[str] = []
    for para in text.split("\n\n"):
        if para.count("**") % 2 == 1:
            idx = para.rfind("**")
            para = para[:idx] + para[idx + 2:]
        cleaned_paras.append(para)
    text = "\n\n".join(cleaned_paras)
    try:
        import markdown as _md
        html = _md.markdown(
            text,
            extensions=["tables", "fenced_code", "smarty"],
            output_format="html5",
        )
    except Exception:
        # Fallback: minimal escape + paragraph wrap so output is at least
        # readable when the markdown library is missing.
        from html import escape as _e
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        html = "".join(f"<p>{_e(p)}</p>" for p in paragraphs)
    if inline_only and html.startswith("<p>") and html.endswith("</p>"):
        html = html[3:-4]
    return html

# Brutalist Editorial Lab Notebook — colour + type system
_BRUTALIST_VARS = """
 :root{
   --ink:#0a0a0d; --paper:#e6e1d6; --paper-dim:#a89e8c;
   --rule:#1c1c20; --rule-bright:#2a2a30;
   --panel:#111114; --panel2:#16161a;
   --accent:#bb8c4d; --accent-dim:#7a5e36;
   --v-pass:#3fb950; --v-minor:#d29922; --v-major:#f0883e; --v-broken:#f85149;
   --v-novel:#a371f7; --v-derivative:#58a6ff; --v-numerology:#8b949e;
   --v-falsified:#db6d28; --v-infra:#8b949e;
 }
 .grain{position:fixed;inset:0;pointer-events:none;z-index:2;opacity:0.035;
   background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");}
 @keyframes reveal{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
 section,.card,.kn-strip,.hero,.findings-row,details.deep{
   animation:reveal 360ms cubic-bezier(.2,.7,.2,1) both;
   animation-delay:calc(var(--i, 0) * 60ms);
 }
 html{scroll-behavior:smooth;}
"""

_EXP_PAGE_CSS = _BRUTALIST_VARS + """
 *{margin:0;padding:0;box-sizing:border-box;}
 html,body{background:var(--ink);}
 body{font-family:'Source Serif 4','Charter','Source Serif Pro',Georgia,serif;
      color:var(--paper);padding:32px 36px 80px;line-height:1.6;
      max-width:1180px;margin:0 auto;font-size:16px;
      font-variant-numeric:tabular-nums;position:relative;}
 a{color:var(--v-derivative);text-decoration:none;border-bottom:1px solid transparent;
   transition:border-color 160ms ease;}
 a:hover{border-bottom-color:var(--v-derivative);text-decoration:none;}
 h1{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
    font-size:34px;line-height:1.15;color:var(--paper);letter-spacing:-0.005em;
    margin-bottom:6px;}
 h2{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
    font-size:20px;color:var(--paper);margin-bottom:14px;letter-spacing:-0.003em;}
 h3{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:11px;
    text-transform:uppercase;letter-spacing:0.16em;color:var(--paper-dim);
    margin:18px 0 8px 0;}
 .md-body p{margin:8px 0 12px 0;}
 .md-body p:first-child{margin-top:0;} .md-body p:last-child{margin-bottom:0;}
 .md-body strong,.md-body b{color:var(--paper);font-weight:600;}
 .md-body em,.md-body i{font-style:italic;}
 .md-body code{font-family:'IBM Plex Mono',monospace;font-size:0.86em;
    background:var(--panel2);padding:1px 5px;border-radius:3px;}
 .md-body pre{font-family:'IBM Plex Mono',monospace;font-size:0.85em;
    background:var(--panel2);padding:10px 12px;border-radius:5px;
    border:1px solid var(--rule);overflow-x:auto;margin:10px 0;}
 .md-body ul,.md-body ol{margin:8px 0 12px 22px;}
 .md-body li{margin:3px 0;}
 .md-body blockquote{border-left:3px solid var(--accent-dim);
    padding:6px 14px;margin:10px 0;color:var(--paper-dim);font-style:italic;}
 .md-body table{border-collapse:collapse;margin:10px 0;font-size:0.92em;}
 .md-body th,.md-body td{border:1px solid var(--rule);padding:6px 10px;
    text-align:left;}
 .md-body th{background:var(--panel2);font-weight:600;}
 .mono{font-family:'IBM Plex Mono',monospace;font-size:0.85em;}
 .head-grid{display:grid;grid-template-columns:1fr auto;gap:24px;
            align-items:start;padding-bottom:18px;
            border-bottom:1px solid var(--rule);margin-bottom:24px;}
 .head-left .tag-display{font-family:'Source Serif 4',Georgia,serif;
    font-size:48px;font-weight:600;line-height:1.05;color:var(--paper);
    letter-spacing:-0.012em;word-break:break-word;}
 .head-left .sub{font-family:'IBM Plex Mono',monospace;font-size:11px;
    text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);
    margin-top:10px;}
 .head-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px;
             min-width:240px;}
 .head-right .back{font-family:'IBM Plex Mono',monospace;font-size:11px;
    text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);}
 .pill{display:inline-block;background:transparent;border:1px solid var(--rule-bright);
       border-radius:1px;padding:3px 10px;font-size:0.72em;color:var(--paper-dim);
       font-family:'IBM Plex Mono',monospace;margin:0 4px 4px 0;
       text-transform:uppercase;letter-spacing:0.12em;}
 .pill.hyp{border-color:var(--v-derivative);color:var(--v-derivative);}
 .pill.grp{border-color:var(--v-novel);color:var(--v-novel);}
 .pill.ds{border-color:var(--accent-dim);color:var(--accent);}
 /* Masthead CTA pills — repo / paper / literature survey (Phase 7) */
 .mast-pill{display:inline-block;padding:3px 10px;margin-right:6px;
            border:1px solid var(--accent-dim);color:var(--accent);
            font-family:'IBM Plex Mono',monospace;font-size:10px;
            text-transform:uppercase;letter-spacing:0.16em;font-weight:600;
            text-decoration:none;}
 .mast-pill:hover{background:var(--accent-dim);color:var(--ink);
                  border-bottom-color:var(--accent-dim);}
 .mast-pill.repo{border-color:var(--v-novel);color:var(--v-novel);}
 .mast-pill.repo:hover{background:var(--v-novel);color:var(--ink);}
 .mast-pill.lit{border-color:var(--v-derivative);color:var(--v-derivative);}
 .mast-pill.lit:hover{background:var(--v-derivative);color:var(--ink);}
 .mast-pill.paper{border-color:var(--v-pass);color:var(--v-pass);}
 .mast-pill.paper:hover{background:var(--v-pass);color:var(--ink);}
 .live-link{display:inline-block;padding:3px 10px;border:1px solid var(--v-pass);
            color:var(--v-pass);font-family:'IBM Plex Mono',monospace;
            font-size:10px;text-transform:uppercase;letter-spacing:0.18em;
            font-weight:600;text-decoration:none;}
 .live-link:hover{background:var(--v-pass);color:var(--ink);}
 /* Footer cross-link row */
 .doc-footer{margin:18px 0 6px 0;font-family:'IBM Plex Mono',monospace;
             font-size:10px;color:var(--paper-dim);letter-spacing:0.12em;
             text-transform:uppercase;}
 .doc-footer a{color:var(--accent);text-decoration:none;
               border-bottom:1px dotted var(--accent-dim);}
 .doc-footer a:hover{color:var(--v-pass);border-bottom-color:var(--v-pass);}
 .verdict-row{display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end;}
 .card{background:var(--panel);border:1px solid var(--rule);
       padding:24px 28px;margin-bottom:24px;position:relative;}
 .card::before{content:"";position:absolute;top:0;left:0;width:48px;
               height:1px;background:var(--accent);}
 .card p{margin-bottom:12px;font-size:0.96em;line-height:1.6;}
 .card ul{margin:6px 0 12px 22px;font-size:0.95em;line-height:1.6;}
 .card ul li{margin-bottom:4px;}
 details.deep{margin:14px 0;border-top:1px solid var(--rule);
              padding-top:14px;}
 details.deep > summary{cursor:pointer;list-style:none;
    font-family:'IBM Plex Mono',monospace;font-size:11px;text-transform:uppercase;
    letter-spacing:0.18em;color:var(--paper-dim);display:flex;align-items:center;
    gap:10px;padding:6px 0;transition:color 160ms ease;}
 details.deep > summary:hover{color:var(--accent);}
 details.deep > summary::-webkit-details-marker{display:none;}
 details.deep > summary::before{content:"▸";display:inline-block;
    transition:transform 200ms ease;color:var(--accent);font-size:14px;}
 details.deep[open] > summary::before{transform:rotate(90deg);}
 details.deep > .body{padding:14px 0 6px 24px;font-size:0.95em;
    border-left:1px solid var(--rule);margin-left:5px;margin-top:6px;
    animation:reveal 280ms ease;}
 table{width:100%;border-collapse:collapse;font-size:0.88em;
       font-variant-numeric:tabular-nums;}
 th{background:transparent;color:var(--paper-dim);text-align:left;
    padding:8px 12px;border-bottom:1px solid var(--rule-bright);
    font-size:10px;text-transform:uppercase;letter-spacing:0.18em;
    font-family:'IBM Plex Mono',monospace;font-weight:500;}
 td{padding:9px 12px;border-bottom:1px solid var(--rule);}
 td.k{color:var(--paper-dim);width:42%;font-family:'IBM Plex Serif',serif;}
 td.v{color:var(--paper);font-family:'IBM Plex Mono',monospace;}
 .formula-chip{background:var(--ink);border:1px solid var(--rule);
               padding:14px 16px;font-family:'IBM Plex Mono',monospace;
               font-size:0.82em;margin-bottom:14px;color:var(--paper);
               word-break:break-all;border-left:2px solid var(--accent);}
 .breakdown td.term{font-family:'IBM Plex Mono',monospace;}
 .pos{color:var(--v-pass);} .neg{color:var(--v-broken);} .mut{color:var(--paper-dim);}
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
 @media(max-width:980px){.charts{grid-template-columns:1fr;}
   .head-grid{grid-template-columns:1fr;} .head-right{align-items:flex-start;}
   .verdict-row{justify-content:flex-start;}}
 .meta{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--paper-dim);
       margin-top:28px;padding-top:18px;border-top:1px solid var(--rule);
       line-height:1.9;letter-spacing:0.04em;}
 .meta code{background:var(--ink);padding:1px 5px;color:var(--paper);
            border:1px solid var(--rule);}
 code{background:var(--ink);padding:1px 5px;font-size:0.92em;
      font-family:'IBM Plex Mono',monospace;border:1px solid var(--rule);}
 pre{background:var(--ink);border:1px solid var(--rule);
     padding:14px 16px;overflow-x:auto;font-family:'IBM Plex Mono',monospace;
     font-size:0.82em;color:var(--paper);white-space:pre-wrap;line-height:1.6;
     border-left:2px solid var(--accent-dim);}
 .reason-section{margin-bottom:16px;}
 .reason-section .lbl{color:var(--accent);font-family:'IBM Plex Mono',monospace;
                      font-size:10px;text-transform:uppercase;
                      letter-spacing:0.18em;margin-bottom:6px;font-weight:600;}
 .reason-section .body{font-size:0.95em;color:var(--paper);
                       white-space:pre-wrap;line-height:1.6;}
 .quote{border-left:2px solid var(--accent);padding:10px 18px;
        background:var(--ink);font-size:0.95em;color:var(--paper);
        margin:10px 0;font-family:'Source Serif 4',Georgia,serif;
        font-weight:400;line-height:1.55;}
 .quote.md-body{font-style:normal;}
 .findings-quote{border-left:2px solid var(--accent);padding:10px 18px;
        background:var(--ink);font-size:0.95em;color:var(--paper);
        margin:10px 0;font-family:'Source Serif 4',Georgia,serif;
        font-weight:400;line-height:1.55;}
 .verdict-label{color:var(--accent);font-weight:600;}
 .empty{color:var(--paper-dim);font-style:italic;font-size:0.9em;}
 .xrefs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
             gap:10px;margin-top:6px;}
 .xref-card{display:block;background:var(--ink);border:1px solid var(--rule);
            padding:12px 14px;text-decoration:none;
            transition:border-color 160ms ease,transform 160ms ease;}
 .xref-card:hover{border-color:var(--accent);transform:translateY(-1px);
                  text-decoration:none;}
 .xref-card .xref-lbl{font-family:'IBM Plex Mono',monospace;font-size:9px;
    text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);
    margin-bottom:6px;}
 .xref-card .xref-tag{font-family:'IBM Plex Mono',monospace;font-size:0.88em;
    color:var(--paper);font-weight:500;}
 .xref-card .xref-meta{font-family:'IBM Plex Mono',monospace;font-size:0.78em;
    color:var(--paper-dim);margin-top:4px;}
 /* Key numbers strip — the "tile bar" near top */
 .kn-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
           gap:1px;background:var(--rule);border:1px solid var(--rule);
           margin-bottom:24px;}
 .kn-tile{background:var(--panel);padding:18px 18px 16px 18px;
          position:relative;overflow:hidden;}
 .kn-tile .kn-val{font-family:'Source Serif 4',Georgia,serif;
                  font-size:28px;font-weight:600;line-height:1.05;
                  color:var(--paper);letter-spacing:-0.005em;}
 .kn-tile .kn-lbl{font-family:'IBM Plex Mono',monospace;font-size:9.5px;
                  text-transform:uppercase;letter-spacing:0.18em;
                  color:var(--paper-dim);margin-top:6px;}
 .kn-tile.tint-pos{box-shadow:inset 3px 0 0 var(--v-pass);}
 .kn-tile.tint-neg{box-shadow:inset 3px 0 0 var(--v-broken);}
 .kn-tile.tint-neu{box-shadow:inset 3px 0 0 var(--accent-dim);}
 /* Composite stacked bar */
 .comp-stack{display:flex;height:34px;border:1px solid var(--rule);
             margin:14px 0 8px 0;font-family:'IBM Plex Mono',monospace;
             font-size:9.5px;}
 .comp-stack .seg{display:flex;align-items:center;justify-content:center;
                  color:var(--ink);font-weight:600;
                  border-right:1px solid var(--ink);padding:0 6px;
                  white-space:nowrap;overflow:hidden;}
 .comp-stack .seg:last-child{border-right:none;}
 .comp-stack .seg.pos{background:var(--v-pass);}
 .comp-stack .seg.params{background:var(--accent);}
 .comp-stack .seg.lat{background:var(--v-derivative);}
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


# Brutalist verdict palette — hairline border, transparent fill, ink-coloured fg.
_VERDICT_COLORS = {
    "PASS":      "#3fb950",
    "MINOR":     "#d29922",
    "MAJOR":     "#f0883e",
    "BROKEN":    "#f85149",
    "NOVEL":     "#a371f7",
    "DERIVATIVE": "#58a6ff",
    "NUMEROLOGY": "#8b949e",
    "FALSIFIED":  "#db6d28",
    "UNFALSIFIABLE": "#a371f7",
    "INFRASTRUCTURE": "#8b949e",
}


def _verdict_badge(label: str, verdict: str, url: str = "",
                   tooltip: str = "") -> str:
    if not verdict:
        return ""
    base = verdict.split("+")[0].strip()
    fg = _VERDICT_COLORS.get(base, "#a89e8c")
    title = _esc(tooltip) if tooltip else _esc(verdict)
    inner = (
        f"<span class='vbadge' style='display:inline-block;padding:3px 10px;"
        f"border:1px solid {fg};color:{fg};font-weight:600;"
        f"font-size:10px;font-family:\"IBM Plex Mono\",monospace;"
        f"text-transform:uppercase;letter-spacing:0.16em;"
        f"transition:transform 160ms ease,filter 160ms ease;background:transparent' "
        f"title='{title}'>"
        f"<span style='opacity:0.65'>{_esc(label)}</span> &nbsp;{_esc(verdict)}"
        "</span>"
    )
    if url:
        return (
            f"<a href='{_esc(url)}' style='text-decoration:none;border-bottom:none;"
            f"margin-right:4px;display:inline-block'>{inner}</a>"
        )
    return f"<span style='margin-right:4px;display:inline-block'>{inner}</span>"


# Tiny hover-scale style for vbadge (injected once via .vbadge:hover)
_VBADGE_HOVER_CSS = (
    " .vbadge:hover{transform:scale(1.05);filter:brightness(1.15);} "
)


# ---------------------------------------------------------------------------
# Audit excerpt — per-hypothesis findings block from audits/G<X>_audit.md
# ---------------------------------------------------------------------------

def parse_audit_findings_block(repo_root: Path, hid: str,
                               group: str) -> str:
    """Return the FULL findings block for ``hid`` from ``audits/G<X>_audit.md``.

    This is the verbatim audit body between ``### Hxx`` headings — longer
    than the 1-sentence summary in ``parse_audit_verdict``.
    """
    if not hid or not group or not group.startswith("G"):
        return ""
    audit_md = repo_root / "audits" / f"{group}_audit.md"
    if not audit_md.exists():
        return ""
    try:
        text = audit_md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    m = re.search(
        rf"^###\s+{re.escape(hid)}\b.*?\n(.+?)(?=^###\s+H\d{{2}}\b|\Z)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not m:
        return ""
    block = m.group(1).strip()
    # Trim ridiculously long blocks
    return block[:4000]


def parse_scicritic_block(md_path: Path | None) -> str:
    """Return the verbatim 'Addendum: Research-Scientist Critique' body."""
    if md_path is None or not md_path.exists():
        return ""
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    m = re.search(
        r"##\s+Addendum:\s*Research-Scientist.*?\n(.+?)(?=\n##\s+(?!#)|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    return m.group(1).strip()[:4000]


# ---------------------------------------------------------------------------
# Per-group SVG visualisation for the aggregate dashboard
# ---------------------------------------------------------------------------

def _verdict_color_for_tag(tag: str, repo_root: Path | None) -> str:
    """Get the canonical verdict colour for a tag (impl-critic verdict)."""
    if repo_root is None:
        return "#58a6ff"
    hid, group = TAG_TO_HYP.get(tag, (None, "Uncategorized"))
    if not hid:
        return "#a89e8c"  # baseline = paper-dim
    try:
        v = parse_audit_verdict(repo_root, hid, group)
        base = (v.get("verdict") or "").split("+")[0].strip()
        return _VERDICT_COLORS.get(base, "#58a6ff")
    except Exception:
        return "#58a6ff"


def _group_scatter_svg(rows: list[pd.Series], group_letter: str,
                       chart_id: str, repo_root: Path | None,
                       width: int = 960, height: int = 280) -> str:
    """Composite-vs-params scatter for a hypothesis group.

    X axis: params (log scale, M), Y axis: composite. Dot colour = impl-critic
    verdict. Baseline highlighted with a hollow ring. Returns "" if <3 rows.
    """
    pts: list[dict] = []
    base_pts: list[dict] = []
    for r in rows:
        try:
            params_m = float(r.get("params", 0) or 0) / 1e6
            comp = float(r.get("composite", 0) or 0)
        except Exception:
            continue
        if params_m <= 0:
            continue
        tag = str(r.get("tag", ""))
        col = _verdict_color_for_tag(tag, repo_root)
        d = {
            "tag": tag, "params_m": params_m, "comp": comp,
            "top1": float(r.get("top1", 0) or 0),
            "seed": r.get("seed", ""),
            "dataset": str(r.get("dataset", "")),
            "color": col,
        }
        if tag.startswith("baseline_"):
            base_pts.append(d)
        else:
            pts.append(d)
    all_pts = pts + base_pts
    if len(all_pts) < 3:
        return ""
    import math
    ml, mr, mt, mb = 64, 28, 36, 44
    pw, ph = width - ml - mr, height - mt - mb
    xs = [math.log10(p["params_m"]) for p in all_pts]
    ys = [p["comp"] for p in all_pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 0.5
    if ymax == ymin:
        ymax = ymin + 0.05
    xspan = xmax - xmin
    yspan = ymax - ymin
    xmin -= xspan * 0.08
    xmax += xspan * 0.18
    ymin -= yspan * 0.08
    ymax += yspan * 0.10

    def sx(lx: float) -> float:
        return ml + pw * (lx - xmin) / (xmax - xmin)

    def sy(y: float) -> float:
        return mt + ph * (1 - (y - ymin) / (ymax - ymin))

    parts: list[str] = [
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='xMinYMin meet' "
        f"xmlns='http://www.w3.org/2000/svg' style='width:100%;height:auto;"
        f"background:#0a0a0d;border:1px solid #1c1c20;display:block;"
        f"margin:8px 0;font-family:\"IBM Plex Mono\",monospace' "
        f"id='{chart_id}'>",
        # title (Newsreader italic via tspan attrs)
        f"<text x='{ml}' y='22' fill='#e6e1d6' font-size='14' "
        f"font-family='Source Serif 4,Georgia,serif' "
        f"font-weight='600'>{_esc(group_letter)} · composite × params</text>",
    ]
    # Gridlines (5 horizontal)
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        yy = sy(yv)
        parts.append(
            f"<line x1='{ml}' y1='{yy:.1f}' x2='{ml + pw}' y2='{yy:.1f}' "
            f"stroke='#1c1c20' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{ml - 6}' y='{yy + 3:.1f}' fill='#a89e8c' "
            f"font-size='9' text-anchor='end'>{yv:.3f}</text>"
        )
    # X ticks (log10 powers)
    for lx in range(int(math.floor(xmin)), int(math.ceil(xmax)) + 1):
        if lx < xmin or lx > xmax:
            continue
        xx = sx(lx)
        parts.append(
            f"<line x1='{xx:.1f}' y1='{mt}' x2='{xx:.1f}' "
            f"y2='{mt + ph}' stroke='#1c1c20' stroke-width='1' "
            f"stroke-dasharray='2,3'/>"
        )
        parts.append(
            f"<text x='{xx:.1f}' y='{mt + ph + 14}' fill='#a89e8c' "
            f"font-size='9' text-anchor='middle'>10^{lx} M</text>"
        )
    # Axis labels (Newsreader italic)
    parts.append(
        f"<text x='{ml + pw - 4}' y='{mt + ph + 30}' fill='#a89e8c' "
        f"font-size='10' font-family='Source Serif 4,Georgia,serif' "
        f"text-anchor='end'>params (M) →</text>"
    )
    parts.append(
        f"<text x='12' y='{mt + ph / 2:.1f}' fill='#a89e8c' "
        f"font-size='10' font-family='Source Serif 4,Georgia,serif' "
        f"text-anchor='middle' "
        f"transform='rotate(-90 12 {mt + ph / 2:.1f})'>composite →</text>"
    )
    # Plot non-baseline dots
    for p in pts:
        cx = sx(math.log10(p["params_m"]))
        cy = sy(p["comp"])
        col = p["color"]
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='{col}' "
            f"fill-opacity='0.85' stroke='#0a0a0d' stroke-width='1'>"
            f"<title>{_esc(p['tag'])} · {p['dataset']} · seed{p['seed']} · "
            f"params {p['params_m']:.3f}M · composite {p['comp']:.4f} · "
            f"top-1 {p['top1']*100:.2f}%</title></circle>"
        )
        # Label (only when short)
        if len(p["tag"]) <= 22:
            parts.append(
                f"<text x='{cx + 7:.1f}' y='{cy + 3:.1f}' fill='#e6e1d6' "
                f"font-size='8.5' font-family='IBM Plex Mono,monospace'>"
                f"{_esc(p['tag'][:22])}</text>"
            )
    # Baseline = hollow ring
    for p in base_pts:
        cx = sx(math.log10(p["params_m"]))
        cy = sy(p["comp"])
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='7' fill='none' "
            f"stroke='#bb8c4d' stroke-width='2'>"
            f"<title>baseline · {_esc(p['tag'])} · {p['dataset']} · "
            f"composite {p['comp']:.4f}</title></circle>"
        )
        parts.append(
            f"<text x='{cx + 10:.1f}' y='{cy + 3:.1f}' fill='#bb8c4d' "
            f"font-size='9' font-family='IBM Plex Mono,monospace' "
            f"font-weight='600'>baseline {p['top1']*100:.1f}%</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _tag_sparkline_svg(history: list[dict] | None,
                       w: int = 100, h: int = 22) -> str:
    """Tiny per-row sparkline of test_top1 vs epoch for the leaderboard."""
    if not history:
        return "<span style='color:#484f58'>—</span>"
    pts = [(r.get("epoch"), r.get("test_top1")) for r in history
           if r.get("test_top1") is not None]
    if len(pts) < 2:
        return "<span style='color:#484f58'>—</span>"
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
    return (
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
        f"xmlns='http://www.w3.org/2000/svg' style='vertical-align:middle'>"
        f"<polyline points='{pts_str}' fill='none' stroke='#bb8c4d' "
        f"stroke-width='1.4'/>"
        f"<circle cx='{last_x:.1f}' cy='{last_y:.1f}' r='1.8' "
        f"fill='#e6e1d6'/></svg>"
    )


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
        f"style='width:100%;height:auto;background:#0a0a0d;border:1px solid "
        f"#1c1c20;display:block;margin-bottom:8px;"
        f"font-family:\"IBM Plex Mono\",monospace' "
        f"id='{chart_id}'>",
        f"<text x='{ml}' y='20' fill='#e6e1d6' font-size='14' "
        f"font-family='Source Serif 4,Georgia,serif' "
        f"font-weight='600'>{_esc(group_letter)} · top-1 distribution "
        f"<tspan fill='#a89e8c' font-size='11' "
        f"font-family='IBM Plex Mono,monospace' font-style='normal'>"
        f"(n={len(bars)})</tspan></text>",
    ]
    # Grid + x-axis ticks at 25 %, 50 %, 75 % of xmax
    for frac in (0.25, 0.5, 0.75, 1.0):
        xv = xmin + (xmax - xmin) * frac
        xx = sx(xv)
        parts.append(
            f"<line x1='{xx:.1f}' y1='{mt}' x2='{xx:.1f}' "
            f"y2='{mt + ph}' stroke='#1c1c20' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{xx:.1f}' y='{mt + ph + 14}' fill='#a89e8c' "
            f"font-size='9' text-anchor='middle'>{xv*100:.0f}%</text>"
        )
    parts.append(
        f"<text x='{ml + pw - 4}' y='{mt + ph + 26}' fill='#a89e8c' "
        f"font-size='10' font-family='Source Serif 4,Georgia,serif' "
        f"text-anchor='end'>top-1 →</text>"
    )
    # Baseline reference line(s) — accent gold
    bl_colors = {"cifar10": "#bb8c4d", "cifar100": "#db6d28"}
    for ds in sorted(datasets_in_group):
        if ds not in BASELINE_TOP1:
            continue
        bx = sx(BASELINE_TOP1[ds])
        col = bl_colors.get(ds, "#bb8c4d")
        parts.append(
            f"<line x1='{bx:.1f}' y1='{mt - 4}' x2='{bx:.1f}' "
            f"y2='{mt + ph + 4}' stroke='{col}' stroke-width='1.4' "
            f"stroke-dasharray='4,3' opacity='0.95'/>"
        )
        parts.append(
            f"<text x='{bx + 4:.1f}' y='{mt - 4}' fill='{col}' "
            f"font-size='9' font-family='IBM Plex Mono,monospace'>"
            f"{_esc(ds)} baseline {BASELINE_TOP1[ds]*100:.1f}%</text>"
        )
    # Bars + labels — colour by impl-critic verdict
    # Cache repo_root from globals; here we receive only rows, so look up via tag
    # We pass repo_root via globals as _ROOT_FOR_SVG below.
    for i, b in enumerate(bars):
        y = mt + i * (bh + gap)
        bx0 = ml
        bx1 = sx(b["top1"])
        col = _verdict_color_for_tag(b["tag"], _ROOT_FOR_SVG.get("root"))
        ds_tag = f"{b['tag']} · {b['dataset']} · seed{b['seed']}"
        parts.append(
            f"<rect x='{bx0:.1f}' y='{y}' width='{bx1 - bx0:.1f}' "
            f"height='{bh}' fill='{col}' opacity='0.85'>"
            f"<title>{_esc(ds_tag)} · top-1 {b['top1']*100:.2f}% · "
            f"params {b['params']/1e6:.3f}M · composite {b['composite']:.4f}</title>"
            f"</rect>"
        )
        parts.append(
            f"<text x='{ml - 6}' y='{y + bh - 3}' fill='#e6e1d6' "
            f"font-size='10' text-anchor='end' "
            f"font-family='IBM Plex Mono,monospace'>{_esc(b['tag'][:24])}</text>"
        )
        parts.append(
            f"<text x='{bx1 + 4:.1f}' y='{y + bh - 3}' fill='#a89e8c' "
            f"font-size='9' font-family='IBM Plex Mono,monospace'>"
            f"{b['top1']*100:.2f}%</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


# Module-level slot for passing repo_root to SVG builders called via maps.
_ROOT_FOR_SVG: dict[str, Path | None] = {"root": None}


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
                               repo_root: Path,
                               sci_badge_html: str = "",
                               impl_badge_html: str = "") -> str:
    """Hypothesis digest with always-visible summary + expandable full digest."""
    if not hid:
        return (
            "<section class='card' style='--i:2'><h2>Hypothesis</h2>"
            "<p class='empty'>No matching hypothesis document — this row "
            "is a baseline reference run.</p></section>"
        )
    md = find_hypothesis_path(repo_root, hid)
    if md is None:
        return (
            f"<section class='card' style='--i:2'>"
            f"<h2>Hypothesis {_esc(hid)}</h2>"
            f"<p class='empty'>(hypothesis doc not located on disk for "
            f"{_esc(hid)})</p></section>"
        )
    digest = parse_hypothesis_doc(md)
    rel = md.relative_to(repo_root).as_posix()
    gh_url = f"https://github.com/dlmastery/nature_inspired_networks/blob/main/{rel}"
    parts: list[str] = [
        f"<section class='card' style='--i:2'>"
        f"<h2>Hypothesis {_esc(hid)} — <i>{_esc(digest['title'] or hid)}</i></h2>"
    ]
    badges = []
    if impl_badge_html:
        badges.append(impl_badge_html)
    if sci_badge_html:
        badges.append(sci_badge_html)
    if badges:
        parts.append(
            f"<div style='margin:2px 0 18px 0'>{''.join(badges)}</div>"
        )
    if digest["oneline"]:
        parts.append(
            f"<p style='font-family:Source Serif 4,Georgia,serif;"
            f"font-size:1.06em;color:#e6e1d6;line-height:1.5;border-left:"
            f"2px solid var(--accent);padding-left:14px;margin:6px 0 16px 0'>"
            f"{_esc(digest['oneline'])}</p>"
        )
    if digest["motivation"]:
        parts.append(
            f"<h3>Motivation</h3><p>{_esc(digest['motivation'])}</p>"
        )
    if digest["formal"]:
        parts.append(
            f"<h3>Formal hypothesis</h3>"
            f"<p style='font-family:Source Serif 4,Georgia,serif;"
            f"color:#e6e1d6'>{_esc(digest['formal'])}</p>"
        )
    # Expandable deep-dive with mechanism / falsifier / predicted / citation
    deep_bits: list[str] = []
    if digest["mechanism"]:
        deep_bits.append(
            f"<h3>Mechanism (because…)</h3>"
            f"<p>{_esc(digest['mechanism'])}</p>"
        )
    if digest["falsifier"]:
        deep_bits.append(
            f"<h3>Numeric falsifier</h3><p>{_esc(digest['falsifier'])}</p>"
        )
    if digest["predicted"]:
        deep_bits.append(
            f"<h3>Predicted Δ</h3><p>{_esc(digest['predicted'])}</p>"
        )
    if digest["citation"]:
        deep_bits.append(
            f"<h3>Primary citation</h3>"
            f"<p class='mono'>{_esc(digest['citation'])}</p>"
        )
    if deep_bits:
        parts.append(
            "<details class='deep'><summary>Read the full design-doc digest"
            "</summary><div class='body'>"
            + "".join(deep_bits)
            + f"<p style='margin-top:14px'>"
            f"<a href='{_esc(gh_url)}'>↗ Full design doc on GitHub</a> "
            f"<span class='mono' style='color:var(--paper-dim);"
            f"font-size:0.78em'>{_esc(rel)}</span></p>"
            "</div></details>"
        )
    else:
        parts.append(
            f"<p style='margin-top:14px'>"
            f"<a href='{_esc(gh_url)}'>↗ Full design doc on GitHub</a> "
            f"<span class='mono' style='color:var(--paper-dim);"
            f"font-size:0.78em'>{_esc(rel)}</span></p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _render_verdict_section(repo_root: Path, tag: str) -> str:
    # FINDINGS.md was moved to paper/ in the 2026-05-29 root-cleanup pass.
    # Prefer the new location; fall back to the legacy path so historical
    # checkouts still resolve correctly.
    findings_md = repo_root / "paper" / "FINDINGS.md"
    if not findings_md.exists():
        findings_md = repo_root / "FINDINGS.md"
    blurb = parse_findings_for_tag(findings_md, tag)
    gh_url = (
        "https://github.com/dlmastery/nature_inspired_networks/blob/main/"
        "paper/FINDINGS.md"
    )
    if not blurb:
        return (
            "<section class='card' style='--i:3'>"
            "<h2>Verdict <span class='mono' style='font-size:0.55em;"
            "letter-spacing:0.18em;color:var(--paper-dim);font-style:normal'>"
            "FINDINGS.md</span></h2>"
            "<p class='empty'>No paragraph in FINDINGS.md mentions this tag "
            "yet.</p>"
            f"<p><a href='{gh_url}'>↗ Browse FINDINGS.md</a></p></section>"
        )
    return (
        "<section class='card' style='--i:3'>"
        "<h2>Verdict <span class='mono' style='font-size:0.55em;"
        "letter-spacing:0.18em;color:var(--paper-dim);font-style:normal'>"
        "FINDINGS.md</span></h2>"
        f"<div class='quote md-body'>{_md_to_html(blurb)}</div>"
        f"<p style='margin-top:14px'><a href='{gh_url}'>↗ Full FINDINGS.md on GitHub</a></p></section>"
    )


def _render_reasoning_section(reasoning_path: Path | None,
                              has_findings_verdict: bool = False) -> str:
    if reasoning_path is None or not reasoning_path.exists():
        if has_findings_verdict:
            return (
                "<p class='empty'>No <code>reasoning.json</code> archived "
                "for this run directory &mdash; see the <b>Verdict</b> card "
                "above for the FINDINGS-derived verdict and the "
                "<b>Implementation-critic audit</b> accordion for the "
                "code-level diagnosis.</p>"
            )
        return (
            "<p class='empty'>No <code>reasoning.json</code> for this run "
            "directory.</p>"
        )
    try:
        data = json.loads(reasoning_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"<p class='empty'>Failed to parse: {_esc(exc)}.</p>"
    if isinstance(data, list):
        data = data[0] if data else {}
    parts: list[str] = []
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
            f"<div class='reason-section'>"
            f"<div class='lbl'>{label}</div>{body}</div>"
        )
    if not parts:
        return "<p class='empty'>reasoning.json is empty.</p>"
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
            f"<pre>{_esc(text)}</pre>"
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
            "<p class='empty'>No <code>config.yaml</code> in run directory "
            "and neither <code>metrics.json</code> nor "
            "<code>experiment_log.jsonl</code> carries recoverable config "
            "overrides.</p>"
        )
    src = "experiment_log.jsonl + metrics.json" if log_row else "metrics.json"
    return (
        f"<p class='mono' style='color:var(--paper-dim);font-size:0.8em;"
        f"margin-bottom:8px'>inferred from {src}</p>"
        f"<pre>{_esc(json.dumps(inferred, indent=2))}</pre>"
    )


def _render_audit_excerpt(hid: str | None, group: str,
                          repo_root: Path) -> str:
    """Return the full audit excerpt body (for an accordion)."""
    if not hid:
        return ""
    audit = parse_audit_verdict(repo_root, hid, group)
    block = parse_audit_findings_block(repo_root, hid, group)
    if not audit["verdict"] and not block:
        return ""
    badge = _verdict_badge("impl-critic", audit.get("verdict", ""),
                           url=audit.get("url", ""),
                           tooltip=audit.get("summary", ""))
    body = block or audit.get("summary") or "(see full audit on GitHub)"
    parts = []
    if badge:
        parts.append(f"<div style='margin-bottom:12px'>{badge}</div>")
    parts.append(f"<div class='quote md-body'>{_md_to_html(body)}</div>")
    if audit.get("url"):
        local = audit.get("local", "")
        local_caption = (
            f" <span class='mono' style='color:var(--paper-dim);"
            f"font-size:0.78em'>{_esc(local)}</span>"
        ) if local else ""
        parts.append(
            f"<p style='margin-top:12px'>"
            f"<a href='{_esc(audit['url'])}'>↗ Full audit on GitHub</a>"
            f"{local_caption}</p>"
        )
    return "".join(parts)


def _render_scicritic_excerpt(md_path: Path | None) -> str:
    """Return the full sci-critic addendum body (for an accordion)."""
    if md_path is None:
        return ""
    sci = parse_scicritic_verdict(md_path)
    block = parse_scicritic_block(md_path)
    if not sci["verdict"] and not block:
        return ""
    badge = _verdict_badge("sci-critic", sci.get("verdict", ""),
                           tooltip=sci.get("raw", ""))
    parts: list[str] = []
    if badge:
        parts.append(f"<div style='margin-bottom:12px'>{badge}</div>")
    if block:
        # Render as preserved-markdown style block
        parts.append(
            f"<div class='md-body' style='border-left:2px solid var(--v-novel);"
            f"padding:8px 14px;background:var(--panel2);border-radius:5px'>"
            f"{_md_to_html(block)}</div>"
        )
    elif sci.get("raw"):
        parts.append(f"<div class='quote'>{_esc(sci['raw'])}</div>")
    return "".join(parts)


def _render_key_numbers_strip(metrics: dict,
                              baseline_top1: float | None,
                              tier: str = "SCREENING",
                              seed_count: int = 1) -> str:
    """Brutalist tile bar of key headline numbers (top of page).

    The Δ vs baseline tile is labelled with the Rule-28 seed-count
    framing — "(n=1, screening)" or "(n=3, evaluation)" — so a reader
    knows whether the claim is screening-tier or evaluation-tier.
    """
    def f(k):
        v = metrics.get(k)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    top1 = f("top1")
    comp = f("composite")
    params = f("params")
    lat = f("latency_ms")
    tiles: list[tuple[str, str, str]] = []
    if top1 is not None:
        tint = "tint-neu"
        if baseline_top1 is not None:
            tint = "tint-pos" if top1 >= baseline_top1 else "tint-neg"
        tiles.append((f"{top1*100:.2f}%", "top-1", tint))
    if comp is not None:
        tiles.append((f"{comp:.4f}", "composite", "tint-neu"))
    if top1 is not None and baseline_top1 is not None:
        delta = (top1 - baseline_top1) * 100
        sign = "+" if delta >= 0 else "−"
        tint = "tint-pos" if delta >= 0 else "tint-neg"
        tier_label = tier.lower()
        delta_lbl = (
            f"Δ vs baseline (n={seed_count}, {tier_label})"
        )
        tiles.append((f"{sign}{abs(delta):.2f}pp", delta_lbl, tint))
    if params is not None:
        tiles.append((f"{params/1e6:.3f}M", "params", "tint-neu"))
    if lat is not None:
        tiles.append((f"{lat:.2f}ms", "latency b=1", "tint-neu"))
    if not tiles:
        return ""
    cells = "".join(
        f"<div class='kn-tile {tint}'>"
        f"<div class='kn-val'>{_esc(v)}</div>"
        f"<div class='kn-lbl'>{_esc(l)}</div></div>"
        for v, l, tint in tiles
    )
    return f"<div class='kn-strip' style='--i:1'>{cells}</div>"


def _render_composite_stacked_bar(metrics: dict, formula: str) -> str:
    """Horizontal stacked bar of composite terms with formula chip below."""
    import math
    try:
        top1 = float(metrics["top1"])
        params_m = float(metrics["params"]) / 1e6
        lat = float(metrics["latency_ms"])
        t_params = -0.05 * math.log10(params_m) if params_m > 0 else 0.0
        t_lat = -0.05 * math.log10(lat) if lat > 0 else 0.0
        total = top1 + t_params + t_lat
        reported = float(metrics.get("composite", total))
    except Exception:
        return (
            f"<div class='formula-chip'>composite ≔ "
            f"<code>{_esc(formula)}</code></div>"
            "<p class='empty'>Insufficient fields to render the composite "
            "stacked bar.</p>"
        )
    # Compute fractional widths from absolute values
    abs_sum = abs(top1) + abs(t_params) + abs(t_lat)
    if abs_sum <= 0:
        abs_sum = 1.0
    w_top = abs(top1) / abs_sum * 100
    w_pp = abs(t_params) / abs_sum * 100
    w_pl = abs(t_lat) / abs_sum * 100
    sign_p = "+" if t_params >= 0 else "−"
    sign_l = "+" if t_lat >= 0 else "−"
    bar = (
        f"<div class='comp-stack'>"
        f"<div class='seg pos' style='width:{w_top:.2f}%' "
        f"title='top-1 = {top1:.5f}'>top-1 +{top1:.4f}</div>"
        f"<div class='seg params' style='width:{w_pp:.2f}%' "
        f"title='-0.05·log10(params_M)'>params {sign_p}{abs(t_params):.5f}</div>"
        f"<div class='seg lat' style='width:{w_pl:.2f}%' "
        f"title='-0.05·log10(latency_ms)'>lat {sign_l}{abs(t_lat):.5f}</div>"
        f"</div>"
    )
    overlay = (
        f"<div style='display:flex;justify-content:space-between;"
        f"font-family:IBM Plex Mono,monospace;font-size:11px;"
        f"color:var(--paper-dim);margin-top:6px'>"
        f"<span>Σ recomputed: <b style='color:var(--paper)'>{total:.5f}</b>"
        f"</span>"
        f"<span>reported: <b style='color:var(--paper)'>{reported:.5f}</b>"
        f" &nbsp;·&nbsp; Δ {(reported-total):+.6f}</span></div>"
    )
    chip = (
        f"<div class='formula-chip' style='margin-top:14px'>composite ≔ "
        f"<code>{_esc(formula)}</code></div>"
    )
    return bar + overlay + chip


def _render_cross_refs_grid(run_dir_name: str, tag: str, dataset: str,
                            results_dir: Path,
                            df: pd.DataFrame | None = None,
                            group_letter: str | None = None,
                            composite: float | None = None) -> str:
    """Grid of xref cards (other seeds / datasets / nearest-in-group)."""
    cards: list[str] = []
    same_ds_dir = results_dir / dataset
    if same_ds_dir.exists():
        for sib in sorted(same_ds_dir.iterdir()):
            if (not sib.is_dir() or not sib.name.startswith(f"{tag}_seed")
                    or sib.name == run_dir_name):
                continue
            href = (
                f"../experiments/"
                f"{run_page_filename(sib.name, dataset=dataset)}"
            )
            seed = sib.name.rsplit("_seed", 1)[-1]
            cards.append(
                f"<a class='xref-card' href='{_esc(href)}'>"
                f"<div class='xref-lbl'>↪ other seed</div>"
                f"<div class='xref-tag'>{_esc(tag)}</div>"
                f"<div class='xref-meta'>{_esc(dataset)} · seed {_esc(seed)}</div>"
                "</a>"
            )
    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == dataset:
            continue
        for sib in sorted(ds_dir.iterdir()):
            if not sib.is_dir() or not sib.name.startswith(f"{tag}_seed"):
                continue
            href = (
                f"../experiments/"
                f"{run_page_filename(sib.name, dataset=ds_dir.name)}"
            )
            seed = sib.name.rsplit("_seed", 1)[-1]
            cards.append(
                f"<a class='xref-card' href='{_esc(href)}'>"
                f"<div class='xref-lbl'>↪ same tag, other dataset</div>"
                f"<div class='xref-tag'>{_esc(tag)}</div>"
                f"<div class='xref-meta'>{_esc(ds_dir.name)} · seed "
                f"{_esc(seed)}</div></a>"
            )
    if (df is not None and group_letter is not None
            and composite is not None and not df.empty):
        neighbours = _nearest_in_group(
            df, group_letter, tag, dataset, composite, k=3,
        )
        for n in neighbours:
            run_name = n["run_dir"].split("/")[-1]
            href = (
                f"../experiments/"
                f"{run_page_filename(run_name, dataset=n['dataset'])}"
            )
            delta = n["composite"] - composite
            sign = "+" if delta >= 0 else "−"
            cards.append(
                f"<a class='xref-card' href='{_esc(href)}'>"
                f"<div class='xref-lbl'>↪ nearest in {_esc(group_letter)}"
                f"</div>"
                f"<div class='xref-tag'>{_esc(n['tag'])}</div>"
                f"<div class='xref-meta'>{n['top1']*100:.2f}% · "
                f"Δ-comp {sign}{abs(delta):.4f}</div></a>"
            )
    if not cards:
        return (
            "<section class='card' style='--i:8'><h2>Cross-references</h2>"
            "<p class='empty'>No companion runs found.</p></section>"
        )
    return (
        "<section class='card' style='--i:8'><h2>Cross-references</h2>"
        f"<div class='xrefs-grid'>{''.join(cards)}</div></section>"
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


def _hillclimb_path_for_tag(repo_root: Path | None, tag: str) -> Path | None:
    """Return ``ideas/<NN>/hillclimb_results.json`` for ``tag`` if it exists.

    The mapping is discovered by reading every
    ``ideas/*/hillclimb_results.json`` and matching the embedded ``tag``
    field. This avoids a hard-coded TAG_TO_IDEA dict and keeps the
    dashboard build self-discovering of new hill-climb sweeps.
    """
    if repo_root is None:
        return None
    ideas_dir = repo_root / "ideas"
    if not ideas_dir.exists():
        return None
    for sub in ideas_dir.iterdir():
        if not sub.is_dir():
            continue
        p = sub / "hillclimb_results.json"
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if str(data.get("tag", "")).strip() == tag:
            return p
    return None


def _render_hillclimb_pill(repo_root: Path | None, tag: str) -> str:
    """Header pill linking to ``ideas/<NN>/dashboard/index.html`` if present.

    Rendered only when a per-hypothesis hill-climb sweep has produced
    ``ideas/<NN>/hillclimb_results.json``. Per CLAUDE.md Rule 27, the
    link is an absolute GitHub-blob URL — the per-experiment page lives
    under ``dashboard/experiments/`` and ``ideas/<NN>/dashboard/`` is NOT
    mirrored into ``docs/``, so a relative ``../../ideas/...`` href would
    404 on the GitHub Pages mirror.
    """
    hc_path = _hillclimb_path_for_tag(repo_root, tag)
    if hc_path is None:
        return ""
    idea_dir = hc_path.parent.name
    href = (
        "https://github.com/dlmastery/nature_inspired_networks/blob/main/"
        f"ideas/{idea_dir}/dashboard/index.html"
    )
    return (
        f"<a class='mast-pill hillclimb' href='{_esc(href)}' "
        "target='_blank' rel='noopener' "
        "title='Per-hypothesis hill-climb dashboard "
        "(Phase-9a, lr × wd × bs × optimizer cube)'>"
        "⛰️ hill-climb</a>"
    )


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
        # Cross-link row: repo · paper · background reading · live demo · GitHub Pages.
        "<div class='doc-footer'>"
        "<a href='https://github.com/dlmastery/nature_inspired_networks' "
        "target='_blank' rel='noopener'>Repo</a> &middot; "
        "<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/PAPER.md' "
        "target='_blank' rel='noopener'>Paper</a> &middot; "
        "<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/NATURE_INSPIRED_NETWORKS.md' "
        "target='_blank' rel='noopener'>Background reading</a> &middot; "
        "<a href='https://dlmastery.github.io/nature_inspired_networks/' "
        "target='_blank' rel='noopener'>Live demo</a> &middot; "
        "<a href='https://dlmastery.github.io/nature_inspired_networks/' "
        "target='_blank' rel='noopener'>GitHub Pages</a>"
        "</div>"
        "<div class='meta'>"
        f"composite formula SHA-256 &nbsp; <code>{_esc(fp)}</code><br>"
        f"epochs run &nbsp; <code>{_esc(epochs)}</code> "
        f"&nbsp;·&nbsp; train wall-clock &nbsp; <code>{train_s_disp}</code> "
        f"&nbsp;·&nbsp; gen-gap (train − test) &nbsp; <code>{gen_gap_disp}</code>"
        f"{sha_html}<br>"
        f"run directory &nbsp; <code>{_esc(run_path_rel)}</code><br>"
        "Generated by "
        "<code>nature_inspired_networks.dashboard.render_experiment_page</code> "
        "&middot; self-contained inline SVG, no external assets "
        "&middot; brutalist editorial lab notebook v3"
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

    # ---- gather audit + sci verdict badges for the header --------------
    md_path = find_hypothesis_path(repo_root, hid) if hid else None
    sci = parse_scicritic_verdict(md_path) if md_path else {
        "verdict": "", "raw": ""
    }
    audit_v = (parse_audit_verdict(repo_root, hid, group)
               if hid else {"verdict": "", "summary": "", "url": ""})
    impl_badge = _verdict_badge(
        "impl", audit_v.get("verdict", ""),
        url=audit_v.get("url", ""),
        tooltip=audit_v.get("summary", ""),
    ) if audit_v.get("verdict") else ""
    sci_badge = _verdict_badge(
        "sci", sci.get("verdict", ""),
        tooltip=sci.get("raw", ""),
    ) if sci.get("verdict") else ""

    # ---- hypothesis section (always visible) ---------------------------
    hyp_html = _render_hypothesis_section(
        hid, group, repo_root,
        sci_badge_html=sci_badge, impl_badge_html=impl_badge,
    )
    if hid is not None and md_path is not None:
        flags["hypothesis"] = True
        if sci.get("verdict"):
            flags["scicritic"] = True

    # ---- audit excerpt (for accordion) ---------------------------------
    audit_excerpt = _render_audit_excerpt(hid, group, repo_root)
    if audit_excerpt:
        flags["audit"] = True

    # ---- sci-critic excerpt (for accordion) ----------------------------
    sci_excerpt = _render_scicritic_excerpt(md_path)

    # ---- FINDINGS verdict (always visible card) ------------------------
    verdict_html = _render_verdict_section(repo_root, tag)
    # FINDINGS.md moved to paper/ in the 2026-05-29 root-cleanup. Try the
    # new location first; fall back to the legacy path.
    _findings_for_flag = repo_root / "paper" / "FINDINGS.md"
    if not _findings_for_flag.exists():
        _findings_for_flag = repo_root / "FINDINGS.md"
    if parse_findings_for_tag(_findings_for_flag, tag):
        flags["verdict"] = True

    # ---- reasoning + config (accordion bodies) -------------------------
    reasoning_path = run_dir / "reasoning.json"
    reasoning_body = _render_reasoning_section(
        reasoning_path if reasoning_path.exists() else None,
        has_findings_verdict=flags["verdict"],
    )
    flags["reasoning"] = reasoning_path.exists()

    config_body = _render_configuration_section(
        run_dir, metrics, results_dir=results_dir,
    )
    if (run_dir / "config.yaml").exists() or "inferred from" in config_body:
        flags["config"] = True

    # Composite for nearest-neighbour distance ranking
    try:
        comp = float(metrics.get("composite"))
    except Exception:
        comp = None
    xrefs_html = _render_cross_refs_grid(
        run_dir_name, tag, dataset, results_dir,
        df=runs_df, group_letter=group, composite=comp,
    )
    if "xref-card" in xrefs_html:
        flags["cross_refs"] = True

    # ---- key numbers strip (per-experiment quick read) -----------------
    baseline_top1 = BASELINE_TOP1.get(dataset)
    # Rule 28 — seed-count framing (n=1 SCREENING vs n>=3 EVALUATION).
    seed_count = _seed_count_for_tag(results_dir, tag, dataset)
    if seed_count < 1:
        seed_count = 1  # at minimum this run itself
    tier = _evaluation_tier(tag, dataset, seed_count)
    key_strip = _render_key_numbers_strip(
        metrics, baseline_top1, tier=tier, seed_count=seed_count,
    )

    # ---- composite stacked bar -----------------------------------------
    composite_stack = _render_composite_stacked_bar(metrics, formula)

    footer_html = _render_footer(metrics, run_dir, repo_root=repo_root)

    # ---- pretty-print metrics JSON for accordion -----------------------
    try:
        metrics_pretty = json.dumps(metrics, indent=2, ensure_ascii=False)
    except Exception:
        metrics_pretty = str(metrics)

    # ---- header pills + verdicts ---------------------------------------
    # Multi-hypothesis aware pill list — combo / loo / pair / slot tags
    # contribute more than one H-ID (per Rule 23 orthogonal-axes stacks);
    # show ALL of them rather than just the leading one.
    all_hids = hypotheses_for_tag(tag)
    if all_hids:
        hyp_pill = "".join(
            f"<span class='pill hyp'>{_esc(h)}</span>" for h in all_hids
        )
    else:
        hyp_pill = "<span class='pill'>baseline</span>"
    grp_pill = (
        f"<span class='pill grp'>{_esc(grp_header)}</span>"
        if group != "Uncategorized" else ""
    )
    ds_pill = f"<span class='pill ds'>{_esc(dataset)}</span>"
    seed_pill = f"<span class='pill'>seed {_esc(seed)}</span>"
    # Rule-28 tier badge: SCREENING (n=1) or EVALUATION (n>=3 Phase-5 gate).
    tier_color = "var(--v-pass)" if tier == "EVALUATION" else "var(--v-minor)"
    tier_badge = (
        f"<span class='pill' style='border-color:{tier_color};"
        f"color:{tier_color}' title='Rule 28 (CLAUDE.md): SCREENING = "
        f"n=1 seed; EVALUATION = n>=3 seeds AND worst-leader-seed > "
        f"best-baseline-seed Phase-5 gate cleared'>"
        f"tier {_esc(tier.lower())} &nbsp;n={seed_count}"
        f"</span>"
    )

    # ---- accordions ----------------------------------------------------
    accordions: list[str] = []
    if sci_excerpt:
        accordions.append(
            "<details class='deep'><summary>Research-scientist critique "
            "(addendum)</summary><div class='body'>"
            + sci_excerpt
            + "</div></details>"
        )
    if audit_excerpt:
        accordions.append(
            "<details class='deep'><summary>Implementation-critic audit "
            f"(audits/{_esc(group)}_audit.md)</summary>"
            f"<div class='body'>{audit_excerpt}</div></details>"
        )
    accordions.append(
        "<details class='deep'><summary>Reasoning blob "
        "(reasoning.json)</summary>"
        f"<div class='body'>{reasoning_body}</div></details>"
    )
    accordions.append(
        "<details class='deep'><summary>Configuration dump</summary>"
        f"<div class='body'>{config_body}</div></details>"
    )
    accordions.append(
        "<details class='deep'><summary>Full metrics JSON</summary>"
        f"<div class='body'><pre>{_esc(metrics_pretty)}</pre></div></details>"
    )

    deep_card = (
        "<section class='card' style='--i:6'>"
        "<h2>Deep inspection</h2>"
        "<p style='color:var(--paper-dim);font-size:0.9em;margin-bottom:6px'>"
        "Audit excerpts, reasoning blob, raw configuration, and full metrics "
        "JSON — expand sections of interest.</p>"
        + "".join(accordions)
        + "</section>"
    )

    page = (
        "<!doctype html>\n"
        "<html lang='en'><head><meta charset='utf-8'>\n"
        f"<title>{_esc(title)} — experiment page</title>\n"
        f"{_EXP_FONT_LINK}\n"
        f"<style>{_EXP_PAGE_CSS}{_VBADGE_HOVER_CSS}</style></head><body>\n"
        "<div class='grain'></div>\n"
        # Asymmetric 3-column header
        "<div class='head-grid' style='--i:0'>\n"
        "  <div class='head-left'>\n"
        f"    <div class='tag-display'>{_esc(tag)}</div>\n"
        f"    <div class='sub'>{hyp_pill}{grp_pill}{ds_pill}{seed_pill}"
        f"{tier_badge}"
        f"&nbsp;·&nbsp; run dir <span class='mono' style='color:var(--paper)'>"
        f"{_esc(run_dir_name)}</span></div>\n"
        # Masthead CTA pills — repo / paper / literature survey / live demo.
        "    <div class='mast-row' style='margin-top:8px;'>"
        "<a class='mast-pill repo' "
        "href='https://github.com/dlmastery/nature_inspired_networks' "
        "target='_blank' rel='noopener'>🔗 source</a>"
        "<a class='mast-pill lit' "
        "href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/NATURE_INSPIRED_NETWORKS.md' "
        "target='_blank' rel='noopener'>📚 background reading</a>"
        "<a class='mast-pill paper' "
        "href='https://github.com/dlmastery/nature_inspired_networks/blob/main/PAPER.md' "
        "target='_blank' rel='noopener'>📄 paper</a>"
        + _render_hillclimb_pill(repo_root, tag) +
        "<a class='live-link' href='https://dlmastery.github.io/nature_inspired_networks/' "
        "target='_blank' rel='noopener'>📡 live</a>"
        "</div>\n"
        "  </div>\n"
        "  <div class='head-right'>\n"
        "    <a class='back' href='../dashboard.html'>← back to dashboard</a>\n"
        f"    <div class='verdict-row'>{impl_badge}{sci_badge}</div>\n"
        "  </div>\n"
        "</div>\n"
        # Always-visible key-numbers strip
        f"{key_strip}\n"
        # Hypothesis (always-visible summary + expandable digest)
        f"{hyp_html}\n"
        # FINDINGS verdict card
        f"{verdict_html}\n"
        # Composite breakdown — stacked bar
        "<section class='card' style='--i:4'>"
        "<h2>Composite-score breakdown</h2>"
        f"{composite_stack}"
        "</section>\n"
        # Training curves card
        "<section class='card' style='--i:5'>"
        "<h2>Per-epoch training curves</h2>"
        f"{charts_html}"
        "</section>\n"
        # Deep-inspection accordions
        f"{deep_card}\n"
        # Quick-glance metrics table
        "<section class='card' style='--i:7'>"
        "<h2>Metrics — quick reference</h2>"
        f"{metrics_table}"
        "</section>\n"
        # Cross references grid
        f"{xrefs_html}\n"
        f"{footer_html}\n"
        "</body></html>\n"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return flags


# ---------------------------------------------------------------------------
# Phase-8 winners (Rule 28 + Rule 34) — exposed for SVG annotation
# ---------------------------------------------------------------------------
PHASE8_WINNERS: set[str] = {"pair_gm_pdw", "slot_act_sine", "sg_only_phi_budget"}

# Per-group colour palette for small-multiples (Phase C / D).
# Distinct hues per CLAUDE.md hypothesis group; baseline + combo override last.
GROUP_COLORS: dict[str, str] = {
    "Baseline":     "#bb8c4d",  # accent gold
    "G1":           "#58a6ff",  # azure (scaling)
    "G2":           "#3fb950",  # green (layer/channel)
    "G3":           "#a371f7",  # violet (topologies)
    "G4":           "#f0883e",  # orange (kernels/attention)
    "G5":           "#d29922",  # amber (optimisation)
    "G6":           "#db6d28",  # ember (topological-bridging)
    "G7":           "#f85149",  # red (cross-paradigm hybrids)
    "G8":           "#8b949e",  # slate (esoteric)
    "Combo":        "#e6e1d6",  # paper (combo/pair/loo/slot stacks)
    "Uncategorized": "#484f58",
}


def _group_for_tag(tag: str) -> str:
    """Resolve a tag to its CLAUDE.md hypothesis-group letter for colouring.

    Combo / pair / loo / slot stacks resolve to the "Combo" pseudo-group so
    they read as a distinct family on the Pareto + ablation small-multiples.
    """
    if tag.startswith("baseline_"):
        return "Baseline"
    if tag.startswith(("combo", "pair_", "loo_", "slot_")):
        return "Combo"
    hp = TAG_TO_HYP.get(tag)
    if hp and hp[1]:
        return hp[1]
    return "Uncategorized"


def _pareto_panels_svg(rows: list[dict],
                       title_prefix: str = "Pareto fronts") -> str:
    """Three side-by-side SVG panels: top-1 vs params / FLOPs / latency.

    Replaces the legacy single PNG `plot_pareto.png` with 3 small-multiples
    per CLAUDE.md Rule 33 (autoresearch-dashboard-comprehension SKILL.md
    Pillar 1). Each panel ~360px wide, log-scale x-axis, shared y-axis
    (top-1 %). Points coloured by hypothesis group (GROUP_COLORS); the
    3 Phase-8 winners (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget`)
    are rendered as larger filled circles with a star (★) marker;
    baseline_resnet20 is explicitly labelled. A dashed convex-frontier
    overlay highlights Pareto-efficient points per panel. Each panel
    carries a 1-sentence "what to read" caption ABOVE; a shared
    group→colour legend sits BELOW. Hover tooltips list tag, top-1,
    axis value, composite, n=seeds.

    rows: list of dicts with keys tag, dataset, seed, top1, params,
    flops, latency_ms, composite. The SVG is dataset-agnostic — it
    plots every row regardless of dataset (typical caller passes the
    CIFAR-10 12-ep sweep).
    """
    import math
    pts: list[dict] = []
    for r in rows:
        try:
            top1 = float(r.get("top1") or 0)
            params = float(r.get("params") or 0)
            flops = float(r.get("flops") or 0)
            lat = float(r.get("latency_ms") or 0)
            comp = float(r.get("composite") or 0)
        except Exception:
            continue
        if top1 <= 0 or params <= 0 or lat <= 0:
            continue
        tag = str(r.get("tag", ""))
        pts.append({
            "tag": tag,
            "top1": top1 * 100,           # display as %
            "params_m": params / 1e6,
            "flops_m": flops / 1e6 if flops > 0 else None,
            "lat": lat,
            "comp": comp,
            "dataset": str(r.get("dataset", "")),
            "seed": r.get("seed", ""),
            "group": _group_for_tag(tag),
            "is_baseline": tag.startswith("baseline_"),
            "is_winner": tag in PHASE8_WINNERS,
        })
    if len(pts) < 3:
        return "<p class='empty'>Insufficient rows for the Pareto small-multiples.</p>"

    panel_w, panel_h = 360, 280
    ml, mr, mt, mb = 50, 14, 32, 38
    pw, ph = panel_w - ml - mr, panel_h - mt - mb

    # Shared y-axis from min/max top1 across all rows.
    y_all = [p["top1"] for p in pts]
    ymin, ymax = min(y_all), max(y_all)
    yspan = ymax - ymin
    ymin -= yspan * 0.06
    ymax += yspan * 0.08

    def sy(y: float) -> float:
        return mt + ph * (1 - (y - ymin) / (ymax - ymin))

    captions = {
        "params": "Which architectures are Pareto-efficient on parameter count.",
        "flops":  "Which architectures are Pareto-efficient on FLOPs (compute).",
        "lat":    "Which architectures are Pareto-efficient on inference latency.",
    }
    axis_labels = {
        "params": "params (M)",
        "flops":  "FLOPs (M)",
        "lat":    "latency b=1 (ms)",
    }

    def _panel(axis_key: str, panel_idx: int) -> str:
        if axis_key == "params":
            vals = [(p, p["params_m"]) for p in pts]
        elif axis_key == "flops":
            vals = [(p, p["flops_m"]) for p in pts if p["flops_m"]]
        else:
            vals = [(p, p["lat"]) for p in pts]
        if len(vals) < 2:
            return f"<div class='ms-panel ms-empty'><i>no data</i></div>"
        xs = [math.log10(v) for _, v in vals if v and v > 0]
        if not xs:
            return f"<div class='ms-panel ms-empty'><i>no data</i></div>"
        xmin, xmax = min(xs), max(xs)
        if xmax == xmin:
            xmax = xmin + 0.5
        xspan = xmax - xmin
        xmin -= xspan * 0.10
        xmax += xspan * 0.14

        def sx(lx: float) -> float:
            return ml + pw * (lx - xmin) / (xmax - xmin)

        parts: list[str] = []
        # Caption ABOVE panel.
        parts.append(
            f"<div class='ms-caption'>Panel {panel_idx + 1}: "
            f"{_esc(captions[axis_key])}</div>"
        )
        parts.append(
            f"<svg viewBox='0 0 {panel_w} {panel_h}' "
            f"preserveAspectRatio='xMinYMin meet' "
            f"width='100%' height='auto' "
            f"xmlns='http://www.w3.org/2000/svg' "
            f"style='background:#0a0a0d;border:1px solid #1c1c20;"
            f"display:block;font-family:\"IBM Plex Mono\",monospace'>"
        )
        # Y gridlines + labels (5 ticks).
        for i in range(5):
            yv = ymin + (ymax - ymin) * i / 4
            yy = sy(yv)
            parts.append(
                f"<line x1='{ml}' y1='{yy:.1f}' x2='{ml + pw}' y2='{yy:.1f}' "
                f"stroke='#1c1c20' stroke-width='1'/>"
            )
            parts.append(
                f"<text x='{ml - 6}' y='{yy + 3:.1f}' fill='#a89e8c' "
                f"font-size='9' text-anchor='end'>{yv:.1f}%</text>"
            )
        # X log-decade ticks.
        for lx in range(int(math.floor(xmin)), int(math.ceil(xmax)) + 1):
            if lx < xmin or lx > xmax:
                continue
            xx = sx(lx)
            parts.append(
                f"<line x1='{xx:.1f}' y1='{mt}' x2='{xx:.1f}' "
                f"y2='{mt + ph}' stroke='#1c1c20' stroke-width='1' "
                f"stroke-dasharray='2,3'/>"
            )
            parts.append(
                f"<text x='{xx:.1f}' y='{mt + ph + 14}' fill='#a89e8c' "
                f"font-size='9' text-anchor='middle'>10^{lx}</text>"
            )
        # Axis labels.
        parts.append(
            f"<text x='{ml + pw - 4}' y='{mt + ph + 28}' fill='#a89e8c' "
            f"font-size='10' font-family='Source Serif 4,Georgia,serif' "
            f"text-anchor='end'>{_esc(axis_labels[axis_key])} →</text>"
        )
        parts.append(
            f"<text x='14' y='{mt + ph / 2:.1f}' fill='#a89e8c' "
            f"font-size='10' font-family='Source Serif 4,Georgia,serif' "
            f"text-anchor='middle' "
            f"transform='rotate(-90 14 {mt + ph / 2:.1f})'>top-1 (%) →</text>"
        )
        # Compute Pareto frontier: maximise (top1) while minimising (axis).
        # A point is on the frontier if no other has both >= top1 AND <= axis.
        pts_with_x = [(p, lv) for (p, v), lv in zip(vals, xs)]
        frontier = []
        for i, (pi, lxi) in enumerate(pts_with_x):
            dominated = False
            for j, (pj, lxj) in enumerate(pts_with_x):
                if i == j:
                    continue
                if (pj["top1"] >= pi["top1"] and lxj <= lxi
                        and (pj["top1"] > pi["top1"] or lxj < lxi)):
                    dominated = True
                    break
            if not dominated:
                frontier.append((pi, lxi))
        # Sort frontier by axis-value ascending.
        frontier.sort(key=lambda t: t[1])
        # Frontier dashed polyline.
        if len(frontier) >= 2:
            fpts = " ".join(
                f"{sx(lxi):.1f},{sy(p['top1']):.1f}" for p, lxi in frontier
            )
            parts.append(
                f"<polyline points='{fpts}' fill='none' stroke='#bb8c4d' "
                f"stroke-width='1.4' stroke-dasharray='5,4' opacity='0.75'/>"
            )
        # Plot each point.
        for (p, lxi) in pts_with_x:
            cx = sx(lxi)
            cy = sy(p["top1"])
            col = GROUP_COLORS.get(p["group"], "#8b949e")
            tooltip = (
                f"{p['tag']} · {p['dataset']} seed{p['seed']} · "
                f"top1 {p['top1']:.2f}% · {axis_labels[axis_key]}="
                f"{10**lxi:.3g} · composite {p['comp']:.4f}"
            )
            if p["is_winner"]:
                # Larger filled circle + star marker.
                parts.append(
                    f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' "
                    f"fill='{col}' stroke='#e6e1d6' stroke-width='1.5'>"
                    f"<title>{_esc(tooltip)} · PHASE-8 WINNER (n=3 seeds)</title>"
                    f"</circle>"
                )
                parts.append(
                    f"<text x='{cx:.1f}' y='{cy + 3.5:.1f}' fill='#0a0a0d' "
                    f"font-size='9' font-weight='700' text-anchor='middle' "
                    f"pointer-events='none'>★</text>"
                )
            elif p["is_baseline"]:
                # Hollow ring + explicit label.
                parts.append(
                    f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5.5' fill='none' "
                    f"stroke='{col}' stroke-width='2'>"
                    f"<title>{_esc(tooltip)} · baseline reference</title>"
                    f"</circle>"
                )
                parts.append(
                    f"<text x='{cx + 8:.1f}' y='{cy + 3:.1f}' fill='{col}' "
                    f"font-size='9' font-family='IBM Plex Mono,monospace' "
                    f"font-weight='600'>baseline</text>"
                )
            else:
                parts.append(
                    f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='3.2' fill='{col}' "
                    f"fill-opacity='0.85' stroke='#0a0a0d' stroke-width='0.6'>"
                    f"<title>{_esc(tooltip)}</title>"
                    f"</circle>"
                )
        parts.append("</svg>")
        return f"<div class='ms-panel'>{''.join(parts)}</div>"

    panels_html = "".join(
        _panel(k, idx) for idx, k in enumerate(("params", "flops", "lat"))
    )
    # Shared legend BELOW the 3 panels.
    legend_groups = ["Baseline", "G1", "G2", "G3", "G4", "G5",
                     "G6", "G7", "G8", "Combo"]
    legend_bits = []
    for g in legend_groups:
        col = GROUP_COLORS[g]
        label = g
        if g == "G1":
            label = "G1 scaling"
        elif g == "G2":
            label = "G2 layer/channel"
        elif g == "G3":
            label = "G3 topologies"
        elif g == "G4":
            label = "G4 kernels/attn"
        elif g == "G5":
            label = "G5 optim/init"
        elif g == "G6":
            label = "G6 topo-bridging"
        elif g == "G7":
            label = "G7 hybrids"
        elif g == "G8":
            label = "G8 esoteric"
        legend_bits.append(
            f"<span class='ms-legend-item'>"
            f"<span class='ms-swatch' style='background:{col}'></span>"
            f"{_esc(label)}</span>"
        )
    # Winner / baseline glyph legend
    legend_bits.append(
        "<span class='ms-legend-item'>★ <em>Phase-8 winner</em> (n=3, EVALUATION)</span>"
    )
    legend_bits.append(
        "<span class='ms-legend-item'>○ baseline_resnet20 (n=3 reference)</span>"
    )
    legend_bits.append(
        "<span class='ms-legend-item ms-frontier'>"
        "<span style='display:inline-block;width:22px;height:0;"
        "border-top:1.5px dashed #bb8c4d;vertical-align:middle;"
        "margin-right:4px'></span>Pareto frontier</span>"
    )
    legend_html = (
        "<div class='ms-legend'>"
        + " ".join(legend_bits)
        + "</div>"
    )
    return (
        f"<div class='ms-grid ms-pareto'>{panels_html}</div>"
        f"{legend_html}"
    )


def _ablation_group_panels_svg(rows: list[dict],
                               baseline_top1: dict[str, float] | None = None,
                               noise_band_pp: float = 1.21,
                               dataset_pref: str = "cifar10") -> str:
    """8 small horizontal-bar panels (one per CLAUDE.md group G1..G8).

    Replaces the legacy single PNG `plot_ablation.png` with 8 small-
    multiples per CLAUDE.md Rule 33 (autoresearch-dashboard-comprehension
    SKILL.md Pillar 1). Arranged in a CSS 4×2 grid; each panel ~280×200.

    Each bar = one tag's Δtop1 (pp) vs the baseline on `dataset_pref`.
    Sorted within group descending by Δ (positive at top). Divergent
    colour scale: green (Δ above noise band), neutral (within noise),
    red (Δ below noise band). Vertical reference line marks the
    empirical noise band ±noise_band_pp (default 1.21pp = 2σ for
    CIFAR-10 pooled σ_seed=0.607pp per paper/STATISTICAL_TESTS.md).
    Per-bar label: tag + Δpp + n=X chip + tier badge. Per-group caption
    ("what this group tested") above each panel.

    rows: list of dicts with tag, dataset, top1, seed.
    baseline_top1: dict[dataset→top1 fraction] for Δ computation;
    defaults to module BASELINE_TOP1.
    """
    if baseline_top1 is None:
        baseline_top1 = BASELINE_TOP1
    base_top1 = baseline_top1.get(dataset_pref)
    if base_top1 is None:
        return (
            f"<p class='empty'>No baseline_resnet20 reference for "
            f"{_esc(dataset_pref)} — ablation panels need a baseline for "
            f"Δ computation.</p>"
        )

    # Group → 1-sentence caption.
    group_captions: dict[str, str] = {
        "G1": "G1 — φ/Fibonacci scaling of depth, width, resolution, "
              "parameter budgets, LR schedules (H01–H10).",
        "G2": "G2 — Fibonacci channel/MLP sizes, golden skip-modulation, "
              "φ-threshold activations, head diversity, ensembles (H11–H20).",
        "G3": "G3 — hexagonal lattices, toroidal closures, Platonic / "
              "icosa equivariance, fractal toroidal (H21–H30).",
        "G4": "G4 — golden-spiral kernels, Fibonacci dilation, vesica "
              "filters, golden-angle rotary, cymatic / harmonic acts (H31–H40).",
        "G5": "G5 — golden-ratio AdamW, φ-weight init, Fibonacci pruning, "
              "φ-dropout, golden-momentum (H41–H50).",
        "G6": "G6 — persistent-homology losses, drop-path anytime, icosa-"
              "unfold bridges, C4-group avg-pool, Betti probes (H51–H60).",
        "G7": "G7 — Liquid / JEPA / KAN / Transformer / GNN cross-paradigm "
              "hybrids (H61–H75). No CIFAR sweep rows yet.",
        "G8": "G8 — Reuleaux constant-width kernels, SIREN sinusoidal, "
              "tetrahedral, radial-12 attention, spectral Hopfield (H76–H84).",
    }

    # Group by G1..G8 from rows. Filter to the requested dataset, exclude
    # baseline_*.
    by_group: dict[str, list[dict]] = {g: [] for g in group_captions}
    seed_counts: dict[str, int] = {}  # tag → seed count across this dataset
    for r in rows:
        try:
            top1 = float(r.get("top1") or 0)
        except Exception:
            continue
        if top1 <= 0:
            continue
        if str(r.get("dataset", "")) != dataset_pref:
            continue
        tag = str(r.get("tag", ""))
        if tag.startswith("baseline_"):
            continue
        g = _group_for_tag(tag)
        # Combo / pair / loo / slot tags belong to their underlying
        # hypothesis group for the per-axis ablation read. Re-map.
        if g == "Combo":
            hp = TAG_TO_HYP.get(tag)
            g = hp[1] if hp and hp[1] else "Uncategorized"
        if g not in by_group:
            continue
        delta_pp = (top1 - base_top1) * 100
        by_group[g].append({
            "tag": tag, "top1": top1, "delta_pp": delta_pp,
            "seed": r.get("seed", ""),
        })
        seed_counts[tag] = seed_counts.get(tag, 0) + 1

    # For tags with multiple seeds, collapse to median.
    for g, bars in list(by_group.items()):
        # Aggregate by tag.
        by_tag: dict[str, list[float]] = {}
        for b in bars:
            by_tag.setdefault(b["tag"], []).append(b["delta_pp"])
        agg = []
        for tag, ds in by_tag.items():
            ds_sorted = sorted(ds)
            mid = len(ds_sorted) // 2
            if len(ds_sorted) % 2 == 1:
                med = ds_sorted[mid]
            else:
                med = 0.5 * (ds_sorted[mid - 1] + ds_sorted[mid])
            agg.append({
                "tag": tag, "delta_pp": med,
                "n": len(ds_sorted),
            })
        agg.sort(key=lambda d: d["delta_pp"], reverse=True)
        by_group[g] = agg

    # Compute global Δ-range for shared x-axis across all 8 panels
    # (so visual comparison across groups is meaningful).
    all_deltas = [b["delta_pp"] for bars in by_group.values() for b in bars]
    if not all_deltas:
        return "<p class='empty'>No non-baseline rows for ablation panels.</p>"
    dmin = min(all_deltas + [-noise_band_pp * 1.2])
    dmax = max(all_deltas + [noise_band_pp * 1.2])
    dspan = dmax - dmin
    dmin -= dspan * 0.06
    dmax += dspan * 0.10

    panel_w, panel_h_min = 290, 150
    ml, mr, mt, mb = 100, 50, 28, 28

    def _panel(g: str, bars: list[dict]) -> str:
        caption = group_captions.get(g, g)
        if not bars:
            return (
                f"<div class='ms-panel'>"
                f"<div class='ms-caption'>{_esc(caption)}</div>"
                f"<div class='ms-panel-empty'>No sweep rows in this group "
                f"on {_esc(dataset_pref)} yet.</div>"
                f"</div>"
            )
        n = len(bars)
        bh = 14
        gap = 5
        panel_h = max(panel_h_min, mt + mb + n * (bh + gap))
        pw, ph = panel_w - ml - mr, panel_h - mt - mb

        def sx(d: float) -> float:
            return ml + pw * (d - dmin) / (dmax - dmin)

        parts: list[str] = []
        parts.append(f"<div class='ms-caption'>{_esc(caption)}</div>")
        parts.append(
            f"<svg viewBox='0 0 {panel_w} {panel_h}' "
            f"preserveAspectRatio='xMinYMin meet' "
            f"width='100%' height='auto' "
            f"xmlns='http://www.w3.org/2000/svg' "
            f"style='background:#0a0a0d;border:1px solid #1c1c20;"
            f"display:block;font-family:\"IBM Plex Mono\",monospace'>"
        )
        # Vertical zero line.
        x0 = sx(0)
        parts.append(
            f"<line x1='{x0:.1f}' y1='{mt - 4}' x2='{x0:.1f}' "
            f"y2='{mt + ph + 4}' stroke='#a89e8c' stroke-width='1.2'/>"
        )
        # Noise band (±noise_band_pp).
        x_pos = sx(noise_band_pp)
        x_neg = sx(-noise_band_pp)
        parts.append(
            f"<line x1='{x_pos:.1f}' y1='{mt}' x2='{x_pos:.1f}' "
            f"y2='{mt + ph}' stroke='#bb8c4d' stroke-width='1' "
            f"stroke-dasharray='3,3' opacity='0.7'/>"
        )
        parts.append(
            f"<line x1='{x_neg:.1f}' y1='{mt}' x2='{x_neg:.1f}' "
            f"y2='{mt + ph}' stroke='#bb8c4d' stroke-width='1' "
            f"stroke-dasharray='3,3' opacity='0.7'/>"
        )
        # X-axis label at bottom.
        parts.append(
            f"<text x='{x0:.1f}' y='{mt + ph + 16}' fill='#a89e8c' "
            f"font-size='9' text-anchor='middle' "
            f"font-family='Source Serif 4,Georgia,serif'>0 pp</text>"
        )
        parts.append(
            f"<text x='{x_pos:.1f}' y='{mt - 4}' fill='#bb8c4d' "
            f"font-size='8' text-anchor='middle'>+{noise_band_pp:.2f}pp band</text>"
        )
        parts.append(
            f"<text x='{x_neg:.1f}' y='{mt - 4}' fill='#bb8c4d' "
            f"font-size='8' text-anchor='middle'>−{noise_band_pp:.2f}pp band</text>"
        )
        # Bars.
        for i, b in enumerate(bars):
            y = mt + i * (bh + gap)
            d = b["delta_pp"]
            x = sx(d)
            # Divergent colour: green above +noise_band, red below -noise_band,
            # neutral grey within.
            if d > noise_band_pp:
                col = "#3fb950"
            elif d < -noise_band_pp:
                col = "#f85149"
            else:
                col = "#8b949e"
            # bar from x0 to x.
            bx0 = min(x0, x)
            bw = abs(x - x0)
            parts.append(
                f"<rect x='{bx0:.1f}' y='{y}' width='{bw:.1f}' "
                f"height='{bh}' fill='{col}' opacity='0.85'>"
                f"<title>{_esc(b['tag'])} · Δ{d:+.2f}pp vs baseline · "
                f"n={b['n']}</title>"
                f"</rect>"
            )
            # Tag label LEFT of bar area.
            tag_disp = b["tag"][:20]
            parts.append(
                f"<text x='{ml - 6}' y='{y + bh - 3}' fill='#e6e1d6' "
                f"font-size='9' text-anchor='end' "
                f"font-family='IBM Plex Mono,monospace'>{_esc(tag_disp)}</text>"
            )
            # Δ value + n chip RIGHT of bar.
            sign = "+" if d >= 0 else "−"
            tier_short = "EVAL" if b["n"] >= 3 and b["tag"] in PHASE8_WINNERS else "SCR"
            value_x = x + (4 if d >= 0 else -4)
            anchor = "start" if d >= 0 else "end"
            parts.append(
                f"<text x='{value_x:.1f}' y='{y + bh - 3}' fill='#a89e8c' "
                f"font-size='9' text-anchor='{anchor}' "
                f"font-family='IBM Plex Mono,monospace'>"
                f"{sign}{abs(d):.2f}pp · n={b['n']} {tier_short}</text>"
            )
        parts.append("</svg>")
        return f"<div class='ms-panel'>{''.join(parts)}</div>"

    panels_html = "".join(
        _panel(g, by_group.get(g, [])) for g in ("G1", "G2", "G3", "G4",
                                                 "G5", "G6", "G7", "G8")
    )
    legend_html = (
        "<div class='ms-legend'>"
        "<span class='ms-legend-item'>"
        "<span class='ms-swatch' style='background:#3fb950'></span>"
        "Δ &gt; +1.21pp (above 2σ noise band → meaningful gain)</span>"
        "<span class='ms-legend-item'>"
        "<span class='ms-swatch' style='background:#8b949e'></span>"
        "|Δ| ≤ 1.21pp (within seed noise → no claim)</span>"
        "<span class='ms-legend-item'>"
        "<span class='ms-swatch' style='background:#f85149'></span>"
        "Δ &lt; −1.21pp (below noise → regression vs baseline)</span>"
        "<span class='ms-legend-item ms-frontier'>"
        "<span style='display:inline-block;width:22px;height:0;"
        "border-top:1.5px dashed #bb8c4d;vertical-align:middle;"
        "margin-right:4px'></span>±1.21pp noise band "
        "(2σ for CIFAR-10 pooled σ_seed=0.607pp per "
        "<code>paper/STATISTICAL_TESTS.md</code>)</span>"
        "</div>"
    )
    return (
        f"<div class='ms-grid ms-ablation'>{panels_html}</div>"
        f"{legend_html}"
    )


def _how_to_read_block() -> str:
    """Render the mandatory 4-bullet 'How to read this dashboard' orientation.

    Required by CLAUDE.md Rule 33 / skill autoresearch-dashboard-comprehension
    SKILL.md Pillar 2. EXACTLY 4 bullets. Inserted between the masthead and
    the headline ribbon so every reviewer sees the framing before any
    numerics.
    """
    return (
        "<section class='how-to-read'>"
        "<h3>How to read this dashboard</h3>"
        "<ul>"
        "<li><b>Numbers shown.</b> top-1 accuracy on "
        "<a href='https://www.cs.toronto.edu/~kriz/cifar.html' "
        "target='_blank' rel='noopener'>CIFAR-10</a> (12 epochs) or "
        "<a href='https://www.cs.toronto.edu/~kriz/cifar.html' "
        "target='_blank' rel='noopener'>CIFAR-100</a> (30 epochs) per the "
        "<code>DS</code> column. <code>composite</code> = "
        "<code>top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)</code> "
        "(SHA-256-fingerprinted per "
        "<a href='https://github.com/dlmastery/nature_inspired_networks/"
        "blob/main/CLAUDE.md#rule-2'>Rule 2</a>).</li>"
        "<li><b>Tiers per "
        "<a href='https://github.com/dlmastery/nature_inspired_networks/"
        "blob/main/CLAUDE.md#rule-28'>Rule 28</a>.</b> "
        "<span class='chip-scr'>SCREENING (n=1)</span> = one seed, "
        "candidate-only — NOT a defensible external claim. "
        "<span class='chip-eval'>EVALUATION (n≥3, Phase-5 gate cleared)</span> "
        "= formally measured. Only the 3 Phase-8 winners "
        "(<code>pair_gm_pdw</code>, <code>slot_act_sine</code>, "
        "<code>sg_only_phi_budget</code>) currently hold the EVALUATION tier.</li>"
        "<li><b>Colours by group.</b> G1 scaling · G2 layer/channel · "
        "G3 topologies · G4 kernels/attention · G5 optimisation · "
        "G6 topological-bridging · G7 hybrids · G8 esoteric. Baseline + "
        "Combo (combo*/pair*/loo*/slot* multi-prior stacks) have distinct "
        "colours on the Pareto small-multiples; group panels of the ablation "
        "matrix break the same data out per hypothesis group.</li>"
        "<li><b>Navigation.</b> click any leaderboard row → independent "
        "per-experiment page (10 sections: hypothesis-doc digest, impl-critic "
        "verdict, sci-critic verdict, FINDINGS verdict, reasoning blob, "
        "config, metrics, composite breakdown, training curves, "
        "cross-references). All numbers traceable to the build-stamp commit "
        "SHA in the footer.</li>"
        "</ul>"
        "</section>"
    )


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
    _ROOT_FOR_SVG["root"] = root
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
""" + _EXP_FONT_LINK + """
<style>
""" + _BRUTALIST_VARS + """
 *{margin:0;padding:0;box-sizing:border-box;}
 html,body{background:var(--ink);}
 body{font-family:'Source Serif 4','Charter','Source Serif Pro',Georgia,serif;
      color:var(--paper);
      padding:32px 36px 80px;line-height:1.55;max-width:1340px;margin:0 auto;
      font-size:16px;font-variant-numeric:tabular-nums;position:relative;}
 a{color:var(--v-derivative);text-decoration:none;
   border-bottom:1px solid transparent;transition:border-color 160ms ease;}
 a:hover{border-bottom-color:var(--v-derivative);text-decoration:none;}
 h1{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
    font-size:42px;line-height:1.10;color:var(--paper);
    letter-spacing:-0.005em;margin-bottom:8px;}
 h2{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
    font-size:22px;color:var(--paper);margin:36px 0 10px 0;
    letter-spacing:-0.003em;}
 h3{font-family:'IBM Plex Mono',monospace;font-size:11px;
    text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);
    margin:14px 0 8px 0;font-weight:600;}
 .sub{font-family:'IBM Plex Mono',monospace;color:var(--paper-dim);
      margin-bottom:18px;font-size:11px;text-transform:uppercase;
      letter-spacing:0.18em;}
 .group-desc{color:var(--paper-dim);font-size:0.93em;margin:4px 0 14px 0;
             max-width:980px;line-height:1.6;
             font-family:'Source Serif 4',Georgia,serif;}
 .ribbon{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:1px;background:var(--rule);border:1px solid var(--rule);
         margin:14px 0 24px 0;}
 .kpi{background:var(--panel);padding:14px 16px;}
 .kpi .label{color:var(--paper-dim);font-size:9.5px;
             font-family:'IBM Plex Mono',monospace;
             text-transform:uppercase;letter-spacing:0.18em;}
 .kpi .value{font-family:'Source Serif 4',Georgia,serif;font-size:26px;
             font-weight:600;margin-top:4px;color:var(--paper);
             letter-spacing:-0.005em;}
 .kpi.positive{box-shadow:inset 3px 0 0 var(--v-pass);}
 .kpi.negative{box-shadow:inset 3px 0 0 var(--v-broken);}
 .kpi.neutral{box-shadow:inset 3px 0 0 var(--accent-dim);}
 .formula-chip{display:inline-block;background:var(--panel);
               border:1px solid var(--rule);border-left:2px solid var(--accent);
               padding:8px 14px;margin:4px 0 18px 0;
               font-family:'IBM Plex Mono',monospace;font-size:11px;
               color:var(--paper);letter-spacing:0.04em;}
 .formula-chip .fp{color:var(--paper-dim);font-size:10px;}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px;}
 .card{background:var(--panel);border:1px solid var(--rule);
       padding:18px 20px;position:relative;}
 .card::before{content:"";position:absolute;top:0;left:0;width:48px;
               height:1px;background:var(--accent);}
 .card img{max-width:100%;height:auto;background:#fff;padding:4px;
           border:1px solid var(--rule);}
 .panel-2col{grid-column:1 / 3;}
 table.runs{width:100%;border-collapse:collapse;font-size:0.82em;
            background:var(--panel);border:1px solid var(--rule);}
 table.runs th{background:transparent;color:var(--paper-dim);text-align:right;
    padding:9px 12px;border-bottom:1px solid var(--rule-bright);
    font-weight:500;text-transform:uppercase;font-size:10px;
    letter-spacing:0.18em;cursor:pointer;
    font-family:'IBM Plex Mono',monospace;}
 table.runs th:first-child{text-align:left;}
 table.runs td{padding:8px 12px;border-bottom:1px solid var(--rule);
    text-align:right;color:var(--paper);
    font-family:'IBM Plex Mono',monospace;font-size:0.96em;}
 table.runs td:first-child{text-align:left;}
 table.runs tr.row-link{cursor:pointer;transition:background 160ms ease;}
 table.runs tr.row-link:hover{background:rgba(187,140,77,0.06);}
 table.runs tr.best-row{box-shadow:inset 3px 0 0 var(--accent);
                        background:rgba(187,140,77,0.045);}
 table.runs tr.best-row td{font-weight:600;}
 .tag-pill{display:inline-block;background:transparent;
           border:1px solid var(--rule-bright);
           padding:2px 9px;font-size:11px;color:var(--paper);
           font-family:'IBM Plex Mono',monospace;letter-spacing:0.06em;}
 .meta{font-family:'IBM Plex Mono',monospace;font-size:10px;
       color:var(--paper-dim);margin-top:28px;padding-top:18px;
       border-top:1px solid var(--rule);line-height:1.9;
       letter-spacing:0.04em;}
 .meta code{background:var(--ink);padding:1px 5px;color:var(--paper);
            border:1px solid var(--rule);}
 .status-done{background:var(--v-pass);}
 .status-impl{background:var(--v-derivative);}
 .status-partial{background:var(--v-minor);}
 .status-running{background:var(--v-minor);animation:pulse 1.5s infinite;}
 .status-queued{background:var(--v-derivative);opacity:0.7;}
 .status-planned{background:#484f58;}
 .status-deferred{background:#4d3a1f;}
 .status-failed{background:var(--v-broken);}
 .status-superseded{background:#6e7681;}
 .hyp-grid-summary{font-family:var(--font-mono,monospace);font-size:0.82em;
   color:var(--paper-dim,#a89e8c);padding:6px 0 12px 0;
   border-bottom:1px solid var(--rule,#1c1c20);margin-bottom:10px;
   letter-spacing:0.02em;}
 .hyp-grid-summary b{color:var(--paper,#e6e1d6);font-weight:600;}
 @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.45;}}
 @keyframes reveal{from{opacity:0;transform:translateY(8px);}
                   to{opacity:1;transform:translateY(0);}}
 section,.card,.ribbon,.group-section,.formula-chip,.headline-ribbon{
   animation:reveal 360ms cubic-bezier(.2,.7,.2,1) both;
   animation-delay:calc(var(--i, 0) * 60ms);}
 html{scroll-behavior:smooth;}
 .group-section{margin-top:32px;background:var(--panel);
                border:1px solid var(--rule);padding:22px 26px;position:relative;}
 .group-section::before{content:"";position:absolute;top:0;left:0;width:64px;
                        height:1px;background:var(--accent);}
 .group-section h2{margin-top:0;color:var(--paper);font-size:22px;}
 .group-empty{color:var(--paper-dim);font-style:italic;font-size:0.93em;
              padding:8px 0;font-family:'Source Serif 4',Georgia,serif;}
 .group-chart{margin:14px 0;}
 table.runs.compact{font-size:0.78em;}
 table.runs.compact th{padding:6px 9px;font-size:9px;}
 table.runs.compact td{padding:5px 9px;}
 .hyp-link{color:var(--v-derivative);font-family:'IBM Plex Mono',monospace;
           font-size:11px;white-space:nowrap;letter-spacing:0.06em;}
 .headline-ribbon{background:var(--panel);border:1px solid var(--rule);
                  border-left:2px solid var(--accent);
                  padding:16px 22px;margin:14px 0 18px 0;
                  font-size:0.96em;color:var(--paper);
                  font-family:'Source Serif 4',Georgia,serif;line-height:1.55;}
 .headline-ribbon b{font-family:'IBM Plex Mono',monospace;
                    font-weight:600;color:var(--accent);font-size:10px;
                    text-transform:uppercase;letter-spacing:0.18em;
                    margin-right:8px;}
 .headline-ribbon h2{font-family:'Source Serif 4',Georgia,serif;
                     font-size:22px;font-weight:600;margin:0 0 10px 0;}
 .headline-ribbon table{border-collapse:collapse;margin:8px 0;font-size:0.92em;
                        width:auto;}
 .headline-ribbon th,.headline-ribbon td{border:1px solid var(--rule-bright);
                                         padding:4px 10px;text-align:left;}
 .headline-ribbon th{background:var(--panel2);font-weight:600;
                     font-family:'IBM Plex Mono',monospace;font-size:0.86em;
                     text-transform:uppercase;letter-spacing:0.08em;
                     color:var(--paper-dim);}
 .headline-ribbon code{background:var(--ink);padding:1px 5px;
                       font-family:'IBM Plex Mono',monospace;
                       border:1px solid var(--rule);font-size:0.86em;}
 .headline-ribbon ul{margin:8px 0 8px 22px;}
 .headline-ribbon li{margin:3px 0;}
 .headline-ribbon p{margin:6px 0;}
 .headline-ribbon blockquote{border-left:3px solid var(--accent-dim);
                              padding:4px 14px;margin:8px 0;color:var(--paper-dim);}
 /* Rule-28 seed-count chips on leaderboard tag cells */
 .n-chip{display:inline-block;padding:1px 6px;font-size:9px;
         font-family:'IBM Plex Mono',monospace;letter-spacing:0.10em;
         text-transform:uppercase;border:1px solid;margin-left:6px;
         vertical-align:middle;}
 .n-chip.eval{color:var(--v-pass);border-color:var(--v-pass);}
 .n-chip.scr{color:var(--v-minor);border-color:var(--v-minor);}
 /* Pages-live ribbon and build-stamp footer */
 .live-link{display:inline-block;padding:3px 10px;border:1px solid var(--v-pass);
            color:var(--v-pass);font-family:'IBM Plex Mono',monospace;
            font-size:10px;text-transform:uppercase;letter-spacing:0.18em;
            font-weight:600;}
 .live-link:hover{background:var(--v-pass);color:var(--ink);
                  border-bottom-color:var(--v-pass);}
 /* Masthead CTA pills — repo / paper / literature survey */
 .mast-pill{display:inline-block;padding:3px 10px;margin-right:6px;
            border:1px solid var(--accent-dim);color:var(--accent);
            font-family:'IBM Plex Mono',monospace;font-size:10px;
            text-transform:uppercase;letter-spacing:0.16em;font-weight:600;
            text-decoration:none;}
 .mast-pill:hover{background:var(--accent-dim);color:var(--ink);
                  border-bottom-color:var(--accent-dim);}
 .mast-pill.repo{border-color:var(--v-novel);color:var(--v-novel);}
 .mast-pill.repo:hover{background:var(--v-novel);color:var(--ink);}
 .mast-pill.lit{border-color:var(--v-derivative);color:var(--v-derivative);}
 .mast-pill.lit:hover{background:var(--v-derivative);color:var(--ink);}
 .mast-pill.paper{border-color:var(--v-pass);color:var(--v-pass);}
 .mast-pill.paper:hover{background:var(--v-pass);color:var(--ink);}
 /* Footer cross-link row */
 .doc-footer{margin:18px 0 6px 0;font-family:'IBM Plex Mono',monospace;
             font-size:10px;color:var(--paper-dim);letter-spacing:0.12em;
             text-transform:uppercase;}
 .doc-footer a{color:var(--accent);text-decoration:none;
               border-bottom:1px dotted var(--accent-dim);}
 .doc-footer a:hover{color:var(--v-pass);border-bottom-color:var(--v-pass);}
 .legend-row{margin:10px 0 14px 0;font-size:0.78em;color:var(--paper-dim);
             font-family:'IBM Plex Mono',monospace;letter-spacing:0.08em;}
 .legend-row .swatch{display:inline-block;width:12px;height:12px;
                     vertical-align:middle;margin:0 4px 0 10px;}
 .hyp-grid-row{display:flex;align-items:center;margin-bottom:4px;
               font-family:'IBM Plex Mono',monospace;font-size:0.78em;}
 .hyp-grid-row .gid{width:34px;color:var(--paper-dim);}
 .hyp-grid-row .cell{width:26px;height:22px;margin-right:3px;
                     display:inline-flex;align-items:center;justify-content:center;
                     font-size:0.7em;color:#0a0a0d;cursor:pointer;
                     border:1px solid var(--rule);font-weight:600;}
 .hyp-grid-row .cell:hover{outline:1px solid var(--accent);}
 .hyp-grid-row .cell.empty{background:var(--rule);color:#484f58;cursor:default;
                            border-color:transparent;}
 code{background:var(--ink);padding:1px 5px;font-family:'IBM Plex Mono',monospace;
      border:1px solid var(--rule);font-size:0.9em;}
 .grain{position:fixed;inset:0;pointer-events:none;z-index:2;opacity:0.035;
   background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");}
 .vbadge:hover{transform:scale(1.05);filter:brightness(1.15);}
 /* ---- Phase E: "How to read this dashboard" 4-bullet orientation block ---- */
 .how-to-read{background:var(--panel);border:1px solid var(--rule);
              border-left:2px solid var(--v-pass);padding:18px 24px;
              margin:18px 0;font-family:'Source Serif 4',Georgia,serif;
              line-height:1.6;}
 .how-to-read h3{font-family:'IBM Plex Mono',monospace;font-size:11px;
                 text-transform:uppercase;letter-spacing:0.18em;
                 color:var(--paper-dim);margin:0 0 12px 0;}
 .how-to-read ul{list-style:none;margin:0;padding:0;
                 display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;}
 .how-to-read li{padding-left:14px;border-left:1px solid var(--rule-bright);
                 color:var(--paper);font-size:14.5px;line-height:1.55;}
 .how-to-read li b{color:var(--paper);font-weight:600;}
 .how-to-read code{background:var(--ink);padding:1px 5px;
                   font-family:'IBM Plex Mono',monospace;font-size:0.88em;
                   border:1px solid var(--rule);}
 .how-to-read a{color:var(--v-derivative);}
 .how-to-read .chip-scr{background:#fff4e5;color:#8a5800;padding:1px 6px;
                        border-radius:3px;font-family:'IBM Plex Mono',monospace;
                        font-size:0.82em;font-weight:600;
                        letter-spacing:0.04em;}
 .how-to-read .chip-eval{background:#e8f4e8;color:#1f5320;padding:1px 6px;
                         border-radius:3px;font-family:'IBM Plex Mono',monospace;
                         font-size:0.82em;font-weight:600;
                         letter-spacing:0.04em;}
 @media(max-width:880px){.how-to-read ul{grid-template-columns:1fr;}}
 /* ---- Phase C / D: small-multiples grid for Pareto + ablation panels ---- */
 .ms-grid{display:grid;gap:14px;margin:12px 0 8px 0;}
 .ms-grid.ms-pareto{grid-template-columns:repeat(3,minmax(0,1fr));}
 .ms-grid.ms-ablation{grid-template-columns:repeat(4,minmax(0,1fr));}
 @media(max-width:1200px){
   .ms-grid.ms-ablation{grid-template-columns:repeat(2,minmax(0,1fr));}
 }
 @media(max-width:880px){
   .ms-grid.ms-pareto{grid-template-columns:1fr;}
   .ms-grid.ms-ablation{grid-template-columns:1fr;}
 }
 .ms-panel{display:flex;flex-direction:column;}
 .ms-caption{font-family:'Source Serif 4',Georgia,serif;font-size:12.5px;
             color:var(--paper-dim);margin-bottom:6px;line-height:1.4;
             min-height:34px;}
 .ms-panel-empty{font-family:'IBM Plex Mono',monospace;font-size:10.5px;
                 color:var(--paper-dim);background:var(--panel2);
                 border:1px dashed var(--rule);padding:18px 14px;text-align:center;}
 .ms-legend{display:flex;flex-wrap:wrap;gap:10px 18px;font-family:'IBM Plex Mono',monospace;
            font-size:10.5px;color:var(--paper-dim);
            margin:6px 0 4px 0;padding:10px 14px;background:var(--panel2);
            border:1px solid var(--rule);letter-spacing:0.04em;}
 .ms-legend em{color:var(--paper);font-style:normal;font-weight:600;}
 .ms-legend code{background:var(--ink);padding:1px 4px;font-size:0.94em;
                 border:1px solid var(--rule);color:var(--paper);}
 .ms-legend-item{display:inline-flex;align-items:center;}
 .ms-swatch{display:inline-block;width:11px;height:11px;margin-right:5px;
            vertical-align:middle;border:1px solid var(--rule-bright);}
</style>
</head><body>
<div class='grain'></div>
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
  // Pages does not serve repo-root .md files (only docs/) and file://
  // blocks fetch in most browsers — both made the previous inline-fetch
  // side-panel uselessly broken. Just navigate to the GitHub-blob URL,
  // which always resolves both as a published doc and in dev.
  var url='https://github.com/dlmastery/nature_inspired_networks/blob/main/hypotheses/'+fname;
  window.open(url,'_blank','noopener,noreferrer');
}
</script>
"""


# ---------------------------------------------------------------------------
# HTML fragment builders (hypothesis grid + ribbons + group sections)
# ---------------------------------------------------------------------------

def _hypothesis_grid_html(index_rows: list[dict],
                          tag_status: dict[str, dict],
                          idea_table_status: dict[str, str] | None = None,
                          run_dirs_root: Path | None = None) -> str:
    if not index_rows:
        return "<i style='color:#8b949e'>hypotheses/INDEX.md not found.</i>"

    # Tier ordering — lower index wins on collision.
    order = ["done", "running", "queued", "impl", "partial",
             "failed", "superseded", "deferred", "planned"]

    # 1) Seed from IDEA_TABLE.md (the canonical per-hypothesis status column).
    #    Falls back to "planned" if the table is missing / un-parseable.
    idea_table_status = idea_table_status or {}
    status_for: dict[str, str] = {}
    for h in index_rows:
        hid = h["id"]
        status_for[hid] = idea_table_status.get(hid, "planned")

    # 2) Overlay EXPERIMENT_LOG.md tier rows (a tag → tier → hyp lookup).
    for tag, meta in tag_status.items():
        idea = meta.get("idea", "")
        for m in re.finditer(r"H(\d{2})", idea):
            hid = "H" + m.group(1)
            if hid in status_for:
                cur = status_for[hid]
                new = meta["status"]
                if order.index(new) < order.index(cur):
                    status_for[hid] = new

    # 3) Promote every hypothesis with at least one COMPLETED CIFAR-10/-100
    #    run directory (metrics.json on disk) from "impl" → "done".
    #    The TAG_TO_HYP map drives the tag → hypothesis edge.
    if run_dirs_root is not None and Path(run_dirs_root).exists():
        seen_runs: set[str] = set()
        for ds_dir in Path(run_dirs_root).glob("*/"):
            for run_dir in ds_dir.glob("*_seed*/"):
                if (run_dir / "metrics.json").exists():
                    # tag is the run-dir name minus the trailing _seed<N>.
                    name = run_dir.name
                    tag = re.sub(r"_seed\d+$", "", name)
                    seen_runs.add(tag)
        for tag in seen_runs:
            hid_pair = TAG_TO_HYP.get(tag)
            if not hid_pair:
                continue
            hid = hid_pair[0]
            if hid and hid in status_for:
                cur = status_for[hid]
                if order.index("done") < order.index(cur):
                    status_for[hid] = "done"

    by_group: dict[str, list[dict]] = {}
    for h in index_rows:
        by_group.setdefault(h["group"], []).append(h)

    # Counts per status — helpful at-a-glance summary.
    counts: dict[str, int] = {}
    for s in status_for.values():
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    summary_bits = []
    for s in ("done", "impl", "partial", "running", "queued",
              "planned", "failed", "superseded", "deferred"):
        n = counts.get(s, 0)
        if n:
            summary_bits.append(f"<b>{n}</b>&nbsp;{s}")
    summary = (
        "<div class='hyp-grid-summary'>"
        f"<b>{total}</b>&nbsp;hypotheses · "
        + " · ".join(summary_bits)
        + "</div>"
    )

    out = [summary]
    out.append("<div class='legend-row'>Legend:")
    for s in ("done", "impl", "partial", "running", "queued",
              "planned", "deferred", "failed", "superseded"):
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
                        table_id: str,
                        history_map: dict[str, list[dict]] | None = None,
                        repo_root: Path | None = None,
                        results_dir: Path | None = None) -> str:
    """Render one ``<section>`` for a hypothesis-group block.

    Now renders 2 diagrams per group (top-1 bars + composite-vs-params
    scatter) and adds a per-row sparkline column when history is available.
    """
    header, desc = GROUP_HEADERS.get(group_letter, (group_letter, ""))
    n = len(rows)
    parts = [
        f"<section class='group-section' style='--i:{hash(group_letter) % 8}'>",
        f"<h2>{_esc(header)} <span style='color:var(--paper-dim);"
        f"font-weight:400;font-size:0.78em;font-style:normal;"
        f"font-family:IBM Plex Mono,monospace'>"
        f"({n} run{'s' if n != 1 else ''})</span></h2>",
        f"<div class='group-desc'>{_esc(desc)}</div>",
    ]
    if not rows:
        parts.append(
            "<div class='group-empty'>No sweep rows yet in this group.</div>"
        )
        parts.append("</section>")
        return "".join(parts)
    # Diagram 1: top-1 horizontal bar chart
    chart = _group_visualisation_svg(
        rows, header, chart_id=f"svg-{table_id}",
    )
    if chart:
        parts.append("<div class='group-chart'>" + chart + "</div>")
    # Diagram 2: composite-vs-params scatter (only if ≥3 rows)
    scatter = _group_scatter_svg(
        rows, header, chart_id=f"svg-scatter-{table_id}", repo_root=repo_root,
    )
    if scatter:
        parts.append("<div class='group-chart'>" + scatter + "</div>")
    parts.append(
        f"<table class='runs compact' id='{table_id}' data-dir='asc'>"
        "<thead><tr>"
    )
    for i, (_k, label) in enumerate(_RUNS_COLS):
        parts.append(
            f"<th onclick=\"sortTable('{table_id}', {i})\">{label}</th>"
        )
    parts.append("<th>Hyp</th>")
    if history_map is not None:
        parts.append("<th>Curve</th>")
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
                # Rule-28 seed-count chip: n=1 (screening) / n>=3 (evaluation)
                n_chip = ""
                if results_dir is not None:
                    ds = str(r.get("dataset", "")) or ""
                    raw_tag = str(r.get("tag", ""))
                    sc = _seed_count_for_tag(results_dir, raw_tag, ds)
                    tier = _evaluation_tier(raw_tag, ds, sc)
                    chip_cls = "eval" if tier == "EVALUATION" else "scr"
                    tip = (
                        f"Rule 28: SCREENING = n=1 seed; EVALUATION = "
                        f"n&gt;=3 seeds AND Phase-5 gate cleared."
                    )
                    n_chip = (
                        f"<span class='n-chip {chip_cls}' title=\"{tip}\">"
                        f"n={sc} {tier.lower()[:4]}</span>"
                    )
                parts.append(
                    f"<td data-v='{_esc(dv)}' style='text-align:left'>"
                    f"{tag_html}{n_chip}</td>"
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
        # Per-tag sparkline cell (test_top1 vs epoch)
        if history_map is not None:
            spark = _tag_sparkline_svg(history_map.get(run_dir_name))
            parts.append(
                f"<td style='text-align:left;padding:2px 6px'>{spark}</td>"
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
                     title: str = "nature_inspired_networks — autoresearch dashboard",
                     subtitle: str = "84-hypothesis dual-track audit on CIFAR-10 + CIFAR-100 / Pareto / hypothesis ledger",
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

    # 2026-05-29 root-cleanup moved EXPERIMENT_LOG.md, IDEA_TABLE.md, and
    # FINDINGS.md into experiments/, hypotheses/, and paper/ respectively.
    # Prefer the new locations; fall back to the legacy paths so historical
    # checkouts still render correctly.
    def _first_existing(*candidates: Path) -> Path:
        for c in candidates:
            if c.exists():
                return c
        return candidates[-1]

    tag_to_tier = parse_experiment_log_tiers(
        _first_existing(root / "experiments" / "EXPERIMENT_LOG.md",
                        root / "EXPERIMENT_LOG.md")
    )
    index_rows = parse_hypothesis_index(root / "hypotheses" / "INDEX.md")
    idea_table_status = parse_idea_table_status(
        _first_existing(root / "hypotheses" / "IDEA_TABLE.md",
                        root / "IDEA_TABLE.md")
    )
    _findings_md = _first_existing(root / "paper" / "FINDINGS.md",
                                   root / "FINDINGS.md")
    findings_blurb = parse_findings_headline(_findings_md)
    findings_metrics = parse_findings_metrics(_findings_md)

    if not df.empty:
        df = df.sort_values(["dataset", "composite"], ascending=[True, False])
        df["__group"] = df["tag"].map(
            lambda t: TAG_TO_HYP.get(t, (None, "Uncategorized"))[1]
        )
        best_idx = set(df.groupby("dataset")["composite"].idxmax().tolist())
    else:
        df["__group"] = pd.Series(dtype=str)
        best_idx = set()

    # Make repo_root available to per-group SVG colour-by-verdict lookup
    _ROOT_FOR_SVG["root"] = root

    # Preload per-tag history for inline sparklines on every leaderboard row
    history_map: dict[str, list[dict]] = {}
    for hj in p.glob("**/history.json"):
        try:
            history_map[hj.parent.name] = json.loads(hj.read_text(encoding="utf-8"))
        except Exception:
            continue

    html = [HTML_HEAD]
    html.append(f"<h1>{title}</h1>")
    html.append(
        f"<div class='sub'>{subtitle}</div>"
    )
    # Masthead CTA pills — repo / paper / literature survey / live demo.
    # Each is a styled call-to-action, opens in a new tab.
    html.append(
        "<div class='mast-row' style='margin:10px 0 6px 0;'>"
        f"<a class='mast-pill repo' "
        f"href='https://github.com/dlmastery/nature_inspired_networks' "
        f"target='_blank' rel='noopener'>🔗 source · dlmastery/nature_inspired_networks</a>"
        f"<a class='mast-pill lit' "
        f"href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/NATURE_INSPIRED_NETWORKS.md' "
        f"target='_blank' rel='noopener'>📚 background reading · nature-inspired networks (state of the field)</a>"
        f"<a class='mast-pill paper' "
        f"href='https://github.com/dlmastery/nature_inspired_networks/blob/main/PAPER.md' "
        f"target='_blank' rel='noopener'>📄 paper</a>"
        f"<a class='live-link' href='https://dlmastery.github.io/nature_inspired_networks/' "
        f"target='_blank' rel='noopener'>📡 live · GitHub Pages</a>"
        "</div>"
    )
    # Secondary doc-link row.
    html.append(
        "<div class='sub' style='margin-top:4px;'>"
        f"<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/FINDINGS.md'>FINDINGS.md</a> &nbsp;·&nbsp; "
        f"<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/experiments/EXPERIMENT_LOG.md'>EXPERIMENT_LOG</a> &nbsp;·&nbsp; "
        f"<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/AUDIT_SUMMARY.md'>AUDIT_SUMMARY.md</a> &nbsp;·&nbsp; "
        f"<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/REVIEWER_CHECKLIST.md'>REVIEWER_CHECKLIST</a> &nbsp;·&nbsp; "
        f"<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/hypotheses/IDEA_TABLE.md'>IDEA_TABLE</a></div>"
    )
    # Phase E (Rule 33): "How to read this dashboard" 4-bullet orientation
    # block — inserted AFTER the masthead but BEFORE the headline ribbon so
    # every reviewer sees the framing before any numerics.
    html.append(_how_to_read_block())

    if findings_blurb:
        # Render the FINDINGS headline through the same markdown converter
        # used everywhere else — the blockquote-stripped table + bold
        # text + code spans all parse correctly.
        headline_html = _md_to_html(findings_blurb)
        html.append(
            "<div class='headline-ribbon md-body' style='--i:1'>"
            f"<b>Headline finding</b>{headline_html}</div>"
        )
    html.append(_composite_formula_chip(df))
    html.append(_ribbon_html(findings_metrics))

    # ------------------------------------------------------------------
    # Phase C: Pareto small-multiples (3 SVG panels, shared y-axis).
    # Phase D: Ablation 8-group panels (one panel per G1..G8, divergent
    #          Δ-vs-baseline colour, ±1.21pp noise band reference line).
    # Both replace the legacy single-PNG cards. The PNG plots still get
    # generated (line above) as a fallback / archival copy, but the live
    # dashboard now embeds the SVG small-multiples directly.
    # ------------------------------------------------------------------
    pareto_rows: list[dict] = []
    if not df.empty:
        pareto_rows = df.to_dict("records")
    html.append("<div class='grid'>")
    html.append(
        "<div class='card panel-2col'>"
        "<h3>Pareto small-multiples · top-1 vs params / FLOPs / latency</h3>"
        + _pareto_panels_svg(pareto_rows)
        + "</div>"
    )
    html.append(
        "<div class='card panel-2col'>"
        "<h3>Ablation matrix · 8 group panels · Δ vs baseline_resnet20 on CIFAR-10</h3>"
        + _ablation_group_panels_svg(pareto_rows, dataset_pref="cifar10")
        + "</div>"
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
        + _hypothesis_grid_html(index_rows, tag_to_tier,
                                idea_table_status=idea_table_status,
                                run_dirs_root=root)
        + "</div>"
    )
    html.append("</div>")  # end .grid

    # ------------------------------------------------------------------
    # Runs sections by hypothesis group (NO modal anywhere)
    # ------------------------------------------------------------------
    html.append(
        "<h2 style='margin-top:48px'>Runs by hypothesis group "
        "<span style='color:var(--paper-dim);font-weight:400;font-size:13px;"
        "font-family:IBM Plex Mono,monospace;font-style:normal;"
        "letter-spacing:0.14em;text-transform:uppercase'>"
        "&nbsp;click any row to expand the experiment page</span></h2>"
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
        # Hide the Uncategorised section when empty (cosmetic cleanup —
        # otherwise a 0-row group renders an empty "No sweep rows yet"
        # box that reviewers will fixate on).
        if g == "Uncategorized" and not rows:
            continue
        html.append(
            _group_section_html(
                g, rows, best_idx,
                table_id=f"runs-{g.lower()}",
                history_map=history_map,
                repo_root=root,
                results_dir=p,
            )
        )

    # Build-stamp footer — Rule 27 + audit fix 5.
    stamp = _build_stamp(root)
    sha_html = ""
    if stamp["sha_short"]:
        sha_html = (
            f"&middot; built at commit "
            f"<a href='{_esc(stamp['url'])}' style='color:#bb8c4d'>"
            f"<code>{_esc(stamp['sha_short'])}</code></a>"
        )
        if stamp["iso_date"]:
            sha_html += f" on <code>{_esc(stamp['iso_date'])}</code> "
    # Cross-link row: repo · paper · background reading · live demo · GitHub Pages.
    html.append(
        "<div class='doc-footer'>"
        "<a href='https://github.com/dlmastery/nature_inspired_networks' "
        "target='_blank' rel='noopener'>Repo</a> &middot; "
        "<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/PAPER.md' "
        "target='_blank' rel='noopener'>Paper</a> &middot; "
        "<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/NATURE_INSPIRED_NETWORKS.md' "
        "target='_blank' rel='noopener'>Background reading</a> &middot; "
        "<a href='https://dlmastery.github.io/nature_inspired_networks/' "
        "target='_blank' rel='noopener'>Live demo</a> &middot; "
        "<a href='https://dlmastery.github.io/nature_inspired_networks/' "
        "target='_blank' rel='noopener'>GitHub Pages</a>"
        "</div>"
    )
    html.append(
        "<div class='meta'>Generated by "
        "<code>nature_inspired_networks.dashboard.render_dashboard</code> "
        "&middot; Brutalist Editorial Lab Notebook v3 "
        "&middot; Source Serif 4 / IBM Plex Mono "
        "&middot; all SVG + CSS inline; no external CDN besides Google Fonts "
        "&middot; leaderboard sectioned by hypothesis group; each row navigates "
        "to its own per-experiment page."
        f"<br>{sha_html}"
        "&middot; live demo: "
        "<a href='https://dlmastery.github.io/nature_inspired_networks/' "
        "target='_blank' rel='noopener'>"
        "https://dlmastery.github.io/nature_inspired_networks/</a>"
        "</div>"
    )

    html.append(HTML_JS)
    html.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html), encoding="utf-8")
