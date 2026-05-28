"""Aggregate experiments/* into the autoresearchspy-rich HTML dashboard.

Usage:
    python scripts/build_dashboard.py [--root experiments] [--out dashboard]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.dashboard import (  # noqa: E402
    load_runs,
    parse_findings_headline,
    render_all_experiment_pages,
    render_dashboard,
)


def _summary_md(results_dir: Path) -> str:
    """Build a small HTML summary of best per dataset."""
    df = load_runs(results_dir)
    if df.empty:
        return "<i style='color:#8b949e'>No runs yet.</i>"
    rows = []
    for ds, g in df.groupby("dataset"):
        best = g.sort_values("composite", ascending=False).iloc[0]
        rows.append(
            f"<li><b>{ds}</b>: top-1 <b>{best['top1']*100:.2f}%</b>, "
            f"params <b>{best['params']/1e6:.3f}M</b>, "
            f"latency <b>{best['latency_ms']:.2f} ms</b>, "
            f"composite <b>{best['composite']:.4f}</b> &nbsp;—&nbsp; "
            f"<code>{best['tag']}</code> seed={int(best['seed'])}</li>"
        )
    return "<ul style='margin-left:18px;font-size:0.9em'>" + "\n".join(rows) + "</ul>"


def _docs_index_html(repo_root: Path) -> str:
    """Build a richer docs/index.html landing page (audit-aware)."""
    return """<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>nature_inspired_networks — autoresearch dashboard</title>
<style>
 *{margin:0;padding:0;box-sizing:border-box;}
 body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;
      color:#c9d1d9;padding:40px 32px;line-height:1.55;}
 a{color:#58a6ff;text-decoration:none;} a:hover{text-decoration:underline;}
 .container{max-width:960px;margin:0 auto;}
 h1{color:#58a6ff;font-size:2.1em;margin-bottom:6px;}
 .sub{color:#8b949e;font-size:0.95em;margin-bottom:24px;}
 .ribbon{background:#161b22;border:1px solid #30363d;border-radius:8px;
         border-left:4px solid #d29922;padding:18px 22px;margin:18px 0 24px 0;}
 .ribbon b{color:#d29922;}
 .audit{background:#241a14;border:1px solid #6d4720;border-radius:8px;
        border-left:4px solid #f0883e;padding:18px 22px;margin:18px 0 24px 0;}
 .audit b{color:#f0883e;}
 .audit code{background:#0d1117;padding:1px 6px;border-radius:3px;color:#c9d1d9;
             font-size:0.92em;}
 .map{background:#161b22;border:1px solid #30363d;border-radius:8px;
       padding:18px 22px;margin-bottom:24px;}
 .map li{margin:6px 0 6px 22px;}
 .map code{background:#0d1117;padding:1px 6px;border-radius:3px;color:#c9d1d9;
           font-size:0.92em;}
 .cta{display:inline-block;background:#1f6feb;color:#fff;padding:12px 24px;
      border-radius:8px;font-weight:600;font-size:1em;border:1px solid #58a6ff;
      box-shadow:0 4px 12px rgba(31,111,235,0.3);transition:transform 0.15s;
      margin-right:10px;margin-bottom:8px;}
 .cta:hover{transform:translateY(-1px);text-decoration:none;background:#388bfd;}
 .cta-secondary{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
                box-shadow:none;}
 .cta-secondary:hover{background:#30363d;color:#fff;}
 .verdict-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
               margin:14px 0;}
 .verdict-cell{background:#0d1117;border:1px solid #30363d;border-radius:6px;
               padding:10px 12px;text-align:center;}
 .verdict-cell .n{font-size:1.6em;font-weight:700;color:#58a6ff;}
 .verdict-cell .lbl{font-size:0.74em;color:#8b949e;text-transform:uppercase;
                    letter-spacing:0.04em;}
 .meta{color:#484f58;font-size:0.78em;margin-top:32px;}
 h2{color:#58a6ff;font-size:1.15em;margin:24px 0 8px 0;}
</style>
</head><body><div class='container'>
<h1>nature_inspired_networks</h1>
<div class='sub'>An autoresearch repository housing 84 nature-inspired neural-network
hypotheses (φ-scaling, hex/Platonic equivariance, fractal recursion, toroidal
closure, Chladni cymatic init, golden-angle modulation, cross-paradigm hybrids,
neutral re-casts of esoteric motifs) — audited adversarially by a dual-track
elite-research critic team and patched by an 8-agent Fixer campaign. ·
<a href='https://github.com/dlmastery/nature_inspired_networks'>GitHub</a> ·
<a href='https://dlmastery.github.io/nature_inspired_networks/'>Pages</a></div>

<div class='audit'><b>⚠ AUDIT IN-PROGRESS · 2026-05-28 ·</b> A dual-track
audit (impl-critic + sci-critic, 8 + 8 parallel agents) just completed.
<b>51 % of 83 implementations landed non-PASS</b>; <b>only 1 of 81 hypotheses
rated NOVEL+TESTABLE</b> (H71 IcosaRoPE3D). The pre-audit headline (H09
φ-budget cross-dataset positive at <code>85.54 % / 58.05 %</code>) was on
broken code — realised stage-param ratio was <code>1:1.41:2.45</code>, not
<code>1:φ:φ²</code>. Fixer-PhiScaling corrected it to <code>1:1.623:2.629</code>
(0.43 % max error); a 31-run post-fix re-run is in flight (~4 h ETA). All
H09/H41/H48 claims are PROVISIONAL until the post-fix numbers land.
<a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/AUDIT_SUMMARY.md'>
Read the full audit summary →</a></div>

<h2>Audit verdict at a glance</h2>
<div class='verdict-grid'>
 <div class='verdict-cell'><div class='n'>41</div><div class='lbl'>impl PASS</div></div>
 <div class='verdict-cell'><div class='n'>24</div><div class='lbl'>impl MINOR</div></div>
 <div class='verdict-cell'><div class='n'>15</div><div class='lbl'>impl MAJOR</div></div>
 <div class='verdict-cell'><div class='n'>3</div><div class='lbl'>impl BROKEN</div></div>
 <div class='verdict-cell'><div class='n'>1</div><div class='lbl'>sci NOVEL</div></div>
 <div class='verdict-cell'><div class='n'>30</div><div class='lbl'>sci DERIVATIVE</div></div>
 <div class='verdict-cell'><div class='n'>40</div><div class='lbl'>sci NUMEROLOGY</div></div>
 <div class='verdict-cell'><div class='n'>5</div><div class='lbl'>sci FALSIFIED+</div></div>
</div>

<a class='cta' href='dashboard/dashboard.html'>📊 Live dashboard &rarr;</a>
<a class='cta cta-secondary'
 href='https://github.com/dlmastery/nature_inspired_networks/blob/main/PAPER.md'>📄 PAPER.md</a>
<a class='cta cta-secondary'
 href='https://github.com/dlmastery/nature_inspired_networks/blob/main/AUDIT_SUMMARY.md'>🔎 AUDIT_SUMMARY.md</a>
<a class='cta cta-secondary'
 href='https://github.com/dlmastery/nature_inspired_networks/blob/main/REVIEWER_CHECKLIST.md'>✅ Reviewer checklist</a>

<h2>What is in this repo</h2>
<div class='map'><ul>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/README.md'>
README.md</a></b> — quick-start, reproduction commands, project orientation.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/PAPER.md'>
PAPER.md</a></b> — post-audit paper draft. <i>The contribution is the protocol,
not the priors.</i></li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/AUDIT_SUMMARY.md'>
AUDIT_SUMMARY.md</a></b> — paper-grade synthesis of impl-critic + sci-critic +
Fixer campaign outputs.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/REVIEWER_CHECKLIST.md'>
REVIEWER_CHECKLIST.md</a></b> — 42-item paper-acceptance gate; <b>39 PASS, 3
gated</b> on the post-fix re-run.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/FINDINGS.md'>
FINDINGS.md</a></b> — curated CIFAR-10/-100 verdicts with the AUDIT NOTICE
prefix marking provisional H09/H41/H48 claims.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/CLAUDE.md'>
CLAUDE.md</a></b> — 25 normative rules. Rules 20-25 added during this campaign
codify auto-checkpoint loops, post-fix re-run, dual-track audit, orthogonal-
axes compounds, dashboard discipline, and Q&A-test correspondence.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/IDEA_TABLE.md'>
IDEA_TABLE.md</a></b> — full 84-hypothesis design space (G1..G8 groups).</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/tree/main/hypotheses'>
hypotheses/</a></b> — committee-grade <code>H&lt;NN&gt;_*.md</code> spec per
hypothesis with the sci-critic addendum appended.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/tree/main/audits'>
audits/</a></b> — per-group impl-critic audit reports (<code>G1_audit.md</code> …
<code>G8_audit.md</code>) with PASS/MINOR/MAJOR/BROKEN verdicts.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/tree/main/skills'>
skills/</a></b> — 17 content-agnostic auto-research skills (the 7 audit-aware
ones — multi-agent-dispatch, critic-team, scicritic-team, fixer-campaign,
combo-ladder, per-experiment-page, auto-checkpoint-loop — were added during
this campaign).</li>
<li><b><a href='dashboard/dashboard.html'>dashboard/</a></b> — live dashboard
sectioned by hypothesis group; every leaderboard row links to a comprehensive
per-experiment page with hypothesis digest, FINDINGS verdict, reasoning,
config, metrics, composite breakdown, and inline-SVG training curves.</li>
</ul></div>

<h2>Reproduce</h2>
<div class='map' style='font-family:Consolas,monospace;font-size:0.85em;
                       white-space:pre-line;color:#c9d1d9'>
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
.\\.venv\\Scripts\\python -u scripts\\run_sweep.py `
   --config configs\\cifar10_quick.yaml --seeds 0 --skip-existing
.\\.venv\\Scripts\\python scripts\\build_dashboard.py
start dashboard\\dashboard.html
</div>

<div class='meta'>Generated by <code>scripts/build_dashboard.py</code> · audit-
aware landing v2 · dashboard theme adapted from dlmastery/autoresearchspy.</div>
</div></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--out", default="dashboard")
    args = ap.parse_args(argv)

    results = Path(args.root)
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = [("Best per dataset (by composite)", _summary_md(results))]

    render_dashboard(
        results,
        out_dir / "dashboard.html",
        extra_sections=extra,
        title="NaturePriorBlock &mdash; autoresearch dashboard",
        subtitle=(
            "Nature-inspired ablations on CIFAR-10 &middot; topology &middot; "
            "71-hypothesis ledger &middot; T0..T6 tier filter"
        ),
        repo_root=repo_root,
    )

    # Per-experiment independent pages (one .html per <tag>_seed<N> run dir).
    exp_out = out_dir / "experiments"
    exp_pages, coverage = render_all_experiment_pages(
        results, exp_out, repo_root=repo_root,
    )
    print(f"[ok] {len(exp_pages)} per-experiment pages written to {exp_out}")
    if exp_pages:
        n = len(exp_pages)
        print(
            "[ok] section coverage (real content vs. fallback): "
            f"hypothesis={coverage['hypothesis']}/{n}, "
            f"verdict={coverage['verdict']}/{n}, "
            f"reasoning={coverage['reasoning']}/{n}, "
            f"config={coverage['config']}/{n}, "
            f"history={coverage['history']}/{n}, "
            f"cross_refs={coverage['cross_refs']}/{n}"
        )

    # Mirror PNG plots into docs/dashboard/ and copy reasoning + betti so the
    # docs mirror is self-contained.
    docs_dir = Path("docs") / "dashboard"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for png in results.glob("plot_*.png"):
        (docs_dir / png.name).write_bytes(png.read_bytes())

    # Also produce the docs/ HTML with side-panel paths adjusted (../hypotheses
    # works the same from both dashboard/ and docs/dashboard/ because both are
    # one directory deep from the repo root).
    dashboard_html = (out_dir / "dashboard.html").read_text(encoding="utf-8")
    (docs_dir / "dashboard.html").write_text(dashboard_html, encoding="utf-8")

    # Mirror the per-experiment pages into docs/dashboard/experiments/ so they
    # are live on GitHub Pages at the same relative path as in dashboard/.
    docs_exp_dir = docs_dir / "experiments"
    docs_exp_dir.mkdir(parents=True, exist_ok=True)
    for fname in exp_pages:
        (docs_exp_dir / fname).write_text(
            (exp_out / fname).read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(f"[ok] {len(exp_pages)} per-experiment pages mirrored to {docs_exp_dir}")

    # docs/index.html landing page with the negative-finding ribbon + CTA.
    (Path("docs") / "index.html").write_text(
        _docs_index_html(repo_root), encoding="utf-8"
    )

    out_main = out_dir / "dashboard.html"
    out_docs = docs_dir / "dashboard.html"
    print(f"[ok] dashboard written to {out_main} ({out_main.stat().st_size} bytes)")
    print(f"[ok] docs mirror at {out_docs} ({out_docs.stat().st_size} bytes)")
    idx = Path("docs") / "index.html"
    print(f"[ok] docs landing at {idx} ({idx.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
