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
    """Build a richer docs/index.html landing page."""
    findings = parse_findings_headline(repo_root / "FINDINGS.md")
    if not findings:
        findings = (
            "Negative result expected: full nature-prior hybrid loses to the "
            "ResNet-20 baseline on CIFAR-10 at 12 epochs."
        )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>nature_inspired_networks — autoresearch dashboard</title>
<style>
 *{{margin:0;padding:0;box-sizing:border-box;}}
 body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;
      color:#c9d1d9;padding:40px 32px;line-height:1.55;}}
 a{{color:#58a6ff;text-decoration:none;}} a:hover{{text-decoration:underline;}}
 .container{{max-width:880px;margin:0 auto;}}
 h1{{color:#58a6ff;font-size:2em;margin-bottom:6px;}}
 .sub{{color:#8b949e;font-size:0.95em;margin-bottom:24px;}}
 .ribbon{{background:#161b22;border:1px solid #30363d;border-radius:8px;
         border-left:4px solid #d29922;padding:18px 22px;margin:18px 0 24px 0;}}
 .ribbon b{{color:#d29922;}}
 .map{{background:#161b22;border:1px solid #30363d;border-radius:8px;
       padding:18px 22px;margin-bottom:24px;}}
 .map li{{margin:6px 0 6px 22px;}}
 .map code{{background:#0d1117;padding:1px 6px;border-radius:3px;
           color:#c9d1d9;font-size:0.92em;}}
 .cta{{display:inline-block;background:#1f6feb;color:#fff;padding:12px 24px;
      border-radius:8px;font-weight:600;font-size:1em;border:1px solid #58a6ff;
      box-shadow:0 4px 12px rgba(31,111,235,0.3);transition:transform 0.15s;}}
 .cta:hover{{transform:translateY(-1px);text-decoration:none;
             background:#388bfd;}}
 .meta{{color:#484f58;font-size:0.78em;margin-top:32px;}}
 h2{{color:#58a6ff;font-size:1.15em;margin:24px 0 8px 0;}}
</style>
</head><body><div class='container'>
<h1>nature_inspired_networks</h1>
<div class='sub'>Nature-inspired neural backbones — φ-scaling, hex/group conv, fractal recursion,
toroidal padding, cymatic init, golden-angle channel gating — evaluated under an
autoresearch protocol on CIFAR-10. ·
<a href='https://github.com/dlmastery/nature_inspired_networks'>GitHub</a> ·
<a href='https://dlmastery.github.io/nature_inspired_networks/'>GitHub Pages</a></div>

<div class='ribbon'><b>Headline negative finding · </b>{findings}</div>

<a class='cta' href='dashboard/dashboard.html'>Live dashboard &rarr;</a>

<h2>What is in this repo</h2>
<div class='map'><ul>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/README.md'>
README.md</a></b> — quick-start, reproduction commands, project orientation.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/MANIFESTO.md'>
MANIFESTO.md</a></b> — research-strict protocol: every claim is hypothesis +
prediction + verdict + learning, signed by composite-formula SHA-256.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/blob/main/IDEA_TABLE.md'>
IDEA_TABLE.md</a></b> — full 71-hypothesis design space (G1..G7 groups) distilled
from the four source documents.</li>
<li><b><a href='https://github.com/dlmastery/nature_inspired_networks/tree/main/hypotheses'>
hypotheses/</a></b> — one committee-grade <code>H&lt;NN&gt;_*.md</code> spec per
row in <code>IDEA_TABLE.md</code>. The dashboard's 71-cell grid links here.</li>
<li><b><a href='dashboard/dashboard.html'>dashboard/</a></b> — the live
autoresearch dashboard: Pareto plots, ablation matrix, training curves,
persistent-homology Betti collapse, hypothesis status heatmap, reasoning-ledger
modal, T0..T6 tier filters.</li>
</ul></div>

<h2>Reproduce</h2>
<div class='map' style='font-family:Consolas,monospace;font-size:0.85em;
                       white-space:pre-line;color:#c9d1d9'>
$env:SSL_CERT_FILE = ".\\.venv\\Lib\\site-packages\\certifi\\cacert.pem"
.\\.venv\\Scripts\\python -u scripts\\run_sweep.py `
   --config configs\\cifar10_quick.yaml --seeds 0 --skip-existing
.\\.venv\\Scripts\\python scripts\\compute_topology.py --seeds 0
.\\.venv\\Scripts\\python scripts\\build_dashboard.py
start dashboard\\dashboard.html
</div>

<div class='meta'>Generated by <code>scripts/build_dashboard.py</code> · dark theme adapted from
dlmastery/autoresearchspy.</div>
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
    exp_pages = render_all_experiment_pages(results, exp_out)
    print(f"[ok] {len(exp_pages)} per-experiment pages written to {exp_out}")

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
