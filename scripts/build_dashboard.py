"""Aggregate experiments/* into the autoresearch-style HTML dashboard.

Usage:
    python scripts/build_dashboard.py [--root experiments] [--out dashboard]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nature_inspired_networks.dashboard import render_dashboard  # noqa: E402


def _summary_md(results_dir: Path) -> str:
    """Build a small Markdown summary of best per dataset (rendered to HTML)."""
    import pandas as pd
    from nature_inspired_networks.dashboard import load_runs
    df = load_runs(results_dir)
    if df.empty:
        return "<i>No runs yet.</i>"
    rows = []
    for ds, g in df.groupby("dataset"):
        best = g.sort_values("composite", ascending=False).iloc[0]
        rows.append(
            f"<li><b>{ds}</b>: top-1 <b>{best['top1']*100:.2f}%</b>, "
            f"params {best['params']/1e6:.3f}M, lat {best['latency_ms']:.2f} ms, "
            f"composite <b>{best['composite']:.4f}</b> "
            f"— <code>{best['tag']}</code> seed={int(best['seed'])}</li>"
        )
    return "<ul>" + "\n".join(rows) + "</ul>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--out", default="dashboard")
    args = ap.parse_args(argv)

    results = Path(args.root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    extra = [("Best per dataset", _summary_md(results))]
    render_dashboard(results, out_dir / "dashboard.html",
                     extra_sections=extra)

    # Also copy plots & write index for docs/
    docs_dir = Path("docs") / "dashboard"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for png in results.glob("plot_*.png"):
        (docs_dir / png.name).write_bytes(png.read_bytes())
    (Path("docs") / "dashboard" / "dashboard.html").write_text(
        (out_dir / "dashboard.html").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Tiny docs/index.html landing
    (Path("docs") / "index.html").write_text(
        """<!doctype html><meta charset='utf-8'>
<title>nature_inspired_networks — autoresearch dashboard</title>
<body style='font-family:sans-serif;margin:40px'>
<h1>nature_inspired_networks</h1>
<p>nature-inspired-priored neural backbones; autoresearch-style ablations.</p>
<p>→ <a href='dashboard/dashboard.html'>Open the live dashboard</a></p>
</body>""",
        encoding="utf-8",
    )

    print(f"[ok] dashboard written to {out_dir/'dashboard.html'}")
    print(f"[ok] docs mirror at docs/dashboard/dashboard.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
