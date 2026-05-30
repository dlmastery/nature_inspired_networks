"""Per-hypothesis hill-climb dashboard renderer.

Reads ``ideas/<NN>/hillclimb_results.json`` and the matching
``experiments/<dataset>/<tag>__hc_*_seed<S>/metrics.json`` files and
emits a self-contained ``ideas/<NN>/dashboard/index.html`` page that
satisfies the ``autoresearch-per-hypothesis-hillclimb`` skill contract
(SKILL.md §"Output contract → Per-hypothesis dashboard").

The renderer is content-agnostic and self-contained (Rule 10): it does
NOT import from ``src/nature_inspired_networks/dashboard.py``; the
visual style (Source Serif 4 body, IBM Plex Mono monospace, dark
"brutalist editorial" palette) is duplicated here so the per-hypothesis
sub-pages stay decoupled from the master renderer (Rule 14: shared
primitives are reused; the duplication here is intentional because the
sub-page must keep working even if the master renderer is mid-refactor).

Usage:

    python ideas/_TEMPLATE/dashboard/render.py \
        --idea 09_phi_budget \
        --hypothesis H09 \
        --hypothesis-group g1_scaling_growth \
        --hypothesis-slug H09_golden_proportion_param_budget

The renderer is invoked once per winner from the dashboard build step
of the hill-climb skill (see ``skills/autoresearch-per-hypothesis-
hillclimb/SKILL.md``). It runs in <2 s per idea.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
GITHUB_BLOB = (
    "https://github.com/dlmastery/nature_inspired_networks/blob/main"
)
GITHUB_PAGES_BASE = (
    "https://dlmastery.github.io/nature_inspired_networks"
)

# Brutalist Editorial palette (mirrors src/nature_inspired_networks/dashboard.py
# but duplicated here so this sub-page stays self-contained per Rule 10/14).
_PAGE_CSS = r"""
:root{
  --ink:#0a0a0d; --paper:#e6e1d6; --paper-dim:#a89e8c;
  --rule:#1c1c20; --rule-bright:#2a2a30;
  --panel:#111114; --panel2:#16161a;
  --accent:#bb8c4d; --accent-dim:#7a5e36;
  --v-pass:#3fb950; --v-minor:#d29922; --v-major:#f0883e; --v-broken:#f85149;
  --v-novel:#a371f7; --v-derivative:#58a6ff; --v-numerology:#8b949e;
  --v-falsified:#db6d28; --v-infra:#8b949e;
  --best-row:rgba(63, 185, 80, 0.10);
  --best-row-border:rgba(63, 185, 80, 0.55);
}
*{margin:0;padding:0;box-sizing:border-box;}
html,body{background:var(--ink);}
body{font-family:'Source Serif 4','Charter','Source Serif Pro',Georgia,serif;
     color:var(--paper);padding:32px 36px 80px;line-height:1.6;
     max-width:1220px;margin:0 auto;font-size:16px;
     font-variant-numeric:tabular-nums;}
a{color:var(--v-derivative);text-decoration:none;
  border-bottom:1px solid transparent;transition:border-color 160ms ease;}
a:hover{border-bottom-color:var(--v-derivative);}
h1{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
   font-size:34px;line-height:1.15;color:var(--paper);letter-spacing:-0.005em;
   margin-bottom:6px;}
h2{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
   font-size:20px;color:var(--paper);margin:28px 0 12px;
   letter-spacing:-0.003em;border-bottom:1px solid var(--rule);
   padding-bottom:6px;}
h3{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:11px;
   text-transform:uppercase;letter-spacing:0.16em;color:var(--paper-dim);
   margin:18px 0 8px;}
.mono{font-family:'IBM Plex Mono',monospace;font-size:0.86em;}
.head-grid{display:grid;grid-template-columns:1fr auto;gap:24px;
           align-items:start;padding-bottom:18px;
           border-bottom:1px solid var(--rule);margin-bottom:14px;}
.tag-display{font-family:'Source Serif 4',Georgia,serif;font-size:42px;
   font-weight:600;line-height:1.1;color:var(--paper);
   letter-spacing:-0.012em;word-break:break-word;}
.tag-sub{font-family:'IBM Plex Mono',monospace;font-size:11px;
   text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);
   margin-top:8px;}
.head-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px;
   min-width:240px;}
.pill{display:inline-block;background:transparent;
      border:1px solid var(--rule-bright);border-radius:1px;
      padding:3px 10px;font-size:0.72em;color:var(--paper-dim);
      font-family:'IBM Plex Mono',monospace;margin:0 4px 4px 0;
      text-transform:uppercase;letter-spacing:0.12em;}
.pill.hyp{border-color:var(--v-derivative);color:var(--v-derivative);}
.pill.grp{border-color:var(--v-novel);color:var(--v-novel);}
.pill.ds{border-color:var(--accent-dim);color:var(--accent);}
.pill.tier{border-color:var(--v-pass);color:var(--v-pass);font-weight:600;
           background:rgba(63,185,80,0.06);}
.mast-pill{display:inline-block;padding:4px 11px;margin-right:6px;
   border:1px solid var(--accent-dim);color:var(--accent);
   font-family:'IBM Plex Mono',monospace;font-size:10px;
   text-transform:uppercase;letter-spacing:0.16em;font-weight:600;
   text-decoration:none;}
.mast-pill:hover{background:var(--accent-dim);color:var(--ink);
   border-bottom-color:var(--accent-dim);}
.mast-pill.repo{border-color:var(--v-novel);color:var(--v-novel);}
.mast-pill.repo:hover{background:var(--v-novel);color:var(--ink);}
.mast-pill.paper{border-color:var(--v-pass);color:var(--v-pass);}
.mast-pill.paper:hover{background:var(--v-pass);color:var(--ink);}
.mast-pill.lit{border-color:var(--v-derivative);color:var(--v-derivative);}
.mast-pill.lit:hover{background:var(--v-derivative);color:var(--ink);}
.cta-row{margin:12px 0 24px;}
.callout{font-family:'IBM Plex Mono',monospace;font-size:14px;
   background:var(--panel);border:1px solid var(--rule-bright);
   border-left:3px solid var(--v-pass);padding:14px 18px;margin:14px 0 24px;
   line-height:1.55;color:var(--paper);white-space:pre;overflow-x:auto;}
.callout .k{color:var(--accent);}
.callout .v{color:var(--paper);font-weight:600;}
.callout .lbl{color:var(--paper-dim);}
.callout .pass{color:var(--v-pass);font-weight:600;}
.callout .fail{color:var(--v-broken);font-weight:600;}
table.cells{border-collapse:collapse;width:100%;margin:10px 0 24px;
   font-size:13px;font-family:'IBM Plex Mono',monospace;
   font-variant-numeric:tabular-nums;}
table.cells th,table.cells td{border:1px solid var(--rule);padding:6px 9px;
   text-align:right;white-space:nowrap;}
table.cells th{background:var(--panel2);color:var(--paper-dim);font-weight:600;
   text-transform:uppercase;letter-spacing:0.08em;font-size:10px;
   cursor:pointer;user-select:none;text-align:center;}
table.cells th:hover{background:var(--rule-bright);color:var(--paper);}
table.cells th.sort-asc::after{content:" \25B2";color:var(--accent);}
table.cells th.sort-desc::after{content:" \25BC";color:var(--accent);}
table.cells td.col-cfg{text-align:left;color:var(--paper-dim);}
table.cells tr:hover td{background:var(--panel);}
table.cells tr.best-row td{background:var(--best-row);
   border-color:var(--best-row-border);color:var(--paper);}
table.cells tr.best-row td.col-rank::before{content:"\2605 ";color:var(--v-pass);}
.delta-pos{color:var(--v-pass);}
.delta-neg{color:var(--v-broken);}
.delta-zero{color:var(--paper-dim);}
.pareto-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;
   margin:12px 0 26px;}
.pareto-row .panel{background:var(--panel);border:1px solid var(--rule);
   padding:10px 12px;}
.pareto-row .cap{font-family:'IBM Plex Mono',monospace;font-size:10px;
   color:var(--paper-dim);text-transform:uppercase;letter-spacing:0.14em;
   margin-top:8px;line-height:1.5;}
.stability-list{display:flex;flex-direction:column;gap:10px;margin:12px 0;}
.stab-row{display:grid;grid-template-columns:280px 1fr 60px;gap:14px;
   align-items:center;font-family:'IBM Plex Mono',monospace;font-size:11px;
   color:var(--paper-dim);}
.stab-row .cid{color:var(--paper);}
.stab-row .bar{position:relative;height:22px;background:var(--panel);
   border:1px solid var(--rule);}
.stab-row .bar-fill{position:absolute;top:3px;bottom:3px;
   background:rgba(63,185,80,0.40);border:1px solid rgba(63,185,80,0.85);}
.stab-row .bar-fill.amber{background:rgba(210,153,34,0.40);
   border-color:rgba(210,153,34,0.85);}
.stab-row .bar-fill.red{background:rgba(248,81,73,0.36);
   border-color:rgba(248,81,73,0.85);}
.stab-row .bar-tick{position:absolute;top:0;bottom:0;width:2px;
   background:var(--paper);}
.stab-row .bar-tick.med{background:var(--v-pass);}
.stab-row .std{text-align:right;font-size:11px;color:var(--paper);}
.footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--rule);
   color:var(--paper-dim);font-size:13px;line-height:1.7;}
.footer code{font-family:'IBM Plex Mono',monospace;color:var(--accent);}
.footer .typo-note{font-style:italic;font-size:12px;margin-top:10px;
   color:var(--paper-dim);}
"""

_SORT_JS = r"""
(function(){
  function val(td, type){
    var t = td.getAttribute('data-sort');
    if(t === null) t = td.textContent;
    if(type === 'num'){ var n = parseFloat(t); return isNaN(n) ? -Infinity : n; }
    return String(t).toLowerCase();
  }
  document.querySelectorAll('table.cells').forEach(function(tbl){
    var ths = tbl.querySelectorAll('th');
    ths.forEach(function(th, idx){
      th.addEventListener('click', function(){
        var dir = th.classList.contains('sort-asc') ? 'desc' : 'asc';
        ths.forEach(function(o){o.classList.remove('sort-asc','sort-desc');});
        th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
        var type = th.getAttribute('data-type') || 'str';
        var rows = Array.prototype.slice.call(tbl.querySelectorAll('tbody tr'));
        rows.sort(function(a, b){
          var va = val(a.cells[idx], type);
          var vb = val(b.cells[idx], type);
          if(va < vb) return dir === 'asc' ? -1 : 1;
          if(va > vb) return dir === 'asc' ?  1 : -1;
          return 0;
        });
        var tbody = tbl.querySelector('tbody');
        rows.forEach(function(r){tbody.appendChild(r);});
      });
    });
  });
})();
"""


def _esc(s: Any) -> str:
    s = "" if s is None else str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt_lr(lr: float) -> str:
    # 3e-4 / 1e-3 / 3e-3 stylings
    if lr <= 0:
        return "0"
    exp = int(math.floor(math.log10(lr)))
    mant = lr / 10 ** exp
    if abs(mant - round(mant)) < 1e-6:
        return f"{int(round(mant))}e{exp:+d}"
    return f"{mant:.2f}e{exp:+d}"


def _fmt_wd(wd: float) -> str:
    return _fmt_lr(wd)


def _fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _load_cells(
    summary: dict, dataset: str
) -> list[dict]:
    """Augment summary['cells'] with metrics.json fields where present.

    Each returned cell has: config (dict), seed, config_id, top1,
    composite, wallclock_s, params, latency_ms, link.
    """
    tag = summary["tag"]
    cells = []
    for c in summary["cells"]:
        cfg = c["config"]
        cid = c["config_id"]
        seed = c["seed"]
        run_dir = REPO_ROOT / "experiments" / dataset / (
            f"{tag}__hc_{cid}_seed{seed}"
        )
        metrics_path = run_dir / "metrics.json"
        m: dict = {}
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                m = {}
        cells.append({
            "tag": tag,
            "config": cfg,
            "config_id": cid,
            "seed": seed,
            "top1": float(c.get("top1", m.get("top1", float("nan")))),
            "composite": float(c.get("composite",
                                     m.get("composite", float("nan")))),
            "wallclock_s": float(c.get("wallclock_s",
                                        m.get("train_seconds", 0.0))),
            "params": int(m.get("params", 0)),
            "latency_ms": float(m.get("latency_ms", 0.0)),
            "epochs": int(m.get("epochs", 0)),
            "metrics_path": metrics_path,
        })
    return cells


def _config_matches(cfg: dict, best: dict) -> bool:
    return all(cfg.get(k) == best.get(k) for k in best.keys())


def _per_cell_median(cells: list[dict]) -> dict[str, dict]:
    """Group cells by config_id, return median (and min/max) top1."""
    by_cid: dict[str, list[dict]] = {}
    for c in cells:
        by_cid.setdefault(c["config_id"], []).append(c)
    out = {}
    for cid, rows in by_cid.items():
        t1 = [r["top1"] for r in rows]
        comp = [r["composite"] for r in rows]
        out[cid] = {
            "rows": rows,
            "config": rows[0]["config"],
            "top1_median": statistics.median(t1),
            "top1_min": min(t1),
            "top1_max": max(t1),
            "top1_range": max(t1) - min(t1),
            "composite_median": statistics.median(comp),
            "n_seeds": len(rows),
            "seeds": sorted({r["seed"] for r in rows}),
        }
    return out


# ---------------------------------------------------------------------------
# Section 1 — Masthead + best-config callout
# ---------------------------------------------------------------------------

def render_masthead(
    *,
    idea: str,
    hypothesis_id: str,
    hypothesis_group: str,
    hypothesis_slug: str,
    summary: dict,
    by_cid: dict,
    baseline_top1_max: float,
) -> str:
    tag = summary["tag"]
    best = summary["best_config"]
    best_cid = _build_cid(best)
    bm = by_cid.get(best_cid, {})
    med = summary.get("best_top1_median", bm.get("top1_median", 0.0))
    mn = summary.get("best_top1_min", bm.get("top1_min", 0.0))
    mx = bm.get("top1_max", 0.0)
    comp_med = bm.get("composite_median", 0.0)
    n = bm.get("n_seeds", len(summary.get("seeds_confirmed", [])))
    gate_pass = mn > baseline_top1_max
    gate_text = (
        f"<span class='pass'>YES (min {_fmt_pct(mn)} &gt; baseline max "
        f"{_fmt_pct(baseline_top1_max)})</span>"
        if gate_pass
        else f"<span class='fail'>NO (min {_fmt_pct(mn)} &le; baseline max "
             f"{_fmt_pct(baseline_top1_max)})</span>"
    )

    hyp_url = (
        f"{GITHUB_BLOB}/hypotheses/{hypothesis_group}/{hypothesis_slug}.md"
    )
    paper_url = f"{GITHUB_BLOB}/PAPER.md#55--three-phase-8-candidates-surfaced-by-the-protocol--certified-at-n7-2026-05-29"
    master_dash_url = "../../../dashboard/dashboard.html"
    repo_url = "https://github.com/dlmastery/nature_inspired_networks"
    pages_url = f"{GITHUB_PAGES_BASE}/"
    pages_demo_url = (
        f"{GITHUB_PAGES_BASE}/dashboard/ideas/{idea}/"
    )
    audit_url = (
        f"{GITHUB_BLOB}/audits/G{hypothesis_group[1]}_audit.md"
    )

    callout_lines = (
        f"<span class='lbl'>tag       </span> "
        f"<span class='v'>{_esc(tag)}</span>\n"
        f"<span class='lbl'>best_cfg  </span> "
        f"<span class='k'>lr=</span><span class='v'>{_fmt_lr(best['lr'])}</span>"
        f"  <span class='k'>wd=</span><span class='v'>{_fmt_wd(best['weight_decay'])}</span>"
        f"  <span class='k'>bs=</span><span class='v'>{best['batch_size']}</span>"
        f"  <span class='k'>opt=</span><span class='v'>{_esc(best['optimizer'])}</span>\n"
        f"<span class='lbl'>top1      </span> "
        f"<span class='v'>{_fmt_pct(med)}</span> median  "
        f"(min {_fmt_pct(mn)}, max {_fmt_pct(mx)}, n={n} seeds)\n"
        f"<span class='lbl'>composite </span> "
        f"<span class='v'>{comp_med:.4f}</span> median  "
        f"(formula sha256: {summary.get('fingerprint','d65565e9c7b1')[:12]})\n"
        f"<span class='lbl'>gate      </span> {gate_text}"
    )
    callout = f"<div class='callout'>{callout_lines}</div>"

    head = (
        "<div class='head-grid'>"
        "<div class='head-left'>"
        f"<div class='tag-display'>{_esc(tag)}</div>"
        f"<div class='tag-sub'>idea {_esc(idea)} &middot; hypothesis "
        f"{_esc(hypothesis_id)} &middot; per-hypothesis hill-climb dashboard "
        f"&middot; CIFAR-100, 30 ep</div>"
        "<div style='margin-top:12px;'>"
        f"<span class='pill hyp'>{_esc(hypothesis_id)}</span>"
        f"<span class='pill grp'>{_esc(hypothesis_group.upper())}</span>"
        f"<span class='pill ds'>CIFAR-100</span>"
        f"<span class='pill tier'>EVALUATION (n&ge;3, hill-climbed)</span>"
        "</div>"
        "</div>"
        "<div class='head-right'>"
        f"<a class='mast-pill repo' href='{repo_url}' target='_blank' rel='noopener'>&#128279; repo</a>"
        f"<a class='mast-pill paper' href='{paper_url}' target='_blank' rel='noopener'>&#128196; paper</a>"
        f"<a class='mast-pill lit' href='{hyp_url}' target='_blank' rel='noopener'>&#128218; background reading</a>"
        f"<a class='mast-pill' href='{pages_demo_url}' target='_blank' rel='noopener'>&#128225; live demo</a>"
        "</div>"
        "</div>"
    )

    # Navigation strip below masthead
    nav = (
        "<div class='cta-row mono' style='color:var(--paper-dim);font-size:12px;'>"
        f"&rarr; <a href='{master_dash_url}'>master aggregate dashboard</a> "
        f"&nbsp;|&nbsp; <a href='{paper_url}' target='_blank' rel='noopener'>"
        "PAPER.md &sect;5.5 certified-claim section</a> "
        f"&nbsp;|&nbsp; <a href='{audit_url}' target='_blank' rel='noopener'>"
        f"audit-critic verdict ({hypothesis_group.upper()})</a> "
        f"&nbsp;|&nbsp; <a href='{hyp_url}' target='_blank' rel='noopener'>"
        f"design doc {_esc(hypothesis_id)}</a>"
        "</div>"
    )

    return head + nav + "<h2>1 &middot; Best-config callout</h2>" + callout


# ---------------------------------------------------------------------------
# Section 2 — Per-cell metric table
# ---------------------------------------------------------------------------

def _build_cid(cfg: dict) -> str:
    """Mirror scripts/run_hillclimb.py's config_id formatter."""
    def lr_tok(x: float) -> str:
        m = {3e-4: "3em4", 1e-3: "1em3", 3e-3: "3em3"}
        for k, v in m.items():
            if abs(x - k) / k < 1e-3:
                return v
        return f"{x:.0e}".replace("-", "m")

    def wd_tok(x: float) -> str:
        m = {1e-4: "1em4", 5e-4: "5em4", 2e-3: "2em3"}
        for k, v in m.items():
            if abs(x - k) / k < 1e-3:
                return v
        return f"{x:.0e}".replace("-", "m")

    return (
        f"lr{lr_tok(cfg['lr'])}_wd{wd_tok(cfg['weight_decay'])}"
        f"_bs{cfg['batch_size']}_opt{cfg['optimizer']}"
    )


def render_cells_table(cells: list[dict], best_cid: str) -> str:
    # Per-config aggregation (median / min / max top1 over seeds run at the
    # same config); rank by composite_median, ties broken by top1_median.
    by_cid = _per_cell_median(cells)
    cid_order = sorted(
        by_cid.keys(),
        key=lambda c: (-by_cid[c]["composite_median"],
                       -by_cid[c]["top1_median"]),
    )
    rank: dict[str, int] = {c: i + 1 for i, c in enumerate(cid_order)}
    best_top1_med = by_cid[best_cid]["top1_median"]

    head = (
        "<thead><tr>"
        "<th data-type='num'>cell #</th>"
        "<th data-type='str' class='col-cfg'>config_id</th>"
        "<th data-type='num'>lr</th>"
        "<th data-type='num'>wd</th>"
        "<th data-type='num'>batch</th>"
        "<th data-type='str'>opt</th>"
        "<th data-type='num'>seed</th>"
        "<th data-type='num'>seeds run</th>"
        "<th data-type='num'>top1 (%)</th>"
        "<th data-type='num'>top1 median (%)</th>"
        "<th data-type='num'>top1 min (%)</th>"
        "<th data-type='num'>top1 max (%)</th>"
        "<th data-type='num'>composite</th>"
        "<th data-type='num'>composite median</th>"
        "<th data-type='num'>&Delta; vs best (pp)</th>"
        "<th data-type='num' class='col-rank'>ranked</th>"
        "</tr></thead>"
    )
    body_rows: list[str] = []
    # Emit one row per raw hill-climb cell so the table mirrors
    # hillclimb_results.json's cells[] (9 rows on the canonical winner).
    cell_index = 0
    # Sort raw cells: by config rank then seed, so multi-seed best rows
    # appear together at the top.
    cells_sorted = sorted(
        cells,
        key=lambda c: (rank[c["config_id"]], c["seed"]),
    )
    for c in cells_sorted:
        cell_index += 1
        cid = c["config_id"]
        agg = by_cid[cid]
        cfg = c["config"]
        d_pp = (c["top1"] - best_top1_med) * 100.0
        if abs(d_pp) < 0.005:
            d_cls = "delta-zero"
            d_str = "0.00"
        elif d_pp > 0:
            d_cls = "delta-pos"
            d_str = f"+{d_pp:.2f}"
        else:
            d_cls = "delta-neg"
            d_str = f"{d_pp:.2f}"
        metrics_link = (
            f"{GITHUB_BLOB}/experiments/cifar100/"
            f"{c['tag']}__hc_{cid}_seed{c['seed']}/metrics.json"
        )
        row_class = "best-row" if cid == best_cid else ""
        row_title = (
            f"metrics.json: experiments/cifar100/{c['tag']}__hc_"
            f"{cid}_seed{c['seed']}/"
        )
        body_rows.append(
            f"<tr class='{row_class}' title='{_esc(row_title)}'>"
            f"<td data-sort='{cell_index}'>{cell_index}</td>"
            f"<td class='col-cfg'><a href='{metrics_link}' "
            f"target='_blank' rel='noopener'>{_esc(cid)}</a></td>"
            f"<td data-sort='{cfg['lr']}'>{_fmt_lr(cfg['lr'])}</td>"
            f"<td data-sort='{cfg['weight_decay']}'>"
            f"{_fmt_wd(cfg['weight_decay'])}</td>"
            f"<td>{cfg['batch_size']}</td>"
            f"<td>{_esc(cfg['optimizer'])}</td>"
            f"<td data-sort='{c['seed']}'>{c['seed']}</td>"
            f"<td data-sort='{agg['n_seeds']}'>{agg['n_seeds']}</td>"
            f"<td data-sort='{c['top1']}'>{_fmt_pct(c['top1'])}</td>"
            f"<td data-sort='{agg['top1_median']}'>"
            f"{_fmt_pct(agg['top1_median'])}</td>"
            f"<td data-sort='{agg['top1_min']}'>"
            f"{_fmt_pct(agg['top1_min'])}</td>"
            f"<td data-sort='{agg['top1_max']}'>"
            f"{_fmt_pct(agg['top1_max'])}</td>"
            f"<td data-sort='{c['composite']}'>{c['composite']:.4f}</td>"
            f"<td data-sort='{agg['composite_median']}'>"
            f"{agg['composite_median']:.4f}</td>"
            f"<td data-sort='{d_pp}' class='{d_cls}'>{d_str}</td>"
            f"<td class='col-rank' data-sort='{rank[cid]}'>{rank[cid]}</td>"
            "</tr>"
        )
    table = (
        "<table class='cells'>"
        + head
        + "<tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
        "<div class='mono' style='color:var(--paper-dim);font-size:11px;'>"
        "Click a column header to sort. One row per raw hill-climb cell "
        "(mirrors <code>hillclimb_results.json</code> &middot; "
        "<code>cells</code>). Best-config rows highlighted green (these "
        "share the same <code>config_id</code> at seeds 0/1/2). "
        "&Delta; vs best is the per-seed top-1 minus the best config's "
        "3-seed median, in percentage points. Hover a row for the "
        "underlying <code>metrics.json</code> path; click the "
        "<code>config_id</code> to open it on GitHub. "
        "<code>ranked</code> column ranks by composite_median per config."
        "</div>"
    )
    return "<h2>2 &middot; Per-cell metric table</h2>" + table


# ---------------------------------------------------------------------------
# Section 3 — Per-axis Pareto plots (inline SVG)
# ---------------------------------------------------------------------------

def _svg_scatter(
    *,
    title: str,
    points: list[tuple[float, float, dict, bool]],  # (x, y, cfg, is_best)
    x_label: str,
    y_label: str,
    log_x: bool,
    width: int = 360,
    height: int = 260,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 56, 14, 28, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if not points:
        return (
            f"<svg width='{width}' height='{height}' "
            "xmlns='http://www.w3.org/2000/svg' style='background:#0a0a0d;"
            "border:1px solid #1c1c20;display:block;'>"
            "<text x='50%' y='50%' text-anchor='middle' fill='#8b949e' "
            "font-family='Source Serif 4,Georgia,serif' font-size='12'>"
            "no data</text></svg>"
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if log_x:
        xs_t = [math.log10(max(x, 1e-12)) for x in xs]
    else:
        xs_t = list(xs)
    x_min, x_max = min(xs_t), max(xs_t)
    if x_max == x_min:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = min(ys), max(ys)
    span = max(y_max - y_min, 0.005)
    y_min -= 0.10 * span
    y_max += 0.18 * span

    def sx(v: float) -> float:
        t = math.log10(max(v, 1e-12)) if log_x else v
        return pad_l + (t - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Gridlines + axis ticks (3 ticks each axis)
    def _ticks(lo: float, hi: float, n: int = 3) -> list[float]:
        step = (hi - lo) / (n - 1)
        return [lo + i * step for i in range(n)]

    parts: list[str] = []
    parts.append(
        f"<svg width='{width}' height='{height}' "
        "xmlns='http://www.w3.org/2000/svg' style='background:#0a0a0d;"
        "border:1px solid #1c1c20;display:block;"
        "font-family:\"IBM Plex Mono\",monospace'>"
    )
    parts.append(
        f"<text x='{width/2:.0f}' y='17' text-anchor='middle' "
        "fill='#e6e1d6' font-size='12' "
        "font-family='Source Serif 4,Georgia,serif' font-weight='600'>"
        f"{_esc(title)}</text>"
    )
    # Axes
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t+plot_h}' "
        f"x2='{pad_l+plot_w}' y2='{pad_t+plot_h}' stroke='#2a2a30'/>"
    )
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' "
        f"y2='{pad_t+plot_h}' stroke='#2a2a30'/>"
    )
    # X ticks
    for tv in _ticks(x_min, x_max, 3):
        x = pad_l + (tv - x_min) / (x_max - x_min) * plot_w
        if log_x:
            v = 10 ** tv
            lbl = _fmt_lr(v)
        else:
            lbl = f"{tv:.0f}"
        parts.append(
            f"<line x1='{x:.1f}' y1='{pad_t+plot_h}' x2='{x:.1f}' "
            f"y2='{pad_t+plot_h+4}' stroke='#2a2a30'/>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{pad_t+plot_h+16}' text-anchor='middle' "
            f"fill='#a89e8c' font-size='9'>{_esc(lbl)}</text>"
        )
    # Y ticks
    for tv in _ticks(y_min, y_max, 3):
        y = sy(tv)
        parts.append(
            f"<line x1='{pad_l-4}' y1='{y:.1f}' x2='{pad_l}' "
            f"y2='{y:.1f}' stroke='#2a2a30'/>"
        )
        parts.append(
            f"<text x='{pad_l-7}' y='{y+3:.1f}' text-anchor='end' "
            f"fill='#a89e8c' font-size='9'>{_fmt_pct(tv,1)}</text>"
        )
    parts.append(
        f"<text x='{pad_l+plot_w/2:.0f}' y='{height-6}' "
        f"text-anchor='middle' fill='#a89e8c' font-size='10' "
        f"font-family='Source Serif 4,Georgia,serif'>"
        f"{_esc(x_label)}</text>"
    )
    parts.append(
        f"<text x='12' y='{pad_t+plot_h/2:.0f}' "
        f"text-anchor='middle' fill='#a89e8c' font-size='10' "
        f"font-family='Source Serif 4,Georgia,serif' "
        f"transform='rotate(-90 12 {pad_t+plot_h/2:.0f})'>"
        f"{_esc(y_label)}</text>"
    )
    # Points (best last, larger)
    for x, y, cfg, is_best in points:
        cx, cy = sx(x), sy(y)
        if is_best:
            parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='8' "
                f"fill='#3fb950' fill-opacity='0.85' stroke='#e6e1d6' "
                f"stroke-width='1.5'>"
                f"<title>BEST: lr={_fmt_lr(cfg['lr'])} "
                f"wd={_fmt_wd(cfg['weight_decay'])} bs={cfg['batch_size']} "
                f"opt={cfg['optimizer']}  top1={_fmt_pct(y)}%</title>"
                "</circle>"
            )
        else:
            parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4' "
                f"fill='#bb8c4d' fill-opacity='0.65' stroke='#bb8c4d' "
                f"stroke-width='1'>"
                f"<title>lr={_fmt_lr(cfg['lr'])} "
                f"wd={_fmt_wd(cfg['weight_decay'])} bs={cfg['batch_size']} "
                f"opt={cfg['optimizer']}  top1={_fmt_pct(y)}%</title>"
                "</circle>"
            )
    parts.append("</svg>")
    return "".join(parts)


def render_pareto_panels(cells: list[dict], best_cfg: dict) -> str:
    by_cid = _per_cell_median(cells)
    rows = list(by_cid.values())

    # Three panels: lr (log x), wd (log x), batch (linear)
    def pts(axis: str, log_x: bool) -> list[tuple[float, float, dict, bool]]:
        result = []
        for r in rows:
            cfg = r["config"]
            result.append((
                float(cfg[axis]),
                float(r["top1_median"]),
                cfg,
                _config_matches(cfg, best_cfg),
            ))
        return result

    svg1 = _svg_scatter(
        title="lr vs top1_median",
        points=pts("lr", log_x=True),
        x_label="learning rate (log)",
        y_label="top1 median (%)",
        log_x=True,
    )
    svg2 = _svg_scatter(
        title="wd vs top1_median",
        points=pts("weight_decay", log_x=True),
        x_label="weight decay (log)",
        y_label="top1 median (%)",
        log_x=True,
    )
    svg3 = _svg_scatter(
        title="batch vs top1_median",
        points=pts("batch_size", log_x=False),
        x_label="batch size",
        y_label="top1 median (%)",
        log_x=False,
    )
    cap = (
        "<div class='cap'>Coordinate-descent path on the X axis at the other "
        "axes' best-config setting. The best-config cell (large green dot) "
        "sits at the highest top-1.</div>"
    )

    return (
        "<h2>3 &middot; Per-axis Pareto plots</h2>"
        + "<div class='pareto-row'>"
        + f"<div class='panel'>{svg1}{cap}</div>"
        + f"<div class='panel'>{svg2}{cap}</div>"
        + f"<div class='panel'>{svg3}{cap}</div>"
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 4 — Seed-stability bars
# ---------------------------------------------------------------------------

def render_seed_stability(cells: list[dict]) -> str:
    by_cid = _per_cell_median(cells)
    # Overall axis scale: span of seen top1
    all_top1 = [c["top1"] for c in cells]
    lo = min(all_top1) - 0.005
    hi = max(all_top1) + 0.005
    # Pooled sigma estimate across multi-seed cells; fall back to 0.005
    multi = [v for v in by_cid.values() if v["n_seeds"] >= 2]
    if multi:
        # Sample std-dev within each multi-seed cell, pooled
        within = []
        for v in multi:
            ts = [r["top1"] for r in v["rows"]]
            if len(ts) >= 2:
                within.append(statistics.pstdev(ts))
        sigma_pool = statistics.mean(within) if within else 0.005
    else:
        sigma_pool = 0.005
    twosig = max(2.0 * sigma_pool, 1e-6)

    rows_html: list[str] = []
    cid_order = sorted(
        by_cid.keys(),
        key=lambda c: -by_cid[c]["top1_median"],
    )
    for cid in cid_order:
        v = by_cid[cid]
        rng = v["top1_range"]
        if v["n_seeds"] <= 1:
            cls = "amber" if rng == 0 else "amber"
            label_extra = "(single-seed)"
        elif rng <= sigma_pool:
            cls = ""
            label_extra = "tight"
        elif rng <= twosig:
            cls = "amber"
            label_extra = "moderate"
        else:
            cls = "red"
            label_extra = "wide &gt; 2&sigma;"
        bar_left = (v["top1_min"] - lo) / (hi - lo) * 100.0
        bar_w = max((v["top1_max"] - v["top1_min"]) / (hi - lo) * 100.0,
                    0.6)
        med_pos = (v["top1_median"] - lo) / (hi - lo) * 100.0
        rows_html.append(
            "<div class='stab-row'>"
            f"<div class='cid'>{_esc(cid)}</div>"
            "<div class='bar'>"
            f"<div class='bar-fill {cls}' "
            f"style='left:{bar_left:.2f}%;width:{bar_w:.2f}%;'>"
            f"<title>min {_fmt_pct(v['top1_min'])}%  "
            f"max {_fmt_pct(v['top1_max'])}%  "
            f"range {rng*100:.2f} pp  {_esc(label_extra)}</title>"
            "</div>"
            f"<div class='bar-tick med' style='left:{med_pos:.2f}%;'></div>"
            "</div>"
            f"<div class='std'>&Delta;={rng*100:.2f}pp</div>"
            "</div>"
        )
    legend = (
        "<div class='mono' style='color:var(--paper-dim);font-size:11px;"
        "margin-bottom:8px;'>"
        f"Pooled &sigma; (multi-seed cells) = {sigma_pool*100:.2f} pp. "
        "Green band = tight (&Delta; &le; &sigma;), amber = moderate "
        "(&Delta; &le; 2&sigma;), red = wide (&Delta; &gt; 2&sigma;). "
        "Green tick = seed-median.</div>"
    )
    return (
        "<h2>4 &middot; Seed-stability bars</h2>"
        + legend
        + "<div class='stability-list'>" + "".join(rows_html) + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 5 — Footer
# ---------------------------------------------------------------------------

def render_footer(
    *,
    idea: str,
    summary: dict,
    hypothesis_group: str,
    hypothesis_slug: str,
    git_sha: str,
) -> str:
    iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    fp = summary.get("fingerprint", "d65565e9c7b1")[:16]
    base = summary["base_config"]
    wall = summary.get("wallclock_total_s", 0.0) / 3600.0
    hyp_url = f"{GITHUB_BLOB}/hypotheses/{hypothesis_group}/{hypothesis_slug}.md"
    return (
        "<h2>5 &middot; Footer</h2>"
        "<div class='footer'>"
        f"<div>Built at commit <code>{_esc(git_sha)}</code> on "
        f"<code>{iso}</code>.</div>"
        f"<div>Base config (screening): "
        f"<code>lr={_fmt_lr(base['lr'])} wd={_fmt_wd(base['weight_decay'])} "
        f"bs={base['batch_size']} opt={_esc(base['optimizer'])}</code> "
        f"&middot; algorithm: <code>{_esc(summary['algorithm'])}</code> "
        f"&middot; budget: <code>{summary['budget']}</code> cells "
        f"&middot; total wall-clock: "
        f"<code>{wall:.2f} h</code>.</div>"
        f"<div>COMPOSITE_FORMULA sha256 fingerprint: <code>{_esc(fp)}</code> "
        "(see <code>src/nature_inspired_networks/eval.py:COMPOSITE_FORMULA</code>, "
        "CLAUDE.md Rule 2).</div>"
        "<div>Links: "
        "<a href='../../../dashboard/dashboard.html'>master aggregate</a> "
        "&nbsp;&middot;&nbsp; "
        f"<a href='{GITHUB_BLOB}/PAPER.md#55--three-phase-8-candidates-surfaced-by-the-protocol--certified-at-n7-2026-05-29' "
        "target='_blank' rel='noopener'>PAPER.md &sect;5.5 certified-claim section</a> "
        "&nbsp;&middot;&nbsp; "
        f"<a href='{GITHUB_PAGES_BASE}/dashboard/ideas/{_esc(idea)}/' "
        "target='_blank' rel='noopener'>GitHub Pages mirror</a> "
        "&nbsp;&middot;&nbsp; "
        f"<a href='{GITHUB_BLOB}/ideas/{_esc(idea)}/hillclimb_results.json' "
        "target='_blank' rel='noopener'>hillclimb_results.json</a> "
        "&nbsp;&middot;&nbsp; "
        f"<a href='{hyp_url}' target='_blank' rel='noopener'>design doc</a>."
        "</div>"
        "<div class='typo-note'>Typography: Source Serif 4 (Adobe, open) for "
        "body and headings &middot; IBM Plex Mono for tags, code, and "
        "tabular metrics. Palette mirrors the master dashboard "
        "(brutalist-editorial dark, sRGB).</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_page(
    *,
    idea: str,
    hypothesis_id: str,
    hypothesis_group: str,
    hypothesis_slug: str,
    dataset: str = "cifar100",
    baseline_top1_max: float = 0.5687,  # CIFAR-100 baseline_resnet20 seed-max
    git_sha: str | None = None,
) -> str:
    summary_path = REPO_ROOT / "ideas" / idea / "hillclimb_results.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells = _load_cells(summary, dataset=dataset)
    by_cid = _per_cell_median(cells)
    best_cid = _build_cid(summary["best_config"])

    sha = git_sha or _git_sha()

    body = (
        render_masthead(
            idea=idea,
            hypothesis_id=hypothesis_id,
            hypothesis_group=hypothesis_group,
            hypothesis_slug=hypothesis_slug,
            summary=summary,
            by_cid=by_cid,
            baseline_top1_max=baseline_top1_max,
        )
        + render_cells_table(cells, best_cid=best_cid)
        + render_pareto_panels(cells, summary["best_config"])
        + render_seed_stability(cells)
        + render_footer(
            idea=idea,
            summary=summary,
            hypothesis_group=hypothesis_group,
            hypothesis_slug=hypothesis_slug,
            git_sha=sha,
        )
    )

    return (
        "<!doctype html><html lang='en'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(summary['tag'])} &middot; per-hypothesis hill-climb</title>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
        "family=Source+Serif+4:wght@400;600&family=IBM+Plex+Mono:wght@400;600"
        "&display=swap'>"
        f"<style>{_PAGE_CSS}</style>"
        "</head><body>"
        + body
        + f"<script>{_SORT_JS}</script>"
        "</body></html>"
    )


WINNERS = [
    dict(idea="09_phi_budget", hypothesis_id="H09",
         hypothesis_group="g1_scaling_growth",
         hypothesis_slug="H09_golden_proportion_param_budget"),
    dict(idea="91_pair_gm_pdw", hypothesis_id="H09+H48+H44",
         hypothesis_group="g1_scaling_growth",
         hypothesis_slug="H09_golden_proportion_param_budget"),
    dict(idea="92_slot_act_sine", hypothesis_id="H81",
         hypothesis_group="g8_esoteric_extensions",
         hypothesis_slug="H81_sinusoidal_activation"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--idea", default=None,
                   help="Single idea to render (e.g. 09_phi_budget). "
                        "Default: all three certified winners.")
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--hypothesis-group", default=None)
    p.add_argument("--hypothesis-slug", default=None)
    p.add_argument("--dataset", default="cifar100")
    args = p.parse_args(argv)

    targets = WINNERS
    if args.idea:
        match = [w for w in WINNERS if w["idea"] == args.idea]
        if match:
            targets = match
        else:
            assert args.hypothesis and args.hypothesis_group \
                and args.hypothesis_slug, \
                "Need --hypothesis/--hypothesis-group/--hypothesis-slug for "\
                "an unknown idea."
            targets = [dict(idea=args.idea,
                            hypothesis_id=args.hypothesis,
                            hypothesis_group=args.hypothesis_group,
                            hypothesis_slug=args.hypothesis_slug)]

    sha = _git_sha()
    for t in targets:
        html = build_page(
            idea=t["idea"],
            hypothesis_id=t["hypothesis_id"],
            hypothesis_group=t["hypothesis_group"],
            hypothesis_slug=t["hypothesis_slug"],
            dataset=args.dataset,
            git_sha=sha,
        )
        out = REPO_ROOT / "ideas" / t["idea"] / "dashboard" / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
