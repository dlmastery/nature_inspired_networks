---
name: autoresearch-dashboard-comprehension
description: Use when designing or revising any dashboard / paper figure / per-experiment visual. Enforces small-multiples over dense overlaid charts, the mandatory "how to read this dashboard" 4-bullet orientation block, the seed-count + tier (SCREENING / EVALUATION) badge on every numeric, multi-hypothesis pill display for combo/pair/hybrid tags, and the no-self-grading-banner discipline.
metadata:
  rules_enforced: [28, 33, 34, 37]
  added: 2026-05-29
  origin_audit: audits/REVIEWER_PASS_DASHBOARD.md
---

# Skill — Dashboard / figure comprehension discipline

## When to use

- Designing any new chart for the dashboard, paper, or README.
- Revising an existing chart that has been flagged as dense /
  illegible / mis-framed.
- Before publishing any dashboard surface (aggregate or per-
  experiment) — verify all 4 pillars below.

## Pillar 1 — Small-multiples over dense overlaid charts (Rule 33)

A chart with 3+ overlaid axes / lines / variants is FORBIDDEN
when the same information fits in 3 side-by-side small-multiples.

### Convert dense → small-multiples

| dense (forbidden) | small-multiples (required) |
|---|---|
| One ablation plot with 20 bars for 20 tags, all groups mixed | One mini-bar-chart per hypothesis group, 8 panels (G1..G8) in a 4×2 grid |
| One training-curve plot with 20 lines | One panel per group, ≤ 5 lines each, shared y-axis |
| One Pareto plot with all variants | One Pareto per group, baseline drawn in each panel for reference |
| One leaderboard table with 50 rows | Group-sectioned tables, ≤ 8 rows each, one per group |

The aggregate dashboard `dashboard/dashboard.html` is already
group-sectioned per Rule 24 — extend the same discipline to
figures.

### Caption contract

Every chart carries a 1-sentence "what to read" caption directly
under it. Example:

> Pareto: top-1 accuracy vs parameter count on CIFAR-10. Lower-
> right is dominated; stars are baselines; circles are
> nature-inspired variants. **What to read: any circle below-and-
> right of all stars is a dominated variant (worse on both axes).**

If the chart needs a paragraph to explain, it's the wrong chart.

## Pillar 2 — "How to read this dashboard" orientation block (Rule 33)

Every dashboard surface opens with EXACTLY 4 bullets at the top of
the page (above the headline ribbon):

```html
<aside class="how-to-read">
  <h3>How to read this dashboard</h3>
  <ul>
    <li><strong>What this page shows:</strong>
        every CIFAR-10/100 sweep row plus its hypothesis context.</li>
    <li><strong>How to interpret colour coding:</strong>
        green = PASS verdict, yellow = MINOR, orange = MAJOR,
        red = BROKEN / NUMEROLOGY / FALSIFIED.</li>
    <li><strong>Screening vs evaluation (Rule 28):</strong>
        rows tagged <code>SCREENING</code> are 1-seed candidates;
        <code>EVALUATION</code> rows are 3-seed final numbers.
        Only EVALUATION rows clear the external-claim gate.</li>
    <li><strong>Drill-down:</strong>
        click any row to navigate to its independent
        per-experiment page (NO modals).</li>
  </ul>
</aside>
```

The 4 bullets are non-negotiable; do not bloat to 6, do not
shrink to 2.

## Pillar 3 — Seed-count + tier badge on every numeric (Rule 34)

Every numeric on every visual surface carries the qualifier.
Recipes:

### KN-strip (per-experiment page header)

```html
<div class="kn-strip">
  <span class="delta">+1.34 pp</span>
  <span class="vs">Δ vs baseline</span>
  <span class="seed-badge n3">n=3</span>
  <span class="tier-chip evaluation">EVALUATION</span>
</div>
```

### Aggregate leaderboard table

Add two columns to every leaderboard:

| tag | composite | top-1 | params | latency | **n** | **tier** |
|---|---:|---:|---:|---:|---:|---|
| pair_gm_pdw | 0.8612 | +1.34 pp | 0.28M | 1.4ms | 3 | EVALUATION |
| sg_only_phi_budget | 0.5741 | +0.25 pp | 0.27M | 1.4ms | 3 | EVALUATION |
| sg_only_golden_adam | 0.5196 | -33.0 pp | 0.27M | 1.4ms | 1 | SCREENING |

### Headline ribbon

```
PHASE-8 winners (n=3 seeds, EVALUATION gate, Holm-Bonferroni α'=0.0167):
  pair_gm_pdw +1.34 pp, slot_act_sine +1.04 pp, sg_only_phi_budget +0.25 pp.
PHASE-8 negatives (n=1 seed, SCREENING tier; not falsifications):
  H41 β-only -1 pp, H22 toroidal UNTESTED_ON_RIGHT_DATASET (see Rule 36).
```

CSS palette for the chips (consistent with academic typography
per Rule 30):

```css
.seed-badge { background: #eef; color: #224; padding: 2px 6px; border-radius: 3px; }
.tier-chip.screening  { background: #fff4e5; color: #8a5800; }
.tier-chip.evaluation { background: #e8f4e8; color: #1f5320; }
```

## Pillar 4 — Multi-hypothesis pills + no-self-grading banners (Rule 33, 37)

### Combo / pair / hybrid pill display

A tag that stacks multiple hypotheses MUST display ALL of them:

```html
<!-- WRONG -->
<div class="h-pill">H09 phi_budget</div>

<!-- RIGHT (combo2_pb_gm = H09 + H48) -->
<div class="h-pills">
  <span class="h-pill">H09 phi_budget</span>
  <span class="plus">+</span>
  <span class="h-pill">H48 golden_momentum</span>
</div>

<!-- RIGHT (pair_gm_pdw = H09 + H48 + H44) -->
<div class="h-pills">
  <span class="h-pill">H09 phi_budget</span>
  <span class="plus">+</span>
  <span class="h-pill">H48 golden_momentum</span>
  <span class="plus">+</span>
  <span class="h-pill">H44 phi_decay_wd</span>
</div>
```

Tag-to-hypothesis mapping lives in `scripts/_tag_hypothesis_map.py`
and is the single source of truth.

### No self-grading banners (Rule 37)

The dashboard / per-experiment page / README banner MUST NOT carry
a self-graded verdict ("ACCEPT" / "FINAL" / "Reviewer-acceptance")
without the explicit qualifier. Pattern:

```html
<!-- FORBIDDEN -->
<div class="banner ok">
  Reviewer-acceptance ACCEPT verdict at commit 0343f35.
</div>

<!-- REQUIRED -->
<div class="banner internal-qa">
  <strong>Internal QA pass</strong>
  (verdict: ACCEPT at commit 0343f35 by same-family critic agent).
  <em>Independent external review pending — see
  <a href="https://github.com/dlmastery/.../blob/main/audits/REVIEWER_PASS_PAPER.md">
    audits/REVIEWER_PASS_PAPER.md</a> (WEAK_REJECT) for the external pass.</em>
</div>
```

When an external reviewer's verdict downgrades an internal verdict,
the dashboard surface MUST reflect the downgrade within the same
commit that processes the audit. Do not let an outdated "ACCEPT"
banner linger after WEAK_REJECT is on file.

## Anti-patterns

- **3 axes on one chart with a 3-paragraph caption** — split it.
- **"How to read" block of 8 bullets** — pick 4, the most load-
  bearing.
- **A leaderboard cell `+1.34 pp` with no `n=X` next to it** —
  Rule-34 violation. Every numeric, every time.
- **Combo tag pill showing only the lead hypothesis** — Rule-33
  violation. Show ALL participating hypotheses.
- **Headline ribbon showing the positive winners but not the
  calibrated negatives** — equal prominence (per Rule 32 spirit
  applied to the dashboard).
- **"ACCEPT" banner without the "Internal QA pass — external
  review pending" qualifier** — Rule-37 violation.

## Cross-references

- CLAUDE.md Rules 28 (screening-vs-evaluation), 33 (small-multiples
  + orientation block), 34 (seed-count badges), 37 (no self-
  grading banners).
- `audits/REVIEWER_PASS_DASHBOARD.md` — origin findings for every
  pillar above.
- `skills/autoresearch-dashboard/` — augmented to call this skill
  out as a prerequisite for any chart.
- `skills/autoresearch-per-experiment-page/` — augmented to embed
  the KN-strip badge pattern in the template.
- `skills/autoresearch-paper-rigor/` — sibling for the statistical
  arm of the same Rule-28 framing.
- `skills/autoresearch-typography-and-rendering/` — sibling for
  the visual palette of the chips and pills.
