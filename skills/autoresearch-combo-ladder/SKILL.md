---
name: autoresearch-combo-ladder
description: Use when designing multi-prior hybrid experiments. Stack ONLY priors that touch orthogonal (non-competing) layers of the training stack. Build an additive 2→N-prior ladder where each next row adds exactly ONE new orthogonal prior — this lets you read off the marginal effect of each axis. The "full hybrid all-on" alternative is forbidden (CLAUDE.md Rule 23).
---

# Skill — Orthogonal-axis additive combo ladder

## When to use

- The user asks about "stacked experiments", "combo hybrids", "5 priors
  together", or whether priors compound.
- After a single-prior sweep has identified a winner (e.g.,
  `phi_budget`) and you want to test if other priors stack on it.
- NEVER stack 3+ priors on the SAME forward path (Rule 23). The
  cautionary tale: `sg_full_fib` stacked 6 priors on the same conv
  block → −11.54 pp from baseline. Anti-compounding is the default.

## The orthogonal axes (this repo's example)

A "layer of the training stack" is a logically independent slot in
the runner. Stacking one prior per slot is non-competing by construction.

| axis | example prior | touches |
|---|---|---|
| Architecture | `phi_budget` / `fib_depth` / `golden_bottleneck` | model class |
| Channel schedule | `fib` / `phi` / `linear` | per-stage widths |
| Momentum schedule | `golden_momentum` | trainer callback per epoch |
| Regulariser | `phi_dropout` | model-level inject before fc |
| Weight decay | `phi_decay_wd` | per-layer wd in optimizer |
| LR schedule | `phi_lr` | scheduler |
| Activation | `sine_act` / `phi_activation` | ReLU → ... walk (mutually exclusive) |
| Init | `golden_spiral_init` / `phi_init` | walk Conv2d/Linear (mutually exclusive) |
| Pruning | `fib_prune` | trainer callback at Fib epochs |
| Inference ensemble | `fib_ensemble` | post-training wrapper |

Mutually-exclusive slots (init, activation) take ONLY ONE prior.

## The ladder

```
N=2: base + 1 orthogonal prior
N=3: base + 2 orthogonal priors
...
N=K: base + (K-1) priors, one per axis
```

Each row adds exactly ONE new axis on top of N-1. Read off `top1(N) -
top1(N-1)` to get the marginal contribution of the newly-added axis.

## Worked example (this repo, May 2026)

Base: `phi_budget` (the only verified single-prior positive at the
campaign cutoff).

```
combo2_pb_gm                          phi_budget + golden_momentum
combo3_pb_gm_pd                       + phi_dropout
combo4_pb_gm_pd_pdw                   + phi_decay_wd
combo5_pb_gm_pd_pdw_plr               + phi_lr
combo6_pb_gm_pd_pdw_plr_fe            + fib_ensemble
combo7_pb_gm_pd_pdw_plr_fe_sa         + sine_act
combo8_pb_gm_pd_pdw_plr_fe_sa_fp      + fib_prune
```

7 rows, ~56 min wall-clock at 12-ep CIFAR-10 on a 4090 Laptop.

## Sweep-row implementation

In `scripts/run_sweep.py:build_matrix`, append rows whose `overrides`
dict spreads a shared base then ADDS one new override key per ladder
step. Use `**dict_shorthand` so the additive structure is visible:

```python
PB = dict(model="phi_budget", phi_model="phi_budget",
          phi_budget_total=270_000, phi_budget_n_stages=3,
          phi_budget_mode="phi")

rows.append(dict(tag="combo2_pb_gm",
                 overrides=dict(**PB, momentum_schedule="golden")))
rows.append(dict(tag="combo3_pb_gm_pd",
                 overrides=dict(**PB, momentum_schedule="golden",
                                dropout="phi_dropout",
                                dropout_cycle="fib", dropout_length=5)))
# ... each next row spreads the previous and adds ONE new key
```

## Selection rules — what NOT to stack

- **Falsified priors** — exclude from combos (e.g., `golden_adam` at
  51.96%; the falsification is real).
- **Model-class alternatives** — you have ONE base architecture; can't
  combine `phi_budget` and `fib_depth` (both are entire models).
- **Mutually-exclusive slot peers** — can't include both
  `phi_activation` and `sine_act` (both replace ReLU). Pick one.
- **Priors flagged BROKEN by the critic** — fix first, then include.

## Output interpretation

- Strict additivity: top1(combo_N) ≈ top1(base) + Σ marginal_i.
- Sub-additivity (combo numbers drift downward): priors interact via
  shared optimization dynamics; ladder reveals which axis is the
  culprit (look at the step where the curve breaks).
- Super-additivity: rare and exciting. Suggests the priors are
  genuinely complementary.

## Anti-patterns

- "Let's just turn everything on and see" → that's `sg_full_fib` /
  `H67 hybrid_full`; both are anti-patterns by the project's own data.
- Two-prior compounds without the ladder context — you can't tell if
  the (positive/negative) effect generalises across other priors.
- Including priors known to be NUMEROLOGY by the sci-critic — wastes
  GPU.

## Cross-references

- CLAUDE.md Rule 23 — orthogonal-axes-only multi-prior compounds.
- `autoresearch-ablation-sweep` — single-prior sweep is the
  prerequisite identifying which priors are stackable.
- `H50 full_fib`, `H67 hybrid_full` — the cautionary "everything on"
  examples.
