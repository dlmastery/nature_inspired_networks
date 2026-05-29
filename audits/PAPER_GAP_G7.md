# Paper-gap audit — Group G7 (Cross-Paradigm Hybrids, H61–H75)

Auditor: paper-gap reviewer (Opus 4.7), 2026-05-29.
Scope: 15 cross-paradigm hybrid hypotheses. **None has a CIFAR
ablation row** — the G7 family is composition-of-primitives and the
sweep has not yet exercised any of these compositions. The most we
can say is whether the *components* have arXiv backing and whether
the *composition* has any precedent.

Classification per row (NO_DATA, IMPL_BUG, IMPL_BUG+COMPOSITION,
DOMAIN, NO_DATA+COMPOSITION_UNTESTED). Every row carries an explicit
"no CIFAR result to compare against" flag.

## Summary

| H   | classification              | components arXiv-backed?  | one-line gap                                                              |
|-----|-----------------------------|---------------------------|---------------------------------------------------------------------------|
| H61 | NO_DATA_COMPOSITION_UNTESTED| yes (CFC, JEPA, NaturePrior) | composition has no precedent; no CIFAR row                              |
| H62 | NO_DATA_COMPOSITION_UNTESTED| yes (toroidal pad, hex mask) | hex-on-attn-mask + torus has no joint paper; no CIFAR row              |
| H63 | NO_DATA_COMPOSITION_UNTESTED| yes (CKA distillation, Platonic) | CKA distillation untested as classifier auxiliary loss              |
| H64 | IMPL_BUG_NO_DATA            | yes (growth, prune)        | growth+prune callback never wired into runner (G7 audit MAJOR)            |
| H65 | NO_DATA_COMPOSITION_UNTESTED| partial (Betti surrogate)  | inherits H51 surrogate gap; no CIFAR row                                  |
| H66 | NO_DATA_COMPOSITION_UNTESTED| yes (Chladni init, MHA)    | cymatic-init MHA never measured against vanilla-init MHA                  |
| H67 | IMPL_BUG_COMPOSITION        | partial                    | 2 of 6 priors silently no-op (GoldenRoPE missing; MetatronGraph arity)    |
| H68 | NO_DATA_COMPOSITION_UNTESTED| yes (Fib widths, GoldenSkip)| world-model with JEPA loss; CIFAR row N/A (sequence task)                |
| H69 | NO_DATA_COMPOSITION_UNTESTED| yes (KAN, GCN)             | KAN-Metatron head; no CIFAR row                                           |
| H70 | NO_DATA_COMPOSITION_UNTESTED| yes (Chladni FFT loss)     | low-data curriculum; no CIFAR row                                         |
| H71 | NO_DATA_COMPOSITION_UNTESTED| yes (RoPE, icosa)          | 3-D RoPE on flat 2-D images is DOMAIN-soft                                |
| H72 | NO_DATA_COMPOSITION_UNTESTED| yes (vesica conv, φ-GELU)  | fractal vesica FFN; no CIFAR row                                          |
| H73 | NO_DATA_COMPOSITION_UNTESTED| yes (Vogel, Metatron)      | spiral PE + Metatron graph PE; no CIFAR row                               |
| H74 | IMPL_BUG_NO_DATA            | no (composition vacuous)   | 13 alphas collapse to scalar sum (G7 audit BROKEN)                        |
| H75 | NO_DATA_COMPOSITION_UNTESTED| yes (SwiGLU, cymatic init) | cymatic-init SwiGLU; no CIFAR row                                         |

Tier counts:
- NO_DATA_COMPOSITION_UNTESTED: **12** (H61, H62, H63, H65, H66, H68, H69, H70, H71, H72, H73, H75)
- IMPL_BUG candidates: **3** (H64, H67, H74) — all three already
  flagged by G7 audit as MAJOR / BROKEN / BROKEN.

---

## Per-hypothesis notes

### H61 — Sacred Liquid JEPA — NO_DATA_COMPOSITION_UNTESTED
Components: Liquid CFC (Hasani 2021 arXiv:2106.13898), JEPA (LeCun
2022), NaturePriorBlock. Composition (CFC + JEPA + Platonic conv) has
no published precedent; no CIFAR row.

### H62 — Toroidal-KV Hex Attention — NO_DATA_COMPOSITION_UNTESTED
Components: Hexagonal-CNN masking (Hoogeboom 2018 arXiv:1803.02108),
toroidal padding. Joint composition on attention masks lacks a
published precedent; no CIFAR row.

### H63 — Platonic Auxiliary Cymatic Teacher — NO_DATA_COMPOSITION_UNTESTED
Components: CKA distillation (Kornblith 2019 arXiv:1905.00414).
Cymatic teacher → student CKA loss is novel composition; no CIFAR row.

### H64 — Dynamic Growth + Pruning Cycle — IMPL_BUG_NO_DATA
Components: Fibonacci pruning (H32), Net2Wider growth (Chen et al.
2016 arXiv:1511.05641). **G7 audit MAJOR**: `GrowthPruningSchedule.step`
is never invoked by `scripts/run_sweep.py` or `runner.py`. The hybrid
exists as a callback but is not wired into training. Any CIFAR row
would reduce to a vanilla ResNet baseline. No row currently exists,
so the bug is latent. Counts as IMPL_BUG.

### H65 — Persistent-Homology Betti-Collapse Loss — NO_DATA_COMPOSITION_UNTESTED
Inherits H51's surrogate gap (the underlying β₀ estimator is the
smooth gap-based surrogate, not real PH). Composition is novel; no
CIFAR row.

### H66 — Cymatic QKV Kernel — NO_DATA_COMPOSITION_UNTESTED
Components: cymatic_init_ (project-local) + standard MHA. Cymatic-init
on attention QKV is novel; no CIFAR row. (The slot_init_cymatic CIFAR
row at top1 0.8540 exercises cymatic init on **conv stems**, not on
attention QKV — that result does not transfer to H66's claim.)

### H67 — Full Paradigm Hybrid — IMPL_BUG_COMPOSITION
**G7 audit BROKEN.** Two of six advertised priors silently no-op:
- `GoldenRoPE` class does not exist (file exports functions, not a
  class) → falls back to standard sinusoidal PE.
- `MetatronGraphLayer(width)` called with single arg; constructor needs
  `(in_dim, out_dim)` → TypeError, caught by `except Exception:` and
  replaced with `nn.Identity()`.
- `which_priors_active` hardcodes 4 flags to True without inspection,
  rendering the test_each_load_bearing_prior_is_engaged a tautology.
- Liquid CFC runs one step from zero state per image, collapsing
  recurrence to an affine + nonlinearity.

Counts as 1 IMPL_BUG (composition-level). No CIFAR row.

### H68 — On-Device World Model — NO_DATA_COMPOSITION_UNTESTED
Components: Fibonacci channels, GoldenSkip (1/φ), GRU JEPA predictor.
World-model task, not CIFAR classification — N/A for image-classification
sweep. NO_DATA.

### H69 — KAN-Metatron Symbolic Head — NO_DATA_COMPOSITION_UNTESTED
Components: KAN (Liu et al. 2024 arXiv:2404.19756) + Metatron GCN
(project-local). Composition novel; no CIFAR row.

### H70 — Cymatic Low-Data Curriculum — NO_DATA_COMPOSITION_UNTESTED
Components: cymatic loss (spectral-bias literature), CE. Curriculum
schedule novel; no CIFAR row.

### H71 — Icosahedral 3-D RoPE — NO_DATA_COMPOSITION_UNTESTED
Components: RoPE (Su et al. 2021 arXiv:2104.09864) + icosahedral
geometry. Rodrigues rotation per-triple is correctly implemented (G7
audit PASS). But CIFAR-10 patches don't have a natural 3-D sequence
position to rotate; the prior is DOMAIN-soft for flat 2-D image
classification. No CIFAR row.

### H72 — Fractal Vesica FFN — NO_DATA_COMPOSITION_UNTESTED
Components: VesicaPiscisConv2d + φ-GELU. Three parallel paths with
radii (b, b/φ, b/φ²). Novel composition; no CIFAR row.

### H73 — Golden Spiral × Metatron PE — NO_DATA_COMPOSITION_UNTESTED
Components: Vogel spiral (golden-angle), Metatron GCN. Concatenated
PE for transformers; no CIFAR row.

### H74 — Metatron Overlap Tying — IMPL_BUG_NO_DATA
**G7 audit BROKEN.** Forward computes `eff_w = W · sum(α_c)` where
the 13 alphas appear only through their **sum**: a single scalar
gate. The "13-circle overlap" composition is reparameterisation-
redundant. Counts as 1 IMPL_BUG. No CIFAR row (so the bug is latent).

### H75 — Harmonic Cymatic SwiGLU — NO_DATA_COMPOSITION_UNTESTED
Components: SwiGLU (Shazeer 2020 arXiv:2002.05202), cymatic init,
PhiGELU. Composition novel; no CIFAR row.

---

## Group-level conclusion

G7 is the group where the **paper gap is dominated by NO_DATA**: 12
of 15 hypotheses have correct (or near-correct) implementations of
well-cited components, but the *composition* has never been
empirically measured against the cited baselines because no sweep
row exercises any G7 hybrid. The 3 IMPL_BUG candidates (H64 H67 H74)
were already caught by the G7 audit at MAJOR/BROKEN tier and are
fixable; none has shipped a CIFAR claim.

**Recommendation.** Before claiming any G7 hybrid as a win, run at
least a 12-epoch CIFAR-10 smoke per hypothesis (Phase 2 of Rule 19),
gated by Phase-1 SOTA smoke. Until that happens, no G7 row can be
classified as PAPER_AGREES or PAPER_DISAGREES — only NO_DATA.
