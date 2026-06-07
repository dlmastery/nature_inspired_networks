# SYNTHESIS — 100 Improvements to Make the Nature-Inspired DL Program Publishable at Top-Tier

**Date:** 2026-06-06
**Source:** five hostile reviewer critiques (R1 ICLR / R2 ICML / R3 NeurIPS / R4 elite researcher / R5 big-tech lab lead)
**All five returned REJECT** — overlapping fatal flaws below.
**Compute envelope:** single laptop RTX 4090, 16 GB VRAM, Windows 11. Total roadmap ~250 GPU-h over 10–12 weeks.

---

## 0 · North star — explicit, non-negotiable

The user's stated goal is **state-of-art nature-inspired methods that move deep-learning forward.**
This synthesis honours that goal. The 100 improvements below are the path to making the
nature-inspired program survive a hostile reviewer at top-tier scale.

**What this synthesis rejects** (explicit guardrail):

- R4's "drop the priors, publish the protocol alone" pivot. The Fixer-mechanism-test
  contract is a real secondary contribution — a methods chapter / appendix, *not*
  the headline. The headline stays nature-inspired priors.
- "Submit to a workshop" as the primary advice. Workshop is a fallback, not the plan.
- Treating the 5 REJECTs as cause to scale down. They are cause to scale **up**:
  iso-FLOPs comparisons, modern architectures, ImageNet-scale, real equivariance,
  honest statistics.

**What "moving the field" actually means for nature-inspired DL** (this synthesis's working definition):

A **single survivor** prior, evaluated at iso-FLOPs against a modern baseline
(RegNetX / ConvNeXt-V2 / ViT-Small), on a benchmark where the symmetry it embodies
is **load-bearing** (rotation-equivariance task for icosahedral; hex-symmetric
texture / wraparound for hex-lattice; non-axis-aligned imagery for golden-angle),
showing a **statistically certified +3 pp or more** at n≥7 over the strongest
literature baseline at that scale. That is the publishable nature-inspired result.

CIFAR-100 ResNet-20 at +1 pp non-iso-FLOPs at n=3 is not it. **We have to climb the
benchmark ladder.** R5's SOTA-migration path is how (see §10 below) — but with
nature-inspired priors as the load, not as the case study.

---

## 1 · Top-10 BLOCKERs — the response to "AI slop"

Every external reviewer hits these in the first 20 minutes. Each must be closed
before any external resubmission.

| # | BLOCKER | Source | Effort | Section |
|---|---|---|---|---|
| B1 | **Phase-9i priors run at 2× baseline FLOPs.** `experiments_modern/cifar100/{pair_gm_pdw,slot_act_sine,sg_only_phi_budget}_seed0/metrics.json` shows `flops=80,816,212` vs baseline `41,224,448`. Composite metric penalises params+latency, **not** FLOPs. Every external "lift" claim is a compute artifact wearing a prior costume. | R5 BLOCKER 1 | XL | §2 |
| B2 | **Modern-recipe baseline 0.6360 below project's own pre-registered floor 0.68.** `convergence/PLAN.md` line 117 says "STOP and debug if median < 0.68." Wave-C2 launched anyway; Phase-9i claim built on a sub-converged baseline. | R5 BLOCKER 2 | L | §2 |
| B3 | **n=3 paired Wilcoxon at theoretical floor p=0.125** is not a statistical test — it is a sign test in a frock. Paper's own `STATISTICAL_TESTS.md §3` derived n≥7 minimum. Phase-9i ships n=3 and calls it "qualitatively binding." | All 5 | M (writing) / XL (n=7 cell) | §4 |
| B4 | **No iso-FLOPs comparator at modern architectures.** RegNetX-200MF is the literature anchor for H09 (the paper's own §1.1 admits "DERIVATIVE+TESTABLE") and remains UNLAUNCHED 6 weeks after being flagged. No ConvNeXt-V2-Femto, no ViT-Small. | All 5 | L | §3 |
| B5 | **CIFAR-only / ResNet-20-only.** Imagenette (1 h/run), Tiny-ImageNet (4 h/run), ImageNet-100 FFCV (13 h/run) are all laptop-feasible. None are run. | All 5 | XL (Wave 0–4) | §3 |
| B6 | **Audit-calibration is Claude-grades-Claude.** Same model family wrote the code, audited it, graded the calibration substrate (`pytorch/vision`, `timm` — in-distribution for Claude). The Fisher p=1.94×10⁻⁵ measures auditor incentive, not protocol validity. $50 of GPT-5/Gemini-3 API closes this. | All 5 | S ($50, 1 day) | §4 |
| B7 | **`slot_act_sine` is SIREN (Sitzmann 2020 arXiv:2006.09661).** Control 2 (§13.2) shows `slot_act_tanh` beats it by +0.48 pp. Yet it sits in the abstract triple as a "nature-inspired winner." | All 5 | S | §6 |
| B8 | **`pair_gm_pdw` is a 3-axis regularizer stack** of which Control 1 shows **~61% is non-φ-attributable**. The φ-residual is +0.61 pp at the n=3 Wilcoxon floor. Abstract claims +1.74 pp φ-prior. | All 5 | S | §6 |
| B9 | **Bello 2021, Wightman 2021, Zheng 2023, Saunders 2022, Madaan 2023, Cohen-Welling 2016, e2cnn (Weiler 2019), e3nn (Geiger 2022) NOT CITED** in PAPER.md References despite being direct prior art for every part of the contribution. | R1, R3 | S | §7 |
| B10 | **No equivariance.** `src/nature_inspired_networks/icosa.py` is a one-shot rotation-pool, not steerable group convolution. H55 PlatonicAttention head bias is provably zero (vertex-transitive centroid identity). Drop "equivariance" terminology or implement it properly via e3nn. | R1, R3 | XL (e3nn) / S (rename) | §6 |

**These ten close the door on the "AI slop" critique. Everything else is the path
to making the paper good, not just defensible.**

---

## 2 · Block A — Iso-FLOPs honesty + recipe debug (P0)

**Goal:** make every reported "lift" iso-compute and re-baseline the modern recipe
so it lands inside the project's own pre-registered band.

### A1 — Iso-FLOPs re-test of the three Phase-9i priors
*What's wrong:* current priors at ~80.8 M FLOPs vs baseline 41.2 M (factor 1.96).
The composite metric (`top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)`) does
not penalise FLOPs.
*What to do:* pin each prior to the baseline 41.2 M FLOP envelope (e.g., reduce
`phi_budget_total` until measured FLOPs match within ±5%). Re-run at n=3 modern
recipe 200 ep. If the lift survives, real result. If it collapses, the +1 pp was
a depth/compute lift.
*Severity:* BLOCKER · *Effort:* L (~30 GPU-h) · *Priority:* P0

### A2 — Add FLOPs to the composite metric AND report it separately
*What's wrong:* composite is currently `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)`.
The Phase-9i confound proves params+latency are insufficient.
*What to do:* extend to `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms) − 0.05·log10(flops_M)`
and re-fingerprint. Update SHA-256 fingerprint. **Always report raw top1, params,
FLOPs, latency separately** in every leaderboard cell (R1 #12 + R4 #10).
*Severity:* BLOCKER · *Effort:* M · *Priority:* P0

### A3 — Add `flops_target` field to YAML config + runner refusal
*What's wrong:* the runner has no FLOP-budget sanity check at config-load time —
which is why B1 went undetected for 4 days.
*What to do:* every YAML carries `flops_target` (default = baseline FLOPs ±10%);
`runner.py` measures FLOPs via `fvcore` at model-build time and **refuses to launch**
if outside band. This would have caught B1.
*Severity:* BLOCKER · *Effort:* M · *Priority:* P0

### A4 — Debug the modern-recipe baseline (0.6360 → ≥0.68)
*What's wrong:* current modern recipe is the ImageNet-tuned 11-trick from Bello/Wightman.
RandAugment(N=2, M=14) is the ImageNet setting; Random Erasing has near-zero effect at
32×32 (Müller & Hutter 2021 NeurIPS).
*What to do:* adopt the He-2019 / "Deep Learning Tuning Playbook" CIFAR-100-specific
recipe: RandAugment(N=1, M=9), drop Random Erasing, Mixup α=0.1 (not 0.2), EMA 0.9999,
warmup 10 ep, 200 ep, LR=5e-4. Re-baseline at n=3 BEFORE any prior claim.
*Severity:* BLOCKER · *Effort:* L (~12 GPU-h) · *Priority:* P0

### A5 — Commit the prior-overlay YAMLs that don't exist
*What's wrong:* `configs/cifar100_modern_200ep.yaml` is committed as baseline; the
prior overlays (`pair_gm_pdw_modern_200ep`, `slot_act_sine_modern_200ep`,
`sg_only_phi_budget_modern_200ep`) **do not exist as YAML files** despite the
runs being checked-in. Cold-clone reproducibility is broken (R5).
*What to do:* commit `configs/cifar100_modern_200ep_{pair_gm_pdw,slot_act_sine,sg_only_phi_budget}.yaml`
with explicit overlay deltas. Hash-link each `metrics.json` to its source YAML.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### A6 — Add iso-FLOPs column to every leaderboard
*What's wrong:* dashboards rank by composite; reviewers cannot see the FLOP gap at
a glance.
*What to do:* add `flops_M`, `flops_target`, `flops_within_band` columns to
`dashboard/dashboard.html`, every per-experiment page, and every paper figure.
Reject any "winner" not iso-FLOPs ±5%.
*Severity:* MAJOR · *Effort:* M · *Priority:* P0

### A7 — Stop the Phase-9 cascade
*What's wrong:* Phase-9a → 9f → 9g → 9h → 9i → 9j is post-hoc analysis-branch
multiplication. Every reviewer reads this as "garden of forking paths animated."
*What to do:* lock the current state. No new phases until §3 Wave-0/1/2/3 lands.
Any future re-analysis is supplementary, not a new headline.
*Severity:* MAJOR · *Effort:* 0 (discipline) · *Priority:* P0

### A8 — Honest renaming of Phase-9i in PAPER.md §5.3
*What's wrong:* "Phase-9i convergence-regime corrective binding" reads as victory
language for what is empirically a non-iso-FLOPs n=3 sign test.
*What to do:* rename to "iso-recipe n=3 diagnostic at non-matched FLOPs (provisional)."
Drop "binding" and "corrective" everywhere these appear (PAPER.md, FINDINGS.md,
README.md, REVIEWER_CHECKLIST.md — currently 12+ instances).
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### A9 — Pre-register the Wave-2 plan BEFORE running it
*What's wrong:* every prior phase has been post-hoc explained. Reviewers will not
trust another phase without a pre-commit hash.
*What to do:* write `pre-registration/wave2_iso_flops_tiny_imagenet.md` with exact
recipe, seeds, decision rule, analysis plan. Commit. Cite that commit hash in
PAPER.md §5 before launching seeds.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### A10 — Drop "EVALUATION-tier" / "formally certified" language until §10 lands
*What's wrong:* README §4.3.1, STATISTICAL_TESTS §0, REVIEWER_CHECKLIST claim
"first formally-certified empirical claims" — but the Phase-9i n=3 is at the
Wilcoxon floor, no iso-FLOPs control, and a baseline below the pre-reg floor.
*What to do:* replace everywhere with "screened candidates pending iso-FLOPs n≥7
confirmation at the modern recipe + RegNetX comparator."
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

---

## 3 · Block B — Nature-inspired claim, executed at top-tier scale (P0/P1)

**Goal:** climb the benchmark ladder to where a nature-inspired result is taken
seriously. This is R5's SOTA-migration path repurposed as the nature-inspired
proving ground.

### B1 — Wave-0: validate the recipe on Imagenette before any prior claim
*What's wrong:* current modern recipe is ImageNet-tuned, fails on CIFAR. Without a
working recipe, nothing else is interpretable.
*What to do:* run baseline ResNet-20-style scaled to 160×160 on Imagenette
(10 classes, 13k images) at n=5 seeds × 10 ep × 3 recipes (legacy / modern-naive /
modern-CIFAR-tuned). Goal: identify the recipe landing in the published Imagenette
small-model band (≥90%).
*Severity:* BLOCKER · *Effort:* M (~5 GPU-h) · *Priority:* P0

### B2 — Wave-1: iso-FLOPs Pareto frontier on Imagenette
*What's wrong:* the paper has zero modern-architecture comparators.
*What to do:* 4 models at iso-FLOPs (~1.0 G at 160²): ResNet-20-Imagenette,
**RegNetX-200MF** (literature anchor for H09), ConvNeXt-V2-Femto, ViT-Small/16.
n=5 seeds, 50 ep, modern-CIFAR-tuned recipe. **This is the table that lets a
reviewer judge whether nature-inspired priors Pareto-dominate the literature.**
*Severity:* BLOCKER · *Effort:* L (~50 GPU-h) · *Priority:* P0

### B3 — Wave-2: Tiny-ImageNet 200-class at iso-FLOPs with 3 priors + baseline
*What's wrong:* CIFAR-100 100 classes is the noise-floor zone for +1 pp claims.
Tiny-ImageNet 200 classes is the sweet spot for a single-4090 study.
*What to do:* baseline + 3 priors (`pair_gm_pdw`, `slot_act_sine`, `sg_only_phi_budget`,
all iso-FLOPs) on Tiny-ImageNet at n=5 seeds × 80 ep. Paired Wilcoxon n=5 + bootstrap
CI + Holm-Bonferroni k=3. **If priors clear at iso-FLOPs on 200-class, the paper has
a real result.** If they collapse, the CIFAR result is a dataset-specific lift.
*Severity:* BLOCKER · *Effort:* L (~80 GPU-h) · *Priority:* P0

### B4 — Wave-3: ImageNet-100 at 160² with FFCV (the headline experiment)
*What's wrong:* zero ImageNet evidence. Every nature-inspired DL paper at top-tier
needs an ImageNet number.
*What to do:* take the Wave-2 winning prior + ResNet-50 baseline, FFCV-package
ImageNet-100, run n=3 seeds × 100 ep at 160² on the 4090 Laptop. FFCV makes
~13 h/seed feasible (Mamba codebase has clean FFCV pipeline). **This single
experiment turns the paper from "12-ep CIFAR ablation" into actual ML.**
*Severity:* BLOCKER · *Effort:* XL (~80 GPU-h) · *Priority:* P0

### B5 — Wave-4: H71 IcosaRoPE3D on Spherical MNIST (the strongest nature-inspired claim)
*What's wrong:* H71 is the sole NOVEL+TESTABLE sci-critic survivor. It is *untested*.
Rotation-equivariance on Spherical MNIST is where icosahedral priors *should*
demonstrate clear advantage if the prior is real.
*What to do:* Spherical MNIST 60×60 (e3nn standard), ViT-Tiny patch-12, rotated
test set, n=5 seeds × 100 ep. Compare to non-equivariant ViT-Tiny baseline AND
to a published e3nn rotation-equivariant ViT. **If H71 lifts non-equivariant ViT
by +3 pp on the rotated test set, the paper has a real "the prior helps where
the symmetry exists" result.**
*Severity:* BLOCKER · *Effort:* L (~15 GPU-h) · *Priority:* P0

### B6 — Hex-lattice priors on a hex-symmetric task
*What's wrong:* HexConv (H10 etc.) tested only on upright CIFAR — where there is
no hex symmetry. Hoogeboom 2018 HexaConv ICLR was demonstrated on aerial imagery
and tiled textures.
*What to do:* run HexConv vs square-conv at iso-FLOPs on **AID** (aerial image
dataset, 30 classes, 600×600, downsample to 128×128) at n=3 seeds × 50 ep.
~12 GPU-h. **If hex priors don't lift on aerial imagery, they don't lift anywhere.**
*Severity:* MAJOR · *Effort:* L (~12 GPU-h) · *Priority:* P1

### B7 — Toroidal closure on wrap-aware tiled-texture dataset (H22 falsifier)
*What's wrong:* H22 was tested on upright CIFAR and verdict "NUMEROLOGY/UNFALSIFIABLE."
The pre-registered falsifier was a wrap-aware dataset. Per Rule 36 the verdict
should be `UNTESTED_ON_RIGHT_DATASET`.
*What to do:* run H22 on tiled-CIFAR-10 (each image is a 2×2 tile of CIFAR images
with wraparound) at n=3 seeds × 50 ep. ~6 GPU-h. **This is the H22 honest test.**
*Severity:* MAJOR · *Effort:* M (~6 GPU-h) · *Priority:* P1

### B8 — Golden-angle modulation on non-axis-aligned imagery
*What's wrong:* golden-angle priors tested on upright CIFAR where axis alignment
is artificial. Golden angle 137.5° is motivated by phyllotaxis on cylindrical /
spherical surfaces.
*What to do:* run golden-angle prior vs grid-conv on rotated CIFAR-10 (test-time
rotations U[0, 360°]) at n=3 seeds × 50 ep. ~5 GPU-h.
*Severity:* MAJOR · *Effort:* M (~5 GPU-h) · *Priority:* P1

### B9 — Fractal / self-similar architectures on multi-scale data
*What's wrong:* fractal-prior hypotheses tested only on single-scale CIFAR-100.
Self-similarity matters when the data is multi-scale (medical, remote sensing).
*What to do:* run one fractal hypothesis (H16 or H19) vs ResNet-20 baseline on
**Camelyon17 patches (lymph node metastases, 96×96, 4 hospitals)** at n=3.
~10 GPU-h. The sister `autoresearchimage` repo already has this loader (sister-repo
parity audit).
*Severity:* MAJOR · *Effort:* L (~10 GPU-h) · *Priority:* P1

### B10 — Multi-task generalization: top-3 priors on Imagenette × Tiny-ImageNet × CIFAR-100
*What's wrong:* even after Wave-0/1/2/3, each prior is evaluated on one task at
a time. R3 #6 + R4 #29.
*What to do:* report the **three priors × three tasks matrix** with consistent
iso-FLOPs + n=3 seeds + modern recipe. Highlights which prior generalizes vs which
is dataset-specific.
*Severity:* MAJOR · *Effort:* — (subsumed by Waves 1-3) · *Priority:* P1

### B11 — Replace `slot_act_sine` headline with a real nature-inspired claim
*What's wrong:* SIREN replication mislabeled as nature-inspired. The catch is the
protocol-positive finding (R3 #5).
*What to do:* in the abstract, replace the third "winner" with **H71 IcosaRoPE3D
Spherical-MNIST lift** (from B5) if Wave-4 succeeds. If Wave-4 fails to lift by
+3 pp, replace with the hex-lattice AID result (B6) if that lifts.
*Severity:* MAJOR · *Effort:* — (subsumed) · *Priority:* P0

### B12 — H09 phi_budget: explicit RegNetX-200MF Pareto comparison
*What's wrong:* paper concedes H09 is "rediscovery of RegNet's `w_m` Pareto region."
RegNet's published `w_m ∈ [2.5, 2.9]` is *not* φ (=1.618).
*What to do:* iso-FLOPs comparison between φ-budget (w_m=φ≈1.618) and RegNetX's
`w_m=2.7` at the same FLOP budget on Imagenette/Tiny-ImageNet/CIFAR-100. n=5
seeds each. **If φ Pareto-dominates RegNet's published optimum, that's a real claim.**
*Severity:* MAJOR · *Effort:* L (~25 GPU-h, subsumed by Wave-1/2) · *Priority:* P0

### B13 — Real icosahedral equivariance via e3nn (not one-shot rotation pool)
*What's wrong:* `src/nature_inspired_networks/icosa.py` is orientation augmentation,
not steerable group convolution. R3 #4 + R1 #1.
*What to do:* build a parallel `nature_inspired_networks_eq/` package on Python 3.10
(separate venv, e3nn requires it) implementing one icosahedral CNN block using
e3nn's `Irreps` + `FullyConnectedTensorProduct`. Test on Spherical MNIST and ModelNet10.
*Severity:* MAJOR · *Effort:* XL (2 weeks) · *Priority:* P1

### B14 — Chladni cymatic init at vibration-mode-matched tasks
*What's wrong:* Chladni-pattern init tested only on CIFAR. Vibration-mode priors
have a real motivation for acoustic / spectral tasks.
*What to do:* run Chladni-init prior on **UrbanSound8K spectrograms (10 classes,
mel-spectrograms 128×128)** at n=3 seeds × 50 ep. ~8 GPU-h.
*Severity:* MAJOR · *Effort:* L (~8 GPU-h) · *Priority:* P1

### B15 — Per-hypothesis hyperparameter hill-climb under modern recipe
*What's wrong:* Phase-9i ran the priors at the baseline's hyperparameters. The
priors were never hill-climbed at the modern recipe. R2 #6.
*What to do:* for the top-3 priors AND baseline, run a 20-trial coordinate-descent
hill-climb over `(lr, wd, batch)` at iso-FLOPs modern recipe. Pre-register the
hill-climb ranges BEFORE seeing any seed result. The current
`autoresearch-per-hypothesis-hillclimb` skill is for this.
*Severity:* MAJOR · *Effort:* XL (~60 GPU-h) · *Priority:* P1

### B16 — Tabulate "screening lift" vs "iso-FLOPs lift" vs "modern-recipe lift"
*What's wrong:* the FINDINGS table conflates screening (n=1, 12 ep), evaluation
(n=7, 30 ep, non-iso-FLOPs), and Phase-9i (n=3, 200 ep, non-iso-FLOPs) into one
row per prior.
*What to do:* per-prior 3-column table showing the lift at each tier. The honest
verdict per prior becomes obvious.
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### B17 — Honest negative-result reporting for non-promoted priors
*What's wrong:* the 81 "not promoted" priors are buried. A reviewer needs to see
the falsifications as first-class results.
*What to do:* add `paper/NEGATIVE_RESULTS.md` listing each non-promoted prior, its
pre-registered falsifier, the falsification verdict, the mechanism that didn't pan
out. **Negative results from disciplined falsification are publishable.**
*Severity:* MAJOR · *Effort:* M (1 day) · *Priority:* P1

### B18 — H71 mechanism pre-registration
*What's wrong:* H71 has no concrete predicted Δ at any task. Untested AND unfalsifiable.
*What to do:* pre-register `Δ_H71 ≥ +3 pp at rotated Spherical MNIST` with seed-7 paired
Wilcoxon. Commit hash referenced in PAPER.md before Wave-4 launches.
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### B19 — Drop H08, H43, H67, H74 from contributions (broken / shape-only)
*What's wrong:* these hypotheses had their implementations or tests caught as
shape-only / mechanism-broken. They survived as "fixed and re-tested" but the
mechanism-pinning tests are weak.
*What to do:* explicitly demote in `IDEA_TABLE.md` to "FALSIFIED_AT_MECHANISM_TEST."
Do NOT list them as Phase-8 winners or contributions.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### B20 — Add scaling-law experiment for the surviving prior
*What's wrong:* nature-inspired DL papers at NeurIPS 2027 carry scaling-law plots.
Bello 2021's recipe lifted ResNet-50 from 76→79.7% on ImageNet — a 4-point lift.
A nature-inspired prior should report what happens at 100k → 200k → 400k → 800k
params under iso-FLOPs.
*What to do:* the survivor prior + RegNetX baseline at 4 size points on Imagenette/
Tiny-ImageNet. ~40 GPU-h. **Scaling-law plot is a top-tier visual artifact.**
*Severity:* MAJOR · *Effort:* L (~40 GPU-h, subsumed by Wave-1/2) · *Priority:* P1

---

## 4 · Block C — Statistical rigor (P0/P1)

**Goal:** every external claim clears Holm-Bonferroni at α=0.05 across the honest
family, with bootstrap CIs and proper power.

### C1 — Phase-9j n=7 paired Wilcoxon at iso-FLOPs modern recipe
*What's wrong:* the current n=3 is at the Wilcoxon floor.
*What to do:* extend each surviving prior to n=7 seeds at iso-FLOPs modern recipe
200 ep (~39 GPU-h per arm × 4 arms = ~156 GPU-h). Report paired Wilcoxon, paired-t
(with normality justification), bootstrap CI (BCa, 10⁴ resamples), Holm-Bonferroni k=3.
*Severity:* BLOCKER · *Effort:* XL (~156 GPU-h, fold into Wave-2) · *Priority:* P0

### C2 — POSI correction across the honest screening family
*What's wrong:* k=3 Holm post-screening is dishonest; the screening universe is k≈40-76.
*What to do:* report explicitly that POSI-corrected α' ≈ 0.001 at k=40, and that the
current n=7 claims do NOT clear POSI. Frame as "two-stage clinical-trial design with
conditional α" (R3 #9, R2 M3).
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### C3 — Cross-family auditor on 10+ findings ($50)
*What's wrong:* every auditor is Claude Opus 4.7.
*What to do:* dispatch GPT-5 + Gemini 3 Pro on the same 10 MAJOR/BROKEN findings
via API. Report verdict-agreement rate. Add to `audits/CROSS_FAMILY_HONEST_REAUDIT.md`
as Section 5. **$50 of API + 1 day.**
*Severity:* BLOCKER · *Effort:* M ($50 + 1 day) · *Priority:* P0

### C4 — Pre-registration of every future experiment
*What's wrong:* Phase-9h → 9i pivot reads as HARKing.
*What to do:* every future experiment files a `pre-registration/<exp>_<date>.md`
with commit hash, recipe, seeds, decision rule. Mandate via CLAUDE.md Rule 39.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### C5 — Stop reporting paired-t at n=3
*What's wrong:* paired-t at df=2 has tails so heavy a single outlier flips the verdict.
*What to do:* drop paired-t for n≤6. Use paired permutation test (achievable
p ≈ 10⁻³ at n=5).
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### C6 — Bootstrap CIs at n=3 explicitly disclaimed
*What's wrong:* bootstrap at n=3 under-covers true parameter by ~2× (DiCiccio &
Efron 1996).
*What to do:* drop bootstrap at n<5. Or report BCa AND disclose the under-coverage
quantitatively.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### C7 — Power analysis section in STATISTICAL_TESTS
*What's wrong:* no paper-grade power analysis exists.
*What to do:* add §16 to `paper/STATISTICAL_TESTS.md`: required n to detect Δ=0.5 /
1.0 / 1.5 pp at σ=0.45 pp, α=0.05 two-sided 80% power. Likely answers: n≥14 / n≥7 /
n≥4. Pre-register the n choice for each future experiment based on expected Δ.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### C8 — Mann–Whitney "p=0.05" at n=3 labeled as the floor
*What's wrong:* at n_a=n_b=3, MW one-sided minimum is 1/C(6,3) = 0.05. Reporting
"clears α=0.05" when at the floor is sleight-of-hand.
*What to do:* every "p at the floor" result labeled "= floor at this sample size,
informationally identical to sign test."
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### C9 — Phase-5 ordinal gate honestly labeled as α=0.125 sign test
*What's wrong:* min(leader) > max(baseline) at n=3 has P=1/8 under H₀.
*What to do:* every Phase-5 PASS labeled with "α=0.125 sign test at n=3."
Replace with paired Wilcoxon + Holm at α=0.05 at n=7 as the EVALUATION-tier gate.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### C10 — Audit-calibration substrate match: research-grade repos at similar maturity
*What's wrong:* `pytorch/vision` vs project-substrate is mature-vs-exploratory.
*What to do:* run the audit doctrine on 30 hypotheses from 3 recent ICML/NeurIPS/ICLR
supplementary code repos (released this calendar year). Expected MAJOR/BROKEN rate
8-18%, comparable to project's 21.7%. Reframe headline: "audit doctrine separates
project-quality from production-quality, not from research-quality." (R2 M1)
*Severity:* BLOCKER · *Effort:* M (1 week, 0 GPU) · *Priority:* P0

### C11 — Known-buggy commit calibration (mutation testing)
*What's wrong:* the 0/62 production-code finding doesn't tell us if the auditor
*can* catch real bugs.
*What to do:* introduce 5 known bugs into `BatchNorm2d.forward`, `Conv2d.__init__`,
`MultiheadAttention`, `RNNBase`. Audit those mutants. If 5/5 caught, sensitivity floor
is solid. If 0/5, the audit is not even Production-quality-aware. (R1 #03)
*Severity:* BLOCKER · *Effort:* M (1 day) · *Priority:* P0

### C12 — Stratified calibration by code-popularity tier
*What's wrong:* `pytorch/vision` is the most-starred repo class; a population-matched
calibration needs popular AND obscure repos.
*What to do:* sample 30 hypothesis-equivalents from <10-star GitHub research repos
matched by topic. Audit at same doctrine. (R4 #8, R2 M2)
*Severity:* MAJOR · *Effort:* L (~30 person-h) · *Priority:* P1

### C13 — Sequential testing correction on n=15 → n=62 calibration growth
*What's wrong:* if the n=62 cohort was assembled after seeing n=15's p≈0.07, this
is sequential testing without correction. (R4 #14)
*What to do:* commit the n=15 verdict timestamp, the n=62 verdict timestamp. Apply
α-spending function (Pocock or O'Brien-Fleming). Report adjusted p.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### C14 — Empirically-derived noise band per dataset
*What's wrong:* "±0.5 pp" rule-of-thumb appears in some places without empirical
backing.
*What to do:* compute per-dataset noise band from project's own multi-seed data.
Add to `paper/STATISTICAL_TESTS.md` §16. Cite in every "X exceeds noise band" claim.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### C15 — Add "what would falsify this paper" section
*What's wrong:* no explicit falsifier — after Phase-9h pivot, implicit answer is
"nothing." R3 #24.
*What to do:* `paper/FALSIFIERS.md` listing 3-5 results that would refute the paper.
Example: "If Wave-2 Tiny-ImageNet returns Δmean < +0.3 pp with bootstrap CI including 0
at n=5, the iso-FLOPs priors claim is refuted."
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

---

## 5 · Block D — Code bugs from R5 (P1)

**Goal:** every BUG R5 found is fixed and has a regression test. Mechanism-pinning
test (Rule 25) applies.

### D1 — Mixup λ-flip silently halves Beta variance (`mixup.py:71-74`)
*What to do:* either remove the flip (timm default) or document; add test asserting
`lam ≥ 0.5` post-flip and that effective α matches expected.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### D2 — CutMix degenerate zero-area boxes (`cutmix.py:43-53`)
*What to do:* add 1000-trial test asserting non-degeneracy rate ≥ 0.9 at α=1.0;
guard `rand_bbox` to reject `(x2-x1)==0 OR (y2-y1)==0` and resample.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### D3 — Random Erasing inplace + value semantics (`random_erasing.py:32, 61-63`)
*What to do:* clarify docstring on Normalize-space value=0 vs pixel-space; add
torchvision-version-parameterised test.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### D4 — `set_seed` missing determinism + worker_init_fn (`runner.py:38-41`)
*What to do:* add `headline_mode=True` argument. In headline mode set
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True,
cudnn.benchmark=False`, and `worker_init_fn=seed_worker` for the DataLoader.
Re-run the n=7 default cert in headline mode; report new σ. **This may
substantially tighten the noise band.**
*Severity:* BLOCKER · *Effort:* M · *Priority:* P0

### D5 — RandAugment per-worker RNG unseeded (`randaugment.py`)
*What to do:* covered by D4's `worker_init_fn`. Add explicit test that two seeded
runs with `num_workers=4` produce bit-identical augmentations.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### D6 — EMA finalization clobbers BN running stats (`train.py:496-502`)
*What to do:* add `_recalibrate_bn(model, train_loader)` post-EMA-load. One fwd
pass with BN momentum override. Standard timm pattern.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### D7 — `train_top1` silently wrong under Mixup (`train.py:393-395`)
*What to do:* when `mixed=True`, evaluate on un-mixed train set once per epoch.
Report both `train_top1_mixed` and `train_top1_clean`. Update `generalization_gap`
to use the clean version.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### D8 — `cudnn.benchmark=True` is global side effect (`runner.py:41`)
*What to do:* move to `enable_fast_mode()` helper. Document benchmark side effects
in CLAUDE.md Rule 6.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### D9 — `composite_score` floors params at 0.001M (`eval.py:130`)
*What to do:* raise `ValueError` on `params < 1000`. Add test.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### D10 — `experiment_log.jsonl` no integrity check (`runner.py:344-352`)
*What to do:* add flock + fsync; assert uniqueness of (tag, seed) on append.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

### D11 — `epochs_to_target` raw vs EMA mismatch (`train.py:488` vs `eval.py:528`)
*What to do:* report both `epochs_to_target_raw` and `epochs_to_target_ema`. Update
schema.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### D12 — H09 phi_budget integer-search realised width tuple in metrics.json
*What to do:* metrics.json now records `realised_widths: [16, 26, 42, 68]` and
`realised_phi_ratios: [1.625, 1.615, 1.619]`. Reviewer can verify the φ-prior is
actually implemented.
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### D13 — Citation regex over/under-matches (`reasoning.py:31-32`)
*What to do:* tighten arXiv regex to require 4.4-5 format; require ≥5 words after
the em-dash for relevance clause; add author-year ID alternative.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### D14 — Pruning + EMA silent strict=False fallback (`train.py:498-502`)
*What to do:* log every dropped key. Raise on shape mismatches. Add test that
H43 fibonacci_prune produces a valid EMA-loaded model.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### D15 — BettiLoss doubles forward compute under autocast (`train.py:378-381`)
*What to do:* cache `stagewise_features` from the main forward; reuse for Betti term.
~2× speedup on Betti-enabled runs.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

---

## 6 · Block E — Nature-inspired priors: depth over breadth (P0/P1)

**Goal:** kill the 84-hypothesis breadth that reads as "AI slop" and replace with
~10 hypotheses defended in depth. Each surviving prior has a theoretical mechanism,
a literature anchor, a pre-registered falsifier on the right dataset, and a
top-tier-scale empirical test.

### E1 — Cull the 84-hypothesis substrate to ≤10 deep ones
*What's wrong:* 84 hypotheses on CIFAR-10 with 80% single-seed coverage signals
LLM breadth-without-depth (R3 #25).
*What to do:* keep 10 hypotheses, one per distinct mechanism family:
1. H09 phi_budget (RegNet Pareto-adjacent width allocation)
2. H22 toroidal_phi_closure (on tiled-texture dataset)
3. H10 hex_phi_radial (on aerial imagery)
4. H71 IcosaRoPE3D (on Spherical MNIST)
5. H16 fractal_self_similar (on multi-scale medical)
6. H88 pair_gm_pdw (admitted as 3-axis regularizer stack with 61% non-φ)
7. H_chladni_init (on spectrograms)
8. H_golden_angle (on rotated CIFAR)
9. H_phi_decay_wd (layer-graded weight decay; the only φ-axis with isolated lift)
10. H_baseline_modern (the recipe itself is documented as a contribution)

Drop the other 74 into supplementary "design-space enumeration." Cut MANIFESTO
references to "84 hypotheses" everywhere.
*Severity:* MAJOR · *Effort:* M (1 week) · *Priority:* P0

### E2 — Per-surviving-hypothesis theoretical mechanism section
*What's wrong:* hypothesis docs are templated but lack rigorous mechanism
derivation (R1 #01).
*What to do:* for each of the 10 surviving hypotheses, add a `## Theoretical
Mechanism` section. For H09: prove φ-allocation minimises FLOPs-per-effective-rank
under a specific objective. For H10: derive HexConv equivariance properly to C₆
group action. For H71: state the SO(3) representation explicitly.
*Severity:* MAJOR · *Effort:* L (1 week per hypothesis · 10 hypotheses · 80 person-h) · *Priority:* P1

### E3 — Literature anchor per surviving hypothesis: be explicit about derivative-vs-novel
*What's wrong:* every NUMEROLOGY verdict is at risk of being uncited prior art.
*What to do:* per surviving hypothesis, the design doc names the closest prior
work AND quantifies the delta (e.g., "H09 differs from RegNet by enforcing exact
φ-ratio vs free Pareto-search; testable claim: φ-specificity vs Pareto-region").
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### E4 — Demote `slot_act_sine` to SIREN-replication-disclosed
*What's wrong:* Control 2 showed tanh > sine; abstract still lists `slot_act_sine`.
*What to do:* drop from abstract triple. Replace with H71 (post-Wave-4) or hex
(post-B6) if either succeeds. SIREN replication mentioned only as protocol catch
in §5.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### E5 — Reframe `pair_gm_pdw` as "3-axis orthogonal regularizer stack"
*What's wrong:* paper sells +1.74 pp as φ-prior; Control 1 shows 61% is non-φ.
*What to do:* abstract: "we identify a 3-axis regularizer stack of which ~39% is
φ-specific; the additive design (Rule 23) is the protocol-positive finding."
Either run the 2³ φ-factorial (B17 below) to isolate the φ-axis contribution, or
demote.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### E6 — 2³ factorial: phi_budget × golden_momentum × phi_decay_wd
*What's wrong:* `pair_gm_pdw` confounds three φ-axes; no factorial decomposition.
*What to do:* 8 cells × 3 seeds CIFAR-100 modern recipe iso-FLOPs. Report marginal
effects of each axis. Likely outcome: only one axis (phi_decay_wd?) carries the
φ-specific signal.
*Severity:* MAJOR · *Effort:* L (~30 GPU-h) · *Priority:* P1

### E7 — Real equivariance: e3nn-based icosahedral block
*What's wrong:* `icosa.py` is one-shot rotation pool, not group convolution.
*What to do:* implement `IcosahedralConv2d` using e3nn's irrep machinery in a
separate Py3.10 venv. Test on Spherical MNIST (B5) and ModelNet10 rotated.
*Severity:* MAJOR · *Effort:* XL (2 weeks) · *Priority:* P1

### E8 — Drop "icosahedral equivariance" from any name where it's not implemented
*What's wrong:* H55, H67 — names imply equivariance, implementation does not.
*What to do:* rename to "icosahedral-pool" or "60-rotation-augmented." Reserve
"equivariant" for blocks that pass an actual equivariance test.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### E9 — Equivariance certification test
*What's wrong:* no test verifies whether any block is actually equivariant.
*What to do:* add `tests/test_equivariance.py` that for each "equivariant" block
checks `‖block(g·x) − g·block(x)‖ < ε` for sampled `g`. Mark all currently-failing
blocks as not-actually-equivariant.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### E10 — Spectral analysis of SIREN-on-CIFAR
*What's wrong:* the SIREN-on-CIFAR result is an interesting protocol catch but has
no spectral story.
*What to do:* compute spectral bias (Rahaman 2019 arXiv:1806.08734) of `slot_act_sine`
vs `slot_act_tanh` vs ReLU on CIFAR-100 random labels. Report which frequencies the
network fits per activation. **Spectral bias plots are top-tier visual artifacts.**
*Severity:* MAJOR · *Effort:* L (~20 GPU-h) · *Priority:* P1

### E11 — H22 toroidal: tiled-texture dataset construction
*What's wrong:* H22 needs a wrap-aware dataset.
*What to do:* construct `tiled-CIFAR-10`: each image is a 2×2 tile with wraparound,
random tile composition. Commit dataset loader. Run H22 on it (B7).
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### E12 — Hex priors on aerial imagery + isolating hex equivariance
*What's wrong:* HexConv tested only on upright CIFAR (B6 fixes the dataset).
*What to do:* additionally verify hex equivariance to C₆ group action via E9's
test. Document the failure mode if not actually C₆-equivariant.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### E13 — Fractal priors on hierarchical/multi-scale data
*What's wrong:* fractal hypotheses tested only on CIFAR.
*What to do:* Camelyon17 (B9) gives multi-scale (96×96 tiles from gigapixel slides).
Run one fractal hypothesis here.
*Severity:* MAJOR · *Effort:* — (subsumed by B9) · *Priority:* P1

### E14 — Golden-angle prior on rotation-augmented data
*What's wrong:* B8 fixes the dataset. Additional ablation needed.
*What to do:* compare golden-angle (137.5°) modulation vs 90° / 60° / random-angle
modulations on rotated CIFAR. Isolates whether 137.5° has a special property.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### E15 — Chladni init on spectrograms (B14) + frequency-domain ablation
*What's wrong:* Chladni motivation is vibration-mode; CIFAR has none.
*What to do:* on UrbanSound8K, additionally compare Chladni-init to random / Xavier
/ orthogonal init. If Chladni helps only on spectrograms and not on CIFAR, that's
the prior-meets-domain signal.
*Severity:* MAJOR · *Effort:* — (subsumed by B14) · *Priority:* P1

---

## 7 · Block F — Literature & citations (P0/P1)

**Goal:** every prior-art citation is correct, every novelty claim has the closest
prior work named, and the LLM-judge / mutation-testing / self-refine literature is
properly engaged.

### F1 — Add Bello 2021 (arXiv:2103.07579) "Revisiting ResNets" to References §8
*What's wrong:* the modern 11-trick recipe IS Bello 2021. Uncited.
*What to do:* add citation. Reframe convergence/PLAN.md as "the Bello-Wightman modern
recipe at 200 ep."
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### F2 — Add Wightman 2021 (arXiv:2110.00476) "ResNet Strikes Back"
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### F3 — Add Zheng 2023 (arXiv:2306.05685) "Judging LLM-as-a-Judge"
*What's wrong:* the audit doctrine is LLM-as-judge; foundational paper uncited.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### F4 — Add Wang 2023 (arXiv:2305.17926) "Large Language Models are not Fair Evaluators"
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### F5 — Add Saunders 2022 (arXiv:2206.05802) "Self-Critiquing Models"
*What's wrong:* "self-auditing" is invented de novo; Saunders 2022 is the lineage.
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### F6 — Add Madaan 2023 (arXiv:2303.17651) "Self-Refine"
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### F7 — Add DeMillo 1978 + Jia 2011 + Tian 2023 (mutation testing lineage)
*What's wrong:* "mechanism-verifying tests" is mutation testing.
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### F8 — Add Cohen-Welling 2016 (arXiv:1602.07576) "G-Equivariant CNNs"
*What's wrong:* foundational equivariance; uncited.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### F9 — Add Weiler 2019 (arXiv:1911.08251) "e2cnn" + Geiger 2022 (arXiv:2207.09453) "e3nn"
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### F10 — Add Sitzmann 2020 (arXiv:2006.09661) "SIREN" prominently for `slot_act_sine`
*What's wrong:* cited in passing; should be the literature anchor with explicit
disclosure of replication.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### F11 — Add Munafò 2017 + Nosek 2018 for pre-registration
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### F12 — Add Gelman & Loken 2013 "Garden of Forking Paths"
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### F13 — Add Panickssery 2024 (arXiv:2404.13076) "LLM Evaluator Self-Preference Bias"
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### F14 — Add Berk 2013 (arXiv:1306.1107) POSI
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### F15 — Add Radosavovic 2020 (arXiv:2003.13678) RegNet — properly engage H09's relationship
*What's wrong:* RegNet is named but the engagement is brief. The honest claim is
"H09 sits in RegNet's Pareto region with a more constrained allocation rule."
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### F16 — Fix arXiv:1902.04615 attribution (Gauge/Icosahedral CNN, ICML 2019, NOT Spherical CNNs)
*What's wrong:* PAPER.md line 233 has the wrong title/venue.
*What to do:* correct to either arXiv:1801.10130 (Spherical CNNs ICLR 2018) or
arXiv:1902.04615 (Gauge/Icosahedral CNN ICML 2019) depending on which the H55
mechanism cites.
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### F17 — Fix Hoogeboom HexaConv venue (ICLR 2018, not ICML 2018)
*Severity:* MINOR · *Effort:* S · *Priority:* P1

### F18 — Verify Islam et al. 2025 arXiv:2510.03511 "Platonic Transformers" exists
*What's wrong:* future-dated arXiv ID with high hallucination risk.
*What to do:* human Google-Scholar verify. If hallucinated, remove citation and the
H55 mechanism story it supports.
*Severity:* BLOCKER · *Effort:* S · *Priority:* P0

### F19 — Audit every arXiv ID post-2024 for hallucination
*What's wrong:* LLM-written citations from memory cluster errors.
*What to do:* automated arXiv API ping for every post-2024 citation. Flag any 404s.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### F20 — Discuss LLM-judge bias literature in §8 Related Work
*What's wrong:* Holm 1979 cited; Zheng 2023 not.
*What to do:* 1-paragraph subsection in §8 with Zheng/Wang/Liu/Chen/Panickssery, discussing
self-preference bias as applied to same-model-family audits.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

---

## 8 · Block G — Paper structure (P1/P2)

**Goal:** the paper fits ICML 9-page limit, lead with the strongest result, no
self-grading banners, hyperparameter table present.

### G1 — Compress abstract from 440 words to ≤200
*Severity:* MAJOR · *Effort:* S · *Priority:* P1

### G2 — Cut PAPER.md from 263 lines to ICML-9-page limit
*What to do:* move audit-calibration math to supplementary. Cut §6 case study to ≤1.5 pp.
*Severity:* BLOCKER for ICML · *Effort:* M (16 person-h) · *Priority:* P1

### G3 — Add hyperparameter Table 1
*What to do:* per-tag table: (model, channel_mode, flags, optimizer, lr, scheduler,
wd, batch, label smoothing, augmentation, AMP, epochs, seeds, hardware).
*Severity:* MAJOR · *Effort:* M · *Priority:* P0

### G4 — Strip self-grading banners from README, PAPER, REVIEWER_CHECKLIST
*Severity:* MAJOR · *Effort:* S · *Priority:* P0

### G5 — Move FINDINGS.md "Phase-9X" journal to audits/HISTORY/
*What's wrong:* FINDINGS reads as project diary, not paper artifact.
*What to do:* per-claim "verdict + one supporting fact" format. Historical narrative
to `audits/HISTORY_2026-05-30.md` etc.
*Severity:* MAJOR · *Effort:* M · *Priority:* P1

### G6 — Move dashboard to supplementary
*What to do:* inline 3-5 hand-curated figures in PAPER body. Dashboard URL goes
under §A.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### G7 — Per-claim cost/value table in §3
*What to do:* claim, GPU-h, n_seeds, statistical power, Δmean, pre-registered α
clearance.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### G8 — Rename internal-jargon phases to descriptive names
*What's wrong:* "Phase-9i convergence-regime corrective binding" is unreadable.
*What to do:* "iso-recipe n=3 diagnostic at non-matched FLOPs" etc. Keep phase
labels only in commit logs.
*Severity:* MINOR · *Effort:* S · *Priority:* P1

### G9 — README elevator pitch ≤4 sentences, honest contribution
*What to do:* "We test 10 nature-inspired priors on iso-FLOPs benchmarks ranging
from CIFAR-100 to ImageNet-100. One prior (H71 IcosaRoPE3D on Spherical MNIST)
lifts +X pp at n=7; one prior is reframed as a 3-axis regularizer stack with 39%
φ-specific residual; eight others falsified at the right dataset. We additionally
release a Fixer-with-mechanism-pinning-test contract for LLM-agent research
pipelines, validated on 18 audit-caught defects."
*Severity:* MINOR · *Effort:* S · *Priority:* P1

### G10 — Cut CLAUDE.md from 38 rules to ≤15 load-bearing rules
*What's wrong:* process inflation (R4 #16).
*What to do:* archive non-load-bearing rules (typography, dashboard mirroring,
link discipline) to `process/historical/`. CLAUDE.md only carries rules that
operationally gate decisions.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

---

## 9 · Block H — Repo hygiene + dashboard truthfulness (P2)

### H1 — Repo-root limited to ≤4 files (CLAUDE.md Rule 31)
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### H2 — Merge `experiments_modern/` under `experiments/modern/`
*What's wrong:* two semi-roots; one unreferenced by FINDINGS.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### H3 — Tag tests with `@pytest.mark.{shape,mechanism,regression}`
*What's wrong:* "780+ tests" badge is misleading — shape-only tests dominate.
*What to do:* tagged breakdown shows ~80 mechanism, ~600 shape, ~100 regression.
Badge reads "80 mechanism + 700 other."
*Severity:* MAJOR · *Effort:* L (1 week) · *Priority:* P1

### H4 — Dashboard: per-page `last_updated` field + auto-invalidate
*What's wrong:* dashboards show stale "screening" verdicts after corrections.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

### H5 — Auto-checkpoint squash post-campaign
*What's wrong:* 1000+ auto-commits clutter history.
*What to do:* `scripts/squash_checkpoints.sh` collapses auto-commits into one
labeled commit per tag after campaign completes.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

### H6 — Dashboard: cut to 1 aggregate + 3 winner pages + 3 falsifier pages
*Severity:* MINOR · *Effort:* M · *Priority:* P2

### H7 — Dashboard "How to read" orientation block + small-multiples
*What's wrong:* dense charts dominate; small-multiples + 4-bullet orientation per
CLAUDE.md Rule 33.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

### H8 — Pydantic TrainConfig schema
*What to do:* `pydantic.BaseModel` with Field constraints for `alpha ≥ 0`,
`decay ≤ 1`. Catches bad configs pre-launch.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

### H9 — Runner `--dry-run` flag
*What to do:* validates config, builds model, runs 1 batch, prints memory + latency
+ expected wall-clock. No full fit.
*Severity:* MINOR · *Effort:* S · *Priority:* P2

### H10 — Runner `--resume` for crash recovery
*What to do:* save optimizer + scheduler + RNG state every epoch to `state.ckpt`.
On launch, resume if present.
*Severity:* MINOR · *Effort:* M · *Priority:* P2

---

## 10 · The 12-week plan (250 GPU-h total)

This is the laptop-realistic execution plan. Every step honours the nature-inspired
north star.

### Week 1-2 · Block A close: iso-FLOPs + recipe debug (~30 GPU-h)
- A1 iso-FLOPs re-test of 3 priors (30 GPU-h)
- A2-A10 paper-writing fixes (parallel, 0 GPU)
- D4 headline-mode `set_seed` (0 GPU)
- D12 H09 realised widths in metrics.json (0 GPU)

**Gate:** iso-FLOPs prior runs commit. If lift survives at iso-FLOPs, proceed.
If collapses, **honest publication of negative result** as Wave-0 entry condition.

### Week 3 · Block B Wave-0: Imagenette recipe validation (~5 GPU-h)
- B1 Wave-0: Imagenette 10ep × n=5 × 3 recipes
- Pre-register the working recipe

**Gate:** recipe lands in published Imagenette band. If not, debug RandAugment /
Random Erasing / Mixup α. Sub-week.

### Week 4-5 · Wave-1: Imagenette iso-FLOPs Pareto (~50 GPU-h)
- B2 Wave-1: 4 models × n=5 × 50ep
- B12 H09 vs RegNetX-200MF Pareto at iso-FLOPs

**Gate:** clean Pareto curve. If nature-inspired prior Pareto-dominates RegNet on
Imagenette, real result. If not, demote H09 from contributions and pivot to B5
(H71 Spherical MNIST) as the headline prior.

### Week 6-8 · Wave-2: Tiny-ImageNet 200-class (~80 GPU-h)
- B3 Wave-2: baseline + 3 priors at iso-FLOPs, n=5
- C1 Phase-9j n=7 confirmation (subsumes ~39 GPU-h for the survivors)
- C3 cross-family auditor on the survivors' mechanisms ($50 API)

**Gate:** paired Wilcoxon n=5 + Holm-Bonferroni k=3 + bootstrap CI. If priors clear,
proceed to Wave-3. If not, **publishable negative**: "Iso-FLOPs nature-inspired
priors do NOT lift on 200-class at modern recipe."

### Week 9-11 · Wave-3: ImageNet-100 FFCV (~80 GPU-h)
- B4 Wave-3: ResNet-50 baseline + winning prior at 160² × n=3
- FFCV-package ImageNet-100 (~2 days setup, sister-repo parity)

**Gate:** ImageNet-100 result for the surviving prior. **This is the figure that
makes the paper publishable at top-tier.**

### Week 12 · Wave-4: H71 IcosaRoPE3D on Spherical MNIST (~15 GPU-h)
- B5 Wave-4: ViT-Tiny baseline + H71 × n=5 × 100ep on Spherical MNIST
- E7 e3nn-based icosahedral block in parallel venv
- E9 equivariance certification test

**Gate:** if H71 lifts non-equivariant baseline by +3 pp, **the strongest
nature-inspired result.** If not, archive H71.

### Week 12 also · paper structure cleanup
- E1 cull 84 → 10 hypotheses
- G1-G9 paper rewrite
- F1-F19 citations
- B17 negative-results section

### Total budget
| Phase | GPU-h | Wall-clock |
|---|---|---|
| Block A | 30 | 2 weeks |
| Wave-0 | 5 | 0.5 week |
| Wave-1 | 50 | 2 weeks |
| Wave-2 | 80 | 3 weeks |
| Wave-3 | 80 | 3 weeks |
| Wave-4 | 15 | 1 week |
| Paper cleanup | 0 | 1 week (parallel) |
| **Total** | **260** | **~12 weeks** |

At 25 GPU-h/week sustainable on a 4090 Laptop with auto-checkpoint loop, this lands
in ~10-12 calendar weeks.

---

## 11 · Closing — how do we get to "all 5 reviewers super impressed"?

The user's stated success criterion: all 5 reviewers recommend top-tier. The
five reviewers above all REJECTED the current submission. Here is what flips each.

| Reviewer | Current verdict | What flips it | Blocked by |
|---|---|---|---|
| R1 ICLR | REJECT | Theoretical mechanism section per surviving hypothesis (E2); equivariance certification (E9); ImageNet evidence (Wave-3) | E2, E7, B4 |
| R2 ICML | REJECT | Iso-FLOPs Pareto across 4 architectures (Wave-1); n=7 Holm-clearance at modern recipe (C1); cross-family auditor (C3) | A1, B2, C1, C3 |
| R3 NeurIPS | REJECT | Modern baselines (B2); novelty over Bello/Wightman/Saunders cited honestly (F1-F6); broader-impact section on automated p-hacking (R3 #15) | B2, F1-F6, paper rewrite |
| R4 Elite researcher | WEAK_REJECT | Pick one nature-inspired prior (H71 or hex) and prove it on the right dataset (B5 or B6); demote everything else; do NOT pivot away from priors | B5, B6, E4, E5 |
| R5 Lab lead | No-hire RS / hire eval-infra | Iso-FLOPs (B1); debug modern recipe (A4); reproducibility plumbing (D4); ImageNet-100 (B4); ALL R5 code bugs fixed (D1-D15) | All of Block A + D + B4 |

**Single path that flips all 5:**

1. Close Block A (iso-FLOPs honesty, recipe debug) — Weeks 1-2.
2. Run Wave-0 → Wave-1 → Wave-2 → Wave-3 → Wave-4 — Weeks 3-12.
3. Land **at least one** of these as a clean iso-FLOPs lift at modern architectures:
   - H71 IcosaRoPE3D on Spherical MNIST (+3 pp over non-eq ViT) — **strongest**
   - φ-budget on Imagenette/Tiny-ImageNet/ImageNet-100 vs RegNetX-200MF — **second-strongest**
   - Hex-lattice on AID aerial imagery — **third**
4. Reframe the paper around the surviving prior, with the protocol as a methods
   chapter (not the headline).
5. Cross-family auditor + n=7 at iso-FLOPs + Bello/Wightman/Saunders citations land.
6. Honest negative results for the 7-9 non-surviving priors. **Negative results
   from disciplined falsification are publishable.**

If Wave-3 + Wave-4 both fail to land a clean +3 pp nature-inspired lift at
iso-FLOPs at modern architecture, the honest paper is:
- "We tested 10 nature-inspired priors at iso-FLOPs modern recipes across
  CIFAR-100 / Imagenette / Tiny-ImageNet / ImageNet-100 / Spherical MNIST.
  None survived. Negative result for the nature-inspired program at this scale."

That negative result is **also publishable** at NeurIPS / ICLR. It is harder to write
but harder to dismiss. The protocol contribution becomes the methods chapter that
shows how the negative result was made trustworthy. Top-tier reviewers respect
disciplined falsification more than weak positive claims.

**The recommended outcome is positive: at least H71 on Spherical MNIST lifts cleanly.
That experiment alone is one of the lowest-cost (15 GPU-h) and highest-prior
nature-inspired claims in the entire 84-hypothesis substrate. It is the experiment
that should run FIRST, not last.**

---

## 12 · Reviewer-objection close-out cross-reference

| Reviewer objection | Closed by | Status |
|---|---|---|
| R1 #1 No theoretical content | E2, E10 (spectral analysis) | P1 |
| R1 #2 ResNet-20 obsolete | B2, B3, B4 (modern arch) | P0 |
| R1 #3 Audit substrate not independent | C3 (cross-family), C10-C11 (matched substrate, mutation) | P0 |
| R1 #4 Phase-9i n=3 at floor | C1 (n=7 cell), A8 (rename) | P0 |
| R1 #5 SIREN demote | E4 | P0 |
| R1 #6 H09 vs RegNet | B12 | P0 |
| R1 #7-#11 citation accuracy | F16-F19 | P0 |
| R1 #12 composite metric | A2 (+ FLOPs term) | P0 |
| R1 #14 POSI | C2 | P0 |
| R1 #15 self-grading banners | G4 | P0 |
| R1 #16 LLM-curated hypothesis cohort | acknowledged in §1 (per R3 #25) + E1 cull | P1 |
| R1 #17 H71 untested | B5 (Wave-4) | P0 |
| R1 #19 Control 1 non-φ 61% | E5 (reframe) | P0 |
| R1 #20 Bello / Wightman | F1-F2 | P0 |
| R1 #26 baseline below SOTA | A4 (recipe debug) + B4 (ImageNet-100) | P0 |
| R1 #30 D&B venue | accepted as fallback; primary target ICML/ICLR | acknowledged |
| R2 M1-M7 methodology | Block A + C3-C13 | P0 |
| R2 B1-B5 baselines | B2-B5 (all Waves) + B12 | P0 |
| R2 S1-S6 stat | C5-C9 | P0/P1 |
| R2 R1-R2 reproducibility | D4, G3 | P0/P1 |
| R2 C1-C2 confounding | A1 (iso-FLOPs), E6 (φ-factorial) | P0/P1 |
| R2 H1-H2 hardware | A1 + D4 worker_init | P0 |
| R2 F1-F9 framing | G1-G10 + E1 + B17 | P0/P1 |
| R3 #1-#3 reframe / Phase-9h pivot | A8 + G1 + protocol-as-secondary | P0 |
| R3 #4 equivariance | E7, E8, E9 | P0/P1 |
| R3 #5 SIREN | E4 | P0 |
| R3 #6 Imagenette/Tiny-ImageNet | B1-B3 (Waves 0-2) | P0 |
| R3 #7 pre-register Phase-9j | C4 + A9 | P0 |
| R3 #8 cross-family auditor | C3 | P0 |
| R3 #9 Holm at screening universe | C2 (POSI) | P0 |
| R3 #10 Bello/Wightman | F1-F2 | P0 |
| R3 #11 RegNetX-200MF | B12 | P0 |
| R3 #15 broader impact | G3 (?) + new ETHICS section TBD | P1 |
| R3 #18 LLM-judge lit | F3, F4, F13, F20 | P0 |
| R3 #19 mutation testing | F7 | P1 |
| R3 #22 self-distillation lineage | F5, F6 | P0/P1 |
| R3 #25 cull 84 to 10 | E1 | P0 |
| R4 #1 protocol-only pivot | **explicitly rejected as guardrail** | — |
| R4 #2 cross-family $20 | C3 | P0 |
| R4 #3 paired iso-recipe experiment | B3 (Wave-2) | P0 |
| R4 #4 pick ONE prior on the right task | B5, B6, B7, B8, B14 (each a candidate) | P0 |
| R4 #5 RegNet-vs-φ | B12 | P0 |
| R4 #7 stop n=3 Wilcoxon | C1, C5 | P0 |
| R4 #15 stop rule | C15 + A7 | P0 |
| R4 #27 mechanism-pinning test contract | secondary methods chapter, not replacement | acknowledged |
| R5 BLOCKER 1 iso-FLOPs | A1, A2, A3, A6 | P0 |
| R5 BLOCKER 2 below-floor baseline | A4 + B1 (Wave-0) | P0 |
| R5 BLOCKER 3 reproducibility | D4, D5, D8 | P0 |
| R5 BUG 1-20 (specific code bugs) | D1-D15 (subset; rest as P2) | P0/P1/P2 |
| R5 SOTA-migration path | §10 plan (Waves 0-4) | P0 |

**All 165 raw reviewer objections collapse to the 100 improvements above.** Closing
P0 items (estimated ~50 of 100) is the minimum to flip at least 3 of 5 reviewers from
REJECT to ACCEPT. Closing P0+P1 (estimated ~85 of 100) flips all 5.

---

## 13 · Decision request

This synthesis is the action plan. Before execution, the user authorizes one of:

(a) **Execute the full 12-week plan** (~250 GPU-h, ~120 person-h paper rewrite).
    Outcome: paper resubmittable at ICML 2028 / ICLR 2028 / NeurIPS 2027.
(b) **Execute Block A + Wave-0 only** (~35 GPU-h, ~2 weeks). Outcome: iso-FLOPs
    honest priors result at default-architecture scale. Workshop submittable.
(c) **Execute Block A + Wave-4 (H71 Spherical MNIST)** (~45 GPU-h, ~3 weeks).
    Outcome: the strongest single nature-inspired claim tested at the right task.
    If H71 lifts, it becomes the headline and paper rewrites around it.
(d) **Reject this synthesis** — execute differently or pivot.

The user's frustration was the right diagnosis. The path forward is the
nature-inspired program at modern scale, with the protocol as the methods
backbone. **Not the protocol alone.**

---

*This synthesis honours the user's stated north star: nature-inspired methods that
move deep-learning forward. The 100 improvements are the path; the 12-week plan is
the schedule; the laptop 4090 is the hardware budget. Awaiting authorization.*
