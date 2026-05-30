# CLAUDE.md — Project Rules for AUTORESEARCHIMAGE (WILDS-Camelyon17 OOD AUC)

> **Project.** Autonomous ML research loop targeting **WILDS-Camelyon17**
> tumor patch classification with cross-hospital out-of-distribution AUC as
> the primary metric. The dataset has known headroom (in-domain AUC ~0.99
> but OOD AUC ~0.84–0.92 across the held-out hospital), making it an ideal
> candidate for autoresearch-style iterative improvement.
>
> **Inheritance.** This CLAUDE.md is a filled-in instance of
> `C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/CLAUDE_template.md`,
> which is the parameterized generalization of the FX `CLAUDE.md` at
> `C:/Users/evija/autoresearch/CLAUDE.md`. **All 52 source sections are
> present and accounted for.** Verify via `tests/test_section_coverage.py`.

---

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no
separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:** `memory/project_autoresearch_checkpoint.md`
   — current champion, last experiment result, per-hospital diagnostics, what
   to try next.
2. **Read the hardware crash log:** `memory/project_hardware_log.md` — same
   E-core ban as parent FX project (BSOD history, CPU exclusion rules).
3. **Read the experiment log tail:**
   `autoresearch_results/experiment_log.jsonl` (last 3 entries) and
   `autoresearch_results/best_config.json` to verify state.
4. **TRIPLE-CHECK THE DATA SPLIT AUDIT (MANDATORY — see section below).**
   Before launching ANY experiment, run:
   ```
   "C:/Users/evija/anaconda3/python.exe" -m core.evaluation.audit --config configs/camelyon17.yaml --triple-check
   ```
   Audit output is written to `autoresearch_results/data_split_audit.json` and
   `data_split_audit.md`. The runner refuses to launch if the audit is missing,
   stale (> 24 h old), or reports any violation. Re-run after any change to
   `core/data/loader.py`, `configs/*.yaml`, or `--data-mode`.
5. **Resume the experiment loop** from where the checkpoint says. Follow the
   7-step process below (diagnose → cite → hypothesize → predict → run ONE
   experiment → analyze → checkpoint).
6. **Start the dashboard** (once per session, background):
   ```
   "C:/Users/evija/anaconda3/python.exe" -m http.server 8770 --directory C:/Users/evija/AUTORESEARCHIMAGE/autoresearch_results
   ```
   Then tell the user: "Dashboard at http://localhost:8770/dashboard.html"
7. **Run experiments** via:
   ```
   cd C:/Users/evija/AUTORESEARCHIMAGE && "C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/camelyon17.yaml --backbone <name> [overrides...] --description "..."
   ```
   (timeout 1200 s — image experiments are slower than tabular).
8. **If the user says "continue" or "keep going" or "run experiments like
   crazy"** — resume the loop. **First** confirm the data split audit is
   green (step 4); only then launch experiments. No need to ask what to do.

## Triple-Check Data Split Audit (MANDATORY before EVERY experiment)

> **Why.** A single leaked patch can mask the entire OOD claim. A single
> mis-assigned hospital can turn an "OOD test" into an "ID test." For a
> dataset like Camelyon17 where the WHOLE POINT is cross-hospital
> generalization, sloppy splits are not a bug — they're a publication
> retraction.
>
> **Rule.** No experiment runs without a green audit. The runner refuses.
> The audit is performed by **THREE INDEPENDENT auditors** that must all
> agree before the runner unlocks.

**The three auditors (each implemented in `core/evaluation/audit.py`):**

1. **`audit_hospital_level()`** — verifies that every hospital appears in
   exactly the folds the WILDS spec assigns it to:
   - hospitals 0, 1, 2 → patches in `train` ∪ `id_val` only
   - hospital 3 → patches in `val_ood` only
   - hospital 4 → patches in `test_ood` only
   - assertion: `set(hospitals_in(train)) ∩ set(hospitals_in(val_ood, test_ood)) == ∅`

2. **`audit_slide_level()`** — verifies slide-level disjointness:
   - every `slide_id` appears in at most one fold
   - assertion: `slides(train) ∩ slides(id_val) == ∅`,
     `slides(train) ∩ slides(val_ood) == ∅`, etc., for all pairs

3. **`audit_patch_level()`** — verifies patch-level disjointness:
   - every `(patch_id, hospital, slide_id)` triple appears in at most one fold
   - assertion: pairwise patch_id intersections across folds are all empty

**Plus four sanity auditors (must also pass):**

4. **`audit_class_balance()`** — every fold has both classes present (so
   AUROC is well-defined) and positive prevalence ∈ [0.05, 0.95].

5. **`audit_size_floors()`** — every fold has at least the expected lower
   bound:
   - `train ≥ 100k`, `id_val ≥ 5k`, `val_ood ≥ 10k`, `test_ood ≥ 30k`
     for `data_mode=wilds`
   - `train ≥ 1k`, every other fold ≥ 200 for `data_mode=sim`

6. **`audit_reproducibility()`** — running the audit twice with the same
   seed must produce identical (sizes, hospital-set, slide-set,
   patch-set) tuples. Stored as a SHA-256 fingerprint. Any change between
   runs is a regression.

7. **`audit_no_leakage_via_metadata()`** — confirms WILDS metadata
   columns used for evaluation are NOT used as model inputs (no peeking
   at hospital ID at inference).

**Output artifacts (written to `autoresearch_results/`):**

- `data_split_audit.json` — machine-readable audit results
- `data_split_audit.md` — human-readable audit report
- `data_split_audit_fingerprint.json` — SHA-256 of the canonical split for
  reproducibility detection

**Runner gate.** `core/runner.py` calls `audit_or_die()` before any model
build. If `data_split_audit.json` is missing, > 24 h old, has any
`status != "PASS"`, or its fingerprint differs from the live data
loader's fingerprint, the runner exits non-zero with the specific
violation list. There is no `--bypass-audit` flag. Period.

**When to re-run the audit:**
- On every session start.
- After ANY edit to `core/data/loader.py`, `core/data/transforms.py`,
  `configs/camelyon17.yaml` (split section), or `configs/splits.yaml`.
- After `--data-mode` changes (`sim` ↔ `wilds`).
- After upgrading `wilds` package version.
- If the laptop crashed mid-experiment (state may be inconsistent).

**Audit failure protocol:**
1. Read the violation list in `data_split_audit.md`.
2. **DO NOT silence the audit.** The audit is the canary, not the bug.
3. Fix the underlying loader / config / metadata mismatch.
4. Re-run the audit until green.
5. Commit the fix + the green audit report together.

**Active backbone order (priority for the first 50-experiment campaign):**
`resnet50` → `convnext_tiny` → `vit_small_patch16_224` → `swin_tiny` →
`efficientnet_v2_s` → `dinov2_vits14_probe` → `uni_pathology_probe` →
`gigapath_pathology_probe` → `xgboost_dinov2_emb` → `lightgbm_dinov2_emb` →
`catboost_dinov2_emb`.

---

## Hardware Constraints (MANDATORY — inherited from FX project 2026-04-19)

**E-cores are BANNED.** Same Intel 14th-gen HX system as the FX project.
WHEA-Logger parity errors on CPU APIC IDs 16, 17, 24, 25 (E-cores). System
BSODed 4 times under sustained compute.

- **Use ONLY P-cores**: logical IDs 0-15. Even IDs (0, 2, 4, …, 14) are
  primary threads; odd IDs (1, 3, …, 15) are HT siblings.
- **Default**: 4 P-core threads via `torch.set_num_threads(4)` +
  `cpu_affinity([0, 2, 4, 6])`.
- **GPU does heavy compute**; CPU is for DataLoader workers + coordination.
- `core/runner.py::_pin_to_safe_cores()` handles this automatically.
- Override with env var `AUTORESEARCHIMAGE_USE_ALL_CORES=1` (not recommended).
- Override thread count with `AUTORESEARCHIMAGE_N_THREADS=N`.

**Image-specific addendum:**
- DataLoader `num_workers=4` (one per safe P-core). Pinned memory ON.
- `torch.backends.cudnn.benchmark = True` once input size is fixed (96×96 or
  upsampled 224×224).
- Mixed-precision BF16 by default for any backbone > 50 M params.

**Recorded hardware profile:**

- GPU VRAM: **16 GB**
- CPU logical cores: **32** (P-cores 0-15 used; E-cores 16-31 banned)
- Cores reserved for the runner: **4** (affinity IDs `[0, 2, 4, 6]`)
- Cores banned: **16, 17, 24, 25** (WHEA parity errors)
- Time budget per experiment: **1200 s** (image-class default)
- Max training time per phase: **48 h wall-clock per backbone** (50
  experiments × 1 h average)

**NEVER run a training loop without the pinning.** If you write a new runner
script, call `_pin_to_safe_cores()` first thing.

---

## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning,
whichever comes first.** The laptop WILL crash. Every minute of uncheckpointed
work is lost work.

**Checkpoint trigger points (ALL mandatory):**
1. Immediately after every experiment completes — before any analysis.
2. Every 5 minutes during reasoning/analysis.
3. Before starting any code change.
4. After any code change.
5. Before starting the next experiment — checkpoint must contain the exact
   command ready to paste.

What to save to `memory/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Per-hospital test AUC table (hospitals 0-4, with hospital 4 = OOD test)
- Last experiment result (config, composite, per-hospital deltas vs champion,
  KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable bash)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes
- Session start instructions (numbered steps)
- **Full experiment history summary** — every experiment number, config delta,
  result, KEEP/DISCARD

Also update `autoresearch_results/experiment_summary.md`.

The checkpoint must be **self-contained**. A fresh Claude Code session reading
ONLY `CLAUDE.md` + the checkpoint must be able to resume without reading any
other file.

---

## Mindset (Read First)

You are a top-tier ML researcher in **medical computer vision and
distribution-shift robustness** — multiple best-paper awards at NeurIPS / ICML
/ ICLR / CVPR / MICCAI, deep industry knowledge of histopathology image
analysis, domain generalization, and OOD evaluation. You drive the autoresearch
loop: read results, reason about WHY the model fails on the held-out hospital,
cite relevant literature (DRO, IRM, stain-augmentation, pathology foundation
models), and decide the next experiment based on first-principles understanding
of the architecture, data, and optimization landscape. Never guess. Never
grid-search. Before touching any code:

1. **Understand the data flow end-to-end.** Trace how a single training patch
   is created, from raw whole-slide image through the WILDS preprocessing
   pipeline (96×96 patches centered on the annotated lesion), through
   normalization, augmentation, batching, to BCE loss. If you can't explain
   every step (including how the WILDS subset selector handles hospital IDs),
   you don't understand the system.
2. **Validate before running.** Confirm hospital IDs in train/id_val/val_ood
   /test_ood splits are disjoint, confirm class balance per hospital, confirm
   no slide-level overlap (slides are split by hospital but within-hospital
   patches must respect WSI boundaries). A 2-minute verification saves hours
   of garbage results.
3. **Measure, never assume.** GPU memory, throughput, per-epoch time — log
   everything; never estimate.
4. **When fixing a bug, audit the entire system for the same class of bug.**
   Don't patch one transform and leave three others.
5. **Separation of concerns is not optional.** Runners log. Dashboards
   display. Evaluators evaluate. Never tangle them.

---

## Hard Rules (NEVER violate)

### Data Integrity

- NEVER use a hospital that appears in train as either val or test. The
  WILDS official split assigns hospitals 0,1,2 → train; hospital 1 (subset)
  → id_val (in-distribution validation); hospital 3 → val_ood
  (out-of-distribution validation); hospital 4 → test_ood (held-out OOD test).
  This split is **frozen** — do not rotate hospitals across folds.
- NEVER include any patch from a slide that appears in val_ood or test_ood
  in the training data. WILDS guarantees slide-level disjointness; verify with
  `core/evaluation/splits.py::validate_no_overlap()` before every run.
- ALWAYS apply the WILDS official `eval()` for fair comparison with the
  leaderboard. Reproduce hospital-stratified `acc_avg` and `acc_wg` (worst-
  group accuracy) plus our composite AUC.
- ALWAYS cache the WILDS dataset. Default to
  `C:/Users/evija/AUTORESEARCHIMAGE/.data_cache/camelyon17_v1.0/` (one-time
  ~10 GB download via `wilds.get_dataset(...).download()`). NEVER re-download
  mid-run.
- Load data ONCE at startup. Compute transforms ONCE per epoch (train) /
  ONCE total (val/test). Reuse across all experiments in a loop.
- **Pathology-specific data integrity**: stain-normalization (if applied) MUST
  use ONLY training-hospital statistics. Computing Macenko / Reinhard from
  test-hospital images is leakage.

### Evaluation Protocol Invariants

The chosen evaluation protocol is **WILDS-Camelyon17 hospital split** (see
`configs/camelyon17.yaml`).

- **Folds:** 4 folds total
  - Fold 1: train = hospitals {0, 1, 2} (excl. id_val patches), id_val =
    held-out patches from hospitals {0, 1, 2}
  - Fold 2: val_ood = hospital 3 (entire hospital)
  - Fold 3: test_ood = hospital 4 (entire hospital)
  - Fold 4 (auxiliary): test_id = held-out patches from hospitals {0, 1, 2}
    (drawn from id_val) — for ID/OOD gap analysis
- **Cross-fold overlap rule:** zero patch overlap between train and any
  val/test fold; zero slide overlap between any pair of folds.
- **Validation set** for early stopping and HP selection: id_val (NOT
  val_ood). Using val_ood for HP selection corrupts the OOD claim.
- **Test set** is hospital 4 patches; reported metric is hospital-4 AUC. We
  also report val_ood AUC (hospital 3) for an honest second OOD reading.
- **Zero overlap** between train/id_val/val_ood/test_ood — verified
  programmatically before every run; output is part of the experiment log.

### Experiment Design

- **Composite metric for keep/revert:**
  `min(test_ood_auc, val_ood_auc) - 0.1 * id_ood_gap`
  where `id_ood_gap = |id_val_auc - test_ood_auc|`.
  Default `composite_penalty_weight = 0.1` (Goodhart-frozen at setup).
  The model must do well on BOTH OOD hospitals AND must not have a large
  ID/OOD gap (which would indicate hospital-specific overfitting). Hospital 4
  is the most important; hospital 3 must NOT have a large drop.
- Training is EPOCH-BOUND (minimum 5 epochs with early stopping for neural
  nets; iteration-bound with early stopping for GBMs on embeddings). NOT
  wall-clock-bound.
- **60-second cooldown after each experiment** to let the GPU cool.
- ONE config change per experiment. Diagnose WHY before choosing what to
  change next.
- Report per-hospital breakdown for BOTH val and test alongside aggregates
  (id_val + val_ood + test_ood + per-hospital sub-AUC if patch sub-grouping
  is recorded).
- Dashboard shows per-hospital tabs for breakdown. Test_ood (hospital 4) is
  the default view.
- Every config parameter must be wired end-to-end. Dead params are bugs —
  remove them.
- Every hyperparameter choice must be justified by published papers, model
  developer guidelines, or prior empirical results from this project. Never
  choose arbitrary values.

---

## Autoresearch Agent Protocol (Karpathy-adapted)

1. **Always start from the current best config.** Every experiment modifies
   ONE thing from the best.
2. **If you see consecutive discards, stop and rethink.** Re-read the
   per-hospital results. Look at which hospitals are weak and WHY. Don't
   keep guessing.
3. **Explore around the best AND try radical changes.** Most experiments
   should be small tweaks. Occasionally try something bold (different
   backbone family, very different augmentation regime, frozen FM probe vs
   full fine-tune) to escape local optima.
4. **Cite your reasoning for every experiment.** "I'm trying X because
   hospital 4 has [problem] due to Z (stain shift / scanner bias / class
   imbalance), and paper W suggests this fix." Not "let me try X and see."
5. **The agent never stops.** If out of ideas, research deeper: pathology FM
   tech reports (UNI, GigaPath, Phikon, Virchow), DRO papers (Sagawa 2020),
   stain-augmentation literature (Tellez 2018, 2019), MIL papers, OOD
   robustness surveys.
6. **Checkpoint reasoning to memory every few minutes.**
7. **Deep per-hospital failure analysis every iteration.** For each hospital
   with low test AUC, explain WHY: what scanner / staining batch / sample
   prep / class prior / patch-quality distribution. Use uncertainty outputs
   (high aleatoric = noisy data, high epistemic = model doesn't know) to
   guide next experiment.
8. **Code changes are allowed.** Save modified versions to `code_versions/`.

---

## Research-Driven Experiment Selection (STRICT — no blind sweeps)

The experiment loop is NOT a grid search. It is a research process. Every
single experiment must follow this exact sequence:

**Step 1 — Diagnose the champion's weakness.** Look at per-hospital test
results. Which OOD hospital is weakest? What's the ID/OOD gap? What does
calibration ECE look like per hospital? What's the predicted-positive
prevalence per hospital vs ground truth? Identify the SPECIFIC failure mode
(e.g. "hospital 4 AUC = 0.81 vs hospital 3 = 0.93; hospital 4 has a lower
predicted prevalence (0.34 vs 0.51 ground truth) suggesting H&E staining
shift the model reads as 'normal'").

**Step 2 — Search the literature.** Examples (image / pathology / OOD):

- Stain shift between hospitals → stain normalization (Macenko et al. 2009,
  Reinhard et al. 2001), stain augmentation (Tellez, Litjens, van der Laak,
  Ciompi 2019 MIA 'Quantifying the effects of data augmentation and stain
  color normalization in convolutional neural networks for computational
  pathology' arXiv:1902.06543), HED color jitter
- Worst-group performance → Group DRO (Sagawa, Koh, Hashimoto, Liang 2020
  ICLR 'Distributionally Robust Neural Networks' arXiv:1911.08731), IRM
  (Arjovsky, Bottou, Gulrajani, Lopez-Paz 2019 arXiv 'Invariant Risk
  Minimization' arXiv:1907.02893), V-REx (Krueger et al. 2021 ICML)
- Pathology-specific representation → UNI (Chen, Ding, Chen, Mahmood 2024
  Nature Medicine 'Towards a general-purpose foundation model for
  computational pathology' arXiv:2308.15474), GigaPath (Xu, Usuyama, Bagga
  et al. 2024 Nature 'A whole-slide foundation model for digital pathology
  from real-world data'), Phikon (Filiot, Ghermi, Olivier et al. 2023
  'Scaling Self-Supervised Learning for Histopathology with Masked Image
  Modeling' bioRxiv:10.1101/2023.07.21.549023), Virchow (Vorontsov et al.
  2024 Nature Medicine 'A foundation model for clinical-grade
  computational pathology' arXiv:2309.07778)
- Architecture ceiling hit → ConvNeXt (Liu, Mao, Wu, Feichtenhofer, Darrell,
  Xie 2022 CVPR 'A ConvNet for the 2020s' arXiv:2201.03545), Swin-V2 (Liu,
  Hu et al. 2022 CVPR 'Swin Transformer V2' arXiv:2111.09883), MaxViT
  (Tu et al. 2022 ECCV 'MaxViT: Multi-Axis Vision Transformer' arXiv:2204.01697)
- Calibration mismatch → temperature scaling (Guo, Pleiss, Sun, Weinberger
  2017 ICML 'On Calibration of Modern Neural Networks' arXiv:1706.04599),
  focal loss (Lin, Goyal, Girshick, He, Dollár 2017 ICCV 'Focal Loss for
  Dense Object Detection' arXiv:1708.02002)
- Imbalanced positive rate per hospital → class-balanced loss (Cui, Jia, Lin,
  Song, Belongie 2019 CVPR 'Class-Balanced Loss Based on Effective Number of
  Samples' arXiv:1901.05555), per-hospital reweighting

**Step 3 — Form a hypothesis and predict the outcome.** Write down: "I
hypothesize that [change X] will improve [test_ood_auc] on [hospital 4]
because [paper/principle]. I predict composite will move from [current] to
approximately [target]."

**Step 4 — Run ONE experiment.**

**Step 5 — Analyze against prediction.** Did the result match?

**Step 6 — Document everything.** Write the full cycle into the experiment
log, journal, and checkpoint.

**Step 7 — Checkpoint.** Ritual close.

**The goal is monotonic improvement.** Every experiment should have a
principled reason. If you're out of ideas for hyperparameters, the answer is
almost always a CODE CHANGE — modify the architecture, augmentation pipeline,
loss function, or feature engineering.

---

## Monotonic Quality Progression (NEVER regress)

The experiment loop must work towards monotonic increase in quality:

- **Never run an experiment you can't justify.**
- **Track the champion lineage.** Document the chain: Exp1 (ResNet50
  baseline) → Exp7 (ConvNeXt-Tiny, +ΔY) → Exp14 (DINOv2 frozen probe, +ΔW)
  → Exp23 (UNI fine-tune, +ΔV) …
- **When you hit a plateau, go deeper.** If 3+ consecutive experiments are
  DISCARD, structural change time: different backbone family, different
  augmentation regime, frozen FM vs full fine-tune.
- **Protect gains.** When trying bold changes, if composite drops > 0.05 AUC
  (5 percentage points), investigate WHY before next thing.
- **Quality ratchet:** once test_ood_auc improves, treat the new level as
  the floor. If a change improves test_ood_auc but regresses val_ood_auc
  below the previous champion, it's a DISCARD — both must improve or hold.
- **Goodhart protection (MANDATORY):** the agent MAY NOT rewrite the
  composite metric formula, the WILDS hospital split, the data integrity
  invariants, or the primary-metric definition mid-project.
  `core/evaluation/composite.py` enforces with a fingerprint hash. Changes
  require an explicit user sign-off documented as a `RULE_CHANGE` entry in
  the checkpoint.

---

## MLOps Documentation Standards (MANDATORY)

You are a strong MLOps engineer. Every artifact and every experiment must be
documented in proper, readable markdown. No exceptions.

**`autoresearch_results/experiment_summary.md`** — the master experiment log.
Updated after EVERY experiment. Format:

```markdown
## Experiment Log — [Backbone] Phase

### Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected composite change]
- **Result:** Composite [X] | Test_OOD AUC [Y] | Val_OOD AUC [Z] | ID/OOD gap [G]
- **Per-hospital test AUC:** H0=[X] H1=[X] H2=[X] H3=[X] H4=[X]
- **Secondary metrics:** Acc=[X] AUPRC=[X] ECE=[X] WorstGroupAcc=[X]
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned]
- **Per-prediction summary:** [see `trade_logs/exp<N>_predictions.csv`]
```

**`autoresearch_results/trade_logs/`** — per-experiment per-prediction detail
(one row per test patch).

**Key documentation principles:**
1. Readable by a human who wasn't there.
2. No orphan artifacts.
3. Consistent formatting. Same table format, same metric names, same
   precision (4 decimal places for AUC, 2 for percentages).
4. Append-only experiment log.

---

## Explainability & Auditability Report (MANDATORY for every NEW BEST)

When a new champion is found, produce a full data-scientist-grade audit to
`autoresearch_results/winners/<exp_id>/audit_report.md`. Not optional — a
medical-imaging model without explainability is un-deployable.

**Required sections (all 14):**

1. **Executive summary** — Champion test_ood AUC (hospital 4), val_ood AUC
   (hospital 3), id_val AUC, AUPRC, accuracy, ECE, all per-hospital metrics.
   Pass/fail per hospital (threshold 0.85 default).
2. **Feature importance via input-attribution methods** — For image
   classification, run **SmoothGrad** (Smilkov et al. 2017) and
   **Integrated Gradients** (Sundararajan, Taly, Yan 2017 ICML
   'Axiomatic Attribution for Deep Networks' arXiv:1703.01365) on 100
   random test_ood patches. Save aggregated attribution heatmaps grouped by
   correct/wrong prediction and by hospital. Save `feature_importance.csv`
   with patch-level summary statistics (mean attribution per channel, edge
   energy, color saturation). Cite: Breiman 2001 'Random Forests' for the
   permutation principle even though we use gradient-based attribution.
3. **Top-N feature analysis (per channel + per attribution-region)** —
   For the top-10 most-attributed image regions across the test set,
   show example patches with overlay heatmaps. Explain what the model is
   looking at (nuclei density, mitotic figures, stroma vs tumor boundary).
4. **SHAP-style local explanations** — For 10 random test_ood patches,
   compute per-patch attribution via Integrated Gradients with 50 steps.
   Save `shap_local.csv` with per-pixel summaries.
5. **Per-hospital feature drift** — For each hospital, compute mean/std of
   pixel statistics (RGB means, RGB stds, V channel mean, S channel std,
   edge energy via Sobel) vs the training set. Hospitals with |Z| > 2 on
   any channel indicate distribution shift. Report top 5 drift dimensions
   per OOD hospital.
6. **Calibration analysis** — Reliability diagram per hospital + ECE.
   Cite Guo et al. 2017. Plot per-hospital reliability curves with 95% CIs
   via bootstrap.
7. **Uncertainty sanity** — Aleatoric vs |error|, confidence vs
   correctness, accuracy per confidence decile, all stratified by
   hospital. Cite Kendall & Gal 2017.
8. **Per-hospital prediction distribution** — Histograms of predicted
   probabilities per hospital, with ground-truth positive rate marked.
   Identify systematic bias.
9. **Error attribution / top-N winners & losers** — For each OOD hospital,
   top-5 most-confidently-correct and top-5 most-confidently-wrong patches
   with overlay heatmaps. Pattern analysis: are errors concentrated on
   border patches, low-cellularity regions, blurry slides?
10. **Risk audit** — Per-class error rates per hospital; false-negative cost
    (missed cancer) vs false-positive cost (over-diagnosis); subgroup
    analysis if any per-slide metadata available; max-confidence-error
    failure modes.
11. **Data pipeline audit** — Reassert: zero patch overlap across folds,
    zero slide overlap across folds, hospital IDs disjoint, transforms
    deterministic on val/test. Rerun `validate_no_overlap()` and include
    output verbatim.
12. **Model config complete dump** — Every hyperparameter + Python version
    + torch version + timm version + numpy version + random seed + WILDS
    version + dataset SHA256.
13. **Known limitations & risks** — Untested scanners (Camelyon17 has 5
    hospitals; production may have 20+); untested staining protocols;
    untested patient demographics; untested magnifications.
14. **Deployment checklist** — Pathologist-in-the-loop monitoring;
    confidence-threshold for auto-flag; retraining cadence (quarterly with
    new hospitals); kill-switch criterion (AUC drop > 0.10 on monitoring
    set in 30 days).

**Implementation:** `core/winner_archive.py::generate_audit_report()`
produces the full report. Runner calls it automatically when
`composite > prev_best`.

---

## Winner Definition (CLARIFICATION)

**"Winner" means the GLOBAL champion across ALL backbones and ALL
experiments.** Not per-backbone. The one single best model (by composite) at
any point in time.

Per-backbone best is tracked separately in the checkpoint but does NOT get
archived to `winners/` unless it is also the global best.

When a new experiment beats the global composite:
1. Save artifacts to
   `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`
2. Include: README.md, config.json, model_checkpoint.pt, code/ (frozen
   snapshot), inference/, reproduction/, audit_report.md (14 sections per
   audit rules), colab_train_and_infer.ipynb
3. Update `best_config.json` at repo root.

---

## Per-Backbone Code Snapshots (MANDATORY)

Before starting experiments on a new backbone, snapshot
`core/backbones/<backbone>.py`, `core/runner.py`, `core/train.py`, and any
modified utilities to `code_versions/<backbone>_start/`.

```
code_versions/
  v1_original/
  resnet50_start/
  resnet50_final/
  convnext_tiny_start/
  ...
```

Rule: never modify code specific to backbone X while experiments on backbone Y
are in progress.

---

## Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)

**Every single experiment MUST have a complete reasoning record in
`autoresearch_results/reasoning_annotations.json` keyed by `experiment_num`.
No experiment ships without one. Orphan entries or "auto-backfilled"
placeholders are a bug.**

Required fields (all non-empty strings unless noted):

| Field | Content | Source |
|-------|---------|--------|
| `diagnosis` | Why THIS experiment now: which champion weakness, which hospital is weakest and why (scanner / stain / cohort), what prior experiments ruled out | Authored by Claude BEFORE running |
| `citations` | Full author/year/venue per Citation Rigor; multiple papers semicolon-separated | Authored before running |
| `hypothesis` | Concrete mechanism: "augmentation X / loss Y / backbone Z will change test_ood_auc via mechanism M" | Authored before running |
| `prediction` | Numeric range: "test_ood_auc from 0.84 to 0.86–0.88; val_ood_auc held; ID/OOD gap from 0.13 to 0.08–0.10" | Authored before running |
| `verdict` | KEEP / DISCARD / NEAR-MISS + composite + delta + per-hospital narrative | Written immediately after results |
| `learning` | What this updates in the mental model; which axis closed/open | Written immediately after results |
| `_manual` | `true` if Claude-authored | Always set |

**Dashboard renders all 7 fields** in the detail panel. Missing or shallow
fields are a regression — fix before next experiment.

**Write cadence — two phases per run:**
1. **BEFORE:** Claude inserts diagnosis/citations/hypothesis/prediction +
   `_manual: true`. Runner refuses to launch otherwise.
2. **AFTER:** Claude appends verdict and learning. Runner's auto-fallback
   only emits `TODO-REWRITE` sentinels.

**Enforcement:** at the start of every experiment cycle, Claude MUST check
that the previous experiment's entry has non-empty verdict/learning AND that
the next experiment's pre-entry is authored.

**Parallel write to `research_journal.md`:**

```markdown
## Exp<N> — <short title>
**Diagnosis:** ...
**Citations:** ...
**Hypothesis:** ...
**Prediction:** ...
**Verdict:** ...
**Learning:** ...
```

JSON is authoritative if they drift.

**`backfill_reasoning.py` rules:** runs only on demand, never overwrites
`_manual: true`, fills only empty fields, logs every overwrite, is NOT a
substitute for authoring annotations before the run.

**Runner's responsibility (`core/runner.py`):** merge CLI flag delta into
the runtime entry without clobbering `_manual: true`; populate verdict/
learning from results as fallback; never emit content placeholders — use
`TODO-REWRITE` + `_needs_rewrite: true` and warn.

---

## Per-Backbone N-Experiment Mandate (MANDATORY, not optional)

**Every backbone gets a full 50-experiment exploration.** Do not stop early.
Mandate:

1. **50 experiments per backbone** — no fewer. If standard HP sweeps plateau,
   explore: architectural variants from arXiv literature through 2026,
   cross-variant combinations, augmentation-pipeline changes (RandAugment,
   AugMix, stain-specific aug), feature engineering (multi-scale patch
   inputs, DINOv2 embeddings + classical head), multi-seed studies on the
   champion to characterize variance, regularization beyond weight decay
   (label smoothing, mixup, stochastic depth, DropPath).
2. **Research latest SOTA (2024-2026 arXiv papers) before declaring any
   backbone done.**
3. **Each experiment must cite its paper/source.**
4. **Document all 50 in `research_journal.md`.**
5. **Only after 50 experiments** may a backbone be declared "done" and
   progression to the next backbone resume.

---

## Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)

**Every backbone picks its OWN epochs, patience, learning rate, batch size,
scheduler, and optimizer from the latest SOTA literature for THAT
architecture.** Defaults below are starting points — iterate per the 7-step
process.

**Before the first experiment on any new backbone, Claude MUST:**

1. **Pull the latest 2024-2026 arXiv / NeurIPS / ICML / ICLR / CVPR / MICCAI
   paper** for the backbone family. Note: epochs, patience, LR + warmup,
   scheduler, optimizer, batch size, weight decay, gradient clipping, loss.
2. **Record the recipe + paper citation** in the reasoning annotation for
   Experiment 1 of that backbone.
3. **Justify the DELTA from the paper.** Different image size? Different
   precision? Different n? Per Smith 2017 scaling rule.
4. **Never assume "ep=50 works for everything."**

### Backbone-Specific Training Recipes (image-class, 2024-2026 SOTA)

The full recipe table is in `sota_catalog.yaml`. Summary:

#### Tier 1 — classical baselines (required floor for every Camelyon17 run)

| Backbone | Epochs | Patience | LR | Warmup | Scheduler | Batch | WD | Opt. | Loss | Image size | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **resnet50** | 10 | 3 | 3e-4 | 1 | cosine | 64 | 1e-4 | AdamW | BCE+pos_weight | 96 | He, Zhang, Ren, Sun 2016 CVPR 'Deep Residual Learning for Image Recognition' (arXiv:1512.03385) |
| **densenet121** | 10 | 3 | 3e-4 | 1 | cosine | 64 | 1e-4 | AdamW | BCE | 96 | Huang, Liu, van der Maaten, Weinberger 2017 CVPR 'Densely Connected Convolutional Networks' (arXiv:1608.06993) |
| **efficientnet_b0** | 15 | 4 | 3e-4 | 1 | cosine | 64 | 1e-5 | AdamW | BCE | 96 | Tan & Le 2019 ICML 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks' (arXiv:1905.11946) |

#### Tier 2 — 2024-2026 SOTA (image classification, vision foundation models)

| # | Backbone | Family | Epochs | Patience | LR | Warmup | Scheduler | Batch | WD | Opt. | Loss | Image size | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **convnext_tiny** | ConvNet (modern) | 20 | 5 | 1e-4 | 5 | cosine | 64 | 5e-2 | AdamW | BCE | 224 (upsample) | Liu, Mao, Wu, Feichtenhofer, Darrell, Xie 2022 CVPR 'A ConvNet for the 2020s' (arXiv:2201.03545) |
| 2 | **convnextv2_tiny** | ConvNet (FCMAE) | 30 | 8 | 1.5e-4 | 5 | cosine | 64 | 5e-2 | AdamW | BCE | 224 | Woo, Debnath, Hu, Chen, Liu, Kweon, Xie 2023 CVPR 'ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders' (arXiv:2301.00808) |
| 3 | **vit_small_patch16_224** | ViT | 30 | 8 | 5e-4 | 5 | cosine | 128 | 5e-2 | AdamW | BCE | 224 | Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai et al. 2021 ICLR 'An Image is Worth 16×16 Words' (arXiv:2010.11929); DeiT-III (Touvron et al. 2022 ECCV arXiv:2204.07118) |
| 4 | **swinv2_tiny** | Swin | 30 | 8 | 1e-4 | 5 | cosine | 64 | 5e-2 | AdamW | BCE | 224 | Liu, Hu, Lin, Yao, Xie, Wei, Ning, Cao, Zhang, Dong, Wei, Guo 2022 CVPR 'Swin Transformer V2: Scaling Up Capacity and Resolution' (arXiv:2111.09883) |
| 5 | **maxvit_tiny** | MaxViT (block + grid attn) | 30 | 8 | 1e-4 | 5 | cosine | 64 | 5e-2 | AdamW | BCE | 224 | Tu, Talebi, Zhang, Yang, Milanfar, Bovik, Li 2022 ECCV 'MaxViT: Multi-Axis Vision Transformer' (arXiv:2204.01697) |
| 6 | **efficientnetv2_s** | EfficientNet (modern) | 25 | 6 | 3e-4 | 3 | cosine | 64 | 1e-5 | AdamW | BCE | 224 | Tan & Le 2021 ICML 'EfficientNetV2: Smaller Models and Faster Training' (arXiv:2104.00298) |
| 7 | **mobilevitv2_175** | Hybrid CNN-ViT | 25 | 6 | 1e-3 | 3 | cosine | 64 | 5e-3 | AdamW | BCE | 224 | Mehta & Rastegari 2023 'Separable Self-attention for Mobile Vision Transformers' (arXiv:2206.02680) |
| 8 | **dinov2_vits14_probe** | Vision FM (frozen probe) | 50 | 10 | 1e-3 | 5 | cosine | 256 | 0 | AdamW | BCE | 224 (×16-multiple) | Oquab, Darcet, Moutakanni et al. 2023 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' (arXiv:2304.07193) |
| 9 | **dinov2_vitb14_lora** | Vision FM (LoRA r=8) | 30 | 8 | 5e-4 | 3 | cosine | 64 | 1e-4 | AdamW | BCE | 224 | DINOv2 paper above + Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen 2022 ICLR 'LoRA: Low-Rank Adaptation' (arXiv:2106.09685) |
| 10 | **uni_pathology_probe** | Pathology FM (UNI, frozen probe) | 50 | 10 | 1e-3 | 5 | cosine | 256 | 0 | AdamW | BCE | 224 | Chen, Ding, Chen, Mahmood et al. 2024 Nature Medicine 'Towards a general-purpose foundation model for computational pathology' (arXiv:2308.15474) — DINOv2-pretrained on 100k WSI |
| 11 | **gigapath_pathology_probe** | Pathology FM (Prov-GigaPath, frozen tile encoder) | 50 | 10 | 1e-3 | 5 | cosine | 256 | 0 | AdamW | BCE | 224 | Xu, Usuyama, Bagga, Zhang, Naumann et al. 2024 Nature 'A whole-slide foundation model for digital pathology from real-world data' — 1.3 B params, 1.4 M WSI |
| 12 | **phikon_pathology_probe** | Pathology FM (iBOT-pretrained ViT-B) | 50 | 10 | 1e-3 | 5 | cosine | 256 | 0 | AdamW | BCE | 224 | Filiot, Ghermi, Olivier et al. 2023 'Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling' (bioRxiv:10.1101/2023.07.21.549023) |
| 13 | **virchow_pathology_probe** | Pathology FM (DINOv2 ViT-H/14) | 50 | 10 | 1e-3 | 5 | cosine | 128 | 0 | AdamW | BCE | 224 | Vorontsov, Bozkurt, Casson et al. 2024 Nature Medicine 'A foundation model for clinical-grade computational pathology' (arXiv:2309.07778) |
| 14 | **ctranspath** | Pathology Swin (mocov3) | 30 | 8 | 5e-4 | 3 | cosine | 64 | 1e-4 | AdamW | BCE | 224 | Wang, Yang, Zhao, Liu, Ouyang, Wang, Shen, Wang, Li, Han 2022 MIA 'Transformer-based unsupervised contrastive learning for histopathological image classification' (arXiv:2205.09048) |

#### Tier 3 — gradient boosted machines on frozen-FM embeddings (each is its OWN backbone)

GBMs are fundamentally different from neural nets: no epochs, no LR schedule,
no batch. Iterations are tree-count. Each GBM has its own paper, its own
hyperparameter language, its own 50-experiment exploration budget. **Do NOT
bundle xgboost/lightgbm/catboost as "the GBM backbone" — they are three
separate architectures.**

For Camelyon17 we run GBMs on **frozen DINOv2-S/14 embeddings** (384-dim per
patch) — this gives a strong tabular-style baseline competitive with full
fine-tuning at a tiny fraction of the compute.

| Backbone | Key HP | Default Start | Regularization | Special feature | Paper |
|---|---|---|---|---|---|
| **xgboost_dinov2_emb** | n_estimators=1500, max_depth=6, lr=0.03, subsample=0.8, colsample_bytree=0.8, early_stop=50 | level-wise trees | reg_lambda=1.0, reg_alpha=0, min_child_weight=1, gamma=0 | 2nd-order Newton boosting | Chen & Guestrin 2016 KDD 'XGBoost' (arXiv:1603.02754) |
| **lightgbm_dinov2_emb** | n_estimators=2000, num_leaves=63, lr=0.03, feature_fraction=0.8, bagging_fraction=0.8, early_stop=50 | leaf-wise trees (GOSS) | reg_alpha, reg_lambda, min_data_in_leaf=20 | GOSS; EFB | Ke, Meng, Finley et al. 2017 NeurIPS 'LightGBM' |
| **catboost_dinov2_emb** | iterations=2000, depth=6, lr=0.03, random_strength=1.0, early_stop=100 | symmetric oblivious | l2_leaf_reg=3, bagging_temperature=1.0 | Ordered boosting | Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin 2018 NeurIPS 'CatBoost' (arXiv:1706.09516) |

**Re-derive for EVERY new variant.** New pathology FMs are released ~quarterly
(GigaPath-2, UNI-2, Virchow-2 expected through 2026); each variant
re-derives its recipe from its OWN paper.

---

## GPU Memory Constraint (MANDATORY — 16 GB VRAM hard cap)

**This laptop has 16 GB of GPU VRAM.** Every backbone selection, every
experiment, every fine-tuning run MUST fit within this budget with headroom.

**Memory budget breakdown (16 GB total):**

| Component | Budget | Notes |
|-----------|--------|-------|
| Model parameters | ≤ 3 GB | FP32 weights; BF16 halves this |
| Optimizer state (AdamW) | ≤ 6 GB | 2 moments at FP32 → ≈ 2× param size |
| Gradients | ≤ 3 GB | Same size as params; freed after step |
| Activations | ≤ 3 GB | batch × C × H × W × depth, scales with bs and depth |
| Reserved / fragmentation | ≥ 1 GB | PyTorch caching allocator overhead |

**Practical parameter ceilings by training mode:**

| Training mode | Max params @ FP32 | Max params @ BF16 | Max params w/ grad-ckpt + BF16 |
|---|---|---|---|
| From-scratch / full FT (Adam states) | ~500 M | ~1.0 B | ~2.0 B |
| LoRA (r=8) | ~1.0 B | ~3.0 B | ~5.0 B |
| Frozen-backbone head probe | ~1.5 B | ~4.0 B | ~7.0 B |
| Inference only | ~4.0 B | ~8.0 B | ~8.0 B |

**Mandatory pre-flight check for any new backbone.** Before launching
Experiment 1, include in the reasoning annotation:

```
Measured/estimated size: N million params
Training mode selected: [from-scratch | LoRA | head-probe | zero-shot]
Image resolution: <H×W>
Expected peak VRAM: <X> GB at bs=<Y>, precision=<FP32|BF16>
Headroom vs 16 GB: <16 - X> GB
Fallback plan if OOM: [bs/2 | BF16 | grad-ckpt | LoRA-only | smaller variant]
```

Without this entry, Experiment 1 does not launch.

**Size-class annotations (for first-experiment reasoning):**

| Backbone | Approx params | Mode fit in 16 GB |
|---|---|---|
| resnet50 | 25 M | full FT FP32 trivially fits at bs=64 |
| densenet121 | 8 M | trivial |
| efficientnet_b0 | 5 M | trivial |
| convnext_tiny | 28 M | full FT fits at bs=64 BF16 |
| convnextv2_tiny | 28 M | same |
| vit_small_patch16_224 | 22 M | full FT at bs=128 BF16 |
| swinv2_tiny | 28 M | bs=64 BF16 |
| maxvit_tiny | 31 M | bs=64 BF16 |
| efficientnetv2_s | 22 M | bs=64 |
| mobilevitv2_175 | 14 M | trivial |
| dinov2_vits14 (probe) | 22 M (frozen) | head-probe trivial; LoRA trivial |
| dinov2_vitb14 | 86 M | LoRA trivially fits; full FT at bs=32 BF16 |
| uni (ViT-L/16) | 304 M | head-probe fits; LoRA fits at bs=32 BF16; full FT requires grad-ckpt |
| gigapath (ViT-G/14) | 1.13 B | head-probe ONLY at our 16 GB (LoRA risky) |
| phikon (ViT-B/16) | 86 M | LoRA fits; head-probe trivial |
| virchow (ViT-H/14) | 632 M | head-probe at bs=64 BF16; LoRA risky |
| ctranspath (Swin-T) | 28 M | full FT |

**Default protocol when adopting a new pathology FM:**
1. Start with a frozen-backbone linear/MLP probe.
2. Run with the default DINOv2-style preprocessing (224×224 BICUBIC, ImageNet
   normalize unless the paper specifies otherwise).
3. If linear probe is promising, add LoRA r=8 on attention QKV projections.
4. Scale to full fine-tune ONLY if smaller modes show signal AND memory math
   works.

**BF16 default** for any backbone > 50 M params being fine-tuned. Use
`torch.autocast(dtype=torch.bfloat16)`. LayerNorm/GroupNorm stay FP32.

**Gradient checkpointing** for fine-tuning models > 200 M params (UNI,
Virchow, GigaPath).

### Epoch-budget rule of thumb (when in doubt)

If the paper's recipe is unclear:

- **Data scaling (Smith 2017):** `epochs ≈ paper_epochs × (paper_n / our_n)^0.5`.
  Camelyon17 train n ≈ 302k; if a paper used n=1.3 M, scale paper_epochs ×
  0.48.
- **Parameter scaling (Kaplan 2020):** larger models need more epochs for
  fixed data: `epochs ≈ base × (our_params / paper_params)^0.2`.
- **Patience as 15% of epochs** when papers don't specify.
- **Warmup = 5–10% of total epochs** for transformer families.

---

## Backbone Isolation Rule

Before starting experiments on a new backbone, snapshot
`core/backbones/<backbone>.py`, `core/runner.py`, `core/train.py` to
`code_versions/<backbone>_start/`. Do NOT modify backbone code specific to
backbone X while experiments on backbone Y are in progress.

---

## Dashboard Backbone Tabs

Dashboard (`autoresearch_results/dashboard.html`) renders a backbone tab bar
above the experiment list. Default view "ALL". Tabs filter the scrollable
experiment list to that backbone's experiments. Click to switch. Camelyon17
adds a per-hospital sub-bar (H0–H4) for granular analysis.

---

## Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)

**Every single experiment updates ALL the following files. If any file is
stale after an experiment completes, that's a regression — stop and fix
before moving on. No "I'll batch-update at the end." No "It's just a variance
check."**

**Ownership — who writes what:**

| File | Written by | When | Content |
|---|---|---|---|
| `autoresearch_results/experiment_log.jsonl` | runner (auto) | every run | composite, test_ood AUC, val_ood AUC, id_val AUC, per-hospital AUC, per-hospital prevalence, ECE, accuracy, AUPRC, worst-group acc, uncertainty, timing, config |
| `autoresearch_results/best_config.json` | runner (auto) | only on new champion | full champion entry |
| `autoresearch_results/best_model.pt` | runner (auto) | only on new champion | weights + transforms_state + config + provenance |
| `autoresearch_results/trade_logs/exp<N>_predictions.csv` | runner (auto) | every run | one row per test patch (patch_id, slide_id, hospital, label, prob, correct, confidence, aleatoric, epistemic) |
| `autoresearch_results/trade_logs/exp<N>_prediction_summary.json` | runner (auto) | every run | per-hospital totals, AUC, AUPRC, accuracy, ECE, conf-stratified accuracy |
| `autoresearch_results/reasoning_annotations.json` | Claude BEFORE + runner AFTER | every run | 7 fields, two-phase write |
| `autoresearch_results/research_journal.md` | Claude | every run, appended | 7-field markdown |
| `autoresearch_results/experiment_summary.md` | Claude | every run, appended | tabular row |
| `memory/project_autoresearch_checkpoint.md` | Claude | every run | champion + history + next-command |
| `autoresearch_results/winners/<backbone>_exp<N>_<desc>/*` | Claude + runner | only on new champion | full archive |
| `autoresearch_results/winners/<...>/audit_report.md` | Claude (via `winner_archive.py`) | only on new champion | 14-section audit |
| `autoresearch_results/winners/<...>/colab_train_and_infer.ipynb` | Claude (via `winner_archive.py`) | only on new champion | self-contained Colab |
| `autoresearch_results/dashboard.html` | Claude (rarely) | only when adding a metric/tab | static |

**Per-experiment ritual (in order, every run):**

1. **Before launch:** insert the next experiment's reasoning entry; runner
   refuses to launch otherwise.
2. **Before launch:** append matching section to `research_journal.md`.
3. **Launch.**
4. **Runner auto-updates:** JSONL, best_config (if champion), best_model
   (if champion), trade_logs CSV + JSON, reasoning fallback fields.
5. **After completion:** Claude rewrites verdict + learning richly.
6. **After completion:** append row to `experiment_summary.md`.
7. **After completion:** update checkpoint.
8. **If new champion:** archive to `winners/...` (full set + audit + Colab).

**Verification at the start of every experiment cycle:**

Before launching Experiment N+1, confirm for Experiment N:

- [ ] `experiment_log.jsonl` has an entry for N
- [ ] `reasoning_annotations.json[N]` has all 7 fields non-empty/non-placeholder
- [ ] `research_journal.md` has a section for N
- [ ] `experiment_summary.md` has a row for N
- [ ] `memory/project_autoresearch_checkpoint.md` references N in its history table
- [ ] `trade_logs/expN_predictions.csv` and `expN_prediction_summary.json` exist
- [ ] If N set a new champion: `winners/<backbone>_expN_<desc>/` exists with all required files

If ANY checkbox is unchecked, stop and fix BEFORE launching N+1.

**Placeholder strings are a bug.** The runner refuses to fabricate pre-run
content. If a pre-run entry is missing, runner inserts `"TODO-REWRITE"` and
`_needs_rewrite: true`.

---

## Citation Rigor (MANDATORY format for `citations` field)

**Every citation string MUST contain, for every paper referenced:**

1. **All authors' surnames** (not just first-author et al. unless > 6 authors)
2. **Year** of publication
3. **Venue** — journal name, conference abbreviation (NeurIPS, ICML, ICLR,
   AAAI, CVPR, ECCV, ICCV, MICCAI, KDD, etc.), or `arXiv` if preprint-only
4. **Full paper title** in single quotes
5. **arXiv ID** in the form `(arXiv:XXXX.YYYYY)` if available — mandatory for
   any paper posted to arXiv
6. **One-sentence relevance note** — why this paper motivates THIS experiment

**Format template:**

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) — one-sentence note on why we cite it here.
```

**Examples of GOOD citations (copy this style):**

> Sagawa, Koh, Hashimoto, Liang 2020 ICLR 'Distributionally Robust Neural
> Networks for Group Shifts' (arXiv:1911.08731) — motivates the worst-group
> training loss to lift hospital-4 AUC.

> Tellez, Litjens, Bándi, Bulten, Bokhorst, Ciompi, van der Laak 2019 MIA
> 'Quantifying the effects of data augmentation and stain color
> normalization in convolutional neural networks for computational
> pathology' (arXiv:1902.06543) — proves stain-augmentation > stain-norm for
> cross-hospital generalization.

> Chen, Ding, Chen, Mahmood et al. 2024 Nature Medicine 'Towards a
> general-purpose foundation model for computational pathology'
> (arXiv:2308.15474) — UNI is DINOv2-pretrained on 100k WSI; expected to
> dominate on Camelyon17 OOD vs ImageNet-pretrained ViT.

**Examples of BAD citations (REJECTED — rewrite required):**

- `"DRO paper"` — missing all 6 elements
- `"(Sagawa2020)"` — parenthetical tag only
- `"see research_journal.md"` — redirects instead of citing

**Arxiv ID lookup discipline.** Fetch via WebSearch / WebFetch on arxiv.org
when missing.

---

## Reasoning Blob Completeness (what "full reasoning" means)

| Field | Minimum content | Word count floor | Must include |
|-------|-----------------|------------------|--------------|
| `diagnosis` | Why THIS now; which hospital weakest; what prior experiments ruled out | ≥ 60 words | Reference to ≥ 1 prior experiment OR per-hospital metric |
| `citations` | Per Citation Rigor | ≥ 40 words single, ≥ 80 multi-paper | Author + year + venue + title + arXiv + relevance |
| `hypothesis` | Mechanism | ≥ 50 words | "mechanism" / "because" / "per [paper]"; specific parameter and value |
| `prediction` | Numeric range on composite + ≥ 1 sub-metric | ≥ 25 words | Range + direction |
| `verdict` | KEEP/DISCARD/NEAR-MISS + composite + delta + per-hospital | ≥ 30 words | Status + composite (4 dec) + ≥ 1 per-hospital |
| `learning` | What's now closed/open; what next | ≥ 40 words | "Axis closed/open" OR "next try: ..." |
| `_manual` | Boolean | — | `true` for Claude-authored |

**Variance-check batches** can share `diagnosis`/`citations` templates but
verdict/learning is per-run-specific.

**Batch updates are forbidden.** Never do 5 experiments then update
journal/summary/checkpoint in one go.

---

## Loss Function Rules

**Default for Camelyon17 (binary classification, ~50% positive rate per
training hospital, but skewed in val_ood/test_ood):**

`BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` computed on the
**training set only** (do not use OOD val for class weighting; it leaks).

Optional alternatives (cite paper when used):

- **Focal loss** (Lin, Goyal, Girshick, He, Dollár 2017 ICCV 'Focal Loss for
  Dense Object Detection' arXiv:1708.02002) — when hospital-4 prevalence is
  very different from training and rare-class recall matters.
- **Class-balanced loss** (Cui, Jia, Lin, Song, Belongie 2019 CVPR
  'Class-Balanced Loss Based on Effective Number of Samples'
  arXiv:1901.05555) — for severe per-hospital imbalance.
- **Group DRO** (Sagawa et al. 2020) — explicitly minimize the worst-group
  loss across hospitals.
- **IRM** (Arjovsky, Bottou, Gulrajani, Lopez-Paz 2019 arXiv 'Invariant Risk
  Minimization' arXiv:1907.02893) — penalty term encouraging
  hospital-invariant representations.
- **V-REx** (Krueger, Caballero, Jacobsen, Zhang, Binas, Zhang, Le Priol,
  Courville 2021 ICML 'Out-of-Distribution Generalization via Risk
  Extrapolation' arXiv:2003.00688) — variance penalty across hospitals.

### Heteroscedastic / uncertainty-aware loss (optional)

For neural binary classification with aleatoric uncertainty: model outputs
logit + log-variance per patch; loss = `BCE_with_uncertainty(logit, y, sigma)`
following Kendall & Gal 2017.

- **Variance-branch dominance** is the #1 failure mode — clamp log_var to
  [-5, 2].
- **Het-loss needs ~50% more epochs** than plain BCE to converge.
- **Monitor uncertainty per hospital:** high aleatoric on H4 = noisy
  staining; high epistemic = need more H4-like data (or augmentation that
  simulates H4-like stain).

Cite: Kendall & Gal 2017 NeurIPS 'What Uncertainties Do We Need in Bayesian
Deep Learning for Computer Vision?' (arXiv:1703.04977).

---

## Winner Archiving Protocol (MANDATORY for every NEW BEST)

Every time a new champion is found, archive ALL artifacts to a self-contained
subdirectory.

**Directory structure:**
`autoresearch_results/winners/<backbone>_exp<N>_<short_description>/`

```
winners/
  convnext_tiny_exp14_stain_aug_seed0/
    README.md
    config.json
    model_checkpoint.pt
    experiment_log_entry.json
    per_fold_results.json
    code/
      backbones/
      evaluation/
      runner.py
      train.py
      reasoning.py
      checkpoint.py
      winner_archive.py
    inference/
      predict.py
      README_inference.md
    reproduction/
      reproduce_log.txt
      seed_variance.json
    audit_report.md
    colab_train_and_infer.ipynb
```

**README.md template for each winner:**
- Model name + experiment number
- Champion composite, test_ood AUC, val_ood AUC, id_val AUC
- Per-hospital test AUC table (H0–H4)
- Per-hospital val AUC table
- Full hyperparameter config
- Architecture description (backbone, classifier head, augmentation pipeline)
- Key insight: WHY this config won
- Training details: epochs run, early stopping epoch, training time, peak VRAM
- Uncertainty metrics per hospital: aleatoric, epistemic, confidence
- Camelyon17-specific secondary metrics: per-hospital accuracy, AUPRC, ECE,
  worst-group accuracy, ID/OOD gap
- Reproduction status: seeds tested, variance observed
- Sample inference code snippet
- **Deployment Strategy** section (see below)

**After archiving:** Rerun the winner. Reproduction log to
`reproduction/reproduce_log.txt`. If composite differs by > 0.02 AUC, flag.

**Model checkpoint MUST be portable:**

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config_dict,
    'transforms_state': {'mean': [...], 'std': [...], 'image_size': 224, 'normalize_kind': 'imagenet'|'pathology'},
    'feature_extractor_id': '<backbone_name>',
    'composite': float, 'test_ood_auc': float, 'val_ood_auc': float,
    'id_val_auc': float, 'description': str,
    'backbone': str, 'experiment_num': int,
    'wilds_split_version': 'camelyon17_v1.0',
}, 'model_checkpoint.pt')
```

**The `predict.py` inference script must:**
1. Load the model checkpoint.
2. Accept a directory of image patches OR a single image.
3. Output: prob_positive, prediction, confidence, aleatoric, epistemic.
4. Include a `__main__` block with a working example using a bundled patch.
5. Print results in a clear table.

**Deployment Strategy section (MANDATORY in every winner README.md):**

For Camelyon17 / pathology binary classification:

1. **Signal generation** — patch-level prob_positive; slide-level via
   max-pool / top-k mean / MIL aggregation
2. **Decision rules** — threshold at p* (calibrated on id_val), with
   pathologist-in-the-loop review for p ∈ [p*-0.05, p*+0.05]
3. **Resource sizing** — inference ~2 ms/patch on RTX 4060 Mobile (16 GB)
   batched at 256
4. **Refresh / retraining cadence** — quarterly with new hospital data;
   continual-learning protocol if a 5th hospital is added
5. **Per-hospital performance table** — AUC + accuracy + ECE per hospital
6. **Risk controls** — monitoring AUC drift on a daily monitoring set; alert
   if 7-day rolling AUC drops > 0.05; kill-switch if > 0.10
7. **Expected performance** — point estimate + 95% bootstrap CI on
   test_ood_auc
8. **Caveats** — staining batch sensitivity; scanner sensitivity;
   magnification sensitivity; patient cohort coverage; class prior shift
9. **Reference to inference code** — `inference/predict.py`

---

## Google Colab Notebook (MANDATORY for every winner)

For every archived winner, generate a self-contained Colab notebook at
`autoresearch_results/winners/<...>/colab_train_and_infer.ipynb`.

**Required cells:**

1. **Setup** — `!pip install wilds timm torch transformers torchvision`
2. **Data download** — `wilds.get_dataset("camelyon17", download=True,
   root_dir="./data")` (or upload a 1k-patch sample for fast demo)
3. **Feature engineering** — define transforms (resize 96→224, ImageNet/
   pathology normalize, optional stain augmentation)
4. **Training** — full loop reproducing the winner config exactly. Print
   per-epoch loss + id_val AUC + (cheap) hospital-3 sub-AUC.
5. **Evaluation** — full evaluation on test_ood (hospital 4), per-hospital
   AUC table, composite computation
6. **Inference** — load checkpoint, score a sample patch, return p, conf,
   aleatoric, epistemic
7. **Visualization** — reliability diagram per hospital + ROC per hospital
   + 5 example patches with attribution overlay
8. **Export** — save final model weights for download

**Notebook principles:**
- Markdown header per cell.
- Champion config dict at the top.
- `torch.manual_seed()` + `np.random.seed()`.
- Summary metric table at the bottom.
- Target runtime: < 5 minutes on Colab T4 (uses a 1k-patch demo subset).
- Self-contained — no imports from `core.*`.

---

## Traditional ML Metrics (MANDATORY for every experiment)

In addition to the primary composite metric, compute and log:

**For Camelyon17 (binary classification with hospital groups):**

- **Per-hospital AUROC** — primary
- **Per-hospital AUPRC** — robust under class imbalance
- **Per-hospital accuracy** at the calibrated threshold
- **Per-hospital ECE** (10-bin reliability)
- **Worst-group accuracy** (`acc_wg`) — WILDS official metric
- **Average accuracy** (`acc_avg`) — WILDS official metric
- **Per-hospital confusion matrix** (TP, FP, TN, FN)
- **Per-hospital precision / recall / F1**
- **MCC** per hospital (balanced under imbalance)
- **Predicted-positive prevalence** per hospital — diagnostic of
  prior-shift handling
- **ID/OOD AUC gap** — `id_val_auc - mean(val_ood_auc, test_ood_auc)`

These must appear in:
1. `core/evaluation/metrics.py::full_report()` output
2. Per-hospital fields in JSONL log entries
3. Dashboard per-hospital tables
4. Winner archive `per_fold_results.json`
5. Experiment summary markdown

---

## Per-Prediction Log (MANDATORY for every experiment)

For EVERY experiment, produce a per-patch prediction CSV.

**Output file:** `autoresearch_results/trade_logs/exp<N>_predictions.csv`

**Columns (one row per test patch):**

| Column | Description |
|---|---|
| patch_id | WILDS patch index |
| slide_id | WILDS slide identifier |
| hospital | Hospital ID (0-4) |
| fold | id_val / val_ood / test_ood |
| label | Ground truth (0=normal, 1=tumor) |
| prob_positive | Model output P(tumor) |
| pred_label | argmax / threshold(prob, t*) |
| correct | 1 iff pred_label == label |
| confidence | 1 - epistemic |
| aleatoric | Aleatoric uncertainty (only if het-loss; else 0) |
| epistemic | Epistemic uncertainty (MC-Dropout 20-pass std) |
| brier | (prob - label)^2 — per-row Brier contribution |

**Per-hospital summary in `exp<N>_prediction_summary.json`:**

- Total patches, correct, wrong per hospital
- AUC, AUPRC, accuracy, ECE per hospital
- Per-hospital confusion matrix (TP/FP/TN/FN)
- Largest single confidently-correct, largest single confidently-wrong
- Streak: max consecutive correct, max consecutive wrong (sorted by patch_id)
- Confidence-stratified accuracy: > 0.9 vs < 0.9 per hospital
- ID/OOD AUC gap

**This data enables:** identifying specific patches/slides where the model
fails; confidence calibration analysis; threshold tuning per hospital;
slide-level MIL aggregation research.

---

## Architecture

- **Autoresearch loop = Claude agent.** Claude reads results, decides what
  to try, calls the runner, reads output. The intelligence is in the agent,
  NOT in Python code. No pre-baked experiment lists.
- Runner (`core/runner.py`) executes ONE experiment per call. Logs JSONL.
- Dashboard (`autoresearch_results/dashboard.html`) reads logs.
- Save checkpoint after every experiment (JSONL append + best_config.json
  overwrite).
- Use relative imports (`from .backbones import ...`).

---

## Validation Checklist (Run Before Every Experiment Session)

1. **Triple-check data split audit (MANDATORY — see "Triple-Check Data Split
   Audit" section above):**
   ```
   "C:/Users/evija/anaconda3/python.exe" -m core.evaluation.audit \
       --config configs/camelyon17.yaml --triple-check
   ```
   All 7 auditors must report `status: PASS`. Report at
   `autoresearch_results/data_split_audit.md`.
2. `core.evaluation.audit.audit_hospital_level()` passes — hospitals 0,1,2
   in train/id_val only; H3 in val_ood only; H4 in test_ood only.
3. `core.evaluation.audit.audit_slide_level()` passes — pairwise slide_id
   intersections across folds are empty.
4. `core.evaluation.audit.audit_patch_level()` passes — pairwise patch_id
   intersections across folds are empty.
5. `core.evaluation.audit.audit_class_balance()` passes — both classes
   present in every fold; positive prevalence ∈ [0.05, 0.95].
6. `core.evaluation.audit.audit_size_floors()` passes — fold sizes meet
   minimums for the active `data_mode` (sim or wilds).
7. `core.evaluation.audit.audit_reproducibility()` passes — second audit
   pass with the same seed yields the same fingerprint.
8. `core.evaluation.audit.audit_no_leakage_via_metadata()` passes — model
   inputs do not include hospital_id, slide_id, or any leak-prone metadata.
9. WILDS-mode count check (only when `data_mode=wilds`):
   train ≈ 302,436, id_val ≈ 33,560, val_ood ≈ 34,904, test_ood ≈ 85,054
   (numbers from WILDS v1.0 official splits; tolerance ±0.5%).
10. Data loaded from `.data_cache/camelyon17_v1.0/` (not re-downloaded).

---

## Project Structure

```
AUTORESEARCHIMAGE/
  CLAUDE.md                      # this file
  README.md
  ARCHITECTURE.md
  pyproject.toml
  configs/
    camelyon17.yaml              # default project config
    splits.yaml                  # WILDS hospital split params
  sota_catalog.yaml              # 14 image backbones + 3 GBM-on-embedding
  core/
    __init__.py
    runner.py                    # one experiment per invocation
    train.py                     # image-aware training loop
    reasoning.py                 # Citation Rigor + Reasoning Blob validators
    checkpoint.py
    winner_archive.py
    backbones/
      base.py
      registry.py
      timm_backbone.py           # generic timm-loaded backbones
      pathology_fm.py            # UNI / GigaPath / Phikon / Virchow / CTransPath
      gbm_on_embedding.py        # XGBoost / LightGBM / CatBoost on DINOv2 embeddings
      sim_backbone.py            # tiny synthetic backbone for smoke tests
    evaluation/
      splits.py                  # WILDSCamelyon17Split + validate_no_overlap()
      metrics.py                 # AUC, AUPRC, ECE, per-hospital, composite
      composite.py               # Goodhart-fingerprinted composite
      uncertainty.py             # MC Dropout + deep ensembles
    data/
      loader.py                  # WILDSDataset wrappers + DataLoader factories
      transforms.py              # ImageNet / pathology / stain-aug pipelines
      stain_norm.py              # Macenko / Reinhard implementations
  autoresearch_results/
    experiment_log.jsonl
    best_config.json
    best_model.pt
    dashboard.html
    experiment_summary.md
    research_journal.md
    reasoning_annotations.json
    trade_logs/
      exp<N>_predictions.csv
      exp<N>_prediction_summary.json
    winners/
      <backbone>_exp<N>_<desc>/
  memory/
    project_autoresearch_checkpoint.md
    project_hardware_log.md
  code_versions/
    <backbone>_start/
    <backbone>_final/
  dashboard/
    dashboard.html               # source; copied into autoresearch_results/
  tests/
    test_smoke.py
    test_section_coverage.py
  examples/
    quickstart_sim.py            # uses sim_backbone for sub-minute smoke
  .data_cache/                   # WILDS Camelyon17 (gitignored, ~10 GB)
```

---

## Key Constants

| Constant | Value | Location |
|---|---|---|
| PRIMARY_METRIC | test_ood_auc | `core/evaluation/composite.py` |
| COMPOSITE_FORMULA | `min(test_ood_auc, val_ood_auc) - 0.1 * id_ood_gap` | `core/evaluation/composite.py` |
| SPLIT_PROTOCOL | WILDS-Camelyon17 hospital split | `configs/splits.yaml` |
| N_FOLDS_OR_GROUPS | 4 (id_val, val_ood, test_ood, test_id_aux) | `configs/splits.yaml` |
| IMAGE_SIZE_DEFAULT | 96 (raw); 224 (upsampled for ViT/DINOv2/UNI) | `core/data/transforms.py` |
| BATCH_SIZE_DEFAULT | 64 (96px) / 32 (224px FT) / 256 (224px frozen probe) | `core/runner.py` |
| EPOCHS_DEFAULT | varies per backbone (see SOTA catalog) | `core/runner.py` |
| PATIENCE_DEFAULT | 15% of epochs | `core/runner.py` |
| LR_DEFAULT | varies per backbone | `core/runner.py` |
| WEIGHT_DECAY_DEFAULT | 1e-4 (CNNs) / 5e-2 (modern transformers) | `core/runner.py` |
| GPU_VRAM_GB | 16 | `core/runner.py` (pre-flight) |
| CPU_RUNNER_AFFINITY | [0, 2, 4, 6] | `core/runner.py::_pin_to_safe_cores()` |
| BANNED_CORES | [16, 17, 24, 25] | hardware policy |
| N_EXPERIMENTS_PER_BACKBONE | 50 | this file |
| MC_DROPOUT_PASSES | 20 | `core/evaluation/uncertainty.py` |
| WILDS_VERSION | 1.0 | `configs/camelyon17.yaml` |

---

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---|---|---|
| Computing stain-normalization stats including val/test hospitals | Leakage; inflates OOD AUC | Compute Macenko/Reinhard stats from training hospitals only |
| Using val_ood (hospital 3) for HP selection | OOD claim corrupted; you've fit to "OOD" | HP selection on id_val only; report val_ood as secondary OOD reading |
| Random-cropping a 96-px patch to < 96 | Loses central lesion; AUC tanks | Crop only on upsampled 224×224, then `RandomResizedCrop(224, scale=(0.7,1.0))` |
| Strong color jitter without HED-stain awareness | Over-distorts H&E; hurts AUC | Use Tellez 2019 HED-stain-shift augmentation OR conservative ImageNet jitter |
| Loading WILDS each run | Minutes wasted | Default `cache_dir=.data_cache/`; load once at runner init |
| Grid sweep on LR | Wasted GPU; uninformed | One change per experiment, diagnose first |
| Running all 4 folds independently | 4× slower for the same metric | One forward pass per checkpoint produces all 4 fold metrics |
| Absolute imports in package | `ModuleNotFoundError` when run as `-m` | Always `from .module import ...` |
| Assuming throughput | Wrong epoch budget | Measure with `time.time()`, log it |
| Monolithic scripts | Can't debug, can't reuse | Runners log. Dashboard reads. Decoupled. |
| Bundling xgboost+lightgbm+catboost | Tier-3 rule violated | Three separate registry entries, three separate budgets |
| Skipping the per-backbone VRAM pre-flight | OOM at experiment 1 | Runner requires the VRAM block in the reasoning annotation |
| Bs change without re-measuring VRAM | OOM mid-cycle | Pre-flight on every bs change |
| Variance check claimed as "champion" | Single-seed luck | 3-seed median check before declaring champion |
| Reporting only test_ood AUC | Misleading; ignores val_ood collapse | Always report `min(test_ood, val_ood)` and the ID/OOD gap |
| Treating image size as a free parameter | Wastes compute on incompatible models | DINOv2 / UNI / GigaPath need 224 (multiple of 14); resnet50 / efficientnet_b0 work at 96; document per-backbone in SOTA catalog |
| Stain-augmentation in val/test transforms | Eval becomes stochastic | Augmentation only in train transform; val/test deterministic |
| Forgetting `model.eval()` for MC Dropout | Aleatoric is wrong; epistemic collapses | Use `enable_dropout()` helper that keeps `eval()` for BN but enables Dropout layers |
| Computing AUC without removing NaN preds | Crashes or undefined | `core.evaluation.metrics.safe_auc()` handles missing |
| Rewriting composite formula mid-project | Goodhart violation | Frozen fingerprint enforced by `composite.py`; require RULE_CHANGE entry |
| Silently dropping a CLAUDE.md section | Rules drift | All 52 sections preserved per `tests/test_section_coverage.py` |

---

## Session Learnings

_Append-only. New session insights go at the bottom, date-stamped._

### Initial setup (2026-04-24)

- **Project created:** 2026-04-24
- **Task type:** binary classification (image, OOD)
- **Primary metric:** test_ood_auc (hospital 4)
- **Composite:** `min(test_ood_auc, val_ood_auc) - 0.1 * id_ood_gap`
- **Split protocol:** WILDS-Camelyon17 hospital split (frozen)
- **Backbones in scope (priority order):** resnet50, convnext_tiny,
  vit_small_patch16_224, swinv2_tiny, efficientnetv2_s, maxvit_tiny,
  mobilevitv2_175, dinov2_vits14_probe, dinov2_vitb14_lora, uni_pathology_probe,
  gigapath_pathology_probe, phikon_pathology_probe, virchow_pathology_probe,
  ctranspath, xgboost_dinov2_emb, lightgbm_dinov2_emb, catboost_dinov2_emb
- **Hardware:** 16 GB VRAM, P-cores 0–15 (E-cores 16,17,24,25 banned)
- **Inherited rules from FX project:** all 52 CLAUDE.md sections preserved.
  Goodhart-protected composite. Citation Rigor + Reasoning Blob Completeness
  word-count floors. Tier-3 GBM-as-three-backbones rule. Winner archive +
  14-section audit + Colab.

### Headroom hypothesis (pre-experiment, to be tested)

- WILDS leaderboard SOTA on Camelyon17 OOD AUC: ~0.92 (ERM-baseline ~0.86,
  GroupDRO ~0.85, IRM ~0.84). Pathology FMs (UNI, GigaPath, Virchow) reported
  on related Camelyon16/CAMELYON17-MIL settings achieve ~0.94–0.97 — strong
  reason to believe Tier-2 pathology FMs will dominate.
- Expected best result this campaign: composite ≈ 0.92–0.95 with
  uni_pathology_probe / gigapath_pathology_probe + stain augmentation +
  GroupDRO loss. Champion lineage hypothesis:
  - Exp1 (resnet50 baseline) → composite ≈ 0.84
  - Exp7 (convnext_tiny + ImageNet aug) → composite ≈ 0.87
  - Exp14 (dinov2_vits14_probe) → composite ≈ 0.89
  - Exp23 (uni_pathology_probe + Tellez stain aug) → composite ≈ 0.92
  - Exp35 (gigapath_pathology_probe + GroupDRO) → composite ≈ 0.94+

This is a **prediction**, not a plan. Each step must be earned by per-fold
diagnosis + literature cite + hypothesis + measured outcome.

---

## Cross-references

| Document | Purpose |
|---|---|
| `C:/Users/evija/autoresearch/CLAUDE.md` | The original FX-project CLAUDE.md. Source of truth for every section. |
| `C:/Users/evija/autoresearch/generalized_ml_autoresearch/CLAUDE.md` | Meta-framework rules for the framework itself. |
| `C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/CLAUDE_template.md` | Parameterized template this file is filled from. |
| `C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/SECTION_MAPPING.md` | Audit log proving 52/52 source sections preserved. |
| `sota_catalog.yaml` (this project) | 14 image backbones + 3 GBM-on-embedding entries. |
| `ARCHITECTURE.md` (this project) | Order-of-build, module I/O, invariants. |
| `tests/test_section_coverage.py` | Audit gate for this CLAUDE.md vs source. |

---

## License

MIT — inherits the parent `autoresearch` repository's license.

## Credits

- FX AutoResearch methodology (the source CLAUDE.md) — Evija Ranti.
- Generalized framework — Claude (hierarchical coordinator), 2026-04-19.
- AUTORESEARCHIMAGE (WILDS-Camelyon17) — Claude, bootstrapped 2026-04-24.
