# ICML 2027 Reviewer Critique — R2
Date: 2026-06-06 · Reviewer lens: methodology + experimental rigor + baselines

## Verdict

**Reject (main track). Borderline-Reject at Datasets & Benchmarks.**

Soundness 2/4 · Presentation 2/4 · Contribution 2/4 · Confidence 4/5 · Overall **4/10**.

This submission is a long, well-organised, internally-honest project journal masquerading as an ICML paper. The narrative is exhausting (a 263-line abstract+body with constant retconning between Phase-9f, 9g, 9h, 9i), the empirical envelope is roughly four interesting cells at n=3 on CIFAR-100 at a non-converged ResNet-20, and the headline statistical claim (Fisher p=1.94e-5 on n=62 audit-calibration) is a strawman comparison whose null is selected to make the project look uniquely buggy. The methodological contribution (Fixer-with-mechanism-pinning-test contract) is real and portable, but it is buried under empirical theatre that an ICML AC will not have patience for. I would not advance this to discussion in its present form.

## Top 3 fatal flaws

1. **Audit-calibration "headline" is a strawman comparison.** The contrast "0/62 third-party MAJOR/BROKEN vs 18/83 project MAJOR/BROKEN" is between (a) production code written by humans over years and battle-tested by millions of users, and (b) speculative LLM-written research code in a single repo. The proper null is "what MAJOR/BROKEN rate does this same audit doctrine assign to *other* exploratory research repos at the same maturity stage" (e.g., other ICML-submission supplementary code, kaggle-winner notebooks, hackathon repos). The Fisher p=1.94e-5 is real arithmetic on a degenerate population pair. `PAPER.md` §4, `paper/STATISTICAL_TESTS.md` §11.

2. **No iso-FLOP / iso-parameter / iso-tuning baseline; the +1.00–+1.24 pp Phase-9i lift is uninterpretable.** The three "winners" change the architecture (`phi_budget` changes channel widths and stage counts → ~284k vs ~272k params), the optimizer (`pair_gm_pdw` changes momentum schedule + per-layer weight decay), and the activation (`slot_act_sine`). None of these variants was matched to baseline on (params, FLOPs, latency) before claiming a lift. The composite metric `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)` is a project-specific kludge that does not substitute for proper iso-cost comparison. Phase-9h showed that ~2 pp of "lift" can come from LR-tuning alone; the inverse is true at modern recipe — Phase-9i never hill-climbed the priors at the modern recipe, so the +1.00–+1.24 pp is a single-config snapshot, not evidence. `paper/STATISTICAL_TESTS.md` §15, `configs/cifar100_modern_200ep.yaml`.

3. **n=3 with Wilcoxon at the theoretical floor (p=0.125) is dressed up as a "qualitatively binding cert" via paired-t (p=0.0028) that the paper itself admits cannot be validated for normality at n=3.** The Phase-9i "all three priors PASS Phase-5 ordinal gate" framing is a one-sided sign test with α = (1/2)³ = 0.125 — a power-zero, family-wise-uncorrected procedure. Reporting Mann–Whitney "p_one = 0.05" at n_a=n_b=3 as a clearance is mathematically vacuous (it's the floor). `paper/STATISTICAL_TESTS.md` §15.2.

## Honest summary

The protocol contribution (audit doctrine + Fixer-with-mechanism-pinning-test + per-experiment-page + auto-checkpoint loop, codified in `CLAUDE.md` Rules 20–38) is genuinely useful infrastructure. The H09 phi_budget 12.6 % realised-ratio drift is a clean single existence proof that the doctrine catches a real bug an unaudited pipeline would have shipped (`PAPER.md` §5.1).

Everything else is overclaimed.

- The 84-hypothesis design space is a non-random LLM-curated catalogue; the "1 NOVEL+TESTABLE / 81 critiqued" is an artifact of selection.
- The headline empirical claim has flipped four times in 4 days (Phase-9a ⇒ Phase-9f ⇒ Phase-9g/9h ⇒ Phase-9i). Each flip is honestly reported in `paper/FINDINGS.md` and `paper/STATISTICAL_TESTS.md` — which is intellectually admirable, but operationally indistinguishable from p-hacking + post-hoc framing. A reviewer cannot tell whether the next phase (Phase-9j n≥7 modern) will flip the verdict again.
- The paper is 263 lines of abstract+body. ICML main is 9 pages. The submission is not camera-ready compressible without cutting one of the two headline claims.
- The single laptop 4090 constraint is real but does NOT excuse the missing modern baselines (ConvNeXt-Tiny, RegNetX-200MF, ResNet-50, ViT-Small/Tiny) on Imagenette or Tiny-ImageNet — these fit in ≤ 6 hours each on a 4090 and would have settled the SOTA-context question.
- The "self-auditing protocol that catches its own headline drift" pitch is novel but the evidence is one anecdote (Phase-9h → Phase-9i). One anecdote is not a load-bearing empirical claim.

## 25+ concrete improvements

Each item: **What's wrong** / **What to do** / **Severity (BLOCKER/MAJOR/MINOR)** / **Effort (h on 4090 + person-h)** / **Priority (1=highest)**.

### Methodology (M)

1. **M1 — Audit-calibration substrate is mis-selected.**
   *What's wrong:* `pytorch/vision` ResNet/Bottleneck/MobileNetV2 are mature production code; comparing their MAJOR/BROKEN rate to an exploratory single-author research repo is a strawman. The right comparison is research code at similar maturity. `PAPER.md` §4.
   *What to do:* Audit 30+ hypotheses from THREE recent ICML/NeurIPS/ICLR supplementary code repos (released this calendar year) using the same doctrine. Report the MAJOR/BROKEN rate. The expected outcome is "8–18%" — comparable to project's 21.7 %. Reframe the headline as "audit doctrine separates project-quality from production-quality" not "from research-quality."
   *Severity:* BLOCKER. *Effort:* 30 person-h, 0 GPU. *Priority:* 1.

2. **M2 — Strawman framing: the 0/62 cell is dispositive only if you also test on hard cases.**
   *What's wrong:* Section 4.1 reports MINOR rates 29 % project vs 34 % calibration, framed as "audit aggressiveness is calibrated." But the project's MINOR floor is higher in absolute terms than its rate suggests because the project has buggier code overall — the MINOR rate is an unreliable calibration when the codebase quality differs.
   *What to do:* Stratify the calibration by code-vintage and audit known-buggy commits in `pytorch/vision` history (e.g., the 2019 SqueezeNet bias-removed bug, the InceptionV3 channel-ordering issue). Report MAJOR/BROKEN agreement on KNOWN-defective historical code as the sensitivity floor.
   *Severity:* MAJOR. *Effort:* 12 person-h, 0 GPU. *Priority:* 2.

3. **M3 — No multiple-comparisons correction across the screening universe.**
   *What's wrong:* 35 single-prior tags × 12-ep CIFAR-10 + 17 candidates × 30-ep CIFAR-100 + 12 control cells + 12 hill-climb rows ≈ 76 statistical comparisons before "k=3 Holm-Bonferroni" was declared. The k=3 Holm family is a post-screening selection. `paper/STATISTICAL_TESTS.md` §3.
   *What to do:* Report family-wise α at the screening universe (k=76 → α/76 ≈ 6.6e-4). The n=7 Wilcoxon floor 0.0078 does NOT clear this. Reframe the cert as "the screening + confirmatory pattern is a two-stage clinical-trial design with conditional α; we make no joint-family claim."
   *Severity:* BLOCKER. *Effort:* 4 person-h. *Priority:* 1.

4. **M4 — Pre-registration claim is undocumented.**
   *What's wrong:* `paper/REVIEWER_CHECKLIST.md` Section A claims Rule-28 screening-vs-evaluation is pre-registered, but the §7.3.1 reclassification of negatives is explicitly post-hoc (`audits/REVIEWER_PASS_PAPER.md` calls this HARKing). Rule 28 was authored AFTER the negatives were observed.
   *What to do:* Either (a) drop the screening-vs-evaluation framing entirely and report 35 row-level results as 35 single-config single-seed observations (which they are), or (b) timestamp Rule 28's git commit and concede that classifications made before that commit are retrospective and so cannot count as pre-registration.
   *Severity:* BLOCKER. *Effort:* 3 person-h. *Priority:* 1.

5. **M5 — Phase-9h vs Phase-9i is dressed up as "self-correction" but is just "we picked a different cell."**
   *What's wrong:* The narrative is "Phase-9h tuned baseline beat priors; Phase-9i restored the priors at iso-modern-recipe; therefore self-correcting." But the Phase-9h cell (lr=0.01, AdamW, 30 ep, default recipe) and the Phase-9i cell (modern 11-trick, 200 ep) are DIFFERENT cells. There is no symmetric hill-climb of the priors at lr=0.01, and no hill-climb of the baseline at modern recipe. The "correction" is just a different choice of comparison cell.
   *What to do:* Run a 2×2 factorial (lr ∈ {1e-3, 1e-2} × recipe ∈ {default, modern}) at n=7 with ALL FOUR arms (baseline + 3 priors) hill-climbed independently in each cell. That's the only honest cross-regime story.
   *Severity:* BLOCKER. *Effort:* ~80 GPU-h (4 cells × 4 arms × 7 seeds at 1–4 h each). *Priority:* 1.

6. **M6 — "Iso-modern + iso-convergence" is not iso-tuning.**
   *What's wrong:* Phase-9i uses the SAME hyperparameters for baseline and priors at the modern recipe (`configs/cifar100_modern_200ep.yaml`). Priors were never hill-climbed at the modern recipe. The Phase-9h LR-tuning confound concern flips: now the priors might be benefiting from baseline mis-tuning at the modern recipe.
   *What to do:* Phase-9j must hill-climb both arms before any iso-recipe claim. Filed as "future work" but the paper claims an empirical lift at iso-modern-recipe before this work was done. Either run it or drop the claim.
   *Severity:* BLOCKER. *Effort:* ~25 GPU-h (4 arms × 4 cells × hill-climb). *Priority:* 1.

7. **M7 — Auditor model-family circularity is acknowledged then waved away.**
   *What's wrong:* `paper/PAPER.md` §1.2 acknowledges all 5 agent roles share Opus 4.7. `audits/CROSS_FAMILY_HONEST_REAUDIT.md` is also Opus 4.7. The "cross-family methodologically-diverse re-audit" (8/10 strict concordant) is intra-family by construction.
   *What to do:* Either spend ~$50 in API credits to dispatch GPT-5 or Gemini-3-Pro on the same 10 findings (the AC synthesis explicitly noted this is the missing piece), or remove the Track-B framing entirely and admit the result is "Opus-self-consistent" rather than "cross-family validated."
   *Severity:* MAJOR. *Effort:* 6 person-h, 0 GPU, $50 API. *Priority:* 2.

### Baselines (B)

8. **B1 — Where are the modern baselines?**
   *What's wrong:* ResNet-20 (2015) is not a 2027 baseline. ConvNeXt-Tiny, ConvNeXt-V2-Femto, RegNetX-200MF, MaxViT-Pico, ViT-S/16, EfficientNetV2-S, MobileNetV4-Small — every one of these is a more honest baseline at 32×32 → 64×64 input and fits in ≤ 6 GPU-h on a 4090. Compute is NOT the excuse.
   *What to do:* Run RegNetX-200MF and ConvNeXt-Tiny on CIFAR-100 at 200 ep with the same modern 11-trick recipe. Compare the three "winners" against THOSE numbers, not against ResNet-20.
   *Severity:* BLOCKER. *Effort:* ~25 GPU-h. *Priority:* 1.

9. **B2 — RegNetX-200MF is the direct literature analog to H09 — and explicitly NOT run.**
   *What's wrong:* H09 phi_budget IS a rediscovery of RegNet's `w_m` Pareto region (Radosavovic et al. CVPR 2020 explicitly prefers `w_m ∈ [2.5, 2.9]`, NOT φ=1.618). The paper concedes this in §1.1 ("DERIVATIVE+TESTABLE"). Control 3b RegNetX-200MF was wired in `controls/PLAN.md` and explicitly REFUSED by the launch allowlist (`paper/STATISTICAL_TESTS.md` §13.0). For a paper that admits H09 is a RegNet rediscovery, not running RegNet is dispositive.
   *What to do:* Run RegNetX-200MF (≈270k params, matched to phi_budget) at n=3 on CIFAR-100 modern recipe 200 ep. If phi_budget does not beat it by ≥ 0.5 pp, drop the H09 claim entirely.
   *Severity:* BLOCKER. *Effort:* ~12 GPU-h. *Priority:* 1.

10. **B3 — Imagenette / Tiny-ImageNet are missing.**
    *What's wrong:* `paper/LIMITATIONS.md` §4 admits "no ImageNet." But Imagenette (160 px, 13k images, 10 classes; ~1 h/run on 4090) and Tiny-ImageNet (64 px, 100k images, 200 classes; ~3 h/run on 4090) are well within the laptop budget and have published baselines at modern recipes. The paper is on CIFAR for hardware reasons; the reasons don't survive contact with Imagenette.
    *What to do:* Run the three winners + baseline at Imagenette 200 ep, n=3. ~25 GPU-h total. If the priors don't survive Imagenette, the CIFAR result is a dataset-overfit story.
    *Severity:* MAJOR. *Effort:* ~25 GPU-h. *Priority:* 2.

11. **B4 — No "non-φ 3-axis regularizer" control was run before claiming `pair_gm_pdw` as a winner.**
    *What's wrong:* Control 1 (`paper/STATISTICAL_TESTS.md` §13.1) ran at n=3 and showed `pair_nonphi_3axis` gets 61 % of the lift WITHOUT the φ-content. The honest reading: most of the +1.74 pp is generic 3-axis regularizer stacking. The paper still markets `pair_gm_pdw` as a "win."
    *What to do:* Extend Control 1 to n=7 against `pair_gm_pdw` paired. If the non-φ residual is < 0.3 pp, drop `pair_gm_pdw` from the headline triple.
    *Severity:* MAJOR. *Effort:* ~7 GPU-h. *Priority:* 1.

12. **B5 — `slot_act_sine` is a SIREN replication, not a nature-inspired prior.**
    *What's wrong:* Control 2 (`paper/STATISTICAL_TESTS.md` §13.2) showed `slot_act_tanh` BEATS `slot_act_sine` by +0.48 pp paired (3/3 positive). The paper itself concedes "SIREN-specific story REFUTED." Yet `slot_act_sine` is listed in the abstract triple as a "Phase-8 winner." This is a category error.
    *What to do:* Drop `slot_act_sine` from the headline. The protocol's catch — that a SIREN replication was almost shipped as a nature-inspired finding — IS the value-add; report it that way.
    *Severity:* MAJOR. *Effort:* 2 person-h. *Priority:* 1.

### Statistical analysis (S)

13. **S1 — Paired-t at n=3 is being reported as if it were credible.**
    *What's wrong:* `paper/STATISTICAL_TESTS.md` §15.2 reports paired-t p_one ∈ {0.0028, 0.0070, 0.0082} at df=2. At df=2, t-distribution tails are extremely heavy; normality cannot be assumed; a single outlier seed flips the verdict.
    *What to do:* Stop reporting paired-t at n=3 entirely. The Wilcoxon floor (0.125) and the sign test (0.125) ARE the n=3 ceiling. Frame Phase-9i as descriptive, not inferential.
    *Severity:* MAJOR. *Effort:* 2 person-h. *Priority:* 1.

14. **S2 — 95 % bootstrap CIs at n=3 are not what they claim to be.**
    *What's wrong:* Bootstrap at n=3 produces a CI that under-covers the true parameter by a factor of ~2 (DiCiccio & Efron 1996). The reported [+0.95, +1.43] / [+0.85, +1.08] / [+0.75, +1.17] are point-estimate-anchored intervals with negligible coverage guarantee.
    *What to do:* Report BCa (bias-corrected accelerated) bootstrap, not percentile, and disclose that BCa at n=3 still under-covers. Or admit "n=3 is too small for inference; report point estimates with σ only."
    *Severity:* MAJOR. *Effort:* 4 person-h. *Priority:* 2.

15. **S3 — Mann–Whitney "p = 0.05" at n_a=n_b=3 is the floor, not a clearance.**
    *What's wrong:* `paper/STATISTICAL_TESTS.md` §15.2 reports MW p_one = 0.05 as if this were achievement. At n=3 vs n=3, MW one-sided minimum is 1/C(6,3) = 1/20 = 0.05. Reporting "clears α=0.05" when you've hit the theoretical floor is informational sleight of hand.
    *What to do:* State explicitly: "MW p=0.05 at n=3 is the theoretical floor; this is informationally identical to 'all 3 leader seeds strictly exceed all 3 baseline seeds.' Not evidence beyond a sign test."
    *Severity:* MAJOR. *Effort:* 1 person-h. *Priority:* 1.

16. **S4 — Phase-5 ordinal gate is a sign test masquerading as a power test.**
    *What's wrong:* "min(leader) > max(baseline)" at n=3 has P=1/8 under H0, i.e., α=0.125. The gate is reported across the project as if it were a high-bar criterion. It's a 12.5% type-I gate.
    *What to do:* Either replace the Phase-5 gate with paired Wilcoxon + Holm-Bonferroni at α=0.05 across the screening universe, or characterize it explicitly as "sign test α=0.125 per row" everywhere it appears (currently inconsistent).
    *Severity:* MAJOR. *Effort:* 6 person-h to edit all references. *Priority:* 2.

17. **S5 — Power analysis is missing for every claim.**
    *What's wrong:* No paper-grade discussion of statistical power at the announced effect sizes. Δ=+1 pp at σ=0.45 pp gives δ/σ≈2.2; at n=7 paired-t power is ~0.75; at n=3 paired-t power is ~0.30. The n=3 binding is under-powered by design.
    *What to do:* Add a power-analysis subsection to `paper/STATISTICAL_TESTS.md` stating required n to detect Δ=0.5 pp / 1.0 pp / 1.5 pp at σ=0.45 pp at α=0.05 two-sided 80 % power. Likely answer: n≥14 for 0.5 pp, n≥7 for 1.0 pp. Phase-9i n=3 is under-powered.
    *Severity:* MAJOR. *Effort:* 4 person-h. *Priority:* 2.

18. **S6 — Phase-9h "tuned baseline beats by +2.27 pp" was over-interpreted, then re-interpreted, then declared "load-bearing methodology."**
    *What's wrong:* The Phase-9h apparent refutation (`paper/STATISTICAL_TESTS.md` §14) was based on n=3 vs n=7 unpaired across different (lr, wd) cells. The "self-correction" (Phase-9i) doesn't reconcile this — it sidesteps it with a different recipe. A reviewer cannot tell whether the priors help or not.
    *What to do:* Run the Phase-9i n≥7 modern extension (filed as future work). Until that lands, the priors' status is "unknown at NeurIPS-α." Don't ship until then.
    *Severity:* BLOCKER. *Effort:* ~39 GPU-h. *Priority:* 1.

### Reproducibility (R)

19. **R1 — Cold-clone reproducibility is plausibly broken on the 11-trick recipe.**
    *What's wrong:* `configs/cifar100_modern_200ep.yaml` references modules (`mixup`, `cutmix`, `random_erasing`, `randaugment`, `ema`) added per `convergence/PLAN.md` Phase 0. CLAUDE.md sets `cudnn.benchmark=True` (non-deterministic). The composite formula and SHA-256 fingerprint exist, but bit-reproducibility on the same hardware is not guaranteed; cross-hardware is impossible (composite includes `log10(latency_ms)`).
    *What to do:* Set `cudnn.deterministic=True` for paper-headline runs; provide an explicit "cold-clone reproduction script" `scripts/reproduce_phase9i.sh` and verify on a clean clone within 24 h of fork.
    *Severity:* MAJOR. *Effort:* 8 person-h, 12 GPU-h. *Priority:* 2.

20. **R2 — No per-hypothesis hyperparameter table in the paper.**
    *What's wrong:* `PAPER.md` §3.4 lists "training defaults" but the per-hypothesis deltas (e.g., what does `pair_gm_pdw` configure?) are never tabulated. A reader has to chase `scripts/run_sweep.py` and YAML configs to reconstruct.
    *What to do:* Add Table-1: per-tag → (model, channel_mode, flags, optimizer-knobs, expected-Δ) row for every claimed result.
    *Severity:* MAJOR. *Effort:* 4 person-h. *Priority:* 2.

### Confounding (C)

21. **C1 — Modern-recipe upgrade leaks into prior runs in a way the legacy baseline didn't get.**
    *What's wrong:* The Phase-9i convergent baseline (0.6360 modern-recipe) is compared to the priors at the SAME modern recipe. But the priors were originally screened at the LEGACY recipe (Phase-2 12-ep). The "transfer" from legacy → modern is asymmetric: each prior carries pre-screening selection bias that the modern baseline does not. The +1 pp lift could be partly a selection-bias survivor effect.
    *What to do:* Screen 30+ random untrained priors at the modern recipe directly. Check the false-positive rate of "any random prior on modern recipe shows +1 pp over baseline." If FPR > 30 %, the Phase-9i lift is a selection artifact.
    *Severity:* MAJOR. *Effort:* ~80 GPU-h. *Priority:* 2.

22. **C2 — `pair_gm_pdw` confounds three axes; attribution is impossible.**
    *What's wrong:* `pair_gm_pdw` = phi_budget × golden_momentum × phi_decay_wd. Control 1 showed 3-axis structure (non-φ) gets 61 % of lift. The remaining 39 % is split across three φ-axes with no factorial decomposition.
    *What to do:* Run the 2³ factorial of φ-budget × golden-momentum × phi-decay-wd (8 cells × 3 seeds) on CIFAR-100 modern 200 ep. Report marginal effects of each axis.
    *Severity:* MAJOR. *Effort:* ~50 GPU-h. *Priority:* 2.

### Hardware-fairness (H)

23. **H1 — ResNet-20 on CIFAR-100 30 ep is below the legitimate noise floor for the claimed effect.**
    *What's wrong:* `paper/SOTA_COMPARISON.md` admits 5.7 pp gap-to-SOTA at 12-ep. Even 30 ep is well below convergence. Claiming a +1.24 pp lift on a 6 pp-below-SOTA baseline is measuring noise that disappears at convergence. Phase-9i at 200 ep partially fixes this but the priors still sit at 0.6485, while ConvNeXt-V2-Tiny under similar recipes reaches ~0.81 on CIFAR-100 (Liu et al. 2023). The claimed contribution lives entirely inside an 18 pp gap from the modern SOTA.
    *What to do:* Either commit to ResNet-20 as the testbed and reframe contribution as "we identify priors that help under-converged small networks" (which is a niche claim), or run on ConvNeXt-V2-Femto / RegNetX-200MF at convergence on CIFAR-100. The 4090 Laptop handles both in ≤ 8 h per seed.
    *Severity:* BLOCKER. *Effort:* ~50 GPU-h. *Priority:* 1.

24. **H2 — `num_workers=0` on Windows is a known performance trap, not a hardware necessity.**
    *What's wrong:* CLAUDE.md §2 mandates `num_workers=0` on Windows because "spawn-start workers wedge." This is a known PyTorch quirk fixable with `persistent_workers=True` + `multiprocessing_context='spawn'`. The paper presents this as a hardware constraint; it's a configuration choice that triples wall-clock and reduces seed coverage. Tiny detail but it's load-bearing for the "we couldn't afford n=7" argument.
    *What to do:* Either fix the dataloader (~6 person-h) and run more seeds, or admit the constraint is configuration, not hardware. The "39 GPU-h to extend to n=7" cost claim in `convergence/PLAN.md` falls by ~2× with proper data loading.
    *Severity:* MINOR but undermines the budget excuse. *Effort:* 6 person-h. *Priority:* 3.

### Framing / contribution (F)

25. **F1 — "Self-auditing protocol that catches its own headline drift" is one anecdote.**
    *What's wrong:* `PAPER.md` §5 lists Phase-9h → Phase-9i as a self-correction cycle. That's n=1 cycle. The narrative is rhetorically compelling but evidentially weak.
    *What to do:* Either find a second self-correction case (apply the doctrine to a sister repo from `dlmastery/autoresearch*` and find an analogous catch), or de-emphasize the self-correction framing to "we believe the doctrine surfaces drift; one well-documented case study; broader claim is future work."
    *Severity:* MAJOR. *Effort:* ~80 person-h to do a sister-repo replication. *Priority:* 2.

26. **F2 — Abstract is 440 words.**
    *What's wrong:* ICML convention is 150–200 word abstracts. The current abstract is denser than three normal abstracts and contains seven separate numeric claims, each with its own caveat.
    *What to do:* Compress to ≤ 200 words. Lead with the H09 catch as the existence proof; demote audit-calibration to one sentence; demote Phase-9h/9i to one sentence.
    *Severity:* MAJOR (presentation). *Effort:* 4 person-h. *Priority:* 1.

27. **F3 — `PAPER.md` is 263 lines body. ICML main is 9 pages.**
    *What's wrong:* The submission is structurally incompatible with ICML 2027 main-track page limits. The current document is ~14 pages of dense Markdown, before LaTeX rendering.
    *What to do:* Cut §6 (case study) to ≤ 1.5 pages; move audit-calibration math to supplementary; lift H09 narrative into §5 with one figure.
    *Severity:* BLOCKER. *Effort:* 16 person-h. *Priority:* 1.

28. **F4 — Self-grading banners on README and PAPER.**
    *What's wrong:* `paper/REVIEWER_CHECKLIST.md` opens with "INTERNAL QA PASS." `README.md` carries a "dual-track-audit pass" badge. These ARE internal QA passes; they read to external reviewers as self-grading theatre.
    *What to do:* Remove badges entirely or relabel as "internal QA pass — independent review pending." (Already partially done in current commit per Rule 37; not yet propagated to README badges.)
    *Severity:* MAJOR. *Effort:* 2 person-h. *Priority:* 2.

29. **F5 — H71 IcosaRoPE3D framed as a contribution despite Control 4 being INCONCLUSIVE.**
    *What's wrong:* `paper/STATISTICAL_TESTS.md` §13.4 reports Δ=+0.18 pp at n_a=3 vs n_b=1 on rotated CIFAR-10. The paper still lists H71 as "the sole NOVEL+TESTABLE survivor" as if it were a contribution. It's a research proposal.
    *What to do:* Drop H71 from contribution list. Mention only as "untested promising future direction." Run a proper n=3 vs n=3 comparison at minimum if you want to keep it.
    *Severity:* MAJOR. *Effort:* ~5 GPU-h, 4 person-h. *Priority:* 2.

30. **F6 — IDEA_TABLE describes H09 as "🏆 CERTIFIED" then footnotes "iso-tuned FAILS Phase-5."**
    *What's wrong:* The badge contradicts the footnote. A reader of `hypotheses/IDEA_TABLE.md` Group G1 row H09 will absorb the certification and skim the qualifier. This is exactly the kind of asymmetric prominence ICML reviewers downgrade for.
    *What to do:* Either downgrade the badge to "📋 SCREENED at default-recipe" or remove badges from IDEA_TABLE entirely.
    *Severity:* MAJOR. *Effort:* 1 person-h. *Priority:* 1.

31. **F7 — `paper/FINDINGS.md` is 1000+ lines of journaling, not a paper artifact.**
    *What's wrong:* The honest journal-of-corrections framing (Phase-9h then Phase-9i then supersession notes) reads as an internal project log. An ICML reviewer will treat the multiple revisions as evidence of post-hoc tweaking.
    *What to do:* Refactor FINDINGS.md into a clean "verdict per claim with one supporting fact" format. Move the historical narrative into `audits/` where it belongs.
    *Severity:* MAJOR. *Effort:* 8 person-h. *Priority:* 2.

32. **F8 — Composite metric is project-specific and never independently validated.**
    *What's wrong:* `top1 − 0.05·log10(params_M) − 0.05·log10(latency_ms)` is a SHA-256-fingerprinted project-internal aggregation. It's not used anywhere else in the literature. Rankings by composite vs raw top-1 can diverge; the paper presents composite rankings as if they were neutral.
    *What to do:* Report raw top-1 first; composite as supplementary. Drop the SHA-256 fingerprinting theatre — it doesn't add empirical credibility.
    *Severity:* MINOR. *Effort:* 3 person-h. *Priority:* 3.

33. **F9 — 51 % non-PASS audit rate as "evidence" is one-sided.**
    *What's wrong:* `PAPER.md` notes the 51 % aggregate non-PASS rate on the project substrate. This is presented as evidence the audit finds real defects. It's equally consistent with hyperactive auditing on speculative code.
    *What to do:* Run the audit doctrine on a known-good third-party research repo of similar maturity AND report the non-PASS rate. If both rates are similar (say 40-60%), the rate is uninformative.
    *Severity:* MAJOR. *Effort:* ~30 person-h. *Priority:* 2.

### Closing-out tractability check

Items 1–8, 11–13, 23, 27, 30 can all be delivered in ≤ 6 months on a single laptop 4090 with disciplined scheduling: ~250 GPU-h spread over ~8 weeks of overnights and ~120 person-h of paper-rewrite + analysis. The path to a publishable D&B paper is clear; the path to ICML main requires either dropping the empirical headline (and reframing as a pure methodology paper) or running the Phase-9j n≥7 modern hill-climb + RegNetX-200MF + ConvNeXt-V2-Femto comparison ladder.

## Closing note

The author is unusually honest about the campaign's evolution. That's both the project's strongest signal of integrity and its weakest signal of finishedness — every reviewer will read the multiple "Phase-9h was wrong → Phase-9i corrects it" supersession notes as a half-baked experiment that has not converged on a stable verdict. The methodological contribution (Fixer + mechanism-pinning-test contract; per-experiment-page; auto-checkpoint loop) is real and portable and deserves publication. The empirical contribution is, charitably, "we find a +1 pp directional signal on under-converged ResNet-20 at CIFAR-100 that may or may not survive proper hill-climbing of a modern baseline." That's a workshop poster, not an ICML main paper.

The author has clearly internalised every criticism a hostile reviewer could make BEFORE submission — `paper/LIMITATIONS.md`, `audits/REVIEWER_PASS_PAPER.md`, `audits/ICML_REVIEWS_2026-05-30/AC_synthesis.md` all anticipate this critique. The problem is that internalising criticism is not the same as resolving it. ICML 2027 rewards papers that DO the controls, not papers that catalogue why the controls aren't done yet.

My recommendation: withdraw, run Phase-9j + RegNetX-200MF + Imagenette + cross-family auditor (≈250 GPU-h + ~$50 API + ~120 person-h), resubmit to ICML D&B 2027 cycle as a Reproducibility / Meta-Research contribution. Or carve out the audit-protocol + Fixer-test discipline as a standalone short paper for the ICML 2027 ML Reproducibility Workshop and let the priors substrate cook for ICML 2028.
