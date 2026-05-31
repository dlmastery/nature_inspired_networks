# H87 — `combo_n4_pair_slot` (4-Axis Orthogonal Stack: φ-Budget + Golden Momentum + φ-Decay-WD + SIREN)

> **One-line claim:** Stacking the two project-certified winners
> `pair_gm_pdw` (H09+H48+H44) and `slot_act_sine` (H09+H81) into a
> single N=4 orthogonal-axis configuration (A7 channel × A15 momentum
> × A12 weight-decay × A8 activation) delivers Δ +1.5 to +2.5 pp over
> the un-tuned CIFAR-100 30-ep baseline; equivalently, it answers
> whether the two certified winners stack productively or saturate.
>
> **Source design space:** G9 combo winners (new group, 2026-05-30) —
> Phase-9d saturation-extension wave from
> `audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md` §3.3.
>
> **Implementation status (this repo):** `○ planned` — all four
> component priors are individually implemented and certified; the
> composition is a new sweep tag in `scripts/run_sweep.py`.

This document is the committee-grade design write-up for hypothesis
H87. Every section below is mandatory per the standard hypothesis
template.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The project's two formally-certified empirical claims at α=0.05 Holm-
Bonferroni (n=7, CIFAR-100 30-ep) are `pair_gm_pdw` (+1.74 pp, three
orthogonal axes: A7 channel + A15 momentum + A12 weight-decay) and
`slot_act_sine` (+1.78 pp, two axes: A7 channel + A8 activation).
Both certify on the same A7 base (H09 φ-budget) but with disjoint
companion axes. The natural extension — also named in the
theoretical-orthogonality analysis (B §6) as the project's pragmatic
Phase-9c next rung — is to combine all four distinct axes into one
N=4 configuration. The natural / sacred motivation is that each of
the four priors is independently motivated by a different geometric
or biological observation: φ-budget (per-stage 1:φ:φ² width ratios
echoing phyllotactic packing), golden momentum (β1 decays at φ^(−1/T)
mirroring the harmonic-mean attractor), φ-decay weight-decay (per-
layer λ = base/φ^k depth-graded regularisation), and sinusoidal
activation (every cell is a vibration; SIREN's `sin(ω·x)` directly
implements the harmonic representation). These four mechanisms touch
**four distinct layers of the training stack** and, per Rule 23, are
guaranteed orthogonal by axis assignment. The empirical question is
whether the additive prediction (+3.52 pp = +1.74 + +1.78) survives
the empirically-documented N=3-saturation finding (A §2): the
combo-ladder marginal first goes mildly negative at N=4.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because the two certified Phase-8 winners `pair_gm_pdw` and
`slot_act_sine` operate on disjoint training-stack axes (A15 + A12 vs
A8; both share the A7 phi_budget base), **mechanism**-wise stacking
them into one N=4 configuration cannot incur same-axis interference;
per Agent A's empirical N=3-saturation finding the marginal benefit
of axis #4 is bounded below by −0.3 pp; per Agent B's max-N analysis
(N=6 CNN-track) the configuration sits below the theoretical
ceiling; we therefore predict Δ ∈ [+1.5, +2.5] pp at n=7 CIFAR-100
30-ep — a result that strictly exceeds the seed-noise floor
(σ_baseline = 0.45 pp) and lies between the strongest existing
winner (+1.78 pp slot_act_sine) and the naïve sum (+3.52 pp).

## 3. Falsifier (≥ 30 words)

If at n=7 CIFAR-100 30-ep `combo_n4_pair_slot` median top-1 is
**less than** `max(pair_gm_pdw, slot_act_sine) − 0.3 pp` = 0.5760
top-1, the "two certified winners stack productively above the N=3
saturation point" claim is REFUTED. Specifically: if median ≤ 0.5760
the N=3 saturation finding (A §2) extends to N=4; if median ≥ 0.5826
(0.5790 + 0.36 % seed-σ margin) the four-axis stack is super-additive.

## 4. Citations (Citation Rigor format, ≥ 80 words multi-paper)

Sitzmann, Martel, Bergman, Lindell, Wetzstein 2020 NeurIPS
'Implicit Neural Representations with Periodic Activation Functions'
(arXiv:2006.09661) — the SIREN paper establishing `sin(ω·x)` as a
spectral-bias-free activation with a published frequency-tuned init;
the H81 component of this stack inherits SIREN's init prescription,
NOT a φ-flavoured re-init, per Agent B §4.3's mechanism-coupling
analysis.

Loshchilov, Hutter 2019 ICLR 'Decoupled Weight Decay Regularization'
(arXiv:1711.05101) — the AdamW reference whose decoupled-WD update
is the substrate H44's per-layer φ-graded λ_k = base/φ^k acts on;
without decoupled WD the H44 prior would be confounded with the
adaptive denominator scaling.

Bello, Fedus, Du, Cubuk, Srinivas, Lin, Shlens, Zoph 2021 NeurIPS
'Revisiting ResNets: Improved Training and Scaling Strategies'
(arXiv:2103.07579) — the source of the "decrease WD when stacking
regularisers" empirical rule that bounds the maximum productive
regulariser stack-depth; this hypothesis's N=4 stack sits well
below the Bello recipe's 8-trick regulariser ceiling.

Tan, Le 2019 ICML 'EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks' (arXiv:1905.11946) — the compound-
scaling reference whose φ-tuned channel-width ratios are the
literature anchor for H09 phi_budget; certified at n=7 in our
project (Δ +1.24 pp).

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

The N=4 stack is implemented as a composition of four already-
existing primitives. Shapes are preserved at every layer (no
geometric / channel-count surgery beyond H09's per-stage allocator).
Per-layer breakdown (ResNet-20 base, 3 stages × 3 blocks):

**Component 1 — H09 phi_budget (A7 channel).**
`src/nature_inspired_networks/phi_scaling.py:PhiBudgetNet`. Per-stage
widths c_k chosen such that ∑ params_k ≈ target B (default 0.27 M)
with ratios c_k : c_{k-1} ≈ φ ≈ 1.618. The default 16/32/64 ResNet-
20 widths become approximately 14/23/37 (post-fix 2026-05-29) under
the budget constraint. **FLOPs:** unchanged at iso-budget (the
allocator preserves total params and re-distributes width).
**Init:** He / Kaiming preserved.

**Component 2 — H48 golden_momentum (A15 momentum-schedule).**
`src/nature_inspired_networks/schedulers.py:GoldenMomentumScheduler`.
β1(t) = β1_0 × φ^(−t/T_max), floored at 1/φ² ≈ 0.382. Default β1_0
= 0.9. **No per-step compute cost** (just modifies the optimizer's
β1 attribute before each step).

**Component 3 — H44 phi_decay_wd (A12 weight-decay).**
`src/nature_inspired_networks/phi_decay.py`. Per-block λ_k = base /
φ^k where k indexes the block depth. Implemented as separate
optimizer param-groups. **No per-step compute cost** (gradient-side
only).

**Component 4 — H81 sinusoidal_activation (A8 activation).**
`src/nature_inspired_networks/sinusoidal_activation.py`. Drop-in
replacement of every ReLU with `sin(ω·x)`, ω learnable per-channel
(init 1.0). **Per-step cost:** sin() is ~2× ReLU cost; total
forward-pass overhead measured at ~7-12 % on RTX 4090.

**Composition order** (the runner.post_build_mutators chain):
1. Build PhiBudgetNet (H09).
2. Swap ReLU → SinusoidalActivation (H81). PRESERVE SIREN init — do
   NOT replace with phi-init (would regress the certified +1.78 pp;
   Agent B §4.3).
3. Build AdamW optimizer with per-block param-groups, λ_k =
   5e-4/φ^k (H44).
4. Build GoldenMomentumScheduler over the AdamW (H48).

The composition is Rule-23-compliant: 4 distinct axes (A7/A15/A12/A8),
2 on-path priors (H09 channel + H81 activation), 2 off-path (H48/H44
both gradient-side).

```python
# Sketch in scripts/run_sweep.py
def make_combo_n4_pair_slot(cfg):
    model = PhiBudgetNet(num_classes=cfg.num_classes, B=cfg.budget)
    swap_relu_to_sin_(model, omega_init=1.0)  # preserves SIREN init contract
    param_groups = phi_decay_wd_groups(model, base_wd=5e-4)
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr)
    sched = GoldenMomentumScheduler(opt, T_max=cfg.epochs)
    return model, opt, sched
```

### 5.2 LLM track (decoder-only Transformer)

The hypothesis is CNN-first; LLM-track adaptation is straightforward.
A7 (channel) maps to per-layer hidden_dim; A8 (activation) maps to
the FFN's pre-residual nonlinearity (replace GELU with `sin(ω·x)`
inside the FFN; see H75 harmonic cymatic SwiGLU for the analogous
construction). A12 (WD) and A15 (momentum) are optimizer-side and
transfer unchanged. **FlashAttention-2 compatibility:** preserved
(no attention-pattern modification). **KV cache:** unchanged.
**Expected impact at 124 M scale:** −0.05 to +0.10 nats validation
perplexity (small; the LLM-track adaptation is not the headline
test).

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.012, +0.022] | top-1 lift dominates; latency cost minor |
| top-1 (CIFAR-100 30-ep, n=7) | [+1.5 pp, +2.5 pp] | decomposes as H09(+1.24) + (H48+H44 in stack: ~+0.5) + (H81 marginal: ~+0.3) − N=4 saturation residual (~−0.5) |
| params | [+0 %, +0 %] | H09 preserves param budget |
| FLOPs | [+5 %, +12 %] | sin() ~2× ReLU FLOP cost amortised over network |
| GPU latency (batch=1) | [+5 %, +12 %] | dominated by sin() overhead |
| rotation-equivariance err | [−0.01, +0.01] | no rotation prior in this stack |
| Betti collapse rate | [no change] | no topological regulariser |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-100 (standard split, no augmentation beyond
  RandomCrop + HFlip — preserving the project's certification regime).
- **Architecture:** ResNet-20 base with `PhiBudgetNet` allocator +
  sinusoidal activation swap.
- **Epochs / batch / precision / seeds:** 30 ep / 128 / bf16 AMP /
  seeds 0..6 (n=7 EVALUATION tier per Rule 28).
- **Optimizer:** AdamW with per-block φ-decay-WD param groups,
  Golden Momentum Scheduler, base LR 3e-3, base WD 5e-4.
- **Composite formula:** SHA-256 `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` (Rule 2).
- **Run-script invocation:**
  ```powershell
  .\.venv\Scripts\python -m nature_inspired_networks.runner `
    --config configs\cifar100_combo_n4_pair_slot.yaml `
    --tag combo_n4_pair_slot --seed <0..6> `
    --root ideas\09_golden_param_budget\experiments\exp_combo_n4_pair_slot\run
  ```
- **Wall-clock estimate:** ~22 min × 7 seeds = ~2.6 GPU-h on RTX
  4090 (16 GB).
- **Archive path:** `ideas/09_golden_param_budget/experiments/exp_combo_n4_pair_slot/run_seed<N>/`.

### 7.2 Idea-targeted experiment (where this prior should SHINE)

The hypothesis is dataset-agnostic; the natural extension is a
CIFAR-10 12-ep screening pass at the same config to verify the
combo doesn't regress on the cheaper benchmark (n=3 seeds, ~1.1
GPU-h). Optional: per-hypothesis hill-climb (cube lr × wd ×
batch × optimizer, budget 25) following Phase-9a protocol — adds
~9 GPU-h if requested.

### 7.3 Cross-paradigm context (LLM track)

Not the primary test of this hypothesis. The LLM-track adaptation
is filed as future work in `ideas/87_combo_n4_pair_slot/AUDIT.md`.

## 8. Cross-references

- Parent design space row: NEW — Group G9, hypothesis H87.
- Parent winners: `pair_gm_pdw` ([H09](../g1_scaling_growth/H09_golden_param_budget.md) + [H48](../g5_optimization_init_reg_nas/H48_golden_momentum.md) + [H44](../g5_optimization_init_reg_nas/H44_golden_regularization.md)) + `slot_act_sine` ([H81](../g8_esoteric_extensions/H81_sinusoidal_activation.md)).
- Research basis: `audits/COMBINATIONS_RESEARCH/[A](../../audits/COMBINATIONS_RESEARCH/A_empirical_stackability.md)` (N=3 saturation evidence), `[B](../../audits/COMBINATIONS_RESEARCH/B_theoretical_orthogonality.md)` §6 (max-N = 6 stack pragmatic N=4 target), `[C](../../audits/COMBINATIONS_RESEARCH/C_literature_survey.md)` §3 (mainstream saturation at 8-12 tricks; our N=4 is below).
- Synthesis doc: `audits/COMBINATIONS_RESEARCH/[D](../../audits/COMBINATIONS_RESEARCH/D_new_hypotheses.md)` §3.3.
- Composes with: any future Class-I bridge (H85/H86) once those land.
- Conflicts with: H10 phi_lr (A §3 ANTAGONISTIC with H48); H42 phi_init (B §4.3 SIREN-init constraint).

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of EfficientNet (Tan 2019) + SIREN (Sitzmann 2020)?**

> EfficientNet's compound-scaling rule fixes a global coefficient φ
> applied to depth/width/resolution; H09 phi_budget instead enforces
> per-stage 1:φ:φ² ratios under a single param budget — a different
> mathematical object. SIREN's published recipe uses sin(ω·x) at the
> network's first layer only with frequency ω=30; H81 here is a
> drop-in replacement of EVERY ReLU with learnable per-channel ω
> init=1.0 — closer to a SIRENified-ResNet than SIREN itself.
> The N=4 composition is a research contribution distinct from
> either component paper.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies the numeric falsifier: median top-1 < 0.5760 at n=7
> CIFAR-100 30-ep refutes the claim. The decision is mechanical at
> the experiment archive's `metrics.json` write time. § 6 gives the
> pre-registered prediction window. There is no aesthetic slack.

**Q: What if the N=3 saturation finding (A §2) is correct and this
combo lands flat?**

> A landing-flat result is itself a publishable empirical answer to
> the saturation-extension question. The Phase-9d brief explicitly
> notes a flat / negative Δ here closes the "more orthogonal axes
> = more lift" hypothesis — that finding has equal scientific value
> to a +2.5 pp lift. The risk is well-bounded and pre-disclosed
> in § 6.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> The `sg_full_fib` catastrophe (A §5, −11.54 pp) stacked SIX priors
> on the SAME conv-block forward path — a Rule 23 violation by
> construction. This hypothesis stacks 4 priors on 4 DISTINCT axes
> with only 2 on-path (Rule 23 cap honored). The empirically-
> documented productive-compounding examples (`pair_gm_pdw` n=7
> +1.74 pp super-additive on three orthogonal axes) are the
> precedent this hypothesis extends.

**Q: How do we know the implementation is correct?**

> Every component prior has its own mechanism-verifying unit tests
> (Rule 25): `tests/test_phi_budget.py`, `tests/test_golden_momentum.py`,
> `tests/test_phi_decay.py`, `tests/test_sinusoidal_activation.py`.
> The composition test `tests/test_combo_n4_pair_slot.py` (TO
> BE ADDED) verifies (a) all four components active simultaneously
> via state inspection, (b) the SIREN init contract is preserved
> (NOT swapped to phi-init), (c) per-block WD groups receive the
> φ-graded λ_k schedule, (d) forward pass uses sin() not ReLU.

## 10. Verification artifacts checklist

- [ ] `ideas/09_golden_param_budget/implementation.py` extended to expose `make_combo_n4_pair_slot()` factory
- [ ] `tests/test_combo_n4_pair_slot.py` ≥ 5 assertions (4 component-active + 1 regression)
- [ ] `ideas/09_golden_param_budget/AUDIT.md` ≥ 3 self-found weaknesses (SIREN-init brittleness, batch-size sensitivity, sin() latency overhead)
- [ ] `ideas/09_golden_param_budget/IMPROVEMENTS.md` records mechanism-coupling fixes if any
- [ ] `ideas/09_golden_param_budget/VERIFY.md` signed on launch date
- [ ] Experiment archive `ideas/09_golden_param_budget/experiments/exp_combo_n4_pair_slot/`
- [ ] Per-seed verification {tests.txt, smoke.txt, gates.txt, reproduction.txt}
- [ ] Row added to `experiments/experiment_log.jsonl`
- [ ] Result reflected in `paper/FINDINGS.md` and dashboard
- [ ] Pre-registration commit hash recorded per Rule 36

## 11. Status journal

(Append-only timeline of what changed when.)

- 2026-05-30 — Created from synthesis of A/B/C research; H87 design doc drafted as Phase-9d wave-1 rank-1 priority.
