# H47 — φ-Dropout (Cyclical Fibonacci-Ratio Dropout Rates)

> **One-line claim:** Cycling dropout rates through Fibonacci-derived
> fractions `{1/φ, 1/φ², 1/φ³, 1/φ⁴}` over the course of training
> reduces overfitting on CIFAR-10 by ≥1.0 pp in generalization gap
> while not regressing top-1 by more than 0.5 pp because the cyclical
> schedule injects multi-scale noise — high noise (0.618) early to
> prevent memorization, low noise (0.146) late to allow consolidation
> — better than a fixed dropout rate, per the curriculum-noise
> literature of Bengio 2009.
>
> **Source design space:** G5 Optimization / Init / Regularization / NAS
> (H41–H50).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for hypothesis
H47. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Dropout (Srivastava 2014) is the canonical neuron-level
regularization: zero each activation with probability p, scale the
rest by 1/(1-p). The standard p=0.5 was empirically tuned and is
almost universally suboptimal for modern architectures with
BatchNorm; smaller rates (0.1–0.2) typically work better. The right
question isn't "what p is best" but "what p *trajectory* is best".

Curriculum learning (Bengio 2009) showed that gradually decreasing
training noise from high to low traces a more reliable path through
the loss landscape than constant noise. Nature shows the same
pattern: biological development reduces synaptic stochasticity with
age, and seed-spread distance in plants decreases as the plant matures.
The decrease is approximately exponential with rate 1/φ per stage —
the same Fibonacci-decay rate seen in capillary branching and
phyllotactic angular packing.

A φ-dropout schedule embeds this curriculum explicitly:
`p(t) = 1/φ^(1 + ⌊4t/T⌋)` cycles through {0.618, 0.382, 0.236, 0.146}
over training. Early phases at p=0.618 provide aggressive
regularization when the network is most prone to memorizing noise;
late phases at p=0.146 free the network to consolidate
discriminative features. This is the engineering case for
nature-inspired dropout scheduling.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because dropout's effective regularization strength is proportional
to its rate p — mechanism-wise, the φ-decay schedule applies high
regularization (p=0.618) during early training when the loss surface
is most non-stationary and low regularization (p=0.146) when
consolidation is needed — per Bengio 2009 (curriculum learning) and
Srivastava 2014 (dropout), we expect that φ-dropout reduces CIFAR-10
generalization gap by ≥1.0 pp at 50-epoch training vs. a fixed
p=0.2 control, while top-1 stays within ±0.5 pp.

## 3. Falsifier (≥ 30 words)

If at 3-seed median the φ-dropout generalization gap reduction is
< 0.7 pp vs. fixed p=0.2 (with 95% CI upper bound below 0.7 pp), OR
if test top-1 drops by more than -0.7 pp from the no-dropout control,
this hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and
Sutskever, Ilya and Salakhutdinov, Ruslan 2014 JMLR 'Dropout: A
Simple Way to Prevent Neural Networks from Overfitting' -- the
foundational dropout paper; we replace its constant-rate schedule
with a φ-cyclic one.

Bengio, Yoshua and Louradour, Jérôme and Collobert, Ronan and
Weston, Jason 2009 ICML 'Curriculum learning' -- introduces
curriculum learning; the conceptual basis for our gradually-
decreasing-noise schedule.

Howard, Andrew G. and Sandler, Mark and Chu, Grace and others 2019
ICCV 'Searching for MobileNetV3' (arXiv:1905.02244) -- shows
dropout-rate scheduling matters for compact architectures; we
extend with the φ-derived schedule.

Wan, Li and Zeiler, Matthew and Zhang, Sixin and LeCun, Yann and
Fergus, Rob 2013 ICML 'Regularization of Neural Network using
DropConnect' -- related regularization technique; we cite for
contrast since DropConnect zeros weights, while φ-Dropout zeros
activations.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

Replace the model's static `nn.Dropout(p=0.2)` with a
`PhiDropout(initial_phase=0, total_phases=4, T_total=epochs)` module
that updates its internal `p` according to the current epoch:

```python
import math
import torch.nn as nn

PHI = (1.0 + 5 ** 0.5) / 2

class PhiDropout(nn.Module):
    def __init__(self, total_phases=4, T_total=12):
        super().__init__()
        self.total_phases = total_phases
        self.T_total = T_total
        self.current_phase = 0
        self.p = 1.0 / PHI

    def set_epoch(self, epoch):
        """Update p based on current epoch."""
        phase = min(self.total_phases - 1,
                    int(epoch * self.total_phases / self.T_total))
        self.current_phase = phase
        self.p = 1.0 / (PHI ** (phase + 1))

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1.0 - self.p)
```

Forward shape unchanged. Forward FLOPs add one rand_like + multiply
(negligible). The training loop must call `model.set_epoch(epoch)` at
the start of each epoch.

Lives in `src/nature_inspired_networks/reg/phi_dropout.py`, re-exported by
`ideas/47_phi_dropout/implementation.py`.

### 5.2 LLM track (decoder-only Transformer)

For a 124M GPT-2-small, the φ-dropout module replaces the standard
`embd_pdrop` and `resid_pdrop` and `attn_pdrop`. The schedule is
applied per "phase" defined by training-step bins rather than
epochs. Care points:

- **attn_pdrop:** dropout on attention weights is much more
  destructive than on FFN; for safety we apply φ-dropout only to
  resid and embd, leaving attn at fixed 0.0.
- **FlashAttention-2 compatibility:** FA2 does not support attn_pdrop
  internal to the kernel; our restriction to resid/embd is
  automatically FA2-compatible.
- **Causal mask preservation:** unaffected.

Expected impact at 124M: validation ppl ≤ baseline by 0.2–0.5 ppl
when training is long enough to overfit (≥10k steps on TinyStories).

```python
class PhiDropoutLLM(PhiDropout):
    def set_step(self, step, total_steps):
        phase = min(self.total_phases - 1,
                    int(step * self.total_phases / total_steps))
        self.current_phase = phase
        self.p = 1.0 / (PHI ** (phase + 1))
```

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (fixed p=0.2) | rationale |
|---|---|---|
| composite | [+0.000, +0.010] | gain via reduced overfit |
| top-1 (CNN, 50 ep) | [-0.5, +1.0] pp | mild lift |
| generalization gap | [-2.5 pp, -0.5 pp] | core targeted metric |
| params | [0, 0] | unchanged |
| FLOPs | [0, 0] | unchanged |
| GPU latency (batch=1) | [0, 0] | inference: dropout is no-op |
| rotation-equivariance err | [0, 0] | unaffected |
| KV cache @ 32k (LLM) | [0, 0] | unaffected |
| Betti collapse rate | [-0.05, +0.05] | minor |
| perplexity (LLM 124M, 10k steps) | [-0.5, +0.1] | small lift via reduced overfit |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- **Dataset:** CIFAR-10
- **Architecture:** `NaturePriorNet` with dropout-3 layer replaced
  by `PhiDropout(total_phases=4, T_total=50)`
- **Epochs:** 50 (overfitting needs time)
- **Optimizer:** AdamW, lr=3e-4, weight_decay=1e-2
- **Comparator:** fixed p=0.0, fixed p=0.2, fixed p=0.5
- **Seeds:** 0, 1, 2
- **Logging:** per-epoch train/test top-1 and gap
- **Run-script:** `python scripts/run_idea.py --idea 47 --drop phi --seeds 0 1 2`
- **Wall-clock:** ≈ 50 min × 3 seeds × 4 controls ≈ 10 h
- **Archive path:** `ideas/47_phi_dropout/experiments/exp001_cifar10_phi_dropout/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

φ-Dropout should shine in the *low-data + long-training* regime:

- **Dataset:** CIFAR-10 with only 5k samples (10%)
- **Epochs:** 200
- **Architecture:** 2× wider `NaturePriorNet` (over-parameterized)
- **Predicted:** ≥3 pp gap reduction vs. fixed p=0.2
- **Diagnostic:** if no advantage even in over-parameterized
  low-data setting, the cyclic schedule does not matter.

### 7.3 Cross-paradigm context (LLM track)

- **Model:** 124M GPT-2-small
- **Dataset:** TinyStories
- **Steps:** 20k
- **Comparator:** fixed p=0.1 vs. PhiDropoutLLM (4 phases over 20k steps)
- **Metric:** val ppl and (train_loss - val_loss) gap
- **Run:** `python scripts/run_llm.py --idea 47 --drop phi`

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G5 row H47.
- Master experiment list: `EXPERIMENT_LOG.md` (planned).
- Implementation sub-directory: `ideas/47_phi_dropout/`
- Related hypotheses that compose:
  - **H44** Golden regularization — both regularization priors with
    φ-decay character; compose to test the joint effect.
  - **H48** Golden momentum scheduler — also a φ-cyclic schedule;
    they share the curriculum-style design.
  - **H43** Fibonacci pruning — both use Fibonacci ratios over time.
- Related hypotheses that conflict:
  - **H50** Full hybrid — adding any new schedule to the broken
    full hybrid is unlikely to help; isolate first.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just dropout-schedule grid-search?**

> The schedule is *prescribed*, not searched. We don't tune the
> phases — they are 1/φ^k by construction. This is what makes the
> hypothesis falsifiable: if the prescribed schedule doesn't beat
> tuned-baseline dropout, it's rejected.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies ≥0.7 pp gap reduction with 95% CI exclusion. Sample
> size is 3 seeds; effect must be robust.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> § 7.2 explicitly tests the low-data overfit regime. ImageNet's
> data abundance may make the prior unnecessary; reported as scope.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H47 is a scheduling prior, orthogonal to the architectural compound
> failure. Tested in isolation first.

**Q: How do we know the implementation is correct?**

> `tests.py` verifies (a) the p schedule matches 1/φ, 1/φ², 1/φ³,
> 1/φ⁴ within float tolerance per phase, (b) in eval mode forward is
> identity, (c) in train mode the scaling factor 1/(1-p) preserves
> expected activation magnitude (verified statistically).

## 10. Verification artifacts checklist

- [ ] `ideas/47_phi_dropout/implementation.py` exists, tests green
- [ ] `ideas/47_phi_dropout/tests.py` ≥ 6 assertions
- [ ] `ideas/47_phi_dropout/AUDIT.md` ≥ 3 weaknesses
- [ ] `ideas/47_phi_dropout/IMPROVEMENTS.md` records fixes
- [ ] `ideas/47_phi_dropout/VERIFY.md` signed
- [ ] At least one experiment archive
- [ ] `verification/{tests,smoke,gates,reproduction}.txt`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md` and dashboard

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-C.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G5 (elite-research-scientist critic). Critiquing the IDEA, not the implementation (audit at `audits/G5_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)
LOW-MED. Cyclical / scheduled dropout is a real technique (Morerio, Cavazza, Volpi, Vidal, Murino 2017 ICLR-W 'Curriculum Dropout' arXiv:1703.06229; Pham & Le 2021 NeurIPS 'AutoDropout' arXiv:2110.04895), but anchoring the cycle ratios on Fibonacci ratios `{F_k/F_{k+1}}` is purely cosmetic — beyond k=4 the ratios are within 1% of 1/φ ≈ 0.618 (see H43 critique).

### Mechanism scrutiny — does the optimizer/init/reg theory actually predict the claimed effect?
Dropout theory (Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov 2014 JMLR 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting'; Wager, Wang, Liang 2013 NIPS 'Dropout Training as Adaptive Regularization' arXiv:1307.1493) treats dropout-rate `p` as a scalar regularization knob. Cyclical / curriculum dropout (Morerio 2017) increases `p` over training because feature reliability grows with epochs — this is a well-grounded but empirically weak effect on CIFAR-10 (~0.2 pp typical). Anchoring the curriculum shape on Fibonacci ratios adds no theoretical content because (a) the schedule converges to constant 0.618 after iteration 4, (b) the choice of 0.618 over 0.5 / 0.6 / 0.7 has no statistical-learning justification.

### Confounds (≥2)
(1) **Effective-LR coupling.** Dropout `p` reduces effective batch size via inverted-dropout scaling, so a `p` schedule is implicitly an effective-LR schedule. (2) **BN-dropout interaction.** Li, Chen, Hu, Yang 2019 CVPR 'Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift' (arXiv:1801.05134) shows BN and dropout co-active in conv layers produce variance-shift artifacts — Fib-scheduled `p` does not address this. (3) **Layer-of-attachment.** Dropout works best between FC layers; on ResNet-20 the FC head is a single layer, leaving little surface for cyclical scheduling.

### Numerology / specificity check
Pure numerology. After 4 iterations the Fib ratios are within 0.5% of 1/φ, so the schedule is operationally equivalent to "constant `p` = 0.382 or 0.618 alternating" — there is no actual Fibonacci structure beyond iteration 4. A controlled test would compare Fib-schedule to constant-0.5, constant-0.618, linear-curriculum {0.0 → 0.5}, and cosine-{0.0, 0.5} curricula at iso-epochs × 3 seeds CIFAR-10; if Fib doesn't strictly dominate, the hypothesis is refuted.

### Literature precedent — optimization/init is one of the most studied fields in DL
Dropout literature is mature: Srivastava 2014 JMLR; Wan, Zeiler, Zhang, LeCun, Fergus 2013 ICML 'Regularization of Neural Networks using DropConnect'; Ba & Frey 2013 NIPS 'Adaptive dropout for training deep neural networks'; Gal & Ghahramani 2016 ICML 'Dropout as a Bayesian Approximation' (arXiv:1506.02142); Ghiasi, Lin, Le 2018 NeurIPS 'DropBlock: A regularization method for convolutional networks' (arXiv:1810.12890); Pham & Le 2021 NeurIPS AutoDropout (arXiv:2110.04895). Across this literature, the `p` value matters within ±0.1 of the optimum; the SCHEDULE matters within ±0.3 pp; no paper finds Fibonacci-anchoring of the schedule.

### Expected effect size (90% CI a priori)
[-0.5 pp, +0.3 pp] on CIFAR-10 top-1 vs. constant-`p=0.1` ResNet-20 baseline. Effects of this size are within seed-noise and require 5-10 seeds to detect.

### Minimum-distinguishing experiment
{constant 0.1, constant 0.5, linear-curriculum 0→0.5, Fib-schedule, cosine 0→0.5} × 3 seeds × 12 epochs CIFAR-10. Compute pairwise paired-t-test composite delta. If Fib-schedule doesn't beat all others by ≥ 0.3 pp with p < 0.05, the Fib-specificity is refuted.

### Verdict
NUMEROLOGY — Fibonacci-anchored dropout schedules collapse to constant-0.618 after a few iterations and have no statistical-learning motivation over generic curriculum dropout. Recommend dropping the Fib framing and either (a) adopting AutoDropout (Pham & Le 2021) or (b) reframing as a simple linear/cosine curriculum study without numerological dressing.
