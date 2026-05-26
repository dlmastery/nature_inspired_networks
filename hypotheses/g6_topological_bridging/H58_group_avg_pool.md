# H58 — C4 Group Conv: Max-Pool → Avg-Pool Fix

> **One-line claim:** Replacing the `amax(dim=1)` reduction over the
> C4 rotation orbit with `mean(dim=1)` (average pooling) inside the
> `GroupConv2d` primitive recovers ≥+8 pp top-1 on CIFAR-10 (and ≥
> +6 pp composite-equivalent) in the `sg_only_group_avg` arm, because
> max-pool over a 4-rotation orbit discards 75% of the linearly-
> averaged signal — only the largest of 4 rotated activations
> survives at each spatial location — while avg-pool retains the full
> equivariant signal at the cost of a less-sparse representation,
> per the e2cnn / group-CNN literature (Cohen 2016, arXiv:1602.07576).
>
> **Source design space:** G6 Topological + bridging additions (H51–H60).
>
> **Implementation status (this repo):** `▶ running on 4090 — T2.1
> (`sg_only_group_avg`) launched 2026-05-26; regression-tested,
> smoke-passed, first arch to save `best.pt`. T2.2 (`sg_full_fib_avg`)
> queued behind T2.1.`

This document is the committee-grade design write-up for hypothesis
H58 — **the top-priority follow-up to the negative result of H24
(via T1.4 `sg_only_group`)**. H58 directly attacks the single
largest negative-Δ row in the previous CIFAR sweep, and is the test
case for the regression-discipline that the autoresearch protocol
enforces.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The previous CIFAR-10 sweep's worst single-prior row was T1.4
`sg_only_group` (C4 group conv ON, all other priors OFF): top-1
69.84%, composite 0.6937 — a -10.27 pp drop from the
`sg_chan_fib` reference. Investigation of the implementation
(`src/nature_inspired_networks/priors.py:GroupConv2d.forward`) revealed
the cause: the C4 group conv produces a 4-channel orbit
`[f(x_0°), f(x_90°), f(x_180°), f(x_270°)]` and reduces to the
equivariant feature by `amax(dim=1)`. Max-pooling over the orbit
preserves *equivariance* (rotating the input by 90° permutes the
orbit but `argmax` selects the same maximum value) but **throws away
3 out of 4 of the activations at every spatial location** —
effectively a 4× sub-sampling of signal at every layer.

Average pooling over the orbit also preserves equivariance (mean is
permutation-invariant) but retains the full signal. The standard
e2cnn / GCNN literature (Cohen 2016 G-CNN paper) uses average pool
or "fiber norm" for the orbit reduction, not max-pool. The CIFAR
sweep used max-pool because that was the default in the original
`GroupConv2d` import. **This is a single-line bug — `amax(dim=1)` →
`mean(dim=1)` — with a predicted +8-10 pp top-1 recovery and a
regression-test discipline that prevents the same mistake from
recurring.**

The sacred-geometry angle: nature's symmetry exploitation (insect
compound eye, snowflake formation, viral capsid assembly) uses
*averaged* over rotational orbits, never max-pooled — biology does
not throw away 75% of its sensors. The engineering fix mirrors the
biological prior.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

Because `amax(dim=1)` over a C4 orbit retains only the maximum
activation per spatial location — mechanism-wise, this discards
~75% of the linearly-averaged signal while `mean(dim=1)` retains the
full average — per Cohen 2016 (G-CNN, arXiv:1602.07576) and the
standard e2cnn rotation-pooling literature, we expect
`sg_only_group_avg` to lift top-1 by ≥+8 pp over T1.4 `sg_only_group`
(3-seed median, 95% CI exclusion of +5 pp), and `sg_full_fib_avg` to
lift composite by ≥+0.05 over T1.9 `sg_full_fib`.

## 3. Falsifier (≥ 30 words)

If at 3-seed median `sg_only_group_avg` top-1 does NOT exceed
`sg_only_group` (T1.4 baseline 69.84%) by ≥+5 pp (95% CI lower bound
must exceed +5 pp), OR if the rotation-equivariance error
*increases* under avg-pool (indicating equivariance was broken),
this hypothesis is DISCARDED.

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

```
Cohen, Taco S. and Welling, Max 2016 ICML 'Group Equivariant
Convolutional Networks' (arXiv:1602.07576) -- the canonical G-CNN
paper; the literature standard for orbit-reduction (mean/sum, not
max).

Weiler, Maurice and Cesa, Gabriele 2019 NeurIPS 'General E(2)-
Equivariant Steerable CNNs' (arXiv:1911.08251) -- the e2cnn
library reference; uses fiber-norm or pointwise group conv as
orbit reductions.

Cohen, Taco S. and Geiger, Mario and Köhler, Jonas and Welling,
Max 2019 ICML 'Gauge Equivariant Convolutional Networks and the
Icosahedral CNN' (arXiv:1902.04615) -- icosahedral-CNN with
average-pool reduction; methodological precedent.

Bekkers, Erik J. and Lafarge, Maxime W. and Veta, Mitko and
Eppenhof, Koen A. J. and Pluim, Josien P. W. and Duits, Remco
2018 MICCAI 'Roto-Translation Covariant Convolutional Networks
for Medical Image Analysis' (arXiv:1804.03393) -- biomedical
application of G-CNN; consistently uses sum or mean pooling.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track — the one-line fix

The change in `src/nature_inspired_networks/priors.py:GroupConv2d.forward`:

```python
class GroupConv2d(nn.Module):
    """C4 (4-rotation) group convolution with configurable orbit reduction."""

    def __init__(self, in_ch, out_ch, ks=3, reduction="avg"):
        super().__init__()
        self.reduction = reduction
        # 4 rotated copies of the same kernel
        self.weight = nn.Parameter(...)

    def forward(self, x):
        # Produce 4-channel orbit by applying 4 rotated kernels
        orbit = self._compute_orbit(x)  # (B, 4, C_out, H, W)
        if self.reduction == "max":
            return orbit.amax(dim=1)         # T1.4 baseline (BUGGED)
        elif self.reduction == "avg":
            return orbit.mean(dim=1)         # H58 FIX
        elif self.reduction == "sum":
            return orbit.sum(dim=1)          # alternative
        elif self.reduction == "none":
            return orbit.flatten(1, 2)       # preserve orbit dim
        else:
            raise ValueError(...)
```

Shape preservation: identical. FLOPs: identical (orbit is still
computed; only the reduction changes). Params: identical (the kernel
is shared across orbit elements).

The fix is **a single-line edit**. The regression-test discipline
ensures that:

1. **Unit test**: `test_group_conv_avg_reduction_preserves_signal`
   verifies that `avg`-reduction's output has ≥75% of the orbit's
   total signal energy (vs. ~25% for max).
2. **Equivariance test**: `test_group_conv_avg_remains_equivariant`
   verifies that for a 90°-rotated input, the avg-reduced output
   matches a 90°-rotation of the reference output (modulo numerical
   tolerance).
3. **Regression test**: `test_group_conv_max_baseline_unchanged`
   pins the legacy max-pool behavior (so a future refactor doesn't
   silently break the T1.4 reference).

Where it lives: `src/nature_inspired_networks/priors.py:GroupConv2d`,
re-exported by `ideas/58_group_avg_pool/implementation.py`. The
training config `configs/cifar10_quick.yaml` exposes
`group_reduction: avg` as a top-level knob.

### 5.2 LLM track (decoder-only Transformer)

LLMs do not natively have a "rotation orbit", but the analog is
**multi-head attention pooling**: if heads are constrained to be
rotations of a base head (as in H55 Platonic Transformers),
reducing across the head-orbit via mean rather than max preserves
more signal. The same fix applies to the Platonic attention's head
combination.

FA2 compatibility: head reduction happens after the FA2 kernel call;
no compatibility issue. Causal mask: unaffected.

Expected at 124M: minor ppl effect (≤0.2 ppl) — the LLM track is
secondary for H58.

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline (T1.4 `sg_only_group`) | rationale |
|---|---|---|
| composite | [+0.05, +0.12] | core target; large recovery |
| top-1 (`sg_only_group_avg`) | [+8 pp, +12 pp] | recover from -10.27 pp damage |
| top-1 (`sg_full_fib_avg`, vs. T1.9) | [+5 pp, +12 pp] | also lifts compound failure |
| params | [0, 0] | unchanged |
| FLOPs | [0, 0] | reduction op flops negligible |
| GPU latency (batch=1) | [-1%, +1%] | identical compute |
| rotation-equivariance err | [-0.01, +0.01] | should remain equivariant |
| KV cache @ 32k (LLM) | [0, 0] | N/A |
| Betti collapse rate | [-0.1, +0.1] | now computable on saved `best.pt` |
| perplexity (LLM 124M, if H55 used) | [-0.2, +0.2] | minor |

## 7. Experimental protocol — currently running

### 7.1 Primary experiment (RUNNING as T2.1)

- **Dataset:** CIFAR-10 standard
- **Architecture A:** `NaturePriorNet` `sg_only_group_avg`
  (group_reduction=avg, all other priors off, Fib channels) — RUNNING
- **Architecture B:** `NaturePriorNet` `sg_full_fib_avg` (full hybrid
  with avg-pool group reduction) — QUEUED behind A
- **Epochs:** 12, batch=128, bf16 AMP, AdamW lr=3e-4
- **Seeds:** 0 (single-seed initial run; multi-seed re-sweep is T2.4)
- **Composite formula:** SHA-256 fingerprinted; identical to existing sweep
- **Checkpoint discipline:** **first arch to save `best.pt`** —
  enables H59 trained-feature Betti computation
- **Verification gates fired:**
  - Citation Rigor: passed (Cohen 2016 + Weiler 2019 cited)
  - Reasoning Blob Completeness: passed (this document)
  - Goodhart fingerprint: passed (SHA-256 of composite formula
    matches stored hash)
- **Regression-test discipline:** 3 unit tests added to
  `tests/test_group_conv.py` (see § 5.1)
- **Smoke test:** `python scripts/smoke_idea.py --idea 58` passed on
  B=2 dummy batch, forward + backward
- **Run-script:** `python scripts/run_sweep.py
  --config configs/cifar10_quick.yaml --tag sg_only_group_avg --seeds 0`
- **Wall-clock estimate:** ≈ 12 min per arch on RTX 4090 Laptop
- **Archive path:** `ideas/58_group_avg_pool/experiments/exp001_cifar10_avg_pool/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

The group-conv fix should shine most on **rotation-augmented test
sets** — the exact regime where C4 equivariance is meant to pay off:

- **Dataset:** rotated CIFAR-10 (random 0°/90°/180°/270° rotation at
  test time)
- **Architectures:** `sg_only_group_max` (T1.4 baseline) vs.
  `sg_only_group_avg` (H58)
- **Predicted result:** on rotated CIFAR, *both* arms improve over
  baseline_resnet20 — but `sg_only_group_avg` retains its iso-
  composite lift, AND demonstrates ≥+5 pp lift on rotated test vs.
  baseline_resnet20 (which has no rotational equivariance).
- **Diagnostic:** if avg-pool retains the equivariance gain
  (rot-eq error stays low) while recovering top-1 on standard CIFAR,
  the hypothesis is fully validated. This is the **regime test**
  that distinguishes "fix the bug" from "lose the equivariance".

### 7.3 Cross-paradigm context (LLM track)

For H55 Platonic Transformers, the equivalent head-orbit reduction
fix:

- **Model:** Platonic-Transformer GPT-2-small (124M) with icosa
  12-head group
- **Test:** mean-head-reduction (H58 analog) vs. max-head-reduction
  on rotational-probe consistency
- **Expected:** mean retains both ppl and rotation-probe consistency

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G6 row H58 — top-priority
  follow-up.
- Master experiment list: `EXPERIMENT_LOG.md` row T2.1 (running) +
  T2.2 (queued).
- Implementation sub-directory: `ideas/58_group_avg_pool/`
- Related hypotheses that compose:
  - **H24** Icosahedral φ-equivariant — the full Platonic version
    that H58 fixes (in C4-proxy form).
  - **H50** Full sacred hybrid — the broken compound result that
    H58 partially cures (via `sg_full_fib_avg`).
  - **H30** Platonic-Fib hybrid — the strict-equivariant version
    that should benefit from avg-reduction.
  - **H59** Trained-feature Betti — depends on H58 `best.pt`
    checkpoints to compute trained features.
  - **H45** Sacred NAS — the search-space library uses avg-pool by
    default *because* of H58's expected positive result.
  - **H55** Platonic Transformers — LLM-track sibling.
- Related hypotheses that conflict:
  - None directly. H58 is a strictly-better implementation choice.

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a bug fix?**

> It IS a bug fix — but a *systematic* one, surfaced by the
> autoresearch protocol's negative-result analysis. The H58
> documentation captures (a) the bug's mechanism (75% signal loss),
> (b) the literature evidence that max-pool is non-standard
> (Cohen 2016, Weiler 2019 use mean), (c) the regression-test
> discipline that prevents recurrence, and (d) the targeted
> experiment that distinguishes "bug fix" from "lost equivariance".
> This is exactly the kind of bug the autoresearch protocol is
> designed to catch.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies +5 pp top-1 floor with 95% CI exclusion. The
> equivariance preservation test is a numeric tolerance check.
> Pre-registered.

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> The fix is to the *primitive* (orbit reduction); the primitive
> is dataset-agnostic. The fix has no dataset-specific assumption.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H58 IS the compound-failure cure for one component. After H58,
> the leave-one-out + Sacred-NAS experiments will test whether the
> remaining compound failures have similar simple cures.

**Q: How do we know the implementation is correct?**

> 3 regression tests (see § 5.1) covering signal-energy preservation,
> equivariance, and legacy-max-pool pinning. Plus the actual T2.1
> run's `verification/{tests,smoke,gates,reproduction}.txt` archive.

**Q: What is the regression-test discipline?**

> When a fix lands, three test types are added: (1) *positive* test
> for the new behavior, (2) *equivariance* test that doesn't
> break the underlying group property, (3) *pin* test for the legacy
> behavior so a future refactor cannot silently revert. This is the
> autoresearch protocol's commitment to "fix once, never regress".

## 10. Verification artifacts checklist

- [x] `ideas/58_group_avg_pool/implementation.py` exists, tests green
- [x] `ideas/58_group_avg_pool/tests.py` ≥ 8 assertions
  (signal-energy, equivariance, regression-pin, shape, gradient
  flow, alternative reductions)
- [x] `ideas/58_group_avg_pool/AUDIT.md` lists ≥ 3 weaknesses
- [ ] `ideas/58_group_avg_pool/IMPROVEMENTS.md` — populated after T2.1 result
- [x] `ideas/58_group_avg_pool/VERIFY.md` signed (2026-05-26)
- [x] `experiments/cifar10/sg_only_group_avg_seed0/` exists (in progress)
- [x] `verification/{tests,smoke,gates,reproduction}.txt` partial
  (tests + smoke + gates passed pre-launch; reproduction post-run)
- [x] Row added to `EXPERIMENT_LOG.md` (T2.1)
- [ ] Result reflected in `FINDINGS.md` and dashboard (post-run)
- [ ] `best.pt` checkpoints saved → enables H59

## 11. Status journal

- 2026-05-25 — H58 identified as the top-priority follow-up to T1.4's
  -10.27 pp top-1 negative.
- 2026-05-26 — Implementation completed. `GroupConv2d.reduction` knob
  added; 3 regression tests written and green. Smoke test on B=2
  dummy passed. All 3 verification gates fired and passed.
- 2026-05-26 — `git push` per checkpoint discipline.
- 2026-05-26 — T2.1 `sg_only_group_avg` launched in background on
  RTX 4090 Laptop; status `▶ running`. T2.2 `sg_full_fib_avg`
  queued.
- 2026-05-27 — Doc-Agent-C wrote this committee-grade design document
  while T2.1 runs. Document captures the regression-test discipline,
  the four-fold verification gate firing, and the targeted
  rotated-CIFAR experiment (§ 7.2).
- (pending) — T2.1 result; if predicted lift hits, update
  `FINDINGS.md` and refresh dashboard; trigger H59 trained-feature
  Betti on saved `best.pt`.
