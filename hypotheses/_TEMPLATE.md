# H{{NN}} — {{Short Title}}

> **One-line claim:** {{≤ 25-word imperative falsifiable claim}}.
>
> **Source design space:** {{which IDEA_TABLE group, e.g., G3 Topologies
> & Graphs (H21–H30)}}.
>
> **Implementation status (this repo):**
> `○ planned | ~ partial | ✓ done | ✗ disproved | ♻ superseded`.

This document is the committee-grade design write-up for hypothesis
H{{NN}}. Every section below is mandatory; the word-count floors are
the same as the autoresearch reasoning-entry gates.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

{{Why does nature use this pattern? Cite the natural example
(phyllotaxis, honeycomb, dodecahedron, Chladni nodes, etc.) and
explain WHY it is optimal at the relevant scale (information packing,
symmetry breaking, energy minimization, periodic encoding, etc.).
Connect to the source PDF / Grok transcript phrasing. Do not stop at
"because nature did it"; explain the physical / informational
optimality.}}

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

{{State the claim in present tense, mechanism + expected effect, with
quantitative bounds. The hypothesis MUST contain the word "mechanism"
or "because" or "per [paper]". Example phrasing:
"Because <prior X> changes <Y> in the model, mechanism-wise <Z>; per
<cited paper>, we expect <effect> on <metric>."}}

## 3. Falsifier (≥ 30 words)

{{Quote the single observation that would DISCARD this hypothesis.
Make it specific and numeric. Examples:
"If composite Δ ≤ -0.005 on CIFAR-10 at 3-seed median, this idea is
DISCARDED."
"If rotation equivariance error on rotated-CIFAR is NOT reduced by
≥ 0.03 versus the priors-off reference, this idea fails its purpose."}}

## 4. Citations (Citation Rigor format, ≥ 40 / 80 words)

{{Every citation in the exact format:

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) -- one-sentence relevance note explaining why we
cite it here.
```

At least one primary citation per hypothesis; multi-paper citations
must reach 80 words total.}}

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track

{{The exact implementation as a PyTorch primitive. Include:
- Shapes (B, C, H, W) before / after
- Computational cost (params, FLOPs) vs. the priors-off reference
- Init / training implications
- A short PyTorch code sketch (≤ 30 lines)
- Where it lives in `src/nature_inspired_networks/` and which
  `ideas/<NN>/implementation.py` re-exports it}}

### 5.2 LLM track (decoder-only Transformer)

{{Per the extended-transcript chunk-4 mapping: how the same prior
applies to a GPT-style decoder layer.
- Where it slots: Token Emb / Pos Enc / MHSA / QKV / RoPE / FFN /
  Residual / RMSNorm / LM head
- FlashAttention-2 compatibility notes
- Causal-mask preservation notes
- Short PyTorch sketch
- Expected impact on perplexity / KV cache / latency at 124M–1B scale}}

## 6. Predicted Δ on 4090 benchmarks (the pre-registered prediction)

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [{{lo}}, {{hi}}] | {{one sentence}} |
| top-1 (CNN) / perplexity (LLM) | [{{lo}}, {{hi}}] | {{one sentence}} |
| params | [{{lo}}, {{hi}}] | {{one sentence}} |
| FLOPs | [{{lo}}, {{hi}}] | {{one sentence}} |
| GPU latency (batch=1) | [{{lo}}, {{hi}}] | {{one sentence}} |
| rotation-equivariance err | [{{lo}}, {{hi}}] | {{one sentence}} |
| KV cache @ 32k (LLM) | [{{lo}}, {{hi}}] | {{one sentence}} |
| Betti collapse rate | [{{lo}}, {{hi}}] | {{one sentence}} |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: {{which Tier-1 dataset}}
- Architecture: {{which scaffold + which sub-modules}}
- Epochs / batch / precision / seeds
- Composite formula + SHA-256 fingerprint
- Run-script invocation (exact)
- Wall-clock estimate on RTX 4090 (16 GB)
- Archive path: `ideas/<NN>/experiments/expNNN_<short>/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

{{Many priors are uninformative on isotropic CIFAR-10. State the
dataset / setup where this prior should actually win, even if it costs
more compute. Examples:
- group conv → rotated-CIFAR / IcoMNIST
- toroidal → wrap-aware synthetic
- cymatic-init → convergence-speed test on small subset
- fractal → deep-net regime at fixed param budget
- φ-LR scheduler → no-tuning baseline on Tiny ImageNet}}

### 7.3 Cross-paradigm context (LLM track)

{{The LLM-track 124M–1B scale protocol per the extended transcript.}}

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` § G{{group}} row H{{NN}}.
- Master experiment list: `EXPERIMENT_LOG.md` Tier {{T}} row {{T.N}}.
- Implementation sub-directory: `ideas/{{NN_short}}/`
- Related hypotheses that compose: {{list of related H-IDs}}
- Related hypotheses that conflict: {{list, if any}}

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of {{cited paper}}?**

> {{Specific answer differentiating this work}}

**Q: How is this falsifiable rather than aesthetic?**

> {{Reference § 3 and pre-registered prediction in § 6}}

**Q: What if the prior helps on CIFAR-10 but hurts on ImageNet?**

> {{State the scope of the claim + § 7.2 targeted experiment}}

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> {{Acknowledge `FINDINGS.md` — yes for the full hybrid; this
> hypothesis's single-prior contribution is the unit of analysis}}

**Q: How do we know the implementation is correct?**

> {{Reference `tests/test_<module>.py` + the
> `ideas/<NN>/experiments/expNNN_<short>/verification/` directory}}

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/<NN>/implementation.py` exists and tests green
- [ ] `ideas/<NN>/tests.py` ≥ N assertions covering forward shape +
      every branch + at least one regression test
- [ ] `ideas/<NN>/AUDIT.md` lists ≥ 3 self-found weaknesses
- [ ] `ideas/<NN>/IMPROVEMENTS.md` records the fixes
- [ ] `ideas/<NN>/VERIFY.md` is signed with a real date
- [ ] At least one experiment archive under
      `ideas/<NN>/experiments/expNNN_<short>/`
- [ ] That archive carries its own `verification/{tests.txt,
      smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Result reflected in `FINDINGS.md` and dashboard

## 11. Status journal

(Append-only timeline of what changed when.)

- {{YYYY-MM-DD}} — Created from template.
- {{YYYY-MM-DD}} — Implementation written, tests green (29/29).
- {{YYYY-MM-DD}} — Experiment expNNN launched on 4090.
- {{YYYY-MM-DD}} — Verdict written, dashboard refreshed.
