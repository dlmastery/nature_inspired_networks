# H83 — CollapseGatedAttention (learnable softmax-temperature attention)

> **One-line claim:** Multi-head attention with a learnable positive
> softmax temperature `tau` (via softplus) that interpolates between
> diffuse and peaked attention, plus an optional inference-time
> `collapse` sharpening toward argmax, matches or improves on
> fixed-temperature attention (within +0.0/−0.0 pp at iso-params on a
> small ViT/TinyStories LM) while learning per-layer temperatures that
> reveal which layers prefer concentrated vs. diffuse attention.
>
> **Source design space:** G8 esoteric extensions; concretely
> implementable bonus idea. Esoteric origin (acknowledged once): the
> double-slit "observer collapse" from wave to particle motivated the
> naming of the high-tau (diffuse, "wave") vs. low-tau (peaked,
> "particle") regimes; the operational object is standard
> scaled-dot-product attention with a learnable temperature.
>
> **Implementation status (this repo):** `● implemented` —
> `src/nature_inspired_networks/collapse_attention.py` +
> `tests/test_collapse_attention.py` (7 assertions, green).

---

## 1. Motivation (≥ 100 words)

Scaled-dot-product attention divides the query-key logits by a fixed
constant `sqrt(d_head)` before the softmax. That constant fixes the
*sharpness* of the resulting attention distribution for the entire
network, yet there is no a-priori reason every layer and head should
want the same concentration: early layers often benefit from broad,
context-mixing ("diffuse") attention while later layers may want
peaked, near-retrieval ("selective") attention. The temperature of a
softmax is exactly the knob that controls this concentration, and a line
of work makes attention concentration explicit and tunable — from
entropy-regularised attention to sparsemax/entmax, which replace softmax
with sparsity-controlled mappings entirely. Making the temperature a
*learnable positive scalar* (parameterised through softplus so it can
never go non-positive) lets each attention module discover its own
preferred concentration by gradient descent, at the cost of a single
parameter per module. A separate forward-time `collapse` knob can
additionally interpolate the post-softmax distribution toward its argmax,
giving a cheap inference-time sharpening control. The hypothesis is that
this added flexibility is at worst neutral (the model can learn
`tau≈1`, recovering the baseline) and potentially helpful, while the
learned per-layer temperatures are themselves interpretable diagnostics.

## 2. Formal hypothesis (≥ 50 words)

Because a fixed `sqrt(d)` scaling forces a single attention sharpness
across all layers and heads, **mechanism**-wise replacing it with a
learnable temperature `tau = softplus(tau_raw)` in `softmax(QKᵀ /
(sqrt(d)·tau))` lets each module tune its own concentration *because*
gradient descent can lower `tau` where peaked retrieval helps and raise
it where diffuse mixing helps; we therefore predict iso-or-better
validation loss/perplexity at +1 parameter per attention module and a
non-degenerate spread of learned `tau` across layers.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median, the learnable-temperature attention scores **worse
than** the fixed-`sqrt(d)` baseline by more than the run-to-run noise
band on the chosen task (ViT CIFAR-10 top-1 or small-LM val PPL),
**OR** if every layer's learned `tau` collapses to within 5 % of 1.0
(showing the model gains nothing from the freedom), the hypothesis is
**DISCARDED**. The `collapse` knob is evaluated at inference only in v0;
training-time collapse scheduling is the targeted follow-up.

## 4. Citations (Citation Rigor format, ≥ 40 words)

```
Martins, Astudillo 2016 ICML 'From Softmax to Sparsemax: A Sparse
Model of Attention and Multi-Label Classification'
(arXiv:1602.02068) -- the neutral precedent for replacing/controlling
the softmax to tune attention sparsity; a learnable temperature is the
smooth, single-parameter analogue of sparsemax's concentration control.

Peters, Niculae, Martins 2019 ACL 'Sparse Sequence-to-Sequence
Models (entmax)' (arXiv:1905.05702) -- the alpha-entmax family that
interpolates between softmax (diffuse) and sparsemax (peaked); the
learnable tau plays the same diffuse<->peaked role with a single
scalar instead of the alpha hyper-parameter.

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
Polosukhin 2017 NeurIPS 'Attention Is All You Need'
(arXiv:1706.03762) -- the scaled-dot-product attention whose fixed
sqrt(d) scaling this module generalises to a learnable temperature.
```

## 5. Mechanism

### 5.1 LLM / Transformer track (primary)

`CollapseGatedAttention` is a standard multi-head self-attention block:
q/k/v/out linear projections, `(B, N, D)` → `(B, N, D)`. The only change
is the logit scaling: instead of `1/sqrt(d_head)` it uses
`1/(sqrt(d_head)·tau)` where `tau = softplus(tau_raw) + 1e-3` (the small
floor guarantees `1/(sqrt(d)·tau)` cannot overflow to inf and produce
NaNs when `tau_raw` is driven strongly negative). `tau_raw` is
initialised so the effective `tau ≈ tau_init` (default 1.0, reproducing
the baseline at step 0). An optional `attn_mask` is added to the logits
before softmax (supports causal masking). A forward-time `collapse ∈
[0,1]` argument linearly interpolates the post-softmax distribution
toward its per-row argmax one-hot, preserving the simplex (rows still
sum to 1). Low `tau` ⇒ peaked ("particle"); high `tau` ⇒ diffuse
("wave"). Param overhead: +1 scalar per attention module.

### 5.2 CNN track

A ViT-Tiny on CIFAR-10 uses the same module unchanged (self-attention
over patch tokens). For a pure-conv ResNet there is no attention to
modify, so the CNN-track applies only to attention-augmented conv
models (ViT / hybrid). Shapes are identical to standard MHA, so it is a
drop-in replacement for `nn.MultiheadAttention` in the
batch-first / self-attention case.

## 6. Predicted Δ (pre-registered)

| metric | Δ vs. fixed-sqrt(d) baseline | rationale |
|---|---|---|
| ViT CIFAR-10 top-1 (12 ep) | [-0.2, +0.8] pp | per-layer tuned sharpness |
| small-LM val PPL | [-0.10, +0.02] nats | tuned concentration |
| params | [+0.0 %, +0.001 %] | one scalar per attn module |
| FLOPs | [+0.0 %, +0.1 %] | one extra softplus + divide |
| learned tau spread (max/min across layers) | [≥ 1.5×] | non-degenerate spread |
| collapse=1.0 eval top-1 | [-3.0, +0.5] pp | sharpening can help or hurt |

## 7. Experimental protocol

1. **Unit tests (Phase 0):** `tests/test_collapse_attention.py` — 7
   assertions: forward shape `(B,N,D)→(B,N,D)`; low `tau` yields sharper
   (higher row-max) attention than high `tau` on shared input; `tau`
   stays positive via softplus even at `tau_raw=-100` with finite
   output; gradient flows to `tau_raw`; attention rows sum to 1;
   `collapse=1` is one-hot and still on the simplex while `collapse=0`
   is more diffuse; additive `-inf` causal mask zeroes future attention.
   All green.
2. **SOTA smoke (Phase 1):** unchanged baseline (ViT or LM) must hit its
   expected band before the variant (Rule 13).
3. **Hypothesis smoke (Phase 2):** one sweep row flips a
   `collapse_attention` flag swapping the attention block for
   `CollapseGatedAttention(tau_init=1.0)`. Log per-layer learned `tau`
   to `history.json`. A separate eval-only row sweeps `collapse ∈ {0,
   0.5, 1.0}`.

## 8. Cross-references

- Attention-axis sibling to H62 (toroidal KV / hex attention) and H66
  (cymatic QKV kernel); H83 modifies only the softmax temperature, so it
  composes orthogonally with both.

## 9. Verification checklist

- [x] `src/nature_inspired_networks/collapse_attention.py` exists.
- [x] Learnable `tau` via softplus (floored for numerical safety);
      `tau_init=1.0` reproduces baseline scaling at init.
- [x] `tests/test_collapse_attention.py` ≥ 4 assertions (has 7), green.
- [x] Low/high-tau sharpness (mechanism), tau-positivity + finite output
      (mechanism), tau-gradient (regression), rows-sum-to-1 (mechanism),
      collapse + causal mask (edge) asserted.
- [x] `# TODO runner wiring:` block describes the attention-swap flag
      without touching `models.py` / `blocks.py`.
- [ ] Phase-2 sweep row (lead wires the flag).

## 10. Status journal

- 2026-05-27 — Implemented + tested (7/7 green) as a standalone G8
  primitive. Added a 1e-3 temperature floor after the unit test caught
  an inf/NaN at extreme negative `tau_raw`; documented in the module.
