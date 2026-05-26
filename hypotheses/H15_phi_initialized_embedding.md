# H15 — phi-Initialized Embedding (token embeddings from golden-spiral lattice)

> **One-line claim:** Initialising nn.Embedding weights as projections
> of the golden spiral lattice in 2D and then expanding to d_model yields
> lower perplexity than Xavier or Gaussian init at the same param count.
>
> **Source design space:** G2 Layer/Channel/Neuron (H11-H20).
>
> **Implementation status (this repo):**
> `o planned`.

This document is the committee-grade design write-up for hypothesis
H15.

---

## 1. Motivation (sacred / natural intuition, >= 100 words)

Token embeddings define the geometry of the input space for a language
model. The standard Xavier/Gaussian init samples each entry independently
from a normal distribution, producing an isotropic but uninformative
prior. Recent work (Mu and Viswanath 2018, "All but the Top") found
that pretrained embeddings have a heavy-tailed structure where the
top-1 principal direction explains a disproportionate fraction of
variance; removing this direction improves downstream tasks. The
golden-spiral lattice is the unique 2D arrangement that minimises
overlap between points while maintaining uniform angular density: each
new point is rotated by 137.5 degrees from the previous one. Projecting
this 2D structure into d_model dimensions via a fixed random orthogonal
matrix preserves the angular non-overlap property in high dimensions.
The hypothesis is that this structured init provides a better starting
prior than isotropic Gaussian because it pre-equips the embedding space
with the phyllotactic angular separation -- making it easier for
training to discover semantic neighborhoods.

## 2. Formal hypothesis (the falsifiable claim, >= 50 words)

Because phi-initialised embeddings are sampled from a golden-spiral
lattice (each token at angle k * 137.5 deg in 2D, then projected to
d_model), the mechanism by which they should reduce perplexity is that
the phyllotactic angular spacing pre-equips the embedding space with
non-overlapping token directions, accelerating convergence. Per Mu and
Viswanath 2018 we expect WikiText-103 validation perplexity to drop
by 0.3-0.8 over Xavier init at 1 epoch training.

## 3. Falsifier (>= 30 words)

If phi-initialised embeddings do NOT reduce WikiText-103 validation
perplexity (3-seed median, 124M decoder, 1 epoch) by at least 0.2 over
Xavier init AND fail to demonstrate a measurable angular-separation
improvement at convergence, the hypothesis is FALSIFIED.

## 4. Citations (Citation Rigor format, >= 80 words)

```
Mu, Jiaqi, Viswanath, Pramod 2018 ICLR 'All-but-the-Top: Simple and
Effective Postprocessing for Word Representations' (arXiv:1702.01417)
-- demonstrates that the geometry of word embeddings has a heavy top-
direction. H15 pre-empts this by initialising along a maximally non-
overlapping lattice.

Mikolov, Tomas, Chen, Kai, Corrado, Greg, Dean, Jeffrey 2013 ICLR
'Efficient Estimation of Word Representations in Vector Space'
(arXiv:1301.3781) -- word2vec; the canonical embedding-quality
baseline against which init schemes are compared.

Vogel, Helmut 1979 Math Biosciences 'A better way to construct the
sunflower head' -- the 137.5 deg golden-angle construction; the
biological precedent for maximally non-overlapping angular lattices.
```

## 5. Mechanism (the deep technical detail)

### 5.1 CNN track (limited applicability)

CNN-track has no embedding tables. The closest analogue is
**positional encoding init**: replace standard sin/cos positional
encodings with golden-spiral PE (cf. H36).

### 5.2 LLM track (the natural home for H15)

For a vocabulary of size V and embedding dim D, generate V points on a
2D golden spiral:

```python
PHI = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = 2 * math.pi / PHI ** 2  # ~ 2.399 rad = 137.5 deg

def golden_spiral_2d(V):
    indices = torch.arange(V).float()
    theta = indices * GOLDEN_ANGLE
    r = torch.sqrt(indices / V)
    return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)
```

Project to D dimensions via a fixed random orthogonal matrix Q (shape
(2, D)):

```python
def phi_init_embedding(V, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    spiral_2d = golden_spiral_2d(V)  # (V, 2)
    # random 2 -> D orthogonal projection
    Q_full = torch.randn(D, D, generator=g)
    Q, _ = torch.linalg.qr(Q_full)
    Q = Q[:, :2]  # (D, 2)
    embedding = spiral_2d @ Q.T  # (V, D)
    # rescale to match Xavier variance
    embedding *= math.sqrt(2 / D)
    return embedding
```

For V = 50k, D = 768: embedding tensor is 50k * 768 = 38.4M params,
same as Xavier baseline. The *initial* embedding has angular separation
ratio (max angle between any two embeddings) / (min angle) approaching
phi**2 = 2.618 by construction. After training, this structure may
persist or wash out.

Causal mask, FlashAttention-2, KV cache: all unchanged (init only).

Variant: initialise *only the first 2 principal axes* with the golden
spiral and leave the remaining D - 2 dimensions Xavier. This is the
"low-rank phi init" variant.

```python
class PhiInitTransformer(nn.Module):
    def __init__(self, vocab_size=50000, d_model=768, init='phi', ...):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        if init == 'phi':
            with torch.no_grad():
                self.embed.weight.data = phi_init_embedding(vocab_size,
                                                              d_model)
        elif init == 'phi_lowrank':
            with torch.no_grad():
                base = phi_init_embedding(vocab_size, 2)
                self.embed.weight.data[:, :2] = base
                # leave [:, 2:] as Xavier
        # ... rest of standard decoder
```

Location: `src/nature_inspired_networks/embedding.py:phi_init_embedding`,
re-exported by `ideas/15_phi_embedding_init/implementation.py`.

Expected impact at 124M scale: WikiText-103 ppl drops by 0.3-0.8 at
1 epoch; gap may close at convergence (more training). KV cache /
latency unchanged.

## 6. Predicted Delta on 4090 benchmarks (the pre-registered prediction)

| metric | Delta vs baseline | rationale |
|---|---|---|
| composite | [-0.002, +0.015] | small ppl drop |
| top-1 (CIFAR-10) | N/A | CNN-track does not apply |
| perplexity (WikiText-103, 1 epoch) | [-0.8, -0.3] | better init |
| perplexity (WikiText-103, converged) | [-0.2, +0.0] | gap may close |
| params | [0, 0] pct | init-only change |
| FLOPs | [0, 0] pct | init-only |
| GPU latency (batch=1) | [0, 0] pct | init-only |
| rotation-equivariance err | N/A | LM task |
| KV cache @ 32k (LLM) | [0, 0] pct | unchanged |
| Betti collapse rate | [+0.02, +0.05] | structured init compresses faster |

## 7. Experimental protocol (the 4090-feasible plan)

### 7.1 Primary experiment

- Dataset: **WikiText-103** (103M training tokens, 0.2M val, vocab ~50k)
- Architecture: 124M decoder (GPT-2-small scaffold)
- Conditions: {Xavier baseline, Gaussian-scaled, phi-spiral full,
  phi-spiral low-rank (first 2 dims only), uniform on sphere}
- Epochs / batch / precision / seeds: 1 epoch (~1.5B tokens), batch
  16, bf16 + grad-ckpt + FlashAttention-2, seeds {0, 1, 2}
- Composite formula: `0.85 * (1 - ppl/30) + 0.15 * (1 - latency/200ms)`
- Run-script: `python scripts/run_llm.py --config
  configs/h15_phi_embed.yaml --seeds 0 1 2`
- Wall-clock: 5 configs * 3 seeds * ~5 hours = ~75 hours (out of
  single-week budget). Compromise: 2 seeds, ~50 hours.
- Archive: `ideas/15_phi_embedding_init/experiments/
  exp001_wikitext_init/`

### 7.2 Idea-targeted experiment (where this prior should SHINE)

TinyStories (50M tokens), where init effects are amplified by the
small dataset. Predict +0.5 ppl gap consistent across seeds. Budget:
1 epoch is ~30 min on 4090.

### 7.3 Cross-paradigm context (CNN-side mirror)

The CNN mirror is H36 (phi-spiral positional encoding). Not run here
to avoid double-counting.

## 8. Cross-references

- Parent design space row: `IDEA_TABLE.md` section G2 row H15
- Master experiment list: `EXPERIMENT_LOG.md` (new T4.x row)
- Implementation sub-directory: `ideas/15_phi_embedding_init/`
- Related hypotheses that compose: H36 (phi-spiral PE), H42 (phi
  weight init), H34 (golden RoPE)
- Related hypotheses that conflict: Xavier baseline

## 9. Committee Q&A (anticipating fake-reviewer pushback)

**Q: Why isn't this just a re-packaging of "All-but-the-Top" (Mu and
Viswanath 2018)?**

> All-but-the-Top is a *post-processing* technique that removes the
> dominant direction from pretrained embeddings. H15 is a *pre-training*
> init: the structure is built in at init and may persist through
> training. The two are complementary, not competing.

**Q: How is this falsifiable rather than aesthetic?**

> Section 3 commits to >= 0.2 ppl reduction at 1 epoch + measurable
> angular-separation improvement at convergence.

**Q: What if the prior helps at 1 epoch but disappears at convergence?**

> Then the claim is restricted to "fast init for warm-start training";
> Status moves to `~ partial`. This is the predicted outcome at
> converged training, but the 1-epoch gap is non-trivial.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H15 is an LM prior; CIFAR has no embedding table. No overlap.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_embed.py` asserts (a) consecutive vocab indices have
> angle 137.5 deg +/- 0.01, (b) variance matches Xavier within 5 pct,
> (c) embedding norm ~= constant across vocab.

## 10. Verification artifacts checklist (committee evidence pack)

- [ ] `ideas/15_phi_embedding_init/implementation.py`; tests green
- [ ] `ideas/15_phi_embedding_init/tests.py` >= 10 assertions
- [ ] `ideas/15_phi_embedding_init/AUDIT.md` >= 3 weaknesses
- [ ] `ideas/15_phi_embedding_init/IMPROVEMENTS.md`
- [ ] `ideas/15_phi_embedding_init/VERIFY.md` signed
- [ ] Experiment archive
- [ ] `verification/{tests.txt, smoke.txt, gates.txt, reproduction.txt}`
- [ ] Row added to `EXPERIMENT_LOG.md`
- [ ] Reflected in `FINDINGS.md`

## 11. Status journal

- 2026-05-27 -- Created from template by Doc-Agent-A.
