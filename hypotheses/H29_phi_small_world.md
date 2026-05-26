# H29 — φ-Small-World

> **One-line claim:** Setting the rewiring probability of a Watts-Strogatz small-world graph to `p = 1/φ ≈ 0.618` produces a GNN with better node-classification accuracy and shorter average path length than `p ∈ {0.1, 0.5, 0.9}` baselines at matched parameter budget.
>
> **Source design space:** G3 Topologies & Graphs (H21–H30).
>
> **Implementation status (this repo):** `○ not started`.

This document is the committee-grade design write-up for hypothesis H29.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The small-world phenomenon (Watts & Strogatz 1998) describes networks with high local clustering AND short average path length — a hallmark of biological neural systems (C. elegans connectome), social networks (six-degrees-of-Kevin-Bacon), and the World Wide Web. Watts–Strogatz construct it by starting from a regular ring lattice and **rewiring** each edge with probability p; the resulting network transitions from regular (p=0) to random (p=1), and "small-world" emerges in a window around p ≈ 0.1–0.3.

The hypothesis here is that **p = 1/φ ≈ 0.618** is the optimal rewiring probability: it is the most-irrational fraction less than 1 (golden-ratio reciprocal), so the rewired-vs-kept partition is maximally aperiodic and produces the most uniform mixture of local and global edges. Sacred-geometry literature emphasizes φ as nature's preferred ratio for balanced subdivision; small-world networks at p = 1/φ should achieve the **best balance of locality and reach**. The biological motivation is the brain: cortical small-world connectivity is empirically estimated at p ≈ 0.5–0.7 (Bullmore & Sporns 2009), bracketing 1/φ.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** 1/φ is the most-irrational fraction less than 1, a Watts-Strogatz graph with rewiring probability p = 1/φ exhibits the optimal balance of local clustering and path length, raising node-classification accuracy on Cora / Citeseer / Pubmed by ≥ +1 pp and reducing average shortest-path length by ≥ 10 % relative to the standard p = 0.1 baseline, per the mechanism of Watts & Strogatz 1998 and Bullmore & Sporns 2009.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on Cora + Citeseer + Pubmed, the p = 1/φ variant fails to lift node-classification accuracy by ≥ 0.5 pp AND fails to show statistically distinct path-length distribution from p = 0.5 (Watts–Strogatz "default plateau"), this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Watts, D. J., Strogatz, S. H. 1998 Nature 'Collective dynamics of
small-world networks' — the foundational small-world paper; defines
the Watts–Strogatz model we modulate.

Bullmore, E., Sporns, O. 2009 Nature Reviews Neuroscience 'Complex
brain networks: graph theoretical analysis of structural and
functional systems' — biological evidence for small-world cortical
connectivity at p ≈ 0.5–0.7.

Kipf, T. N., Welling, M. 2017 ICLR 'Semi-Supervised Classification
with Graph Convolutional Networks' (arXiv:1609.02907) — GCN
architecture we use for node classification.

Sen, P., et al. 2008 AI Magazine 'Collective Classification in
Network Data' — Cora / Citeseer / Pubmed dataset citation.

Hu, W., et al. 2020 NeurIPS 'Open Graph Benchmark' (arXiv:2005.00687)
— OGB methodology.
```

## 5. Mechanism

### 5.1 CNN / GNN track

Generate a Watts–Strogatz random graph with `p = 1/φ`, use as a fixed adjacency for a GCN / GAT, and compare to `p ∈ {0.1, 0.5, 0.9}` controls.

```python
# ideas/29_phi_small_world/implementation.py
import networkx as nx
PHI = (1+5**0.5)/2

def ws_phi_graph(N, k=6, seed=0):
    return nx.watts_strogatz_graph(N, k, p=1.0/PHI, seed=seed)

class PhiSmallWorldGCN(nn.Module):
    def __init__(self, N, F_in, F_out, k=6):
        super().__init__()
        G = ws_phi_graph(N, k)
        A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=torch.float32)
        A = A + torch.eye(N)  # self-loops
        D = A.sum(1).rsqrt()
        self.register_buffer("A_hat", (D.view(-1,1)*A*D.view(1,-1)))
        self.lin = nn.Linear(F_in, F_out)
    def forward(self, x):
        return self.lin(self.A_hat @ x)
```

- Params: identical to standard GCN.
- FLOPs: identical (adjacency density same as p=0.5 control).
- Init: standard.

### 5.2 LLM track

For Transformers, p = 1/φ rewiring maps onto **sparse attention pattern**: each token connects to its k local neighbours plus rewired global "shortcut" links sampled at p = 1/φ. Combined with H62 (toroidal KV + hex), this gives long-range information flow with sub-linear cost.

```python
def phi_small_world_mask(L, k=8):
    mask = torch.zeros(L, L, dtype=torch.bool)
    # local k-ring
    for i in range(L):
        for d in range(1, k//2 + 1):
            mask[i, (i-d) % L] = True
            mask[i, (i+d) % L] = True
    # rewire with p = 1/phi
    for i in range(L):
        for j in range(L):
            if mask[i,j] and torch.rand(1).item() < 1/PHI:
                mask[i,j] = False
                k_new = torch.randint(0, L, (1,)).item()
                mask[i, k_new] = True
    # apply causal
    mask = mask & torch.tril(torch.ones_like(mask, dtype=torch.bool))
    return mask
```

Expected at 124 M: **perplexity neutral**, KV cache reduces 5–8×, latency improves at long context.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (Cora) | [+0.005, +0.020] | small but real graph-topology effect |
| Node-class acc (Cora, primary) | [+1.0 pp, +3.5 pp] | direct claim |
| Avg shortest-path length | [-10 %, -25 %] | small-world property |
| params | [0, 0] | adjacency change only |
| FLOPs | [-5 %, +5 %] | depends on rewiring sparsity |
| GPU latency | [≈1.0×, ≈1.05×] | trivial |
| KV cache @ 32 k (LLM) | [-70 %, -85 %] | sparse pattern |
| Betti collapse rate | [+0.05, +0.20] | small-world topology accelerates collapse |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** Cora + Citeseer + Pubmed (transductive node classification).
- **Architecture:** 2-layer GCN with Watts–Strogatz adjacency at p ∈ {0.1, 1/φ, 0.5, 0.9}.
- **Epochs / batch / precision / seeds:** 200 epochs, full-batch, fp32, 3 seeds × 4 p-values.
- **Composite:** Accuracy (0.7), path-length (0.2), params (0.1).
- **Run-script:** `python ideas/29_phi_small_world/experiment.py --p 0.1 0.618 0.5 0.9 --seeds 0 1 2`.
- **Wall-clock:** ~10 min/seed × 3 × 4 = 120 min.
- **Archive:** `ideas/29_phi_small_world/experiments/exp001_cora_*/`.

### 7.2 Idea-targeted experiment

1. **Cora / Citeseer / Pubmed** — classic small-world testbed.
2. **OGB node-property prediction** (ogbn-arxiv).
3. **Social-graph datasets (Reddit, Pokec)** — naturally small-world.
4. **Synthetic small-world generated at varying p** — confirm theoretical path-length curve.

### 7.3 Cross-paradigm context (LLM track)

WikiText-103 at 124 M with `phi_small_world_mask` attention pattern. Train 100 k steps, evaluate perplexity and per-context-length latency curves.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G3 row H29.
- Master: planned Tier 3.
- Sub-dir: `ideas/29_phi_small_world/`.
- Composes: H21 (hex topology), H23 (Platonic graph — opposite structured-prior), H62 (LLM toroidal KV + hex attn).
- Conflicts: H23 / H24 / H30 (structured Platonic priors are opposite of random rewiring).

## 9. Committee Q&A

**Q: Why p = 1/φ specifically, not 1/π or any other irrational < 1?**

> 1/φ is the slowest-converging continued fraction (all 1s), making it the most "uniformly irrational". This produces the maximally-aperiodic rewiring pattern. 1/π is a control we will also run to discriminate; if 1/π gives similar results, the φ-specificity collapses.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives node-class accuracy ≥ +0.5 pp AND path-length distribution differs from p=0.5. Either failure discards.

**Q: What if p = 0.5 is statistically indistinguishable from p = 1/φ?**

> That is a legitimate negative result — the φ-specificity is small (0.618 vs 0.5 is only 0.118 absolute difference), so we pre-commit to a Wilcoxon rank-sum test at α = 0.05 to discriminate.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H29 is graph-not-image and tested as a single prior; compounding is H62 / H67.

**Q: How do we know the implementation is correct?**

> `tests/test_phi_small_world.py::test_clustering_coefficient` asserts the clustering coeff at p=1/φ falls in [0.05, 0.20] (Watts–Strogatz predicted range). `test_path_length` asserts avg path length < 3 × log(N)/log(k). `test_reproducible_seed` ensures the rewiring is deterministic given seed. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/29_phi_small_world/implementation.py` tests green
- [ ] `ideas/29_phi_small_world/tests.py` ≥ 5 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 3
- [ ] FINDINGS reflects result

## 11. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B.
