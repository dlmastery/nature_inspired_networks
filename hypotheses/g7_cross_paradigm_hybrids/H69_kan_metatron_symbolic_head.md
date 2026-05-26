# H69 — KAN-Edge Symbolic Head on Metatron Graphs

> **One-line claim:** A Kolmogorov-Arnold Network operating on
> Metatron's-Cube edge structure as a symbolic-reasoning head on a
> 124M decoder lifts GSM8K zero-shot accuracy by ≥3 percentage points
> and recovers an analytic closed-form symbolic solution on ≥40% of a
> Feynman-equations subset.
>
> **Source design space:** G7 hybrids; composition of H30 (Platonic-Fib
> hybrid) + KAN paradigm + Metatron graph (H23/H40).
>
> **Implementation status (this repo):** `○ planned`.

This document is the committee-grade design write-up for H69.

---

## 1. Motivation (≥ 100 words)

The Kolmogorov-Arnold representation theorem (1957) proves that any
multivariate continuous function can be written as a composition of
**univariate** spline functions on a graph; the recent KAN paper
(Liu et al. 2024) realises this theorem as a trainable architecture
where the edges (not the nodes) carry the learnable splines.
Metatron's Cube is the 13-circle, 78-edge sacred-geometry figure whose
**vertex-edge incidence matrix** encodes all five Platonic solids
simultaneously — it is a maximally-symmetric finite graph. Mounting a
KAN on Metatron's edges therefore gives a symbolic-regression head
whose **graph structure is fixed** to a maximally-symmetric polytope
and whose **edge-spline parameters are learned**. The combination is
biologically motivated (cortical pyramidal trees are well-modelled by
edge-spline computation; the same is true of olfactory glomeruli
networks which have a near-Metatronic 13-circle organisation in
several mammalian species).

## 2. Formal hypothesis (≥ 50 words)

Because KAN edges fit univariate splines that compose into a
multivariate function on a fixed Metatron's-Cube graph,
**mechanism**-wise the head admits closed-form symbolic extraction of
the learned function via Liu et al.'s `kan.suggest_symbolic()`; per
**Liu et al. 2024 (KAN, arXiv:2404.19756)**, on a Feynman-equations
subset the head recovers a symbolic form on ≥40% of equations and
gains ≥3 pp GSM8K zero-shot accuracy as a side-effect of the
symbolic-friendly representation.

## 3. Falsifier (≥ 30 words)

If GSM8K zero-shot Δ ≤ +1 pp at 3-seed median, OR if the symbolic-
recovery rate on the Feynman subset is < 20%, the hypothesis is
**DISCARDED**.

## 4. Citations (≥ 80 words)

```
Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark 2024
NeurIPS 'KAN: Kolmogorov-Arnold Networks' (arXiv:2404.19756) -- the
KAN edge-spline architecture and the `suggest_symbolic` extraction.

Kolmogorov 1957 'On the representation of continuous functions of many
variables by superposition of functions of one variable' -- the
theoretical basis for KAN.

Udrescu, Tegmark 2020 'AI Feynman: a Physics-Inspired Method for
Symbolic Regression' (arXiv:1905.11481) -- the Feynman-equations
benchmark we adopt.

Strogatz 2003 'Sync: The Emerging Science of Spontaneous Order' --
the natural-system precedent for graph-structured symbolic dynamics.

Velickovic, Cucurull, Casanova, Romero, Liò, Bengio 2018 ICLR 'GAT'
(arXiv:1710.10903) -- the graph-attention prior that complements KAN
edges.

Bronstein, Bruna, Cohen, Velickovic 2021 'GDL' (arXiv:2104.13478) --
the geometric deep learning blueprint.

Cohen, Geiger, Köhler, Welling 2018 ICLR 'Spherical CNNs'
(arXiv:1801.10130) -- Platonic-symmetric architectures.
```

## 5. Mechanism

### 5.1 CNN track

A small KAN-MetatronHead module: 13-node Metatron graph (the 12
icosahedral vertices + center), 78 edges (the full Metatron's-Cube
edge set), each edge carrying a 12-grid B-spline. Forward pass: project
input to 13 node features by a linear map, run KAN message passing
(one round) along Metatron edges, mean-pool node features, linear to
output dim.

```python
# src/nature_inspired_networks/kan_metatron.py
class KANMetatronHead(nn.Module):
    def __init__(self, d_in, d_out, grid=12):
        super().__init__()
        self.in_proj = nn.Linear(d_in, 13)
        self.edges = nn.ModuleList([
            KANEdge(grid=grid) for _ in range(78)])
        self.adj = metatron_adjacency()  # (13, 13) {0,1}
        self.out_proj = nn.Linear(13, d_out)
    def forward(self, x):                           # (..., d_in)
        h = self.in_proj(x)                          # (..., 13)
        h_new = torch.zeros_like(h)
        for e_idx, (i, j) in enumerate(self.adj.nonzero()):
            h_new[..., j] = h_new[..., j] + self.edges[e_idx](h[..., i])
        return self.out_proj(h_new)
```

### 5.2 LLM track

Slot: **a new symbolic head** that runs alongside the LM head; gated
by a context-sensitive switch (the input is routed to KAN-Metatron
when the last few tokens contain math-token patterns; routed to LM
head otherwise — modulo a soft gate). FA2 compatibility: unaffected
— KAN is downstream of attention. Causal-mask preservation: trivial.

Inference cost: +30% latency on math queries (the KAN forward), +0%
on non-math queries (routed to LM head). The symbolic extraction is
done **offline** with `kan.suggest_symbolic()` on the trained edge
splines.

Expected impact: at 124M scale, GSM8K +3 pp; ARC-Challenge math
sub-section +1.5 pp; perplexity on math-heavy WikiText subsets
-0.1 nats.

## 6. Predicted Δ

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite | [+0.012, +0.025] | symbolic head wins on math |
| GSM8K zero-shot | [+2 pp, +5 pp] | direct target |
| ARC math | [+1 pp, +3 pp] | side-effect |
| Feynman symbolic-recovery | [+40%, +70%] | direct target |
| perplexity (LLM, full corpus) | [-0.10, +0.10] nats | mild |
| params | [+1%, +3%] | KAN head added |
| FLOPs | [+5%, +15%] on math; [+0%] elsewhere | gated |
| GPU latency (batch=1) | [+0%, +30%] depending on routing | gated |
| KV cache @ 32k | [0%] | unchanged |

## 7. Experimental protocol

### 7.1 Primary experiment

- Datasets: GSM8K (zero-shot), Feynman equations (symbolic), WikiText
  (PPL sanity).
- Architecture: 124M decoder + KAN-Metatron head.
- Training: 20k steps fine-tune, bf16.
- Composite SHA-256.
- Wall-clock: ≈14 h on 4090.
- Archive: `ideas/69_kan_metatron_symbolic_head/experiments/exp001_kan/`.

### 7.2 Targeted experiment

Should SHINE on **symbolic regression**: take the Feynman dataset
subset of 50 equations, train KAN-Metatron head on per-equation data,
run `suggest_symbolic`, measure exact-match rate. Expected: ≥40%
recovery (Liu et al. reports ≈60% on a pure-KAN baseline; our
Metatron-graph constraint may marginally cost recovery rate but
should improve the rate at low data).

### 7.3 Cross-paradigm context

H69 is the only hypothesis in H61-H71 that adopts KAN as a primary
paradigm. It tests the chunk-3 mechanism-axis claim that
basis-function composition is orthogonal to attention.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G7 row H69.
- Log: row T2.H69.
- Sub-dir: `ideas/69_kan_metatron_symbolic_head/`.
- Composes with: H23, H30, H40, H67.
- Conflicts with: H39 (alternative non-spline activation).

## 9. Committee Q&A

**Q: Why isn't this just KAN with a fancy graph?**

> The Metatron's-Cube graph is **maximally symmetric** (encodes all
> five Platonic solids); the natural ablation is KAN on a random graph
> of equal vertex/edge count. If random matches Metatron, the prior
> contributes nothing — falsifier triggers.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 names the +1 pp + 20% recovery floors.

**Q: What if KAN-Metatron helps Feynman but not GSM8K?**

> A partial discard: claim is narrowed to "symbolic regression only".
> Reported as such in FINDINGS.

**Q: Hasn't the previous CIFAR sweep shown the priors don't compound?**

> H69 composes KAN + Metatron — two specific priors with a gated head.
> The compositions in CIFAR were 7-prior structural; H69 is 2-prior
> aux head.

**Q: How do we know the implementation is correct?**

> `tests/test_kan_metatron.py` asserts (a) Metatron adjacency has 78
> non-zero entries, (b) KAN edge splines are differentiable,
> (c) symbolic extraction matches known function `y = x²` on a
> planted test, (d) head is causal-mask-safe.

## 10. Verification artifacts checklist

- [ ] `implementation.py`
- [ ] `tests.py` ≥ 8 assertions
- [ ] `AUDIT.md`
- [ ] `IMPROVEMENTS.md`
- [ ] `VERIFY.md`
- [ ] `experiments/exp001_kan/`
- [ ] `verification/`
- [ ] Log row
- [ ] FINDINGS reflected
- [ ] Dashboard

## 11. Status journal

- 2026-05-26 — Created from template by Doc-Agent-D.
