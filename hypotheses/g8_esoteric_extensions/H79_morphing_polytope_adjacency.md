# H79 — Morphing Polytope Adjacency (Jitterbug)

> **One-line claim:** A learnable convex morph between cuboctahedron and
> icosahedron adjacencies gives a dynamic 12-node graph topology.
>
> **Source design space:** G8 Esoteric Extensions.
>
> **Implementation status (this repo):** `✓ done` (primitive + tests).

---

## 1. Motivation (≥ 100 words)

Buckminster Fuller's "jitterbug" is a continuous transformation that
twists a cuboctahedron — the *vector equilibrium*, the unique polyhedron
whose 12 vertices are equidistant from the centre and from each other —
into an icosahedron, by rotating its eight triangular faces. Crucially
both solids have exactly 12 vertices, so the transformation is a
continuous deformation of a fixed 12-point graph: only the *edges* change
(the cuboctahedron has 24 edges, the icosahedron 30). This is a clean
physical analogue of a graph whose connectivity can adapt smoothly along
a one-parameter family — exactly what a continual-learning system wants
when the optimal relational structure drifts. The esoteric vector-
equilibrium / jitterbug motif is the mystical motivation; the
implementation is a plain convex interpolation of two nearest-neighbour
adjacencies with a single learnable, sigmoid-gated mixing parameter.

## 2. Formal hypothesis (≥ 50 words)

Because the layer's adjacency is `A(t) = (1−t)·A_cubocta + t·A_icosa`
with `t = sigmoid(t_raw)` learnable, the optimiser can slide the message-
passing topology continuously between the two 12-vertex polytopes;
mechanism-wise this makes the relational inductive bias itself trainable
rather than fixed. Per Battaglia 2018, a learnable adjacency is a valid
relational bias, so we expect the dynamic topology to match or beat a
fixed-topology graph layer when the task's optimal connectivity is
unknown a priori.

## 3. Falsifier (≥ 30 words)

If, on a 12-node graph classification task, the morphing layer with
learnable `t` does not match or exceed the better of the two fixed-`t`
endpoints (`t=0` and `t=1`) by ≥ 0 composite at 3-seed median, the
learnable morph adds nothing and the hypothesis is DISCARDED.

## 4. Citations (≥ 80 words)

Kipf, T. N., Welling, M. 2017 ICLR 'Semi-Supervised Classification with
Graph Convolutional Networks' (arXiv:1609.02907) — supplies the
symmetric-normalised `D^{-1/2} A D^{-1/2}` message-passing rule the
MorphingGraphLayer uses on the morphed adjacency. Battaglia, P. W.,
Hamrick, J. B., Bapst, V. 2018 'Relational inductive biases, deep
learning, and graph networks' (arXiv:1806.01261) — frames a learnable
adjacency as a first-class relational inductive bias, justifying making
the jitterbug mixing parameter `t` a trained scalar rather than a fixed
hyperparameter, and motivating dynamic topology for adaptation.

## 5. Mechanism

### 5.1 CNN / vision track
Reshape a pooled feature into 12 node slots → one/two
`MorphingGraphLayer(in, out)` → head. Endpoint adjacencies are cached as
buffers (cuboctahedron 24 edges, icosahedron 30, verified via the
nearest-neighbour edge set); per forward only the convex blend + degree
normalisation runs. One extra scalar parameter (`t_raw`). Shapes
`(B,12,in)→(B,12,out)`. Lives at
`src/nature_inspired_networks/morphing_adjacency.py`.

### 5.2 LLM track (decoder-only)
Use as a 12-slot relational mixer over grouped channel-blocks of a token
representation (a tiny GNN side-path), with `t` learned per layer. No
causal-mask interaction (it mixes feature groups, not sequence positions)
and no KV-cache cost.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. fixed-topology | rationale |
|---|---|---|
| composite | [−0.002, +0.005] | adaptive topology, task-dependent |
| top-1 (CNN, graph head) | [−0.2, +0.4] pp | small relational head effect |
| params | [+1, +1] scalar | single `t_raw` over the fixed-graph layer |
| FLOPs | [+0%, +0.1%] | one extra blend |

## 7. Experimental protocol

### 7.1 Primary
ResNet-20 + 12-node morphing graph head, CIFAR-10 12-epoch smoke, bf16,
batch 256, seeds 0/1/2. Archive:
`ideas/79_morphing_adjacency/experiments/exp001_smoke/`.

### 7.2 Targeted (where it should shine)
A two-phase continual task where the optimal 12-node connectivity flips
mid-training; the learnable `t` should track the shift while fixed-`t`
baselines cannot.

### 7.3 Cross-paradigm (LLM)
12-group relational mixer on a 124M decoder; measure perplexity vs. a
fixed-adjacency variant.

## 8. Cross-references

- Parent: G8 esoteric extensions.
- Sibling fixed graph: `src/nature_inspired_networks/platonic_graph.py` (H23),
  `platonic_fib.py` (H30).
- Composes with: H78 (toroidal node codes), H77 (graph→attention head).

## 9. Committee Q&A

**Q: Why isn't this just MetatronGraphLayer with learnable weights?**
> MetatronGraphLayer learns per-edge magnitudes around a *fixed* pattern;
> here a single scalar morphs between two *distinct* polytope edge sets
> (24- vs 30-edge), so the topology family, not just weights, is the
> object of learning.

**Q: How is this falsifiable?**
> §3: the learnable morph must not underperform the better fixed endpoint
> — and both endpoints are exactly recoverable (`t=0`, `t=1`, tested).

**Q: What if `t` just collapses to an endpoint?**
> That is a valid (and informative) outcome — it means a fixed topology
> was optimal; the §7.2 continual task forces a non-trivial trajectory.

**Q: Priors don't compound — why bother?**
> Single-prior unit (one scalar), measured against fixed-topology
> baselines, not inside a hybrid.

**Q: How do we know it is correct?**
> `tests/test_morphing_adjacency.py` (5 tests): both vertex sets 12×3,
> symmetric zero-diagonal adjacencies with the exact 24/30 edge counts,
> `t=0`/`t=1` recover the polytope adjacencies, forward shape, and the
> learnable `t` carries a finite non-zero gradient confined to (0,1).

## 10. Verification checklist

- [x] Primitive `morphing_adjacency.py` exists, tests green (5/5).
- [x] Vertex sets 12×3; adjacencies symmetric (tested).
- [x] `morph(0)`=cubocta, `morph(1)`=icosa (tested).
- [x] Learnable `t` has gradient, sigmoid-gated to (0,1) (tested).
- [ ] Experiment archive (deferred — graph module, no sweep row).

## 11. Status journal

- 2026-05-27 — Created; primitive + 5 unit tests green.

---

## Addendum: Research-Scientist Critique (2026-05-27)

*Reviewer: SciCritic-G8 (elite-research-scientist critic). Critiquing
the IDEA, not the implementation (audit at `audits/G8_audit.md`).*

### Prior plausibility (LOW/MED/HIGH + why)

LOW. The general idea (learnable adjacency / topology) is real
(Kipf, Welling 2017 ICLR 'Semi-Supervised Classification with Graph
Convolutional Networks' (arXiv:1609.02907); Battaglia, Hamrick,
Bapst 2018 'Relational inductive biases, deep learning, and graph
networks' (arXiv:1806.01261); Liu, Simonyan, Yang 2019 ICLR 'DARTS:
Differentiable Architecture Search' (arXiv:1806.09055)). But
restricting the family to a 1-parameter convex interpolation between
*exactly* the cuboctahedron and *exactly* the icosahedron is
hyper-specific with no task-level motivation: CIFAR-10 has no
12-vertex relational structure to morph between. The plausibility
collapses once one asks: what about the 12-vertex truncated
tetrahedron, or the 30-edge dodecahedron, or any other 12-vertex
graph?

### Mechanism scrutiny — does the NEUTRAL recast match the cited real technique?

PARTIALLY. The GCN propagation (Kipf 2017 arXiv:1609.02907) and
learnable-adjacency framing (Battaglia 2018 arXiv:1806.01261) are
faithfully implemented. But the broader literature on learnable
graph topology (DARTS arXiv:1806.09055; graph attention networks —
Veličković, Cucurullo, Casanova, Romero, Liò, Bengio 2018 ICLR
'Graph Attention Networks' (arXiv:1710.10903)) makes the *whole*
adjacency learnable, not a 1-parameter convex blend between two
fixed endpoints. The author's recast is a *constrained* learnable
topology — strictly weaker than the cited literature.

### Does the esoteric origin contaminate the implementation or framing?

YES, decisively. The choice of cuboctahedron ↔ icosahedron is
entirely the Buckminster Fuller "jitterbug" / "vector equilibrium"
mystical motif. Without that motif, no rational principle selects
these two 12-vertex polyhedra out of the dozens of plausible
candidates. The doc itself opens §1 with "Buckminster Fuller's
jitterbug" — the engineering motivation *is* the esoteric framing
in this case.

### Confounds (≥2)

1. **Reshape-to-12 confound.** Reshaping a pooled feature into 12
   node slots is itself an architectural choice; the relational
   message passing might help (or hurt) regardless of which two
   adjacencies are blended. The fair baseline is a 12-node graph
   with a *learned* full adjacency (n=12, 66 free off-diagonal
   weights), not a 1-scalar morph.
2. **Endpoint asymmetry.** Cuboctahedron has 24 edges; icosahedron
   has 30. A learnable convex blend monotonically increases edge
   density with `t`, so the "morph" is partially a *density* knob,
   not a pure topology shift. A density-matched ablation (uniform
   12-vertex graphs with 24 and 30 edges drawn randomly) is needed.
3. **Single scalar.** A 1-parameter family cannot reach most
   12-vertex graphs at all; the optimiser is restricted to a line
   in graph space, and the line is chosen by mystical motif.

### Numerology / specificity check

The number 12 appears because the cuboctahedron and icosahedron
share 12 vertices (a coincidence Fuller fetishised). 12 is not
intrinsic to CIFAR-10 or any image task. The framing privileges
two polytopes over hundreds of 12-vertex graphs (Petersen graph,
truncated tetrahedron, snub disphenoid, cubic graphs, etc.) with no
defensible reason beyond the jitterbug aesthetic. This is the
clearest numerology case in G8: a *graph-family* prior chosen by
sacred-geometry coincidence.

### Literature precedent — was the neutral recast already known?

The *generalisation* (learnable graph topology) is heavily explored
(DARTS arXiv:1806.09055; GAT arXiv:1710.10903; NRI — Kipf, Fetaya,
Wang, Welling, Zemel 2018 ICML 'Neural Relational Inference for
Interacting Systems' (arXiv:1802.04687)). The *specific* recast
(convex blend of two named platonic adjacencies) is not in the
literature because no one would propose it without the jitterbug
prior.

### Expected effect size (90% CI a priori)

CIFAR-10 12-ep top-1 vs. fixed cubocta adjacency: [−0.5 pp, +0.1
pp]. The single scalar cannot meaningfully outperform either
endpoint on a non-graph-structured task; at best `t` collapses to
the better endpoint (an outcome the doc concedes is "valid").

### Minimum-distinguishing experiment

Three-arm sweep on a graph-classification benchmark (e.g.
TUDatasets / MUTAG, or a synthetic 12-node graph task): (a) fixed
cubocta adjacency, (b) fixed icosa adjacency, (c) morphing. The
hypothesis lives only if (c) > max(a, b) by ≥ 0.5 pp at 3-seed
median. CIFAR-10 image classification is the *wrong* benchmark —
images have no native 12-vertex graph structure.

### Verdict

NUMEROLOGY — the cuboctahedron↔icosahedron-only morph family is
selected by the jitterbug motif; the general "learnable adjacency"
prior is real but already known and is a strict superset of this
1-scalar restriction.
