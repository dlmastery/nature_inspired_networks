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
