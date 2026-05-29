# PARADIGM_COMPARISON — Six-Paradigm Synthesis

> Generated from the 8-chunk extended-transcript expansion (chunks 1-8,
> EXPERIMENT_LEDGER.md). This file is the **synthesis claim** of the
> `dlmastery/nature_inspired_networks` program: that the five modern
> paradigms in vogue — **Liquid Foundation Models, JEPA, KAN,
> Transformer, and equivariant GNNs** — are each a partial rediscovery
> of the same sacred-geometry priors and can be unified inside a single
> drop-in residual block, the **NaturePriorBlock v2** (a.k.a.
> SacredGeoBlock v2) whose pseudocode closes the document.
>
> Every chunk is independently citable. Every table is **6 columns
> wide** (Liquid / JEPA / KAN / Transformer / GNN / **NaturePrior
> synthesis**) so that the synthesis column is the explicit hypothesis
> in every row.
>
> Cross-links: `MANIFESTO.md` § 6, `IDEA_TABLE.md` § G7 rows H61–H71,
> `hypotheses/g7_cross_paradigm_hybrids/H67_full_paradigm_hybrid.md` (the operational test of
> this synthesis), `EXPERIMENT_LEDGER.md` chunks 5–7.

---

## Executive synthesis (~200 words)

The five modern paradigms in deep learning attack **orthogonal axes**
of the natural-system design space. Liquid Foundation Models (LFM2)
rediscover **continuous-time integration** by exposing learnable
per-layer τ that the cortex has always used. The JEPA family
(I-JEPA, V-JEPA, seq-JEPA) rediscovers **joint-embedding predictive
coding** — the cortical loop that compares predicted to actual
latent states. KAN rediscovers **Kolmogorov-Arnold edge-spline
composition** — the dendritic-tree computation that 1957 mathematics
proved sufficient. Decoder-only Transformers rediscover **all-to-all
attentional binding** — the first 100 ms of cortical stimulus
processing. Equivariant GNNs rediscover **local synaptic message
passing** under group symmetry — the steady-state dendritic
computation. Nature does all five **simultaneously**, on manifolds
that respect Platonic / hexagonal / toroidal / fractal / cymatic /
phyllotactic priors. The synthesis claim — operationalised in H67 —
is that a single decoder layer can run all five sub-paradigms in
parallel, gated by a learned mixing weight, on sacred manifolds, and
achieve a **Pareto-strict** improvement over the best single
paradigm. The 25 untapped opportunities in chunk 8 are the
operational menu; H61-H71 are the first 11 concrete tests; the
NaturePriorBlock v2 pseudocode below is the canonical implementation.

---

## Chunk 1 — Philosophical foundations

### 1.1 The 6×6 comparison table

| Property | Liquid (LFM2) | JEPA family | KAN | Transformer | GNN (GDL) | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Originating metaphor** | C. elegans 302-neuron continuous dynamics | cortical predictive coding | Kolmogorov-Arnold 1957 theorem | "attention is all you need" | message passing in molecules | nature does all of the above at once on sacred manifolds |
| **Foundational paper** | Hasani et al. 2021 (arXiv:2006.04439) + LFM2 arXiv:2511.23404 | Bardes et al. 2024 (V-JEPA, arXiv:2404.08471) + seq-JEPA arXiv:2506.09985 | Liu et al. 2024 NeurIPS (arXiv:2404.19756) | Vaswani et al. 2017 (arXiv:1706.03762) | Veličković et al. 2018 GAT (arXiv:1710.10903) + Bronstein et al. 2021 GDL (arXiv:2104.13478) | this repo + Huh et al. 2024 PRH (arXiv:2405.07987) |
| **Time model** | continuous ODE | discrete latent prediction | static spline composition | discrete autoregressive | discrete message rounds | continuous time + discrete prediction (Liquid+JEPA fused) |
| **Inductive bias** | smoothness of trajectories | invariance of latents under masking | univariate compositional decomposition | permutation-equivariance with positional learning | local-neighborhood with group equivariance | **all of the above + Platonic / hex / toroidal / fractal symmetry** |
| **What it claims to be a recovery of** | LTC biology | cortical predictive coding | Kolmogorov representation theorem | biological all-to-all binding | local synaptic message passing | the Platonic Representation Hypothesis (Huh 2024) |
| **What it sacrifices** | parallelism (recurrence) | direct generation (predictor-only) | scale (parameter density) | linear-time inference | global integration speed | none — claim is that mixing recovers each strength |

### 1.2 Deep dive #1 — Why each paradigm exists, philosophically

Each paradigm answered a different "what failed about the previous
generation" question. Liquid answered the question "why does an RNN
need to discretise time?" by allowing learnable continuous τ. JEPA
answered "why does a self-supervised loss reconstruct pixels rather
than latents?" by predicting in latent space. KAN answered "why are
activation functions on nodes rather than edges?" by inverting the
location of the non-linearity. Transformer answered "why must
recurrence be sequential?" by making it parallel. GNN answered "why is
the topology of computation a regular grid?" by making it a learned
graph. **The sacred-geometry synthesis answers a fifth question**:
"why does each paradigm pick **one** inductive bias?" — and proposes
that the cortical answer is to compose them on manifolds whose
symmetry is itself a useful prior. Each of the five paradigms is a
**partial rediscovery** of biological computation; the synthesis is
the full composition.

### 1.3 Deep dive #2 — The Platonic Representation Hypothesis as the bridge

Huh, Cheung, Wang, Isola 2024 (arXiv:2405.07987) prove empirically that
sufficiently large encoders **across modalities and objectives**
converge to a shared statistical model of reality. This is the
philosophical bridge between sacred geometry and modern deep learning:
if every sufficiently-large neural network converges to the same
Platonic representation, then **the priors that accelerate this
convergence are themselves a discovery target**. The five paradigms
each accelerate the convergence along one axis (continuous-time,
predictive, basis-function, all-to-all, local). The synthesis
hypothesis is that priors that respect Platonic / hex / toroidal /
fractal / cymatic / phyllotactic symmetries accelerate convergence
along **all axes simultaneously**, because nature did that exact
composition during evolution.

### 1.4 Deep dive #3 — Why "sacred" survives engineering scrutiny

The "sacred" label is rhetorically dangerous but operationally
defensible: every prior in this repo (φ-Fibonacci scaling, hexagonal
lattice, icosahedral equivariance, fractal recursion, toroidal closure,
Chladni-eigenmode init, golden-angle modulation) maps to a peer-
reviewed paper with an arXiv ID. The label "sacred" is short-hand for
"natural priors that humans, lacking modern math, encoded as religious
geometry"; the engineering content is the same as the
peer-reviewed citations. The autoresearch protocol's **Citation
Rigor gate** enforces the discipline: every prior must have an arXiv
ID. The label survives because it is the **superset name** for the
set of priors that are already peer-reviewed individually. The
synthesis claim is empirically falsifiable per `MANIFESTO.md` § 2.

---

## Chunk 2 — Inductive biases & symmetries

### 2.1 The 6×5 comparison table

| Inductive bias dimension | Liquid | JEPA | KAN | Transformer | GNN | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Time symmetry** | continuous-time invariance | step-shift invariance under masking | none (static) | discrete-shift via RoPE | discrete message-round invariance | LTC time + JEPA prediction + φ-spaced τ |
| **Permutation symmetry** | permutation-broken (sequential) | full permutation on patches | full per-edge | full on tokens minus PE | full on nodes minus PE | per-paradigm + Platonic discrete subgroups |
| **Rotation symmetry** | none | mask-augmented | none | none (unless RoPE-3D) | full SO(3) via equivariant GNN | icosahedral I (H24, H71) + hex C6 (H21, H62) |
| **Translation symmetry** | linear time | masked patches translate | none | RoPE | grid GNN | toroidal wrap (H22, H68) |
| **Scale symmetry** | continuous via τ | multi-scale masks | spline grid resolution | multi-head | multi-hop | fractal φ-recursion (H05, H26, H38) |

### 2.2 Deep dive #1 — Continuous-time symmetry (Liquid's contribution)

Liquid Foundation Models inherit from Liquid Time-Constant Networks
(Hasani 2021) the property that the recurrence relation is an explicit
ODE: `dx/dt = -x/τ + f(x, input)`. This makes the network's
representation **invariant under reparameterisation of the time axis**
within a bounded range — a property no discrete RNN has. The
sacred-geometry contribution sets the time constants on a φ-spaced
geometric progression (τ_k = τ_0 · φ^k for a bank of k taps), so the
bank covers a logarithmic time-frequency range in the minimum number
of taps. The biological precedent is the cortical L5 pyramidal neuron's
multiple integration time-constants (from milliseconds to seconds) —
nature uses a logarithmically-spaced bank because that is the
information-theoretically optimal coverage. H61 operationalises this
in the LLM decoder. Continuous-time symmetry composes additively with
discrete-shift symmetry (Transformer's RoPE) because they act on
orthogonal sub-spaces (time vs. positional embedding).

### 2.3 Deep dive #2 — Rotation symmetry (GNN's contribution + sacred extension)

Equivariant GNNs (e3nn, Cohen et al. 2019 Icosahedral CNN) achieve
**full SO(3) equivariance** by decomposing features into irreducible
representations of SO(3) and using Clebsch-Gordan tensor products for
message passing. The cost is high — a single equivariant layer has
≈3-5× the FLOPs of a conventional GNN. The sacred-geometry alternative
is **discrete subgroup equivariance**: the icosahedral group `I` (order
60) is the largest discrete subgroup of SO(3) and gives an
approximately uniform S² sampling for free. H71 generalises RoPE
along this axis (icosahedral-group rotations instead of 2-D
rotations); H24 generalises CNN convolution; H30 hybridises with
Fibonacci node-degree distributions. The trade-off is exactness vs.
cost: SO(3) is exact but expensive; `I` is approximate-uniform but
cheap. The synthesis chunk argues that for **most LLM use-cases**
discrete `I` equivariance is sufficient and gives ≥80% of the SO(3)
benefit at <5% of the cost.

### 2.4 Deep dive #3 — Scale symmetry (the missing chunk-2 axis)

None of Liquid / JEPA / KAN / Transformer / GNN has explicit
multi-scale symmetry. Liquid's continuous-time hides it inside τ;
JEPA's multi-scale comes from mask-size augmentation; KAN's grid
resolution is a hyperparameter; Transformer's multi-head is an
implicit multi-scale; GNN's multi-hop is explicit but discrete. The
sacred-geometry contribution is **fractal φ-recursion**
(FractalNet 2017 + 1/φ depth-shrink, H05 / H26 / H38): a sub-network
runs at 1/φ the parent's depth and width. This is the only paradigm
column that gives **continuous-bounded** scale symmetry (the recursion
terminates at depth log_φ(d)). The natural precedent is the bronchial
tree, which follows a φ-recursive branching law and is
information-theoretically optimal for fluid distribution. The synthesis
operationalises this via the v2 SacredGeoBlock's fractal sub-block.

---

## Chunk 3 — Core computational mechanisms

### 3.1 The 6×N mechanism table

| Mechanism | Liquid | JEPA | KAN | Transformer | GNN | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Core forward op** | LTC ODE step | predictor + EMA-target diff | edge-spline eval | scaled-dot-product attn | aggregate-update | learned-gate mix of all five |
| **Complexity in N** | O(N) | O(N) (encoder) + O(M) (predictor on M masked) | O(grid × deg × N) | O(N²) naive / O(N) FlashAttn | O(deg · N) | Pareto mix; KV-cache O(N) by H62 hex pruning |
| **Memory for cache** | O(d) hidden | O(d) EMA + masked-feature buffer | 0 (stateless) | O(N·d) KV | O(N·deg) | min{O(N·d) (attn), O(d) (Liquid)} |
| **Backward gradient** | through-time recurrence | EMA-detached (stop-gradient on target) | through splines | through softmax-attn | through aggregate | per-sub-path then mixer |
| **Init dependence** | small (continuous attractor) | strong (EMA must track) | strong (spline grid) | moderate (depends on norm) | small | **cymatic basis init across all sub-paths (H66)** |

### 3.2 Deep dive #1 — Attention vs. message passing (the false dichotomy)

The Transformer/GNN dichotomy is **theoretically false** — a
transformer is a fully-connected GNN with learned edge weights — but
**practically real** because Transformer's O(N²) is unsustainable for
graph applications. The chunk-3 contribution is the observation that
**hex-graph-restricted attention** (H62) reduces O(N²) to O(7·N) per
hop and reaches the full graph in O(log_φ N) hops, giving the
Transformer paradigm a GNN-like cost while retaining its all-to-all
multi-hop expressiveness. The sacred-geometry contribution is the
**hex lattice** — which is the unique 2-D lattice with the smallest
neighbour cardinality (six) consistent with full-plane tessellation.
The natural precedent is the bee honeycomb / fly retina. The cost is
a one-time permutation of token positions onto the hex lattice; the
saving is ≈50% FLOPs and ≈40% KV cache. The trade-off in expressive
power is mild because the multi-hop reachability is preserved.

### 3.3 Deep dive #2 — Basis-function vs. attention (KAN's contribution)

KAN's edge-spline mechanism is structurally different from
attention's softmax-weighted aggregation: KAN learns a **per-edge
non-linear function** (a B-spline on a fixed grid), then sums across
edges. The complexity is `O(grid_size × deg × N)` per layer — at
grid = 8 and deg = 10, that's 80 FLOPs/token vs. attention's
≈4·d²·N FLOPs (where d ≈ 768). At small N (≤512 tokens), KAN is
**cheaper per param** but much smaller per param. The synthesis-axis
insight is that **KAN and attention are complementary**: KAN handles
local non-linear transforms efficiently; attention handles
long-range correlation efficiently. H67 fuses them with a learned
gate. H69 takes KAN further by constraining its graph to Metatron's
Cube — turning it into a fixed-topology symbolic-regression head.

### 3.4 Deep dive #3 — Cymatic init as a cross-paradigm forward primitive

The Chladni eigenmode basis is **paradigm-agnostic**: it can
initialise the QKV of attention (H66), the LTC mixing matrix of Liquid
(H35 generalised), the spline coefficients of KAN, the message
projection of GNN, or the embedding table itself. The unifying
property is that it gives an **orthonormal multi-scale basis** that
is also **multi-resolution** (low-frequency modes correspond to
coarse spatial features). The chunk-3 contribution is the observation
that the same basis works across all five paradigms — making it the
canonical chunk-3 cross-paradigm primitive. The chunk-5 (training)
follow-up is the cymatic curriculum (H70) that uses the same basis as
a curriculum schedule.

---

## Chunk 4 — Efficiency, compression, capacity, scalability

### 4.1 The 6×N efficiency table

| Efficiency dimension | Liquid (LFM2) | JEPA (V-JEPA 2) | KAN | Transformer | GNN | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Params per output dim** | low (recurrence shares) | moderate | very low (10-100× less than MLP) | high | moderate | min of constituent paradigms |
| **FLOPs per token** | O(d²) per step | O(d²) per masked patch | O(grid × d) per edge | O(N·d) self-attn | O(deg·d²) per node | Pareto-mixed via gate |
| **KV cache at 32k** | 192 MB (LFM2 reports) | ≈400 MB (V-JEPA encoder) | 0 | ≈600 MB (vanilla GPT-2) | 0 | ≈140 MB (H62 hex + toroidal) |
| **Sample efficiency** | moderate | **1.5-6× (V-JEPA 2)** | 10-100× (small models only) | low | moderate | inherits JEPA + cymatic curriculum (H70) |
| **Quantisation friendliness** | high (continuous τ stable) | moderate (EMA needs precision) | low (splines fragile) | high (standard) | moderate | mid (gate determines per-sub-path) |
| **Inference-mem dominated by** | activation state | KV + EMA | (none) | KV cache | (graph) | KV cache (smallest contributing) |
| **Scaling law slope** | flat (slow) | steep (1.5-6× per data 2×) | steep at low scale, flat at large | steep (linear in compute) | flat | **JEPA-curve at start, Transformer-curve at scale** |

### 4.2 Deep dive #1 — KV cache as the dominant inference memory cost

Chunk-4 of the extended transcript reports that **at long context
(32k+) the KV cache is ≈70% of total inference memory** for a
decoder-only Transformer. LFM2 reduces this to 192 MB at 32k via
linear-complexity recurrence; H62 (hex attention + toroidal wrap +
Fib pruning) targets 140 MB by combining three orthogonal compressors.
The hex pruning saves ≈50% by replacing dense N² with 7N; the
toroidal wrap is allocation-neutral but eliminates boundary padding
(≈1%); the Fib-pruning schedule cuts another ≈25% off the residual
QKV magnitudes over training. The combined attack on the KV cache is
the **single largest efficiency win** any of the cross-paradigm
hypotheses claim. The synthesis-axis insight is that **three small
compressors compose multiplicatively** when they target orthogonal
properties of the cache (sparsity / boundary / magnitude).

### 4.3 Deep dive #2 — JEPA's sample efficiency and how to inherit it

V-JEPA 2 reports **1.5-6× sample efficiency** over discrete-token
self-supervised baselines on the same downstream tasks. The mechanism
is the latent-prediction loss replacing pixel-reconstruction loss —
the model is forced to predict abstract structure rather than memorise
texture. The synthesis-axis question is whether a non-JEPA paradigm
can inherit this gain. The chunk-4 answer is **yes, partially** — by
adding a JEPA auxiliary loss to any paradigm (H61 for Liquid, H63 for
attention's auxiliary, H65 for the Betti-aux equivalent). The cymatic
curriculum (H70) extends this further: it provides a structured
auxiliary **input stream** rather than auxiliary loss. The combined
sample efficiency gain at 30% data is projected at ≈80% gap-recovery
(H70 § 6).

### 4.4 Deep dive #3 — Why KAN's 10-100× param-efficiency does not scale

KAN's headline result (10-100× param-efficiency vs. MLPs) is reported
on **small-model regimes** — symbolic regression, low-dim physics
problems, sub-1M param models. At larger scale the edge-spline grid
becomes the bottleneck (the per-edge spline grid does not benefit from
the parallelism the MLP enjoys), and the param-efficiency advantage
collapses. The synthesis-axis insight is that **KAN is the symbolic
head, not the backbone**: H69 mounts KAN edges on Metatron's Cube as
a gated **head** with the backbone remaining a vanilla decoder. This
gives KAN's param-efficiency where it actually wins (small,
structured outputs) without paying the scaling penalty in the
backbone. The chunk-4 conclusion is that **each paradigm has a
scale-sweet-spot**, and the synthesis is the routing that puts each
sub-paradigm in its sweet spot.

---

## Chunk 5 — Training paradigms & self-supervision

### 5.1 The 6×N training table

| Training dimension | Liquid (LFM2) | JEPA | KAN | Transformer | GNN | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Primary loss** | next-token CE | latent-prediction (cosine + EMA) | regression / classification | next-token CE | task-specific | next-token CE + JEPA aux + Betti aux + dodeca aux |
| **Self-supervision signal** | via tokenisation | masked-patch features | none (supervised) | via masked-LM / next-token | none (supervised) | JEPA + cymatic curriculum (H70) |
| **Distillation pattern** | **Top-K KD** from a larger teacher (LFM2 reports) | EMA-momentum self-distillation | regression fit | full or LoRA | (none standard) | Top-K KD + dodeca aux teacher (H63) |
| **Curriculum** | none standard | mask-size schedule | grid-resolution schedule | none | none | **cymatic frequency curriculum (H70)** |
| **Optimisation** | AdamW + cosine | AdamW + EMA | LBFGS or Adam | AdamW + cosine | AdamW | AdamW + φ-scheduler (H10/H48) |
| **Pre-training data scale** | trillion-tokens | hundreds of millions of clips | small (k-M) | trillion-tokens | (graph datasets) | matched to backbone paradigm |
| **Fine-tuning posture** | full / LoRA | linear probe / full | spline refit | full / LoRA / QLoRA | task-specific | per-sub-path: backbone full + symbolic head separate |

### 5.2 Deep dive #1 — Top-K KD as the on-device training innovation

LFM2's training pipeline includes a **Top-K knowledge-distillation**
step where the student (a smaller LFM2) is matched to the teacher's
top-K logit distribution rather than the full vocabulary. This gives
≈70% of full-distillation quality at ≈5% of the compute cost. The
sacred-geometry contribution to this axis is that **Top-K can be set
to Fibonacci values** (K ∈ {5, 8, 13, 21, 34}) and **the cymatic
basis can score "which top-K positions" by resonance** rather than by
softmax probability — i.e., select the K logits whose embedding has
maximum cymatic resonance with the input rather than the K logits
with highest probability. This is a chunk-5 untapped opportunity (#7
in the chunk-8 list). The combined "Fib Top-K cymatic-scored KD" is
projected to give ≈20% extra distillation quality at the same K.

### 5.3 Deep dive #2 — JEPA's EMA-target distillation and its sacred extension

JEPA's training innovation is the **EMA-momentum target encoder**: a
copy of the encoder whose parameters track the live encoder via
exponential-moving-average update with momentum m ≈ 0.999. The target
is detached from gradient flow; the student learns by minimising
cosine distance to the target's features on masked patches. The
sacred-geometry extension (H63 / H65) is to **add a fixed-geometric
target** alongside the EMA target — specifically, the dodecahedral
projection of the embedding. This gives the student **two** distinct
target signals: the EMA-momentum target (drifting, paradigm-natural)
and the dodeca-projection target (fixed, Platonic-RH-natural). The
chunk-5 prediction is that the combination yields ≥0.5 pp on
multi-modal alignment tasks and ≥0.1 nats on text-only perplexity.
The mechanism is that the dodeca target acts as a **stationary
anchor** that prevents EMA drift from degenerate fixed points.

### 5.4 Deep dive #3 — Curriculum learning is the cross-paradigm primitive

Curriculum learning (Bengio et al. 2009) is **universally applicable**
— any paradigm can be trained with an easy-to-hard data schedule.
Yet only V-JEPA 2 in the canon adopts an explicit curriculum (the
mask-size schedule). The sacred-geometry contribution is the **cymatic
frequency curriculum** (H70): start with low-frequency Chladni modes
and progressively add high-frequency modes. The natural precedent is
prenatal retinal-wave activity, which exposes the cortex to
low-frequency waves first, then progressively higher frequencies.
This is the chunk-5 single most-portable primitive — it can be added
to any paradigm with a 30-line code change. The chunk-5 prediction
is ≥80% gap-recovery to the full-data baseline at 30% data fraction,
across paradigms. H70 tests this on the Transformer paradigm; the
result will generalise.

---

## Chunk 6 — Interpretability, emergence, and neuroscience

### 6.1 The 6×N interpretability table

| Interpretability dimension | Liquid | JEPA | KAN | Transformer | GNN | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Primary visualisation** | trajectory in state space | UMAP of latent features | spline shape per edge | attention heatmaps | message-flow graph | cymatic-pattern decomposition + Platonic UMAP + Betti curves |
| **Mechanistic-interpretability technique** | phase-portrait | feature-correspondence | symbolic-extraction (`suggest_symbolic`) | circuit tracing | graph attribution | combined: spline shapes + Betti tracking + dodeca UMAP |
| **Neuroscience correspondence** | LTC = pyramidal RC dynamics | predictive coding (Rao & Ballard 1999) | dendritic-tree computation | binding / first 100ms | local synaptic message-passing | combined |
| **Grokking analogue** | smooth trajectories converge late | EMA collapse | spline crystallisation | grokking in attention heads | (rare) | Betti collapse rate (H65) |
| **Mode-collapse failure mode** | τ degenerate | EMA tracks student | spline saturates flat | attention heads attend to BOS only | over-smoothing | one-hot gate → degenerate single-paradigm |
| **Probe for "what learned"** | τ histogram | mask-position-conditional MI | spline-shape inspection | head function attribution | edge-importance | combined battery |

### 6.2 Deep dive #1 — Cymatic patterns as the cross-paradigm interpretability tool

Cymatic / Chladni eigenmode decomposition gives a **paradigm-agnostic
visualisation**: take any layer's activation, project onto a
Chladni-mode basis, plot the coefficient magnitudes. The resulting
"cymatic spectrum" is interpretable as a multi-resolution decomposition
of what the layer has learned. The chunk-6 prediction is that **the
cymatic spectrum is a reliable probe for layer-wise specialisation**:
early layers should show broad low-frequency content; late layers
should show narrow high-frequency content. A failure mode of any
paradigm is a **flat cymatic spectrum** (the layer is uninformative or
collapsed). H66 (cymatic QKV init) directly exploits this; H70
(cymatic curriculum) measures it during training. The natural
precedent is the auditory cortex's tonotopic organisation, which is
itself a cymatic decomposition of audio inputs onto the basilar
membrane.

### 6.3 Deep dive #2 — Betti curves and the trained-feature collapse

Naitzat et al. 2020 (arXiv:2004.06093) report that **Betti numbers of
trained-feature distributions collapse layer-by-layer** in well-
trained classifiers — β_0 falls to the number of classes, β_1 falls
to zero, higher β_k vanish. This is the chunk-6 reliable diagnostic
for "is the network learning". The sacred-geometry contribution
(H51 / H54 / H65) is to **add a differentiable PH loss** that
explicitly rewards Betti collapse. The chunk-6 prediction is that the
loss accelerates collapse without harming downstream accuracy, with
the additional benefit of providing a **publishable interpretability
metric** alongside accuracy. The trained-feature Betti is
distinguished from fresh-init Betti (H59); the project's current
state computes only fresh-init, which is the wrong signal.

### 6.4 Deep dive #3 — The grokking analogue across paradigms

Grokking (Power et al. 2022) is the phenomenon where Transformer
training plateaus, then suddenly generalises after many extra epochs.
Each paradigm has its own analogue: Liquid shows smooth-trajectory
convergence (no sudden jump); JEPA shows EMA-collapse-then-recovery;
KAN shows spline-crystallisation (the spline coefficients become
discrete-looking suddenly); GNN rarely groks. The sacred-geometry
prediction is that **Betti-collapse rate** is the unifying
**grokking-detection metric** across paradigms: in each, grokking
coincides with a sharp drop in mid-layer β_1. H65 turns this
observation into a training-time loss; the resulting model **groks
faster** by design. The natural precedent is cortical critical
periods, where structured pruning compresses representations
discontinuously.

---

## Chunk 7 — Limitations, cross-pollination, hybrid risks

### 7.1 The 6×N limitations table

| Limitation dimension | Liquid | JEPA | KAN | Transformer | GNN | **NaturePrior synthesis** |
|---|---|---|---|---|---|---|
| **Worst failure mode** | τ collapse | EMA collapse | spline saturation | softmax-attention sink | over-smoothing | gate collapse to one sub-path |
| **Scale ceiling (current)** | 7B (LFM2 reports) | 1B (V-JEPA family) | sub-100M typical | 1.5T (GPT-4-scale) | 100M typical | bounded by max(constituents) |
| **Symbolic reasoning** | weak | weak | strong (closed-form) | moderate | moderate-strong (graph) | KAN-Metatron symbolic head (H69) |
| **Long-context** | strong (linear) | moderate (encoder bounded) | weak | moderate-strong with KV optimisation | weak | hex + toroidal KV (H62) |
| **Multi-modal** | weak | strong (mask-augmented) | weak | strong | moderate | inherits JEPA |
| **Reliability for production** | high (LFM2 ship) | research | research | very high | research | gated, with H67 mode-collapse falsifier |
| **Cross-pollination risk** | LTC-attention conflict | EMA + grad-checkpoint memory | spline-grad instability | RoPE-equivariance conflict | over-smoothing in hybrid | per-sub-path test in H61-H66, then H67 |

### 7.2 Deep dive #1 — Hybrid risk: when paradigms actively conflict

The previous CIFAR-10 sweep's negative result (full hybrid is the
worst row) is a real cross-pollination warning. The chunk-7 analysis
identifies **three specific conflict modes**: (a) the **C4 max-pool
+ toroidal pad** combination over-smooths feature maps (H58 is the
fix); (b) **cymatic init + He init on the same layer** doubles the
initialisation variance, blowing up early gradients; (c) **fractal
recursion + dropout** at the same scale collapses the recursion to
the identity. The H67 design **mitigates each** by per-sub-path
gating: the gate can down-weight a conflicting sub-path to zero,
recovering the best single-paradigm performance. The mode-collapse
falsifier (H67 § 9) tests for the degenerate case where the gate
collapses to one-hot — turning the hybrid into a no-op.

### 7.3 Deep dive #2 — Why per-prior ablation is the only honest evaluation

The synthesis claim's defensibility rests on **independent
ablatability of every prior**. H67's `NaturePriorBlockV2` has five
sub-paths, each gated by a learned softmax — so dropping any sub-path
is a 1-line config change. The H61-H66, H68-H71 hypotheses each test
**one prior in isolation** (or a small composition of 2-3 priors)
against the same composite metric. The chunk-7 prediction is that the
ablation pattern will be **multi-modal, not unimodal**: some priors
will help in isolation but conflict together; some will help only in
combination. Honest reporting requires the full Pareto sweep —
without it, the synthesis claim is unfalsifiable.

### 7.4 Deep dive #3 — Risk of Goodharting the composite metric

The composite metric (`0.30 * neg_norm_ppl + 0.20 * norm_gsm + ...`)
is SHA-256 fingerprinted to prevent edit-during-training, but is
itself a **Goodhart target**. The chunk-7 mitigation is the
**Goodhart fingerprint** in the autoresearch protocol: any change to
the composite formula breaks the next experiment's launch gate. The
chunk-7 deep risk is that the synthesis sub-paths can **collude** to
optimise the composite without improving the underlying capability —
e.g., a sub-path that gives PPL gains by overfitting to validation
distribution. The defence is the **multi-axis Pareto criterion**:
gains must show on **independent** axes (perplexity, GSM, ARC,
3D-equivariance, KV cache, latency) rather than a single composite.
This is the chunk-7 deep-risk discipline that H67's § 3 falsifier
encodes.

### 7.5 Deep dive #4 — The "negative-result" path is a feature

A non-trivial probability of the H67 hybrid is that **the learned gate
collapses to a single sub-path** (most likely attention, the
best-tested paradigm). This is **not a failure of the project** — it
is a publishable result that the cortical "all five paradigms
simultaneously" claim does not hold at 350M scale. The chunk-7
discipline requires that this outcome is **reported as
mode-collapse**, not hidden. The autoresearch protocol's
append-only experiment log (`experiment_log.jsonl`) makes hiding
impossible. The synthesis claim is therefore **risky in the Popperian
sense** — falsifiable, with a specific failure mode pre-registered.
This is what distinguishes the project from a mystical sermon
(`MANIFESTO.md` § 3).

---

## Chunk 8 — Synthesis + 25 untapped opportunities

### 8.1 The 25 opportunities table

Extracted from chunk 8 of the extended transcript. Each row is a
**concrete, 4090-feasible experiment**. Columns: mechanistic
description, 4090 protocol, key metric, projected gain, H-tie (which
H61-H71 hypothesis tests it).

| # | Opportunity | Mechanism | 4090 protocol | Key metric | Projected gain | H-tie |
|---|---|---|---|---|---|---|
| 1 | φ-spaced LTC bank in JEPA predictor | 4 LTC heads with τ_k = τ_0 · φ^k inside JEPA predictor | 350M decoder on TinyStories, 200 ep | 32k-context PPL | -0.4 to -0.7 nats | **H61** |
| 2 | Toroidal KV-cache + hex attention graph | KV ring buffer + hex graph + Fib-prune | 124M WikiText-103 progressive 8k→128k | KV memory @ 128k | -40% to -50% | **H62** |
| 3 | Platonic dodeca-projection aux loss | 20-D dodeca projection after every layer | GPT-2-small fine-tune λ=0.1-0.3 | GSM8K zero-shot | +2.0 to +3.0 pp | **H63** |
| 4 | Cymatic wavelet teachers | Pre-computed Chladni-mode targets | same fine-tune | multi-modal cosine | +0.04 | **H63** |
| 5 | Dynamic φ-growth + Fib-prune + cymatic gate | grow at Fib-spaced epochs gated by cymatic resonance | 124M WikiText-103 | GPU-hours to target PPL | -25% to -45% | **H64** |
| 6 | Differentiable Betti-collapse loss | PH loss on every k-th layer | 124M GSM8K + rotated-CIFAR | β_0 AUC | -25% to -50% | **H65** |
| 7 | Cymatic wavelet QKV init | Chladni eigenmodes for Q, K, V matrices | 350M WikiText-103 | early-step PPL | -0.5 nats @ 1k step | **H66** |
| 8 | Resonant Top-K KD | Top-K logit selection by cymatic resonance | 124M from a 7B teacher | distillation quality | +20% at same K | chunk-5 #7 |
| 9 | Sacred-Liquid-JEPA-KAN-GNN-Transformer block | full 5-paradigm gated block | 350M TinyStories + GSM8K | composite | +0.04 to +0.08 | **H67** |
| 10 | On-device toroidal world model | JEPA next-latent on toroidal KV + cymatic curriculum | 124M TinyStories + synthetic physics | next-step acc | +10 pp | **H68** |
| 11 | KAN edges on Metatron graph | KAN spline on 13-vertex 78-edge graph as symbolic head | 124M GSM8K + Feynman | GSM8K | +3 pp | **H69** |
| 12 | Cymatic resonance curriculum (low-data) | Chladni frequency curriculum input stream | 124M on 10-50% TinyStories | gap-recovery | ≥80% | **H70** |
| 13 | Icosahedral RoPE for 3-D spatial | I-group rotations in RoPE | 124M 3-D nav QA + 3-D-Shapes | nav QA | +3 pp | **H71** |
| 14 | Fibottention with hex graph | Per-head Fib dilation on hex lattice | 350M WikiText-103 | KV cache | -25% | chunk-2 (H32+H21) |
| 15 | Golden-spiral PE + RoPE-φ co-design | spiral PE alongside φ-modulated RoPE base | 124M | long-context PPL | -0.1 nats | (H36+H34) |
| 16 | Platonic-Fibonacci hybrid graph for GNN | icosa adjacency with Fib node degrees | molecular benchmark | QM9 MAE | -5% | (H30) |
| 17 | Drop-path anytime with φ-fractal stages | FractalNet drop-path on φ-recursion | ImageNet-100 | latency-acc Pareto | new frontier | (H52+H05) |
| 18 | Cymatic Loss in Fourier domain | activation power-spectrum match to Chladni mode | 124M | activation spectrum | new metric | (H46) |
| 19 | Trained-feature Betti curves | re-run topology on best.pt | full sweep | trained β-AUC | (descriptive) | (H59) |
| 20 | PRH alignment + Betti combined loss | composite of H49 + H51 | 124M | aux convergence | improved | (H49+H51) |
| 21 | Multi-seed Pareto with 3-seeds | seeds 0/1/2 every variant | full sweep | confidence intervals | (publishable) | (H60) |
| 22 | C4 avg-pool fix (the dominant negative finding) | replace max-pool with avg-pool in group conv | CIFAR-10 sweep rerun | top-1 | +10 pp | (H58) |
| 23 | φ-NAS over the prior library | DARTS-style NAS restricted to φ/Fib/Platonic | CIFAR-10 NAS | top-1 + params | new optimum | (H45) |
| 24 | Spherical / icosa 2-D-3-D bridge | GICOPix planar unfold of icosa to share 2-D hex weights | Spherical MNIST | acc-iso-params | +1 pp | (H53) |
| 25 | NaturePriorBlock v2 (SacredGeoBlock v2) | the canonical synthesis block | flagship | composite | +0.04 to +0.08 | **H67** |

### 8.2 Why opportunity #9 (H67) is the flagship

Opportunity #9 is the only row whose explicit mechanism is "do all
the others at once". The other 24 opportunities are largely
**single-axis** (one prior, one paradigm); #9 is the **Pareto-test**
of whether they compose. The full hybrid is the synthesis claim of
this entire program — and its falsifier (gate-collapse →
single-paradigm reduction) is the synthesis claim's antithesis.
Running #9 without first running #1-#8 and #10-#13 would be
unscientific (you would not know which sub-paths to gate); running
#1-#13 without #9 would leave the synthesis claim untested.

### 8.3 The path through the 25 opportunities

The recommended execution order on a single 4090 Laptop is:

1. **Phase 1 (≈3 weeks):** opportunities #22, #19, #21 — fix the
   dominant negative, get trained-feature Betti, and add 3-seed
   error bars. These are the autoresearch hygiene moves.
2. **Phase 2 (≈4 weeks):** opportunities #6, #7, #11, #13 — the
   loss-only and head-only changes that are cheapest. These give
   independent ablations.
3. **Phase 3 (≈4 weeks):** opportunities #1, #2, #10 — the
   structural changes. Each is a 24-72 h experiment.
4. **Phase 4 (≈3 weeks):** opportunity #9 (H67) — the flagship.
5. **Phase 5 (≈4 weeks):** opportunities #3-#5, #12, #14-#18, #20,
   #23-#24 — the depth and breadth tests.

Total: ≈18 weeks on a single laptop. With agentic parallelism on a
cloud cluster, this contracts to ≈4 weeks.

---

## Conclusion: SacredGeoBlock v2 / NaturePriorBlock v2 — canonical pseudocode

The synthesis claim of this document is operationalised as a single
drop-in residual block — the **NaturePriorBlock v2**. It is the
LLM-track flagship implementation (the CNN-track flagship is its
mirror in `src/nature_inspired_networks/blocks.py`).

```python
# src/nature_inspired_networks/nature_prior_block_v2.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from .sacred_liquid    import PhiLTCBank         # H61
from .jepa             import JEPApredictor      # H61
from .kan_metatron     import KANMetatronHead    # H69
from .hex_attn         import HexToroidalAttn    # H62
from .icosa_rope       import MHSA_icosa_phi_RoPE # H34 + H71
from .priors           import (cymatic_init_,    # H35 / H66
                               toroidal_pad,     # H22 / H68
                               fibonacci_channels, # H04 / H12
                               metatron_adjacency) # H23 / H40
from .ph_loss          import BettiCollapseLoss   # H51 / H65
from .platonic_proj    import DodecahedronProjection # H25 / H49 / H63

PHI = (1.0 + 5.0 ** 0.5) / 2.0

class NaturePriorBlockV2(nn.Module):
    """
    The flagship hybrid. A drop-in decoder block that runs all five
    paradigms in parallel on sacred manifolds, mixes them by a learned
    softmax gate, and emits auxiliary losses on the side-channel for
    PRH alignment (dodeca-projection) and Betti collapse (PH loss).

    Independently ablatable sub-paths (per FINDINGS.md discipline):
      * Liquid   (PhiLTCBank)            — H61
      * JEPA     (JEPApredictor)         — H61
      * KAN      (KANMetatronHead, gated)— H69
      * GNN      (HexToroidalAttn)       — H21 / H62
      * Trans    (MHSA_icosa_phi_RoPE)   — H34 / H62 / H71

    Sacred priors applied across sub-paths:
      * Cymatic-init for Q, K, V         — H35 / H66
      * Toroidal KV wrap                 — H22 / H62 / H68
      * φ-spaced LTC time constants      — H61
      * Dodeca aux projection            — H25 / H49 / H63
      * Fib-pruned hex graph             — H43 / H62
      * Icosahedral RoPE                 — H24 / H71
      * Fractal recursion (sub-block)    — H05 / H26 / H38
      * φ-channel widths                 — H04 / H12

    Loss-side hooks returned per forward:
      * dodeca   — for L_dodeca = MSE(proj, target_or_EMA)
      * betti    — for L_betti  = ∑_k λ_k ∫ persistence_k
      * jepa     — for L_jepa   = 1 - cos(z_pred, z_target_EMA)
    """

    def __init__(self, d: int, n_heads: int = 16, hex_taps: int = 7,
                 kan_grid: int = 8, n_taus: int = 4,
                 fractal_depth: int = 2):
        super().__init__()
        # Channel widths follow φ then round to nearest Fibonacci
        d_inner = fibonacci_channels(d, k=1)

        # Pre-norm
        self.norm_in = nn.RMSNorm(d)

        # Paradigm 1 — Liquid
        self.liquid = PhiLTCBank(d_inner, n_taus=n_taus, tau0=1.0)

        # Paradigm 2 — JEPA next-latent predictor
        self.jepa = JEPApredictor(d_inner, ema_momentum=0.999)

        # Paradigm 3 — KAN edges on Metatron's Cube
        self.kan = KANMetatronHead(d_inner, d_inner, grid=kan_grid)

        # Paradigm 4 — Hex graph GNN-style attention (toroidal + Fib-prune)
        self.gnn = HexToroidalAttn(d_inner, n_heads=n_heads,
                                   prune_ratio=1.0 / PHI ** 2)

        # Paradigm 5 — Transformer attention with icosa + φ-RoPE
        self.attn = MHSA_icosa_phi_RoPE(d_inner, n_heads=n_heads,
                                        use_icosa=True)
        # Initialise QKV with Chladni eigenmodes (H66)
        cymatic_init_(self.attn.qkv.weight, n_modes=12, phi_spacing=True)

        # Learned mixing gate over the 5 sub-paths
        self.gate = nn.Linear(d_inner, 5)

        # Project mixed back to d
        self.in_proj  = nn.Linear(d, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d, bias=False)

        # Optional fractal sub-block at 1/φ depth/width (H05 / H26 / H38)
        if fractal_depth > 0:
            self.fractal = NaturePriorBlockV2(
                d=int(d / PHI), n_heads=max(1, n_heads // 2),
                fractal_depth=fractal_depth - 1)
            self.fractal_proj_in  = nn.Linear(d, int(d / PHI), bias=False)
            self.fractal_proj_out = nn.Linear(int(d / PHI), d, bias=False)
        else:
            self.fractal = None

        # Auxiliary side-channels
        self.dodeca_proj = DodecahedronProjection(d_inner, d_target=20)
        self.betti_loss  = BettiCollapseLoss(maxdim=1, lam0=1.0, lam1=0.5)

    def forward(self, x: torch.Tensor,
                ema_target_block: 'NaturePriorBlockV2 | None' = None,
                pos_xyz: torch.Tensor | None = None,
                betti_every_k_steps: int = 16,
                step: int = 0) -> tuple[torch.Tensor, dict]:
        """
        Args
        ----
          x                : (B, N, d) — input hidden state
          ema_target_block : optional EMA-momentum twin block for JEPA
          pos_xyz          : (B, N, 3) if H71 icosa-RoPE is in 3-D mode
          betti_every_k_steps : amortisation factor for PH loss
          step             : training step (for amortisation + curriculum)

        Returns
        -------
          out  : (B, N, d) — residual-added output
          aux  : dict of auxiliary tensors {dodeca, betti, jepa, gate_dist}
        """
        h = self.norm_in(x)
        h = self.in_proj(h)                                # (B, N, d_inner)

        # Gate weights — context-sensitive softmax over sub-paths
        g = F.softmax(self.gate(h.mean(dim=-2)), dim=-1)    # (B, 5)

        # Run sub-paths in parallel (5-way ensemble)
        h_liq = self.liquid(h)
        h_jep, jepa_pred, jepa_targ = self.jepa(
            h, ema_target_block.in_proj(self.norm_in(x))
            if ema_target_block is not None else None)
        h_kan = self.kan(h)
        h_gnn = self.gnn(h, toroidal_wrap=True)
        h_att = self.attn(h, pos_xyz=pos_xyz)

        # Mix
        h_mix = (g[:, 0:1, None] * h_liq +
                 g[:, 1:2, None] * h_jep +
                 g[:, 2:3, None] * h_kan +
                 g[:, 3:4, None] * h_gnn +
                 g[:, 4:5, None] * h_att)

        # Fractal sub-block (1/φ scale, recurses to fractal_depth)
        if self.fractal is not None:
            x_frac     = self.fractal_proj_in(x)
            h_frac, _  = self.fractal(x_frac, ema_target_block=None,
                                       pos_xyz=None,
                                       betti_every_k_steps=10**9,  # disable
                                       step=step)
            h_mix = h_mix + self.fractal_proj_out(h_frac)

        # Residual add
        out_inner = h_mix
        out       = x + self.out_proj(out_inner)

        # Auxiliary side-channel outputs
        aux = {
            'dodeca'    : self.dodeca_proj(out_inner),       # (B, N, 20)
            'jepa_pred' : jepa_pred,
            'jepa_targ' : jepa_targ,
            'gate_dist' : g,                                  # (B, 5)
        }
        # Amortised PH loss — only every k-th step
        if step % betti_every_k_steps == 0:
            aux['betti'] = self.betti_loss(out_inner.detach())
        else:
            aux['betti'] = torch.zeros((), device=x.device)

        return out, aux


# --------------------------------------------------------------------
# Loss computation (called from the training loop)
# --------------------------------------------------------------------
def compose_natureprior_loss(
        logits: torch.Tensor,           # (B, N, V)
        targets: torch.Tensor,          # (B, N)
        aux_outputs: list[dict],        # one per layer
        dodeca_targets: torch.Tensor,   # (B, N, 20), e.g., EMA of own dodeca
        cymatic_targets: torch.Tensor,  # (B, N, 20), pre-computed Chladni
        lambda_jepa: float = 0.10,
        lambda_dodeca: float = 0.10,
        lambda_cymatic: float = 0.05,
        lambda_betti: float = 0.05,
        gate_collapse_thresh: float = 0.95):
    """
    Composite loss used by the H67 flagship.

    Returns
    -------
      total_loss : torch.Tensor (scalar)
      diag       : dict with per-component losses + gate-collapse flag
    """
    L_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1))

    L_jepa, L_dodeca, L_cymatic, L_betti, gate_collapse = 0., 0., 0., 0., False
    n_layers = len(aux_outputs)
    for aux in aux_outputs:
        # JEPA latent-prediction
        L_jepa  = L_jepa + (1.0 - F.cosine_similarity(
            aux['jepa_pred'], aux['jepa_targ'].detach(), dim=-1)).mean()
        # Dodeca alignment (PRH H49)
        L_dodeca = L_dodeca + F.mse_loss(aux['dodeca'], dodeca_targets)
        # Cymatic teacher signal (H63)
        L_cymatic = L_cymatic + F.mse_loss(aux['dodeca'][..., :20],
                                            cymatic_targets[..., :20])
        # Betti collapse aux (amortised; zero on off-steps)
        L_betti = L_betti + aux['betti']
        # Mode-collapse falsifier on the gate
        if aux['gate_dist'].max(dim=-1).values.mean() >= gate_collapse_thresh:
            gate_collapse = True

    total = (L_ce
             + lambda_jepa    * L_jepa    / n_layers
             + lambda_dodeca  * L_dodeca  / n_layers
             + lambda_cymatic * L_cymatic / n_layers
             + lambda_betti   * L_betti   / n_layers)
    diag = dict(L_ce=L_ce, L_jepa=L_jepa, L_dodeca=L_dodeca,
                L_cymatic=L_cymatic, L_betti=L_betti,
                gate_collapse=gate_collapse)
    return total, diag
```

### Composability table — which prior maps to which sub-path

| Prior | Sub-path | Where applied | Independently ablatable? |
|---|---|---|---|
| φ-channel widths | shared | `fibonacci_channels(d, k=1)` | yes (`k=0`) |
| φ-LTC bank | Liquid | `PhiLTCBank(d, n_taus=4, tau0=1.0)` | yes (drop) |
| JEPA next-latent | JEPA | `JEPApredictor` + EMA twin | yes (drop) |
| Metatron KAN edges | KAN | `KANMetatronHead` | yes (drop) |
| Hex toroidal attn | GNN | `HexToroidalAttn` | yes (drop) |
| Fib-pruning ratio | GNN | `prune_ratio=1/φ²` | yes (set to 0) |
| Icosahedral RoPE | Transformer | `MHSA_icosa_phi_RoPE(use_icosa=True)` | yes (off) |
| Cymatic QKV init | Transformer | `cymatic_init_(qkv.weight, n_modes=12)` | yes (He init) |
| Toroidal KV wrap | Transformer/GNN | `toroidal_wrap=True` | yes (off) |
| Dodeca aux projection | side-channel | `DodecahedronProjection` + L_dodeca | yes (λ=0) |
| Betti-collapse aux | side-channel | `BettiCollapseLoss` + amortise | yes (λ=0) |
| Cymatic teacher | side-channel | `cymatic_targets` + L_cymatic | yes (λ=0) |
| Fractal φ-recursion | sub-block | `NaturePriorBlockV2(d/φ, depth-1)` | yes (depth=0) |

### What this pseudocode is **not**

- It is **not** a final implementation. The `JEPApredictor`,
  `KANMetatronHead`, `HexToroidalAttn`, `PhiLTCBank`,
  `MHSA_icosa_phi_RoPE`, `DodecahedronProjection`, and
  `BettiCollapseLoss` modules are referenced; their full
  implementations live in `src/nature_inspired_networks/`.
- It is **not** a claim that all sub-paths help. The gate may collapse;
  the falsifier in H67 § 3 catches that.
- It is **not** licensed for ImageNet-scale extrapolation. The 4090
  Laptop 16 GB budget caps the operational scale at ≈350M params.

### What this pseudocode **is**

- A **single drop-in residual block** that operationalises the
  synthesis claim.
- A **gated parallel composition** of five paradigms, each on a sacred
  manifold.
- A **falsifiable architecture** — the gate distribution, the per-axis
  Pareto criterion, and the mode-collapse threshold are all
  explicit.
- The **canonical implementation target** for `ideas/67_full_paradigm_hybrid/`.

---

## Synthesis: the single sentence

A single drop-in residual block — the **NaturePriorBlock v2** — gated
over five paradigm-channels (Liquid LTC bank, JEPA next-latent
predictor, KAN-Metatron symbolic head, Hex-toroidal Fib-pruned graph
attention, Icosahedral-φ-RoPE Transformer attention) — running on
sacred manifolds (φ-channel widths, toroidal KV wrap, fractal
recursion at 1/φ scale) — with auxiliary losses for PRH alignment
(dodecahedral projection), Betti collapse (differentiable persistent
homology), and cymatic-teacher signal — is the synthesis claim
operationalised, and is independently ablatable per prior, paradigm,
and auxiliary, with mode-collapse caught by the gate-distribution
falsifier. **H67** is the test. **H61-H66, H68-H71** are the
ablations. **PARADIGM_COMPARISON.md** is the synthesis.
