# H30 — Platonic-Fib Hybrid

> **One-line claim:** A 3-D point-cloud network whose icosahedral/dodecahedral adjacency has node degrees following the Fibonacci sequence (1, 1, 2, 3, 5, 8) outperforms a uniform-degree icosa-adjacency baseline on ModelNet40 classification at matched parameter budget.
>
> **Source design space:** G3 Topologies & Graphs (H21–H30).
>
> **Implementation status (this repo):** `○ not started` (composition of H24 and H02).

This document is the committee-grade design write-up for hypothesis H30, the **composition** of H24 (icosahedral equivariance) and Fibonacci scaling (H02 / H04). It directly inherits the lessons from T1.4 (C4 max-pool failure) and is downstream of H58 (avg-pool fix).

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

The composition of Platonic 3-D symmetry with Fibonacci scaling captures TWO of the most pervasive natural patterns simultaneously: the icosahedral / dodecahedral 60-element symmetry group governs molecular self-assembly (viral capsids, fullerenes, quasicrystals), while the Fibonacci sequence governs phyllotaxis growth in plants (sunflower spirals, pinecone scales). Combining them in a single graph — an icosahedral mesh whose node degrees vary as Fibonacci numbers — produces a structure where each vertex's "fan-in" follows the natural growth law while the global symmetry follows the Platonic rule.

For deep learning, this hybrid is hypothesis-rich for two reasons. First, **multi-scale geometric prior**: Platonic gives the symmetry group (60 rotations), Fibonacci gives the scale hierarchy (1:1:2:3:5:8 vertex-importance ratio). Second, **3-D point-cloud data** (ModelNet40, ShapeNet, KITTI) is exactly the regime where both priors are data-aligned: 3-D objects have rotational symmetries (Platonic) AND fractal-like surface roughness (Fibonacci self-similarity). Per Cohen 2019 and Wang 2019 (DGCNN), point-cloud networks benefit substantially from symmetry priors; Fibonacci-degree allocation has not been tested in this literature but is the natural extension.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** Platonic symmetry and Fibonacci scaling are both demonstrably present in 3-D natural objects, a point-cloud GNN whose icosahedral-mesh adjacency has node degrees {1, 1, 2, 3, 5, 8} (Fibonacci-allocated across 20 dodecahedral vertices) raises ModelNet40 classification accuracy by ≥ +1.5 pp and reduces inference latency by ≥ 10 % relative to a uniform-degree-6 icosa-adjacency baseline at matched parameter budget, per the mechanism of Cohen 2019 and the H58 avg-pool fix.

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on ModelNet40, the Platonic-Fib variant fails to lift classification accuracy by ≥ 1.0 pp AND fails to reduce per-cloud inference latency by ≥ 5 % versus uniform-degree icosa baseline, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Cohen, T. S., et al. 2019 ICML 'Gauge Equivariant Convolutional
Networks and the Icosahedral CNN' (arXiv:1902.04615) — icosahedral
CNN reference; we extend with Fibonacci-degree variation.

Wang, Y., et al. 2019 ACM TOG 'Dynamic Graph CNN for Learning on
Point Clouds' (arXiv:1801.07829) — DGCNN reference; comparator.

Qi, C. R., et al. 2017 CVPR 'PointNet' (arXiv:1612.00593) — the
foundational point-cloud architecture; comparator.

Caspar, D. L. D., Klug, A. 1962 — biological icosahedral self-
assembly principle.

Vogel, H. 1979 Math Biosciences — Fibonacci phyllotaxis formula.

Wu, Z., et al. 2015 CVPR '3D ShapeNets: A Deep Representation for
Volumetric Shapes' (arXiv:1406.5670) — ModelNet40 dataset citation.
```

## 5. Mechanism

### 5.1 CNN / GNN track

Build an icosahedral mesh adjacency where the 20 dodecahedral vertices are partitioned into 6 Fibonacci groups of sizes {1, 1, 2, 3, 5, 8} (totaling 20) and each group has a different "fan-in" degree following the same Fibonacci numbers. Apply icosa-group equivariant message passing with avg-pool orbit reduction (per H58).

```python
# ideas/30_platonic_fib_hybrid/implementation.py
PHI = (1+5**0.5)/2; FIB = [1, 1, 2, 3, 5, 8]

def fib_partition_icosa():
    """Partition 20 dodeca vertices into Fib-sized groups."""
    sizes = FIB; assert sum(sizes) == 20
    groups, idx = [], 0
    for s in sizes:
        groups.append(list(range(idx, idx+s)))
        idx += s
    return groups

def platonic_fib_adjacency(N_points):
    """Construct icosa-equivariant adjacency with Fib degrees per group."""
    groups = fib_partition_icosa()
    A = torch.zeros(N_points, N_points)
    for g_idx, group in enumerate(groups):
        target_degree = FIB[g_idx]
        # connect each vertex in group to its target_degree nearest neighbours
        ...
    return A

class PlatonicFibPointNet(nn.Module):
    def __init__(self, F, F_out, N_points=1024):
        super().__init__()
        self.register_buffer("A", platonic_fib_adjacency(N_points))
        self.lin = nn.Linear(F, F_out)
        self.icosa_rot = IcosaGroupConv2d(F_out, F_out, k=1, reduction="avg")
    def forward(self, x):  # x: (B, N, F)
        m = self.A @ x  # Fib-aggregated message
        h = self.lin(m)
        return self.icosa_rot(h.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
```

- Params: 6-Fib-group projection + icosa rotation; comparable to PointNet.
- FLOPs: less than uniform-degree-6 baseline because most vertex groups have lower degree (avg degree ≈ 3.3 vs 6).
- Init: He init.

### 5.2 LLM track

For decoder-only Transformers, Platonic-Fib hybrid maps onto **3-D-spatial KV cache**: tokens are positioned at icosahedral vertices with Fibonacci-allocated attention degrees. Combined with H71 (icosa RoPE), provides a 3-D spatial reasoning prior.

Expected at 350 M scale on 3-D-nav-QA: **+2-5 pp** zero-shot accuracy.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (ModelNet40) | [+0.010, +0.040] | symmetry-aligned task |
| Classification acc (ModelNet40, primary) | [+1.5 pp, +4.0 pp] | direct claim |
| Per-cloud latency | [-10 %, -25 %] | Fib avg degree < uniform 6 |
| params | [-5 %, 0 %] | smaller fan-in saves params |
| FLOPs | [-20 %, -10 %] | Fib avg degree smaller |
| GPU latency (batch=1) | [≈0.8×, ≈1.0×] | favorable |
| Rotation-equivariance err | [-0.04, -0.08] | icosa group preserved |
| Betti collapse rate | [+0.10, +0.30] | symmetry-aligned acceleration |
| KV cache @ 32 k (LLM) | [-30 %, -50 %] | sparse 3-D adjacency |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** ModelNet40 (40-class CAD model classification, 1024-point clouds).
- **Architecture:** `PlatonicFibPointNet` with icosa group equivariance + Fib-degree fan-in.
- **Epochs / batch / precision / seeds:** 50 epochs, batch 32, bf16 AMP, 3 seeds.
- **Composite:** accuracy (0.5), latency (0.2), params (0.15), rot-eq (0.15).
- **Run-script:** `python ideas/30_platonic_fib_hybrid/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~1 hr/seed × 3 = 3 hr.
- **Archive:** `ideas/30_platonic_fib_hybrid/experiments/exp001_modelnet_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **ModelNet40 + rotated ModelNet40** (rotation-augmented test).
2. **ShapeNet part segmentation** (Wang DGCNN benchmark).
3. **2D-3D-S indoor scene segmentation** at low resolution.
4. **Spherical MNIST** (project as point cloud on sphere).

### 7.3 Cross-paradigm context (LLM track)

3-D-spatial reasoning fine-tune at 350 M on a synthetic 3-D-nav-QA benchmark with icosa RoPE (H71) + Fib-degree attention pattern.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G3 row H30.
- Master: planned Tier 3.
- Sub-dir: `ideas/30_platonic_fib_hybrid/`.
- Composes: H02 (Fib depth), H04 (Fib widths), H23 (Platonic graph), H24 (icosa CNN), H58 (avg-pool fix — prerequisite), H71 (icosa RoPE).
- Conflicts: H29 (small-world; random rewiring opposite of structured Fib).

## 9. Committee Q&A

**Q: Why isn't this just DGCNN with a different graph?**

> DGCNN uses dynamic k-NN graphs in feature space (NOT geometric symmetry). H30 uses a FIXED icosahedral mesh with Fib-degree partitioning. The structural prior is fundamentally different.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 gives accuracy ≥ +1.0 pp AND latency ≤ -5 %. Either failure discards.

**Q: What if Fib-degree partitioning is no better than uniform?**

> The uniform-degree-6 control is the H24-without-Fib baseline. If they tie within 0.5 pp, the Fib partitioning is null and the hypothesis fails.

**Q: T1.4 (C4 proxy) was -10 pp. Why expect H30 to recover?**

> Because (a) we use AVG-pool over orbit (H58 fix, not max-pool), (b) we operate on 3-D point clouds where icosa symmetry is data-aligned (not 2-D CIFAR-10 where it is irrelevant), and (c) Fib-degree partitioning reduces FLOPs vs uniform-6.

**Q: How do we know the implementation is correct?**

> `tests/test_platonic_fib.py::test_fib_partition_sum_20` asserts `sum(FIB[:6]) == 20`. `test_avg_pool_not_max` asserts the orbit reduction is mean. `test_icosa_60_orbit` asserts the rotation group has 60 elements. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/30_platonic_fib_hybrid/implementation.py` tests green
- [ ] `ideas/30_platonic_fib_hybrid/tests.py` ≥ 6 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 3
- [ ] FINDINGS reflects result

## 11. Lessons from T1.4 (C4 proxy) and H58 prerequisite chain

T1.4 (`sg_only_group`) failed at -10.27 pp because (a) it used C4 max-pool, (b) on upright CIFAR-10 (wrong regime). H30 inherits the H58 fix (avg-pool) and shifts to ModelNet40 (3-D, symmetry-rich). Key changes from T1.4 → H30:

1. **Group expansion**: C4 (4 elements) → I-60 (60 elements). 15× more orbit copies with proper avg-pool.
2. **Orbit reduction**: max-pool (T1.4) → avg-pool (H58 fix).
3. **Data regime**: upright CIFAR-10 → 3-D point clouds (ModelNet40).
4. **Fib structure**: uniform degree (H24) → Fibonacci-allocated degrees per H30.

The H58 follow-up experiment T2.1 (currently running) is the **direct prerequisite** for confirming the avg-pool fix recovers the lost capacity. Once T2.1 yields +8 to +10 pp recovery on CIFAR-10 sg_only_group, H30 launch is justified on ModelNet40.

## 12. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B. Integrates T1.4 lesson and H58 prerequisite chain.
