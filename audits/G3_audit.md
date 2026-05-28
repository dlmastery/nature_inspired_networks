# G3 Audit — Topologies & Graphs (H21–H30)

Reviewer: skeptical PC/NeurIPS-style audit, treating shape-only tests as
MINOR and mechanism-verifying tests as required for PASS. Per-hypothesis
findings cover (1) mechanism vs. formal claim, (2) math (PHI drift,
axis/dim, RNG), (3) test rigor, (4) citation alignment, (5) falsifier
reachability, (6) hidden bugs.

## Summary table

| H | Verdict | One-line reason |
|---|---|---|
| H21 hex_phi | MAJOR | φ-radial weighting in the doc is NOT in the implementation; k=3 mask is only 180-sym (acknowledged in code, but original H21 formal hypothesis still demands 6-fold isotropy from the default radius=1 path). |
| H22 toroidal | MAJOR | φ-scaling of wrap distance from the doc is absent in `toroidal_pad`; only zero-φ circular pad is implemented. T1.6 was negative on the partial; full claim is untested. |
| H23 platonic_graph | MAJOR | Doc claims 78-edge Metatron's Cube graph; implementation has 24 undirected edges (ring lattice). Doc's center↔inner and inner-ring are weight 1, but the missing 54 long-line connections (which define Metatron's Cube) are absent. |
| H24 icosa | MAJOR | `icosahedral_rotations()` is mathematically correct (60 elements, det=+1, full closure verified by test). BUT `IcosaConv1d.forward` reduces with `amax(dim=1)` (max-pool) — directly violates the doc's mandated avg-pool + the T1.4 H58 lesson; the 60-rotation buffer is registered then unused (orbit is only a cyclic shift inside Fib-channel groups). Equivariance is C8×C13×C21, not I-60. |
| H25 dodeca_latent | PASS | 20 vertices with correct golden-ratio coordinates `(±1,±1,±1), (0,±1/φ,±φ), (±1/φ,±φ,0), (±φ,0,±1/φ)`; all on sphere √3. Tests verify count, sphere, uniqueness, distance-loss semantics. icosa-symmetry test promised in doc Q&A is missing (MINOR sub-issue). |
| H26 fractal_toroidal | PASS | Faithful FractalNet-style recursion with toroidal conv at every node; φ-shrink correctly drops mid-width by 1/φ; tests cover recursion depth, φ-shrink existence, stride, and a real toroidal-vs-zero-pad differential. |
| H27 spiral_graph | MINOR | Vogel formula correct (`r_k=sqrt(k+1)`, `θ_k=k·golden_angle`); test verifies the 2-D formula coordinate-by-coordinate. But: the doc claims first 2 emb columns = spiral / remaining d-2 = Xavier; implementation lifts the 2-D spiral into D via a seeded random orthonormal projection — different structure but isotropy-equivalent. Doc Q&A's `test_nn_distance_uniform` (5× lower variance vs Xavier) is not implemented. |
| H28 cymatic_hex | MAJOR | Per-channel modulation factor `cos(ω·t + φ·c·t)` where `c` is the **output-channel** index, but H28's formal claim modulates the **6 peripheral hex taps** with `cos(ω·t + k·φ)` (`k = neighbour index`). Mechanism mismatch: the entire structural assertion ("cymatic resonance of the seven hex-stencil tap weights") is not realised; the impl multiplies the whole masked kernel by a single per-channel scalar. Strong points: t=0 regression to static HexConv2d, ω/t learnable with sin(0)=0 gradient guard, mask corners preserved. |
| H29 phi_small_world | MINOR | Watts-Strogatz construction correct; `p=None` defaults to `1/φ` faithfully; tests verify symmetry under rewiring across p∈{0, 0.1, 1/φ, 0.5, 0.9, 1.0}, edge-count preservation, seed determinism. Missing: doc Q&A promises `test_clustering_coefficient` and `test_path_length` to verify actual small-world properties; only structural correctness is tested, not the WS observable that the falsifier rests on. |
| H30 platonic_fib | MAJOR | Doc claims 20 dodecahedral vertices, Fib partition {1,1,2,3,5,8}=20; impl uses 12 icosahedron vertices with `(1,1,2,3,5)=12`. Different polyhedron, different Fib sum. Worse: the doc's central mechanism — "Apply icosa-group equivariant message passing with avg-pool orbit reduction (per H58)" — is absent. Impl is a plain GNN with k-NN-by-distance adjacency; no rotation equivariance, no group orbit. The "Platonic" property reduces to "vertex coordinates happen to be icosahedral". |

Tier counts: PASS = 2, MINOR = 2, MAJOR = 6, BROKEN = 0.

---

## H21 — Hexagonal φ-Packing

- **Module**: `src/nature_inspired_networks/priors.py:HexConv2d`, `hex_kernel_mask`.
- **Verdict**: MAJOR.
- **Mechanism vs claim**: Formal hypothesis §2 requires the "six peripheral weights scaled by 1/φ relative to the centre" — the `HEX_RADIAL` constant in the doc's `5.1 CNN track` (lines 67–71 of H21). The implemented `HexConv2d` carries only the binary mask; **no φ-radial scaling buffer exists**. The hypothesis tested in CIFAR-10 sweep (T1.3) was a strictly weaker variant (uniform 7-tap mask) and the doc explicitly flags this: "the previous CIFAR-10 sweep (T1.3) used UNIFORM weights across those 7 taps; this hypothesis extends … with φ-radial energy distribution." The extension is not in the code.
- **Math**: `hex_kernel_mask(3)` zeros `m[0,2]` and `m[2,0]` (sum=7). This is **only 180°-symmetric**, not 6-fold. The H21.v2 docstring (lines 344–351 of `priors.py`) now openly acknowledges this and offers `hex_kernel_radius=2` (5×5, 19 cells, true hex-radius-2 by `|q|+|r|+|q+r|≤4`) as the "true 6-fold isotropic" path. The radius-2 mask math is sound — verified the inequality matches the standard axial-coord hex disc. But the **default path** (radius=1) is the one wired into the legacy NaturePriorBlock and the one that ran in T1.3, so the formal claim's 6-fold isotropy argument never matches the default code path.
- **Test rigor**: `test_priors.py` covers (a) mask zero corners, (b) radius-2 mask cell count = 19, (c) zero corners survive into the **effective** masked kernel `conv.weight * mask` (good — checks the forward path), (d) radius-2 path also forward-shape-preserving, (e) unsupported `hex_kernel_radius=3` rejected. No φ-radial weighting test (because the feature does not exist). No 60° equivariance test (despite doc §9 promising `test_hex_60deg_eq` to within 1e-4). Test discipline is structural; the mechanism claimed by the hypothesis is **not** verified.
- **Citation alignment**: Hoogeboom 2018 (arXiv:1803.02108) supplies the mask; the φ-radial extension is the H21 increment and is unimplemented. The doc accurately cites; the code does not deliver the increment.
- **Falsifier reachable?**: The falsifier requires top-1 lift ≥ 1.0 pp AND rot-eq-err drop ≥ 0.02 on rotated-CIFAR for the **φ-weighted** variant. Because φ-radial is not implemented, the falsifier cannot be exercised; running the current code would only re-test T1.3.
- **Hidden bugs**: None destructive — but the mismatch between doc §5.1's `HEX_RADIAL` matrix (with 1/φ on neighbours, He-init `gain = 1.0 / sqrt(1 + 6·(1/φ)²) ≈ 0.71`) and the absent code is itself a doc-vs-impl defect.
- **Fix**: Add an `phi_radial: bool = False` (default off, preserving T1.3 row) constructor flag that registers a second buffer `(HEX_MASK * HEX_RADIAL)`. Add unit test asserting `w_eff[centre] / w_eff[neighbour] == φ` to within tolerance, plus a 60° equivariance test (rotated input vs rotated output) — the doc already enumerates exactly these.

## H22 — Toroidal φ-Closure

- **Module**: `src/nature_inspired_networks/priors.py:toroidal_pad`.
- **Verdict**: MAJOR.
- **Mechanism vs claim**: Formal hypothesis §2 requires "wrap distance is φ-scaled — i.e., the effective receptive field wraps … with period φ·W and φ·H". The doc's §5.1 code snippet uses `eff_pad = int(round(PHI * pad)) if phi_scale else pad`. The current `toroidal_pad(x, pad)` accepts a single `pad` argument and applies `F.pad(... mode="circular")` without any φ-scaling option.
- **Math**: For `pad=1`, `round(φ·1) = round(1.618) = 2`, so φ-scaling would change behaviour from a single-pixel wrap to a two-pixel wrap (visible already on a 3×3 conv). The doc's own §9 Q&A even concedes "at this discretization the φ-scaling is INDISTINGUISHABLE from a 2-pixel wrap" — which means **the feature is testable and observable**, but it is not implemented. Wrapping correctness itself is fine: `F.pad(..., mode="circular")` is the canonical way.
- **Test rigor**: `test_toroidal_pad_is_circular` checks shape `(1,1,6,6)` and that the left-pad column equals the right edge — a single mechanism check on the wrap direction. Good but minimal. No test for the φ-scaling claim (because it is absent). Doc Q&A promises `test_phi_scaling` (asserts effective wrap = `round(PHI*k)`) and `test_translation_eq_on_torus` (equivariance on the torus) — neither is implemented.
- **Citation alignment**: Pittorino 2022 (arXiv:2202.03038) supports flatter-landscape claim; CircleNet (Schubert 2019) for circular-pad. Both are about plain circular padding. The φ-scaled wrap is the H22 increment; unimplemented.
- **Falsifier reachable?**: Requires the φ-scaled wrap to evaluate against tiled-data top-1 + boundary-err. Cannot be tested without the φ-scaling code path.
- **Hidden bugs**: None functional — but a configuration sweep tag `sg_toroidal_phi` would currently run identical-to-`sg_only_toroidal`, silently re-running T1.6.
- **Fix**: Add `phi_scale: bool = False` argument to `toroidal_pad`; multiply `pad` by `int(round(PHI*pad))` when set. Add a `test_phi_scaling` unit test asserting the pad amount equals `int(round(PHI*k))` and a tiled-input regression that boundary outputs match the wrap-position content.

## H23 — Platonic φ-Graph (Metatron-Cube)

- **Module**: `src/nature_inspired_networks/platonic_graph.py`.
- **Verdict**: MAJOR.
- **Mechanism vs claim**: Doc §1 — "Metatron's Cube … yields a **78-edge graph** that connects every vertex of the 13 circle centers." Doc Q&A §9 — "`test_metatron_edge_count` asserts |E|=78." Implementation builds 24 undirected edges (4 classes × 6) — the **lattice subset**, not the full Metatron's-Cube line collection. The choice between 1.0 (center↔inner + inner-ring) and 1/φ (outer-ring + radial spokes) is an internally consistent 2-class partition but it is **not** what the doc claims. The geometry literature defines Metatron's Cube precisely as connecting every pair of 13 centres (C(13,2)=78), out of which 24 form the local hex / radial pattern and the remaining 54 are the long "Platonic-projection" lines that justify the artifact's name.
- **Math**: Adjacency is symmetric (verified by test), zero-diagonal, edge weights in `{0, 1, 1/φ}`. Symmetric normalisation `D^{-1/2} A D^{-1/2}` is Kipf-Welling standard. `_effective_adj` symmetrises the learnable edge gate (`0.5*(g + g.t())`). Math is internally clean.
- **Test rigor**: Good — symmetry, weight-set membership, distinct value counts (24 ones, 24 inv-phi as directed entries), shape, learnable-gate path matches fixed at init then diverges under perturbation, gradient flow, rejection of wrong N. The tests **enforce the implementation's 24-edge choice**; they would also pass on a non-Metatron 24-edge graph. There is **no graph-theoretic test** (e.g., assert that every pair of vertices in the doc's 78-line Metatron set is connected) that would distinguish implementation from doc.
- **Citation alignment**: Battaglia 2018, Cohen 2018, Gilmer 2017 all support generic relational inductive bias / graph nets. None of the cited papers define Metatron's Cube; the geometric grounding is from sacred-geometry tradition and is mis-cited in the formal text vs the actual lattice subset.
- **Falsifier reachable?**: Falsifier is ROC-AUC ≥ +0.5 pp on ogbg-molhiv. The current 24-edge graph can run; outcome would not, however, test the **Metatron** hypothesis as claimed — it would test the lattice-hex-with-φ-rings adjacency.
- **Hidden bugs**: None functional. The learnable edge gate uses `0.5*(gate + gate.t())` ensuring symmetry — solid. The `clamp(min=1e-6)` on degree prevents div-by-zero on isolated nodes (no isolated nodes here, but defensive code is fine).
- **Fix**: Either (a) extend `metatron_cube_adjacency` to the full 78-edge complete connectivity with a 2- or 3-class φ partition (short-orbit / mid-orbit / long-orbit) and update tests to assert |E|=78; or (b) update H23 doc to claim the 24-edge hex+radial subset rather than the full Metatron line set, and rename to "Metatron-Lattice" to be honest.

## H24 — Icosahedral φ-Equivariant CNN

- **Module**: `src/nature_inspired_networks/icosa.py`.
- **Verdict**: MAJOR.
- **Mechanism vs claim**: Two distinct problems.
  1. The **doc's central mandate** (§5.1 + §11 + §9 Q&A) is **avg-pool orbit reduction** ("orbit reduction at the end of each block is AVG-pool over the 60 orbit elements (not max-pool, per the H58 lesson)"). The implementation reduces with `orbits.amax(dim=1)` at line 230 of `icosa.py`. Direct contradiction with the documented prerequisite chain. NOTE: there is a later editorial in `priors.py:GroupConv2d` docstring (lines 263–272) reversing the avg-pool stance based on a follow-up CIFAR-10 ablation that found max better on that data — but H24's `IcosaConv1d` still uses max-pool while the H24 hypothesis doc continues to mandate avg-pool. The repo has not reconciled the two narratives.
  2. The "60-element icosahedral group" is constructed correctly as a buffer (`icosahedral_rotations()` is mathematically rigorous — see math below) but is **unused in the forward of `IcosaConv1d`**. The `forward` constructs a `(60, C_out)` permutation table where each of the 60 group elements is mapped to a **cyclic shift within each Fibonacci channel group**. The effective equivariance is therefore `C_8 × C_13 × C_21` (three independent cyclic groups), not I-60. The doc claims "shared 1-D conv across the 60 icosahedral orbit copies" — the rotations are not applied to the conv weights at all.
- **Math**: `icosahedral_rotations()` is the cleanest piece of math in G3. Construction uses two generators (5-fold around vertex (0,1,φ) and 2-fold around edge midpoint), BFS closure, asserts exactly 60 elements emerge. Test `test_group_closure_under_multiplication` verifies `R_i @ R_j ∈ G` for all 3600 pairs to 1e-5 — this is the *defining group axiom* and the test is correct. `test_60_rotations_each_det_plus_one` verifies det=+1 (rotations, not reflections) and orthogonality `R R^T = I`. Math: PASS.
- **Test rigor**: Mathematical rigor for the rotation group is exemplary (60 elements, det=+1, orthogonality, closure). But `IcosaConv1d` tests only cover (a) forward shape with `out_channels = sum(hidden_sizes) = 42`, (b) AssertionError on mismatched out_channels. No equivariance test (`f(R·x) ≈ R·f(x)`), no avg-vs-max guard, no test that the 60 rotations actually flow through the forward pass. The doc Q&A explicitly promises `test_avg_pool_not_max` — absent. The shape-only tests do not catch the C-cyclic-vs-I-60 mechanism gap.
- **Citation alignment**: Cohen 2019 (arXiv:1902.04615) and Esteves 2018 (arXiv:1711.06721) accurately cited as references for the full icosa-CNN. The implementation is explicitly labeled a "lightweight equivariance proxy" in the docstring — so the author was honest about the proxy nature, but the doc's formal claim still requires the full I-60 equivariance.
- **Falsifier reachable?**: Spherical MNIST falsifier (top-1 ≥ +5 pp) is not testable with the current proxy because `IcosaConv1d` is a 1-D operator over a length-L feature; the doc's CNN-track is `(B, C, 5·H, H)` on a GICOPix unfold. The proxy cannot be wired into the CIFAR-10 NaturePriorBlock as-is; the doc's "TODO runner wiring" at the bottom of `icosa.py` admits this. Falsifier unreachable until a true `IcosaGroupConv2d` lands.
- **Hidden bugs**:
  - `_already_in` does an O(N) linear scan; the BFS has a safety cap of 200 iterations. Safe but slow at construction time (~1 second for the 60-element closure). Acceptable.
  - In `IcosaProjection.forward`, the result is `(B, N, 3) @ (3, 60) = (B, N, 60)`, which is a length-60 dot product with the rotated `ê_z` directions. Doc says this is "equivariant" — it is, up to permutation, *if* the input is rotated by an element of I; this is geometrically correct.
- **Fix**: (a) Replace `orbits.amax(dim=1)` with a configurable `reduce` flag and default to `"mean"` per the doc, or reconcile the doc to follow the GroupConv2d empirical finding. (b) Add an honest `test_equivariance_under_60_rotation` that perturbs the input by an element of `icosahedral_rotations()` and verifies the output transforms by a corresponding permutation. (c) Either rename `IcosaConv1d` to `FibCyclicConv1d` (honest about C-cyclic proxy) or implement the actual rotation-applied-to-kernel forward.

## H25 — Dodecahedral Latent

- **Module**: `src/nature_inspired_networks/dodeca_latent.py`.
- **Verdict**: PASS.
- **Mechanism vs claim**: Doc §5.1 — 20 vertices at golden-ratio coords; soft-assignment to vertices via softmax; convex combination produces a 3-D latent. Implementation is faithful: 8 cube + 4 (0,±1/φ,±φ) + 4 (±1/φ,±φ,0) + 4 (±φ,0,±1/φ) = 20. The doc's stated coordinate set matches the impl line-for-line.
- **Math**: All 20 vertices lie on a common sphere of radius √3:
  - Cube vertex (±1, ±1, ±1): norm² = 3.
  - (0, ±1/φ, ±φ): norm² = 0 + 1/φ² + φ² = 0.382… + 2.618… = 3 (uses identity `1/φ² + φ² = 3` via `φ² = φ + 1`).
  - The other two cyclic permutations: identical.
  Verified. The `vertex_distance_loss(z, idx)` correctly indexes V[idx] and uses MSE.
- **Test rigor**: Tests verify (a) unique-vertex count 20 (deduplicated by 6-d.p. rounding), (b) sphere radius √3, (c) post-normalisation unit norm, (d) projector forward shapes `(B,3) + (B,20) + (B,out_dim)`, (e) softmax sum = 1, (f) vertex-distance-loss zero on target and positive on perturbation, (g) input-shape rejection. Solid mechanism verification.
- **Citation alignment**: van den Oord 2017 (VQ-VAE), Snell 2017 (ProtoNets), Cohen 2019 icosa-equiv, Huh 2024 (PRH), Hendrycks 2017 (OOD baseline) — all directly relevant.
- **Falsifier reachable?**: Falsifier is OOD-AUC ≥ +1.0 pp + rotation-instability ≤ -0.03. Implementation is straightforwardly droppable into a `models.py` head — the `TODO runner wiring` comment is the only thing missing for an actual experiment. The falsifier is testable.
- **Hidden bugs**: None. One MINOR sub-gap: doc §9 promises `test_icosa_symmetry` (assert the 60-element icosa group permutes the vertex set). Not implemented. Adding it would strengthen the mechanism (the dodeca vertex set IS icosa-symmetric by construction; a one-liner test would lock that in).
- **Fix**: (Optional) Add `test_icosa_symmetry` using `icosahedral_rotations()` from H24's module — for each `R ∈ I`, assert `R @ V.T` (or `V @ R.T`) permutes V to within tolerance. This would tie H25 to H24 mathematically.

## H26 — Fractal Toroidal

- **Module**: `src/nature_inspired_networks/fractal_toroidal.py`.
- **Verdict**: PASS.
- **Mechanism vs claim**: Doc §5.1 specifies FractalNet-style two-branch recursion with toroidal padding at every conv and 1/φ width shrink on the deep branch, restored by a 1×1 projection. Implementation matches: `a = _ToroidalConv(c_in, c_out, k=3, stride)`, `b = FractalToroidalBlock(b1(x))`, merge `0.5*(a+b)`. `b1` projects to `c_mid = max(8, int(c_out / PHI))`, then recursive sub-block at width `c_mid`, then `_ToroidalConv(c_mid, c_out, k=1)` restores width. φ-shrink: for c_out=16, c_mid = max(8, int(16/1.618)) = max(8, 9) = 9; 9 < 16, so projection branch is created. Correct.
- **Math**: `_ToroidalConv` uses `toroidal_pad(x, pad)` before a `padding=0` Conv2d — correctness depends on `toroidal_pad`, which is itself correct (circular pad on the last two dims; see H22 finding). BN inside `_ToroidalConv` is sensible. Mean-merge with `0.5*(a+b)` preserves variance approximately for uncorrelated branches (matches FractalNet's drop-path-friendly contract).
- **Test rigor**: Six tests covering (a) depth=2 shape preservation, (b) toroidal-vs-zero pad **really differ** on a boundary-loaded input (this is a true mechanism check, not a shape check), (c) depth=1 falls back to a single conv path (asserts `not hasattr(block, "a")`), (d) depth=3 recursion, structural inspection of `block.b2.b2.depth == 1`, gradient finiteness, (e) `phi_shrink=True` creates `b_project`, `phi_shrink=False` does not, (f) stride=2 halves spatial dims. Excellent test discipline — the recursion structure and the φ-shrink choice are both directly observed.
- **Citation alignment**: Larsson 2017 FractalNet (arXiv:1605.07648) for fractal recursion; Pittorino 2022 (arXiv:2202.03038) for toroidal landscapes. Both directly relevant.
- **Falsifier reachable?**: Falsifier is tiled-CIFAR top-1 ≥ +2.0 pp + Betti-β₁ collapse ≥ +0.05 vs planar-fractal baseline. Block is wireable into a standard 3-stage scaffold.
- **Hidden bugs**: None visible. One stylistic note: the BN inside `_ToroidalConv` means BN statistics are computed AFTER the circular wrap content has folded into the feature map — biologically OK, but a planar-vs-toroidal A/B that didn't carefully match BN scaffolding could yield apparent gains attributable to BN-vs-pad interaction. Worth noting in the AUDIT.md for the idea sub-project.
- **Fix**: None required. (Optional) Add a Betti / persistent-homology unit test on a tiny toy if the project ever wires a real PH library.

## H27 — Golden Spiral Graph

- **Module**: `src/nature_inspired_networks/spiral_graph.py`.
- **Verdict**: MINOR.
- **Mechanism vs claim**: Doc §5.1 says "first 2 columns are the golden-spiral lattice scaled to unit variance, and the remaining `d-2` columns are Xavier-init". Implementation generates the 2-D spiral, normalises per-coordinate, then **lifts to D dimensions via a seeded random orthonormal projection** (`_seeded_orthonormal_projection` uses `torch.linalg.qr` on a `(M, M)` Gaussian then slices to `(in_dim, out_dim)`). The lifted embedding spreads spiral structure across all D dimensions rather than keeping it isolated in dim-0,1. This is structurally different from the doc but preserves the *isotropy claim* (orthonormal projection preserves pairwise distances up to a global scale), so the formal hypothesis "maximally isotropic discrete sampling" is upheld in a stronger form.
- **Math**: Vogel formula `r_k = sqrt(k+1)`, `θ_k = k · golden_angle`. `GOLDEN_ANGLE_RAD = 2π·(1 - 1/φ) ≈ 2.3999632…`. Per-coordinate standardisation (mean→0, std→1) and optional `scale` are clean. Orthonormal projection via QR is correct (singular values ≡ 1 on the smaller axis).
- **Test rigor**: `test_golden_spiral_2d_layout_matches_vogel_formula` directly verifies `r_k = sqrt(k+1)`, `θ_k = k * GOLDEN_ANGLE_RAD` for k=0..4 to 1e-5 — this is a true mechanism test, not a shape test. Other tests: seed determinism, no-NaN over varied (N,D), shape preservation, identity-return semantics, layer-level forward and gradient flow. Solid. The doc Q&A promises `test_nn_distance_uniform` (5× lower NN-distance variance than Xavier) — not implemented; this is the test that would actually establish the "isotropy" claim against the Gaussian baseline.
- **Citation alignment**: Vogel 1979, Glorot/Bengio 2010 (Xavier), Dwivedi 2020, Hu 2020 OGB — all relevant.
- **Falsifier reachable?**: Falsifier is `ogbg-molhiv` ROC-AUC ≥ +0.5 pp + convergence-epochs ≤ -10 %. Initialization helper is droppable into any `nn.Embedding` (the `GoldenSpiralEmbedding` wrapper in the doc's LLM-track is one-liner).
- **Hidden bugs**: For `N=1`, `spiral.std(dim=0, keepdim=True)` is undefined; the impl guards with `if N > 1` and `.clamp(min=1e-8)`. For very small N (N<D), the orthonormal projection is built at M=max(N,D) then sliced. Correct.
- **Fix**: Add the missing `test_nn_distance_uniform` (compute NN distance variance for spiral vs Xavier at matched N,D — assert spiral variance ≤ 0.2 × Xavier). Either align the doc text to describe the orthonormal-lift strategy, or rebuild the helper as `[spiral_2d | xavier_(d-2)]` to match the doc.

## H28 — Cymatic Hex Resonance

- **Module**: `src/nature_inspired_networks/cymatic_hex.py`.
- **Verdict**: MAJOR.
- **Mechanism vs claim**: Formal hypothesis §2 — "modulating the 7 hex-stencil taps with `w_k(t) = w_0,k · cos(ω·t + k·φ)`, k = neighbour index". Implementation modulates per **output channel** `c`, not per **tap** `k`. The factor `cos(ω·t + φ·c·t)` is one scalar per output channel that multiplies the entire masked kernel. So all 7 taps of a given output channel oscillate **in phase**; the doc's whole "cymatic resonance of the 6 peripheral taps" structure is absent.
- **Math**: Modulation: `phase = ω·t + φ·c·t = t·(ω + φ·c)`. At `t=0` every channel reduces to `cos(0)=1` — exactly the static HexConv2d. Gradient w.r.t. `ω` at `t=0` is `-t·sin(ω·t) = 0`, and the implementation initializes `ω=1.0` (not 0.0) — but that does not rescue the zero gradient (the zero comes from `t=0`, not `ω`). The test cleverly bumps `t=0.3` before checking grad flow, acknowledging this. Math is internally consistent for **the implemented per-channel scheme**, not for the doc's per-tap scheme.
- **Test rigor**: Good for the **as-implemented** behaviour:
  - `test_t0_reduces_to_static_hex_conv2d` — important regression: at init, output exactly matches static HexConv2d. **PASSES the doc's own §11 lesson 1 ("Dynamic resonance keeps the structure ALIVE through training") because the static point is recoverable.**
  - `test_omega_and_t_learnable_with_grad_flow` — bumps t to 0.3 first (clever workaround for the sin(0)=0 issue) and verifies both `t.grad` and `omega.grad` are non-zero.
  - `test_hex_corner_mask_preserved_in_effective_kernel` — confirms modulation does NOT leak past the mask (corners stay zero even at t=0.5, ω=2.0).
  - `test_per_channel_phi_spacing_distinct_modulation` — verifies adjacent channels see different modulation factors (PHI is irrational → consecutive c values give distinct cos values). This is the test that **enforces the implementation's per-channel choice**; it does not test the doc's per-tap claim.
  - `test_radius2_19tap_mask_path_works` — propagates radius-2 mask through the wrapper.
  - **Missing**: per-tap modulation test (because feature is absent). No spatiotemporal forward (the entire H28 narrative is about video / Conv3d — but the impl is `Conv2d` only).
- **Citation alignment**: Chladni 1787, Sussillo & Abbott 2009 (arXiv:0903.4537), Hoogeboom 2018, Tran 2018 — all cited correctly. The Sussillo & Abbott "Lyapunov stabilization" argument is the theoretical justification; per-channel oscillation does provide one form of Lyapunov-style stabilization, so the citation is NOT entirely lost — it just no longer maps cleanly to "the 7 hex taps".
- **Falsifier reachable?**: Falsifier is UCF-101 top-1 + seed-variance. Current Conv2d implementation cannot be tested on UCF-101 (3-D video) without significant rework. The hypothesis as documented (Conv3d, spatiotemporal) is **unreachable** from the current code; the Conv2d wrapper might be useful for static CIFAR-10 but is no longer testing the H28 claim.
- **Hidden bugs**: None destructive. The `_modulation` returns shape `(out_channels,)` and broadcasts correctly to `(O, 1, 1, 1)` × `(O, I, K, K)`. The toroidal flag is inherited from `HexConv2d` (line 124).
- **Fix**: (a) For doc-fidelity, expose a per-tap modulation: a 7-element phase buffer `[0, φ, 2φ, …, 6φ]` × t, with the centre tap held at 1.0 (so the cymatic resonance acts on the 6 peripheral taps as the doc claims) and broadcast against the masked kernel positions. (b) Add a Conv3d variant for the actual UCF-101 falsifier, or descope the H28 doc to a static CIFAR-10 claim. (c) Add explicit test `test_centre_tap_unmodulated` to nail down the doc's "6 peripheral taps modulated, centre tap static" structure.

## H29 — φ-Small-World

- **Module**: `src/nature_inspired_networks/small_world.py`.
- **Verdict**: MINOR.
- **Mechanism vs claim**: Doc §2 — rewiring probability `p = 1/φ ≈ 0.618`. Implementation: `phi_small_world_adjacency` defaults `p=None → 1.0/PHI`. Correct. Watts-Strogatz construction: ring lattice of degree k, then rewire each "right-side" edge `(i, (i+j)%n)` for j=1..half with probability p, picking a new non-neighbour target — this is the standard WS algorithm with a 1:1 swap (drop one edge, add one), preserving the edge count.
- **Math**: PHI value, p=1/φ, k must be even, symmetry maintained because the impl drops both `A[i,r]` and `A[r,i]`, then adds both `A[i,cand]` and `A[cand,i]`. Self-loops excluded (`cand == i` rejected). Duplicate-edge avoidance (`A[i, cand]` already True → rejected). The retry budget is `n_nodes` attempts; if exhausted (dense graph) the original edge is kept — a documented WS edge case.
- **Test rigor**: Five tests covering (a) shape and dtype (bool) and zero diagonal, (b) **symmetry across p ∈ {0, 0.1, 1/φ, 0.5, 0.9, 1.0}** — this is a real mechanism test, sweeping the parameter, (c) total edge count preservation (`n*k` directed entries for p=0; same count for arbitrary p), (d) GNN forward shape + adjacency determinism on identical seed, (e) rejection of wrong node count. Solid structural verification. **Gap**: doc Q&A promises `test_clustering_coefficient` (in [0.05, 0.20] at p=1/φ — the canonical WS regime indicator) and `test_path_length` (< 3·log(N)/log(k)) — neither implemented. These are the tests that would actually confirm the graph IS small-world rather than merely an arbitrary symmetric adjacency.
- **Citation alignment**: Watts & Strogatz 1998 Nature; Bullmore & Sporns 2009 NRN (the empirical p≈0.5-0.7 cortical claim); Kipf-Welling 2017 GCN; Sen 2008 (Cora); Hu 2020 OGB. All correctly cited.
- **Falsifier reachable?**: Falsifier is node-class accuracy on Cora + path-length distribution. Adjacency helper is drop-in for a standalone GCN run, but the layer wrapper (`PhiSmallWorldGNN`) does not address the Cora benchmark (it expects a `(B, N, D)` batch with fixed N — Cora is a single 2708-node graph). For the falsifier to be reached on Cora, the adjacency would need to be loaded directly into a Kipf-Welling GCN; doable but not provided.
- **Hidden bugs**: Minor: the rewiring loop walks `for j in range(1, half+1): for i in range(n_nodes): ...` — this is one valid WS sequencing; an earlier rewire can change `A[i, (i+j) %n]` to False, in which case the loop "leaves it" (line 95–96). That biases against multiply-rewiring the same node's edges, but is consistent with the canonical WS algorithm in NetworkX (`watts_strogatz_graph`). Not a defect.
- **Fix**: Add `test_clustering_coefficient` and `test_path_length` per doc §9. Both are one-pass tensor operations on `A`. The clustering coefficient test would discriminate p=1/φ from p=0.5 (or fail if they are indistinguishable), which is exactly what the doc §3 falsifier requires.

## H30 — Platonic-Fib Hybrid

- **Module**: `src/nature_inspired_networks/platonic_fib.py`.
- **Verdict**: MAJOR.
- **Mechanism vs claim**: Two compounding mismatches.
  1. **Vertex set / Fib partition**: Doc §2 — "20 dodecahedral vertices" with Fib partition `{1,1,2,3,5,8}` (sum 20). Implementation uses **12 icosahedron vertices** with Fib partition `(1,1,2,3,5)` (sum 12). Different polyhedron (dodeca is the dual; vertices and faces are swapped), different Fib length. The impl's docstring acknowledges 12 icosa vertices explicitly, but the H30 hypothesis doc continues to claim dodeca/20.
  2. **Icosa-group equivariance**: Doc §5.1 — "Apply icosa-group equivariant message passing with avg-pool orbit reduction (per H58)" with `self.icosa_rot = IcosaGroupConv2d(F_out, F_out, k=1, reduction="avg")`. The implementation `PlatonicFibPointConv` is a **plain GNN** message-passing layer (`A_norm @ X · W_msg + X · W_self`). There is no group orbit, no rotation, no avg-pool. The "Platonic" property reduces to "the vertex coordinates happen to be icosahedral".
- **Math**: `icosa_vertices()` generates the 12 vertices via cyclic permutations of `(0, ±1, ±φ)`. All on sphere of radius `sqrt(1+φ²)`. Test directly verifies. Vertex set is mathematically correct. The k-NN adjacency `fib_nearest_neighbors(verts, fib_counts)` uses `torch.cdist` for pairwise euclidean and `torch.topk(d[v], k=k_eff, largest=False)` — correct nearest-k selection. Symmetric OR-merge `(A + A.T) > 0`. Diagonal zeroed defensively. Math: clean.
- **Test rigor**: Six tests:
  - `test_icosa_vertices_12_on_unit_sphere` — count, sphere radius, pairwise distinct.
  - `test_fib_adjacency_edge_count_bounded_by_2_sum_fib_counts` — derives the directed→symmetric bound `sum(fib_counts) ≤ A_sym.sum() ≤ 2·sum(fib_counts)` and checks parity. Strong test of the construction's behaviour.
  - `test_platonic_fib_pointconv_forward_shape` — `(B, 12, in_dim) → (B, 12, out_dim)`, gradient flow.
  - `test_complete_graph_regression_with_uniform_max_fib_counts` — `fib_counts=(11,)*12` recovers `K_12` (the all-ones-minus-identity adjacency). Excellent boundary-case verification.
  - `test_oversized_fib_counts_rejected` — rejects len > N.
  - `test_layer_buffer_persistence_across_device_move` — registered buffers exist.
  - **Missing**: no equivariance test (because no equivariance is implemented), no test of the doc's 20-vertex dodeca claim (because the impl is 12 icosa). The doc Q&A's promised `test_avg_pool_not_max` and `test_icosa_60_orbit` are not testable on the current code.
- **Citation alignment**: Cohen 2019 icosa-CNN, Wang 2019 DGCNN, Qi 2017 PointNet, Caspar-Klug, Vogel 1979, Wu 2015 ModelNet40 — all relevant. None of these citations support a 12-vertex Fib k-NN GNN without rotation equivariance; the implementation does not deliver what they support.
- **Falsifier reachable?**: Falsifier is ModelNet40 accuracy + per-cloud latency. The current code provides a 12-vertex GNN layer; ModelNet40 typically uses 1024 sampled points (point clouds) — there is no mapping from a 1024-point cloud to the 12 icosa vertices in the code, nor any sampling / pooling glue. The doc-stated falsifier is unreachable from the current code; only a synthetic 12-node toy could be run.
- **Hidden bugs**: None functional. `torch.topk` on an `inf`-masked diagonal correctly excludes self. The `(A + A.t()) > 0` symmetrisation is a Python-level boolean comparison promoted back to float — works on CPU; would also work on CUDA.
- **Fix**: (a) Reconcile doc and code: either (i) update H30 doc to declare 12-icosa-vertex with Fib `(1,1,2,3,5)` AND drop the icosa-equivariance claim — i.e., demote to "Fib-degree k-NN over icosa vertices"; or (ii) implement the doc: 20 dodeca vertices (use H25's `dodecahedron_vertices()`!) with Fib `(1,1,2,3,5,8)` AND a real rotation-equivariant message pass using `icosahedral_rotations()` from H24. (b) Add a rotation-equivariance test once the equivariance is implemented. (c) Add a ModelNet40-bridge utility (sample 1024 points → assign to nearest icosa/dodeca vertex → aggregate).

---

## Group concerns (cross-cutting G3)

1. **The Cohen-2019 group-equivariance promise is unfulfilled in G3.** H24 has the math (60 rotations, det=+1, full closure) but does not apply rotations to the conv kernel (cyclic permutation of channels only). H30 inherits the promise but implements a plain GNN. The most damning consequence: T1.4's catastrophic -10.27 pp result on `sg_only_group` cannot be redeemed by the current code, because the current code does not implement the icosa-equivariant block the H24/H30 docs describe. The whole "H58 fix → H24/H30 graduation" prerequisite chain is therefore not yet executable in this repo.

2. **φ-extensions promised by the doc are routinely absent in the code.** H21 (φ-radial weighting), H22 (φ-scaled wrap), H23 (78-edge Metatron with multi-class φ weighting), H28 (per-tap φ phases) all describe a φ-rich mechanism in the formal hypothesis and ship a φ-agnostic or partial implementation. The single-prior CIFAR-10 sweep results documented in the H21/H22 §11 sections (T1.3 and T1.6 negatives) were honest in disclosing the partial nature — but the **upgrade** the docs propose has not landed yet. Any sweep row added with a `_phi_` suffix today would silently re-run the partial.

3. **Doc Q&A `test_…` claims serve as a check-list of missing tests.** Many hypotheses claim specific named tests in §9 that are not in the corresponding `tests/test_*.py`: H21 (`test_hex_phi_radial_factor`, `test_hex_60deg_eq`), H22 (`test_phi_scaling`, `test_translation_eq_on_torus`), H23 (`test_orbit_partition`, with the implied |E|=78), H24 (`test_avg_pool_not_max`, true equivariance), H25 (`test_icosa_symmetry`), H27 (`test_nn_distance_uniform`), H28 (per-tap φ phases), H29 (`test_clustering_coefficient`, `test_path_length`), H30 (`test_avg_pool_not_max`, `test_icosa_60_orbit`). Treating doc Q&A as the test specification would uncover the mechanism mismatches mechanically.

4. **`GroupConv2d` docstring vs `IcosaConv1d` choice has diverged.** `priors.py:GroupConv2d` was updated with an empirical finding ("max is BETTER than mean on CIFAR-10 by 4-6 pp; the avg-pool hypothesis is DISCARDED — the cure is data, not the reduction op"), but `icosa.py:IcosaConv1d` still uses `amax(dim=1)` while the H24 hypothesis doc still mandates avg-pool. The repo has two contradictory stories about pool reduction, and the choice silently affects every G3 hypothesis that composes with H24 (i.e., H30, and H50/H67 hybrids downstream). This needs a single repo-wide decision and a doc reconciliation pass.

5. **Mathematical math-rigor inside G3 is uneven.** H24's group closure (verified by 3600-pair `R_i @ R_j ∈ G` test), H25's dodeca √3-sphere proof, H30's 12-icosa cyclic permutation, H27's Vogel-formula coordinate-by-coordinate test — these are excellent. But the *use* of those mathematical objects in the surrounding modules (`IcosaConv1d.forward`, `PlatonicFibPointConv.forward`) does not deliver the geometric prior the math was constructed for. The math is real; the application is partial.

---

## Follow-ups (ranked by audit-severity)

1. **(MAJOR, H21+H22)** Implement the doc-promised φ extensions in `priors.py`: `HexConv2d.phi_radial` and `toroidal_pad(..., phi_scale)`. Add tests `test_hex_phi_radial_factor`, `test_phi_scaling`. Re-run T1.3 and T1.6 as `sg_only_hex_phi` and `sg_only_toroidal_phi` rows. Effort: half a day.

2. **(MAJOR, H24+H30)** Decide pool reduction repo-wide. If max-pool stands (per GroupConv2d empirical), update H24/H30 docs to drop the "avg-pool per H58 lesson" mandate and instead claim "max-pool with data-aligned target (Spherical MNIST / IcoMNIST / rotated)". If avg-pool stands (per H24 doc), fix `IcosaConv1d.forward`. Either way, add `test_equivariance_under_60_rotation` to verify the rotations are actually applied.

3. **(MAJOR, H23)** Decide Metatron scope: full 78-edge connectivity (with a 3-class φ partition: center↔inner, inner-ring + radial, outer-ring + Platonic-long-lines) OR honest renaming to `MetatronLatticeGraph` (24 edges). Either way, update the doc's |E|=78 line.

4. **(MAJOR, H28)** Add a per-tap modulation factor `cos(ω·t + k·φ)` with k indexing the 6 peripheral hex taps; keep centre tap unmodulated; either drop the spatiotemporal/Conv3d claim from H28 or add a `CymaticHexConv3d`.

5. **(MAJOR, H30)** Choose between dodeca/20-vertex Fib `{1,1,2,3,5,8}` or icosa/12-vertex Fib `(1,1,2,3,5)`. Update doc to match impl, or extend impl to 20-vertex dodeca + reuse H25's `dodecahedron_vertices()`. Add a real rotation-equivariant message pass once the polyhedron is fixed.

6. **(MINOR, H25)** Add `test_icosa_symmetry` linking H25 to H24's `icosahedral_rotations()` to lock in the icosa-invariance of the dodeca vertex set.

7. **(MINOR, H27)** Add `test_nn_distance_uniform`: assert NN-distance variance of golden-spiral lift is ≤ 0.2 × Xavier baseline at matched N, D. Optionally also commit to whether the lift is the orthonormal-projection variant (current code) or the `[spiral_2d | xavier_(d-2)]` variant (current doc).

8. **(MINOR, H29)** Add `test_clustering_coefficient` and `test_path_length` — both one-pass on the boolean adjacency. Without these, the falsifier (statistical distinct path-length distribution at p=1/φ vs p=0.5) is not verifiable from the unit test layer.

---

*Generated 2026-05-27 by skeptical-PC G3 audit. Tier counts: PASS=2, MINOR=2, MAJOR=6, BROKEN=0.*
