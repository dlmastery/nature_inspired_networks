# G8 Audit — Esoteric Extensions (H76–H84)

**Auditor role:** NeurIPS PC member / skeptical reviewer.
**Scope:** 9 hypotheses (H76–H84), neutral recasts of esoteric ideas.
**Method:** Read design doc + `src/.../<module>.py` + `tests/test_<module>.py`. Audit whether the neutral recast actually matches the cited technique, the math is correct, and tests verify mechanism (not just shape).
**Tiers:** PASS | MINOR | MAJOR | BROKEN.

---

## Tier counts (9 hypotheses)

| Tier | Count | IDs |
|---|---|---|
| **PASS** | 7 | H76, H77, H78, H79, H80, H82, H83 |
| **MINOR** | 2 | H81, H84 |
| **MAJOR** | 0 | — |
| **BROKEN** | 0 | — |

No contamination by mystical framing was detected: every implementation reduces to a real, citable mechanism (group conv, relative Fourier bias, S¹×S¹ embedding, GCN message passing, masked conv, SIREN, fixed sparse attention, learnable-τ softmax, modern Hopfield).

---

## Per-hypothesis findings

### H76 — TetrahedralDualPathBlock — PASS
- **Mechanism:** Two `GroupConv2d` paths (`reduce='max'` and `reduce='mean'`) fused by `β·A + (1−β)·B` with `β = sigmoid(β_raw)`, initialised to `logit(beta_init)` so `β(0) == beta_init`. Residual gated on shape match. This is the prescribed "Merkaba" recast: two complementary C4 orbit reductions and a learnable convex merge.
- **Math:** β confined to (0,1) without clamping; β=0 → mean-only, β=1 → max-only.
- **Tests** (`test_tetra_dualpath.py`, 7 tests): explicitly verifies `β→1` matches `path_max`, `β→0` matches `path_mean`, max-path ≠ mean-path even with weight-tied paths (so the *reduction* genuinely differs), gradient reaches `β_raw`, residual skip is disabled on channel/stride mismatch, and `beta_init=0.7` is honoured. Mechanism-verifying, not shape-only.
- **Verdict:** No defects. Strong recast.

### H77 — RadialSymmetry12Attention — PASS
- **Mechanism:** Buffer `head_angles[h] = 2π·(h mod 12)/12`; relative bias `bias[h,i,j] = cos(angles[h] + 2π·(j−i)/L) / PHI`, added to logits before softmax. `n_heads` enforced as multiple of 12; QKV/proj are standard.
- **Math:** Bias depends on `delta = j − i`, **not** a constant per-head (which would be softmax-invariant). The cited Shaw et al. 2018 (`arXiv:1803.02155`) result is the justification, correctly applied.
- **Tests** (`test_radial12_attention.py`, 5): verifies `head_angles` is a buffer (not a parameter), `radial=False` exactly equals plain MHA reconstructed from the module's own projections, `radial=True` ≠ `radial=False` (so the bias bites), `n_heads` validator rejects 1/5/13/10, and cyclic shift `(roll(N,N),(1,2))` leaves the bias invariant — i.e. position-periodic. Mechanism-verifying.
- **Minor cosmetic:** division by `PHI` (≈1.618) is a dimensionless rescale of the bias magnitude; not load-bearing. Documented in the docstring.
- **Verdict:** No defects.

### H78 — ToroidalLatent — PASS
- **Mechanism:** `Linear(in_dim → 2)` produces `(θ₁,θ₂)`; embedded as `(cos θ₁, sin θ₁, cos θ₂, sin θ₂) ∈ T² ⊂ ℝ⁴`; `Linear(4 → out_dim)` returns to feature space. `toroidal_distance` wraps each angle into [−π,π] via `remainder(diff+π, 2π) − π` and L2-combines.
- **Math:** Each (cos,sin) pair has unit norm by construction; angle wrap is min(|d|, 2π−|d|) exactly.
- **Tests** (`test_toroidal_latent.py`, 5): explicit unit-norm check on both pairs (`pair1.norm()` ≈ 1), wrap-around distance check `dist(0.1, 2π−0.1) ≈ 0.2√2` (and NOT ≈ 2π√2), shape, determinism, and bad-shape rejection. Mechanism-verifying.
- **Verdict:** No defects.

### H79 — MorphingPolytopeAdjacency — PASS
- **Mechanism:** Cuboctahedron vertices (permutations of (±1,±1,0)); icosahedron vertices (cyclic permutations of (0,±1,±φ)) — both at 12 nodes, both deterministic. Edges = minimum-distance pairs (with 1e-4 tol). `A(t) = (1−t)·A_cubocta + t·A_icosa`, sigmoid-gated learnable `t = sigmoid(t_raw)`. Message passing is symmetric-normalised GCN (Kipf-Welling) with skip term.
- **Math:** Tests confirm A_cubocta has 24 undirected edges (matrix sum 48 with symmetric self-pair counting) and A_icosa has 30 edges (sum 60). At `t=0` returns exactly `A_cubocta`; at `t=1` exactly `A_icosa`. Both adjacencies symmetric with zero diagonal. The degree-normalisation in `_morphed_adj` recomputes `D` from the *fractional* `A(t)`, which is the correct per-step normalisation.
- **Tests** (`test_morphing_adjacency.py`, 5): edge counts, symmetry, zero diagonal, endpoint match, midpoint = avg, learnable-t gradient. Mechanism-verifying.
- **Verdict:** No defects. The Fuller "jitterbug" framing maps cleanly to a convex interpolation of two GCN adjacencies.

### H80 — ConstantWidthConv2d (Reuleaux) — PASS
- **Mechanism:** `reuleaux_mask(k)` = intersection of 3 disks at equilateral-triangle vertices, each disk radius = triangle side length `s = R·√3` with `R = (k−1)/2`. Soft mode = product of sigmoids of (s − dist) per disk; hard mode = boolean AND. Mask is normalised so peak = 1.0 and applied at every forward as `weight * mask` (so masked taps get zero gradient — gradient does flow to all non-masked taps).
- **Math:** A Reuleaux triangle is *defined* as the intersection of three disks centred at the equilateral-triangle vertices with disk radius equal to the side length. Implementation matches the definition exactly. Centre value = 1 (inside all three disks). Corners < 1 (suppressed). Disk radius `= s` (the side), not `s/2` or `R`, is correctly chosen.
- **Tests** (`test_constant_width_kernel.py`, 6): corner values < 1e-2, centre = 1.0, coverage strictly between 5 (tiny-disk floor) and k² (square), corner gradient ≪ centre gradient (mask is in the forward, not init), and param count equals plain `nn.Conv2d` (mask is a buffer). Mechanism-verifying.
- **Verdict:** No defects.

### H81 — SinusoidalActivation (SIREN) — MINOR
- **Mechanism:** `sin(ω · x)` with a learnable scalar `ω` (or per-channel vector). `swap_relu_with_sine` walks a model and replaces every `nn.ReLU` with a fresh `SinusoidalActivation`. Cites Sitzmann et al. 2020 NeurIPS SIREN (`arXiv:2006.09661`) correctly.
- **Math:** Element-wise `sin`; shape preserved; per-channel `ω` broadcasts along configurable `dim`. Tests verify `sin(0)=0`, periodicity `act(x) == act(x + 2π/ω)`, learnable parameter receives gradient, swap helper correctly replaces nested ReLUs, channel-0 with `ω=0` yields zero output, `learnable=False` stores a buffer.
- **Concern (MINOR, the requested SIREN canonical):** The default `omega_init=1.0` is **not** the canonical SIREN first-layer recipe of `ω₀=30`. The module docstring acknowledges this explicitly ("SIREN's canonical first-layer value is `30.0`") and the swap-helper signature exposes `omega_init`, so the canonical recipe is reachable via `swap_relu_with_sine(model, omega_init=30.0)`. The TODO runner-wiring comment also flags "a second sweep row flips that one number" for the ω₀=30 stem variant. Because the default deviates from the canonical SIREN recipe used in the literature, an out-of-the-box smoke run is **not** a faithful SIREN reproduction; only the documented `omega_init=30.0` variant is. This is a documentation/discoverability nit rather than a correctness bug.
- **Verdict:** MINOR — code is correct, default is not the canonical SIREN ω₀=30 but is labelled in the docstring. Recommend the CIFAR sweep row use `omega_init=30.0` (with sub-rows at 1.0 for ablation).

### H82 — VoronoiSparseAttention — PASS
- **Mechanism:** `voronoi_adjacency(N, seed)` samples N deterministic points in the unit square, builds true Delaunay adjacency via `scipy.spatial.Delaunay` when available, otherwise falls back to symmetric kNN with default `knn=6` (≈ mean 2-D Delaunay degree). Self-loops added; matrix symmetrised. The mask is registered as a buffer; attention logits at masked positions are set to `finfo.min` before softmax.
- **Math:** Mask is binary {0,1}, symmetric, self-loops present, sparse (`<0.2·N²` at `N=64` by test). Seed-deterministic (same seed → identical mask; different seed → different mask).
- **Tests** (`test_voronoi_attention.py`, 5): explicit symmetry check, self-loop check, sparsity check (`edges < 0.2·N²`), minimum-1-neighbour invariant, seed determinism positive and negative, mismatched-N rejection, masked forward shape preserved and finite, kNN fallback path for `N=3 < 4`.
- **Verdict:** No defects. Note: when SciPy raises an exception the fallback silently uses kNN — this is a documented design choice (the import is guarded). Could log a warning, but not a correctness bug.

### H83 — CollapseGatedAttention — PASS
- **Mechanism:** Standard MHA with `logits = (Q·Kᵀ) / (√d_head · τ)`, where `τ = softplus(τ_raw) + 1e-3`. `τ_raw` initialised so `softplus(τ_raw) = tau_init`. Optional post-softmax `collapse ∈ [0,1]` interpolates toward one-hot argmax: `attn ← (1−c)·attn + c·onehot(argmax(attn))`. Cites Martins 2016 (sparsemax) and Peters 2019 (entmax) as the temperature/sparsity precedents.
- **Math:**
  - `softplus(z) = log(1+e^z) > 0` strictly, so τ > 0 always.
  - The **1e-3 floor** on τ is in place at lines 86 and 126 (`F.softplus(self.tau_raw) + self._TAU_FLOOR`). This is the requested floor (added after a NaN was caught when τ drifted to ~0). Verified present in both the `tau` property and the forward path — i.e. no code path can produce τ=0.
  - Low τ produces sharper softmax (because the scale factor `1/(√d · τ)` grows, amplifying logit differences); high τ produces diffuse softmax. The test `test_low_tau_sharper_than_high_tau` checks this empirically: at τ=0.2 vs τ=5.0, the mean row-max is strictly higher for low τ.
- **Tests** (`test_collapse_attention.py`, 7): shape; low-τ vs high-τ sharpness; positivity even at `τ_raw=−100` (and forward stays finite); `τ_raw.grad` non-zero; row sums = 1; `collapse=1` yields row-max=1 with rows summing to 1; additive causal mask blocks future positions. Mechanism-verifying, including the floor's NaN-prevention role.
- **Verdict:** No defects. The 1e-3 floor is present as required.

### H84 — SpectralHopfieldMemory — MINOR
- **Mechanism:** Modern continuous Hopfield (Ramsauer et al. 2020, `arXiv:2008.02217`). `store(patterns)` saves a real bank `(M, dim)` as a buffer. `retrieve(query)` computes:
  1. `s_patt = [Re(rfft(patterns)) ‖ Im(rfft(patterns))]` — real spectral features `(M, F)` with `F = 2·(dim//2 + 1)`.
  2. `s_query = [Re(rfft(query)) ‖ Im(rfft(query))]`.
  3. `scores = β · s_query @ s_pattᵀ`.
  4. `weights = softmax(scores, dim=-1)`.
  5. `out = weights @ patterns` — readout in the **signal domain**.
- **Math:**
  - `irfft(rfft(x))` is lossless for real inputs (`test_fft_roundtrip_lossless` confirms `‖rt − x‖∞ < 1e-5`).
  - Associative recall (`test_associative_recall_returns_stored_pattern`) confirms a near-target query at β=20 retrieves the closest stored pattern (argmin of squared distance to the bank). High-β vs low-β sharpness directly verified via softmax mass on the target.
- **Concern (MINOR — spectral metric is *not* the same as signal-domain dot product):** Because `rfft` for real signals duplicates the non-DC/non-Nyquist energy in conjugate pairs and the implementation stacks `[Re ‖ Im]` (so the Nyquist bin's zero-imag contributes 0 and the DC bin's zero-imag contributes 0), the spectral inner product:
  `<s_query, s_patt> = Re(<RFFT(q), RFFT(p)>) = Re(Σ_k Q*_k · P_k) over k in {0,…,dim//2}`,
  which by the Hermitian symmetry of the full DFT of a real signal equals `(1/2)·<full-DFT(q), full-DFT(p)> + (DC and Nyquist correction)`. Up to Parseval scaling this is *proportional* to signal-domain dot product, but the per-bin weighting is non-uniform (DC and Nyquist count once; other bins count once but the rfft already drops half the bins, which would normally be doubled to recover Parseval — the implementation does NOT apply that ×2 correction). So the "spectral basis" is **not an isometry** of the signal basis as the docstring claims; it is an **alternative inner product** that downweights DC/Nyquist relative to mid-frequencies. This still gives a valid Hopfield similarity (and the associative-recall test passes), but the docstring's "exact linear reparameterisation of the energy landscape" / "isometry of the complex inner product up to the usual rfft scaling" is imprecise — the scaling is *not* uniform across bins.
  In practice, for randomly-drawn patterns the relative ordering of pairwise similarities is preserved with very high probability (which is why the retrieval test passes), but for adversarially-crafted patterns with heavy DC or Nyquist content the spectral basis could in principle rank differently than a raw dot product.
- **Tests** (`test_spectral_hopfield.py`, 5): associative recall, output shape (batched and 1-D), high-β vs low-β softmax sharpness on the matching pattern (explicit `w_lo < 0.9 < w_hi`), FFT round-trip lossless, retrieve-before-store error.
- **Verdict:** MINOR — the implementation is correct as a Hopfield variant on a real-spectral feature space, the associative-recall mechanism works as advertised at high β, but the docstring's framing ("isometry of the complex inner product") oversells the relationship: it is a non-uniformly-weighted similarity, not a Parseval isometry. The functional claim (retrieval finds the matching pattern at high β) is verified by test. Suggested doc tweak: state that spectral matching is an "alternative similarity metric in a real-spectral feature space" rather than "an exact linear reparameterisation".

---

## Three most damning findings (rank-ordered)

1. **H84 SpectralHopfield — "isometry" claim overstated.** The docstring asserts that stacking `[Re(rfft) ‖ Im(rfft)]` gives "an isometry of the complex inner product up to the usual rfft scaling" and "an exact linear reparameterisation of [the] energy landscape." It is not: `rfft` drops the conjugate-symmetric upper half of the bins, so the spectral inner product downweights non-DC/non-Nyquist energy by ×2 relative to a Parseval-correct mapping. Retrieval still works (test confirms argmin-on-target at β=20), but the "energy landscape is preserved" framing is imprecise and could mislead a reviewer.

2. **H81 SinusoidalActivation — default `omega_init=1.0` is not the canonical SIREN recipe.** Sitzmann et al. 2020 use `ω₀=30` for the first layer (and `ω=1` elsewhere is itself an active choice, with the dense layers using a specific uniform initialisation). The default of `1.0` is closer to a "small-signal linear-approx" regime than to the published SIREN spectral-bias-killing regime. The docstring and the TODO runner comment do flag this, but an out-of-the-box CIFAR sweep row would NOT reproduce SIREN faithfully. Mitigation is one config flag, so this is a documentation/discoverability nit, not a code bug.

3. **H82 VoronoiSparseAttention — SciPy fallback is silent.** When `scipy.spatial.Delaunay` is unavailable or raises (e.g. coplanar / degenerate point set, although random points in the unit square should avoid this), the module silently falls back to a symmetric `knn=6` adjacency without warning. The kNN adjacency is a reasonable Delaunay proxy (mean 2-D Delaunay degree ≈ 6), but a downstream run cannot distinguish between true Delaunay and the proxy. A `warnings.warn(...)` on the exception path would make the provenance explicit; not a correctness bug but a reproducibility annoyance.

---

## Summary

All 9 G8 hypotheses are faithful neutral recasts of real, citable techniques. No mystical framing leaked into the math. The two MINOR findings (H81 default, H84 docstring) are documentation/discoverability issues, not correctness bugs. The H83 `1e-3` τ-floor requested by the user is present and verified. Tests across G8 are mechanism-verifying (not shape-only) — they explicitly check the load-bearing invariants (β endpoints, T² unit-norm, polytope edge counts, Reuleaux corner suppression, learnable-τ sharpness, FFT round-trip).
