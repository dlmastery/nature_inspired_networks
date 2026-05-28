# G4 Audit — Kernels / Attention / Filters (H31 – H40)

> Reviewer: NeurIPS-PC adversarial pass.
> Doctrine: doc claim → src mechanism → test rigor → math → citation → hidden bugs.
> Shape-only tests are MINOR; mechanism-verifying tests are required for PASS.

---

## Summary

| Verdict | Count | Hypotheses |
|---|---|---|
| PASS    | 5 | H32, H34, H36, H39, H40 |
| MINOR   | 4 | H33, H35, H37, H38 |
| MAJOR   | 1 | H31 |
| BROKEN  | 0 | — |

Headline: G4 is the strongest group implementation-wise. Attention/RoPE math is correct, tests are mostly mechanism-verifying, and the pentagonal bias is — surprisingly — actually relative-position (NOT the constant-additive trap I expected). The notable defect is **H31**, where the implementation silently substitutes a generic exponential for the spec's *golden-ratio* growth, and the angular schedule is uniform rather than golden-angle — both core to the hypothesis. The H35 banded-basis has a latent bug when `n_modes > k*k` (duplicate-vector cycle defeats orthonormality), and the H38 even-kernel cropping introduces a subpixel spatial shift. None block training; one (H31) silently mis-tests its own claim.

---

## H31 — Golden-Spiral Kernel Init — VERDICT: **MAJOR**

- **Module:** `src/nature_inspired_networks/inits.py` (`golden_spiral_mask`, `golden_spiral_init_`, `apply_golden_spiral_init`).
- **Mechanism vs claim:** The design doc and module docstring both promise `r(θ) = r0 · φ^(θ/(π/2))` (φ-growth per quarter turn) sampled at the **golden-angle step** `Δθ = 2π(1 − 1/φ) ≈ 137.508°` (`inits.py:8-10`, doc §5.1). The implementation does **neither**:
  - Growth: `b = log(r_max/r0)/theta_max; r = r0·exp(b·θ)` (lines 79, 83). This calibrates `b` to fill the kernel grid — `b` is a function of `k`, not of φ. For `k=5, n_turns=2, r0=0.3, r_max=2`: `b ≈ 0.151`, and `r(π/2) / r(0) = exp(b·π/2) ≈ 1.27`, not φ=1.618. So the "φ growth per quarter turn" claim is silently dropped.
  - Angular schedule: `theta = i · theta_max / (n_samples - 1)` (line 82) — **uniform** spacing across `[0, 2π·n_turns]`, not the golden-angle increments documented at line 41 ("traced at golden-angle increments `Δθ = 2π·(1 − 1/φ)`"). Phyllotaxis is what makes the spiral self-similar; uniform sampling kills the prior.
- **Math:** The post-init variance rescale `mask /= sqrt(mean(m²))` does correctly target `mean(m²)=1`, so He-variance is preserved on average — that part is mathematically fine. But the spiral itself is φ-free.
- **Test rigor:** Six tests, all shape / non-negativity / non-degeneracy / variance / determinism / 4-D guard. **No test asserts φ-growth radii**, no test asserts golden-angle phyllotactic step, no test compares against a known φ-spiral reference. The doc's own Committee Q&A (§9, "How do we know the implementation is correct?") promises a `test_phi_growth` asserting "successive points are at φ-scaled radii" — that test does not exist.
- **Citation:** Olshausen 1996, He 2015 cited with full form. Fine.
- **Falsifier:** Reachable in principle (top-1 ≥ +0.3 pp AND 3-ep loss ≤ −5 %), but useless: the experiment cannot falsify the φ-spiral claim because the implementation does not encode it.
- **Hidden bug:** Beyond the math mismatch, `(o + i) % 2` style sign-flipping is irrelevant here, but the inner loop `if v > mask[y, x].item()` calls `.item()` per sample inside a Python loop — N×k² CPU syncs. Cold-start cost only (init runs once), so cosmetic.
- **Fix:**
  1. Replace `b = math.log(r_max / r0) / theta_max` with the φ-anchored constant `b = math.log(PHI) / (math.pi / 2)` (so `r(θ + π/2) = φ · r(θ)`). Adjust `r0` so the spiral still terminates near `r_max` after `n_turns`.
  2. Replace uniform angular stepping with `theta = i * GOLDEN_ANGLE` (or `i · GOLDEN_ANGLE`-stepped sampling with rejection outside the disc) so the angular schedule is phyllotactic.
  3. Add a test `test_golden_spiral_radii_grow_by_phi_per_quarter_turn` and `test_golden_spiral_angle_step_is_golden_angle`.
- **Group concern:** None of the H31 conv smoke rows (`sg_only_golden_spiral_init`) can be interpreted as evidence for/against the *φ-spiral* hypothesis; they only measure a kernel-grid-shape-aware multiplicative mask on top of He. The published verdict (if any) needs an asterisk.

---

## H32 — Fibottention (Fibonacci-Dilated Attention) — VERDICT: **PASS**

- **Module:** `fibottention.py` (`fibonacci_dilations`, `wythoff_pattern`, `Fibottention`).
- **Mechanism vs claim:** `(j − i) % d == 0` mask, additive `-inf` on disallowed entries, softmax-normalised over kept positions, per-head Fibonacci dilation `[1, 2, 3, 5, 8, 13, …]`. Matches the doc and Rajagopalan 2024 description.
- **Math:** Additive `-inf` (not multiplicative) preserves softmax normalisation across kept keys. `head_dim` scaling `√(D/H)` is correct. Self-position is naturally included since `0 % d == 0`.
- **Test rigor (strong):** Asserts (a) canonical sequence `[1,2,3,5,8,13,…]`; (b) Fibonacci recurrence holds; (c) shape & dtype; (d) self-position attended; (e) **density ≈ 1/d** within ±20 %; (f) **coprime-dilation off-diagonal intersection density ≈ 1/(d₁·d₂)** — the Wythoff non-overlap claim has a numerical test (gcd-Fibonacci pairs); (g) dense fallback `[1]*H` numerically matches a hand-rolled MHA built from the same weights (`atol=1e-6`); (h) sparse forward differs from dense fallback (so the mask is actually applied); (i) cache reuse via `data_ptr()` check.
- **Citation:** Rajagopalan 2024 (arXiv:2406.19391), Wythoff 1907 in docstring. Adequate.
- **Falsifier:** Reachable through composite delta on CIFAR/LLM benchmarks.
- **Hidden bugs:** None observed. The cache-reuse test is among the more rigorous in the repo.

---

## H33 — Vesica Piscis Filter — VERDICT: **MINOR**

- **Module:** `vesica_piscis.py` (`vesica_kernel_mask`, `VesicaPiscisConv2d`, `vesica_phi_offsets`).
- **Mechanism vs claim:** Three binary discs *shifted horizontally* by `(i − (n−1)/2) · 0.5 · offset` pixels. Each path applies a masked Conv2d; outputs are summed with a learnable `scales` vector initialised at `1/n_circles`.
- **Math:** Disc pixel count ≈ π r² within ±4 pixels (tested). Centre-of-mass biasing left/centre/right (tested). Multiplicative `weight * mask` preserves grad on the unmasked positions.
- **Issue (MINOR, geometry):** The classical **Vesica Piscis** is the lens-shaped intersection of *two equal circles whose centres each lie on the other's circumference* — a 2-disc geometric primitive with a specific overlap area `2r²(π/3 − √3/4)`. The implementation is a **row of 3 horizontally-shifted discs** — a 1-D approximation of the *Flower of Life* tiling. None of the tests check that adjacent discs satisfy the actual vesica condition `|c₁ − c₂| = r` (and the default `offset=1.0, radius=2.0` gives centre spacing `0.5`, not `radius=2`). The mechanism *is* multi-scale overlapping reception, which matches the broader claim, but the name overpromises the geometry.
- **Test rigor:** Shape, binary, disc-area-≈-π·r², COM shift, adjacent-disc-overlap (`m[0]·m[1] ≥ 1`), `n_circles ∈ {1, 2, 3, 5}` works, **trainable scales verified end-to-end** (`scales.grad.abs().sum() > 0`, manual sum-of-paths matches forward), `vesica_phi_offsets` φ-decay verified. Solid.
- **Citation:** Szegedy 2015, Olshausen 1996, Chen 2019 OctConv — all cited in full form.
- **Hidden bug:** Default `padding = kernel_size // 2 = 2` for `kernel_size=5`. Stride=1 — output shape preserved (tested). Bias defaults `False`.
- **Fix (cosmetic):** Either rename to `OverlappingDiscFilter` or modify `vesica_kernel_mask` so adjacent discs are spaced exactly at `radius` pixels (the vesica-piscis canonical layout). Add a test asserting the canonical vesica overlap pixel count for n_circles=2.

---

## H34 — Golden-Angle Rotary (RoPE-φ) — VERDICT: **PASS**

- **Module:** `golden_rope.py` (`GOLDEN_ANGLE`, `golden_angle_rope_freqs`, `apply_golden_rope`, `_rotate_half`).
- **Mechanism vs claim:** `freq_k = base^(−2k/dim)` with `base=φ` default; per-pair phase offset `k · GOLDEN_ANGLE mod 2π`; rotation `q_rot = q·cos + rotate_half(q)·sin`. Matches the doc.
- **Math:** `_rotate_half(x)[0::2] = −x[1::2]; [1::2] = x[0::2]` gives the correct 2-D rotation per pair: `q'_{2i} = q_{2i}·cosθ − q_{2i+1}·sinθ`, `q'_{2i+1} = q_{2i+1}·cosθ + q_{2i}·sinθ`. Norm-preserving. The constant phase offset cancels in differences `angle(p₂) − angle(p₁) = (p₂ − p₁)·freq_k`, so relative-position equivariance is exact (verified by test).
- **Test rigor (strong):** Phase progression `k·GA mod 2π` per pair (numerical); φ-base geometric decay; odd-dim guard; shape preserved on `(B,H,N,D)`; **per-pair norm preservation** (the load-bearing rotation-is-orthogonal test); **differ-from-base-10000-RoPE** numerically; **relative-equivariance under position shift** (q·k dot at (0,5) == dot at (7,12)). Two shape-validation guards. Mechanism-verifying.
- **Citation:** Su 2021 (arXiv:2104.09864) in module header. Adequate.
- **Hidden bug:** `freqs.to(device=q.device, dtype=q.dtype)` casts to `q.dtype` — under bf16 AMP the cos/sin precision drops, but that is also true of stock RoPE and is acceptable.

---

## H35 — Cymatic / Chladni Init — VERDICT: **MINOR**

- **Module:** `priors.py` (`chladni_modes`, `chladni_modes_banded`, `cymatic_init_`).
- **Mechanism vs claim:** `chladni_modes_banded(n_modes, k, band, seed)` samples `n_modes` `(m,n)` pairs uniformly from `[band[0], band[1]]²`, builds `sin(m·X)·sin(n·Y)` modes, then **QR-orthonormalises** across modes (line 142). With `orthonormalize=True, band=(2, 5)`, `cymatic_init_` writes signed copies of these orthonormal modes per `(out_c, in_c)`, He-variance-rescaled.
- **Math:** QR on `(k², n_modes)` with `mode="reduced"` returns up to `min(n_modes, k²)` orthonormal columns. Per-input filters across output channels are signed copies of distinct orthonormal modes, so the cross-channel Gram has off-diagonal magnitude bounded by `≈ 0` (clean) or ≤ |off-diag of basis| (numerical). The test checks `< 0.5` — generous tolerance but adequate.
- **Issue (MINOR, latent bug):** When `n_modes > k²`, `chladni_modes_banded` cycles literal copies: `out = out.repeat(reps, 1, 1)[:n_modes]` (line 150). The duplicated rows are NOT mutually orthogonal — they are **identical**. The test (`test_h35v2_chladni_modes_banded_orthonormal`) uses `out_c=4, k=5` (16 cells, well under k²) so the cycle path is never exercised. In practice `out_c` for early conv layers can be 32, 64 with k=3 → `k²=9 < 32 < 64`, so this path **does fire** during real training and silently violates the orthonormality contract. Discovery requires reading line 146-150.
- **Test rigor:** Orthonormality (`< 1e-4` off-diag — strong), per-input decorrelation across out channels (`< 0.5` off-diag — loose), legacy bit-identical reproducibility via `Generator(0xC1A171C)`, differs-from-legacy under new flags. Mechanism-verifying for small `out_c`, blind for the cycle path.
- **Citation:** Chladni (implicit). The module text references "2-D wave equation eigenmodes" but no formal `Chladni 1787` or modern equivalent (e.g., Tuan & Chen 2023 on Chladni-pattern eigenfunctions). MINOR citation gap.
- **Falsifier:** Reachable via composite.
- **Fix:** Either (a) raise an error when `n_modes > k²` ("Chladni basis exhausted; reduce out_c or increase k"); or (b) sign-flip the duplicate copies deterministically so at least the duplicates are anti-symmetric. Add a `test_banded_cycles_raise_or_signflip` to lock it.

---

## H36 — Golden-Spiral PE — VERDICT: **PASS**

- **Module:** `spiral_pe.py` (`golden_spiral_pe`, `GoldenSpiralPE`).
- **Mechanism vs claim:** Trajectory `(cos(k·GA), sin(k·GA), k/N)` with `GA = 2π(1 − 1/φ)`, then a `(3, D)` orthonormal lift (fixed or learnable). Matches the doc.
- **Math:** Unit circle for the first two coords (`cos² + sin² = 1` — tested); monotone z (tested); golden-angle phase increment (tested numerically); QR-based fixed projection is deterministic.
- **Test rigor (strong):** Shape (default + learnable raw 3-vec); determinism across two calls; **consecutive positions distinct**; unit-circle invariant; **golden-angle progression** `(ang[1:] − ang[:-1]) mod 2π ≈ GA` (numerical); dim-≥-3 guard; module forward `y == x + PE`; learnable Linear receives grad; non-learnable path has no `nn.Linear`; capacity guard. Mechanism-verifying.
- **Citation:** Phyllotaxis (implicit). Doc has full citations.

---

## H37 — Pentagonal φ-Attention — VERDICT: **MINOR**

- **Module:** `pentagonal_attention.py` (`pentagonal_head_groups`, `PentagonalAttention`).
- **Mechanism vs claim:** Per-head additive bias `(1/φ) · cos(angle_h + 2π·(j − i)/L)`, **relative-position** in `(j − i)`, where `angle_h = 2π·(h % 5)/5` is the head's dodeca-vertex phase. Bias is a buffer (non-learned). Matches the doc, and — encouragingly — the implementation **avoided** the constant-additive trap (a position-independent additive bias would be softmax-invariant). The `_relative_bias` builds `cos(angles + 2π·rel/L)` where `rel = j − i`.
- **Math:** Correct relative-position dependence. The `1/φ` scaling sets the bias magnitude (could otherwise dominate the QK logits at small `√head_dim`).
- **Issue (MINOR, equivariance):** The cyclic-shift equivariance test (`test_h37_rotational_symmetry_cyclic_shift`) holds *only because* `rel = j − i` is invariant under `(i, j) → (i + Δ, j + Δ)` modulo `L`. The implementation uses `rel = idx.view(1,L) − idx.view(L,1)` with `idx = arange(L)` — this is the *unshifted* relative position, not a circular relative. Under a sequence shift the linear projections shift the tokens, but the bias matrix is *static* (indexed by absolute `i, j`). So `attn(x_shifted) == roll(attn(x))` would require the bias map to also be shift-equivariant. Because `bias = cos(angle_h + 2π·(j − i)/L)` depends only on `(j − i) mod L` AND because the test rolls both q and k uniformly via `torch.roll`, equivariance does hold — but only on cyclic shifts that match the implicit `L` period. **Sub-`L` shifts break it.** Test only checks `shift = L // 5 = 2` on `L = 10`. Adversarial robustness narrow.
- **Test rigor:** Group partitioning, n_heads-multiple-of-5 guard, forward shape, buffer-not-Parameter check, period-5 across heads, **cyclic-shift equivariance** at `shift = L/5`, **bias-zeroed changes output** (proves bias is wired into softmax inputs). Mechanism-verifying.
- **Citation:** Cohen 2019 ICML 'Icosahedral CNN', Dosovitskiy 2021 ViT, Vaswani 2017 — full form.
- **Hidden bug:** None functional. The 5-fold "dodeca-vertex" framing is a stretch: only 5 distinct head angles emerge, and `cos(0 + θ) = cos(θ)` so heads in group 0 see the standard cosine bias, while groups {1,4} and {2,3} produce conjugate-pair biases (the 5-fold symmetry collapses to ~3 distinct cosine profiles via even symmetry). Worth noting in design doc.
- **Fix:** Add a test for non-`L/5` shift to confirm general translation equivariance, OR document the cyclic-only equivariance. Optionally distinguish cos/sin per group to recover full 5-fold expressivity.

---

## H38 — Fractal Golden Filter (3+5+8 paths) — VERDICT: **MINOR**

- **Module:** `fractal_filter.py` (`FIB_KERNELS`, `FractalGoldenFilter`).
- **Mechanism vs claim:** Three Fibonacci-sized paths (3, 5, 8). Each path = `Conv2d(in→mid, k)` then `Conv2d(mid→out, 1)`. Per-path learnable scale `α` initialised to `[1, 1/φ, 1/φ²]` normalised to sum 1. Sum the three paths.
- **Math:** Path additive aggregation `Σ α_i · proj_i(conv_i(x))`. Phi-decay verified.
- **Issue (MINOR, even-kernel crop introduces half-pixel shift):** For `k=8` (even), `pad = k // 2 = 4`. `nn.Conv2d` with input `H` outputs `H + 2·4 − 8 + 1 = H + 1`. The implementation crops `y[..., :H, :W]` (line 129), removing the **trailing** row/col. This means the effective receptive field of the k=8 path is centred at *(input position + 0.5)* relative to the k=3 and k=5 paths. So adding the three paths is summing kernels with a **subpixel-shifted receptive field** on the k=8 branch. Real consequence: the k=8 path's contribution to output pixel (i, j) responds to input pixels (i−4..i+3), not (i−3.5..i+3.5). This is a known pitfall of even kernels and is not flagged in any test.
- **Test rigor:** Shape preservation including odd input sizes (7, 15, 17, 32); all 3 paths active (zero-α perturbation test); single-path filter works; **alpha is φ-decay normalised** (tested numerically); exact parameter-count match (`Σ in·mid·k² + mid·out + 3` α's); invalid-kernel-size guards; **gradient flows through every path's conv AND alpha**. Mechanism-verifying for the additive aggregation.
- **Citation:** Larsson FractalNet 2017 (arXiv:1605.07648), Mandelbrot 1982, Szegedy 2015, Chen OctConv 2019 — full form.
- **Falsifier:** Reachable.
- **Hidden bug:** The even-kernel crop is functional (no NaN, no shape break) but introduces an unstated half-pixel shift between the three paths. For a "fractal multi-scale" claim where alignment matters, this is a real but subtle defect.
- **Fix:** Either replace `(3, 5, 8)` with `(3, 5, 7)` (odd-only) — same Fibonacci-ish spacing without parity asymmetry — or apply asymmetric padding for k=8 (`F.pad(x, [3, 4, 3, 4])` before `Conv2d(..., padding=0)`) so the effective centre lines up at the input pixel.

---

## H39 — PhiGELU — VERDICT: **PASS**

- **Module:** `activations.py` (`phi_act`, `PhiGELU`, `swap_relu_with_phigelu`).
- **Mechanism vs claim:** `x · sigmoid(β · x)` with `β = φ` default. SiLU when `β = 1`. Buffer when `learnable=False`, Parameter when `learnable=True`. Drop-in for `nn.ReLU` via `swap_relu_with_phigelu` (named-children recursion).
- **Math:** `f'(0) = σ(0) + 0 = 0.5` — numerically verified. Monotonic on `x > 0` (numerically verified over `linspace(0, 5, 200)`). `f(0) = 0` exactly. Reduces to SiLU at β=1 (`atol=1e-6`).
- **Test rigor (strong):** Zero at origin; monotonic on positive x; **reduces to SiLU exactly at β=1**; spot-check `x · σ(φx)` formula; module-shape preservation; **f'(0) ≈ 0.5** via autograd; learnable-as-Parameter; buffer-when-not-learnable; drop-in shape match vs ReLU; **state_dict round-trip preserves β** regardless of learnable flag. Mechanism-verifying across all four design-doc properties.
- **Citation:** Implicit Ramachandran 2017 Swish, Hendrycks 2016 GELU — not cited in module text. MINOR gap; docs may have these.
- **Hidden bug:** `swap_relu_with_phigelu` only touches `nn.ReLU` submodules, not `F.relu(...)` in hand-written `forward`. Documented in the docstring as expected behavior. Fine.

---

## H40 — Metatron Kernel Overlap — VERDICT: **PASS**

- **Module:** `metatron_kernel.py` (`metatron_basis_kernels`, `MetatronConv2d`).
- **Mechanism vs claim:** 13 disc masks (1 centre + 6 inner-hex at distance `k/4`, angles `i·π/3` + 6 outer at distance `k/2.5`, angles `i·π/3 + π/6`). All radii `k/4`. Learnable `α ∈ R¹³`, learnable `W ∈ R^{13, out, in, k, k}`, `W_eff = Σ_c α_c · basis_c · W_c`.
- **Math:** Inner-hex centre at distance `k/4` with disc radius `k/4` ⇒ central pixel lies **on the boundary** (`d² = (k/4)² ≤ (k/4)²` ⇒ included by `≤`). So central pixel hit by 1 (centre) + 6 (inner) = 7 discs ≥ 3. Outer at `k/2.5` with radius `k/4`: for `k=7`, `r_outer = 2.8 > 1.75 = r_disc`, so outer discs do NOT cover the centre — consistent with the geometry (outer Metatron ring should not enclose the centre). The full Metatron-Cube central-pixel overlap is preserved.
- **Effective-kernel variance:** Init `W ~ N(0, sqrt(2/fan_in)/sqrt(n_circles))`. After summing over 13 circles with shrinking α (φ-decay), the effective kernel variance is comparable to a standard He-init Conv2d. Reasonable.
- **Test rigor (strong):** Basis shape `(13, k, k)` and binary `{0, 1}`; **central-pixel overlap ≥ 3** across `k ∈ {7, 9, 11}`; truncation at `n_circles ∈ {1, 7, 13}` preserves the canonical 1 + 6 + 6 structure (`b13[:7] == b7`, `b13[:1] == b1`); guards on `k < 3`; forward shape; effective_kernel shape `(out, in, k, k)`; learnable α (Parameter) changes output; α init = `1/φ^c` numerical match; **basis is buffer not Parameter, but appears in state_dict**; grad flows through α AND W. Mechanism-verifying.
- **Citation:** Qiu 2018 DCFNet, Cohen-Welling 2016, Hoogeboom 2018 HexaConv, Mandelbrot 1982 — full form.
- **Hidden bug:** None observed. `n_circles > 13` cycles the 13 canonical centres (duplicate discs) — flagged in the docstring as "stress-test only". Fine.

---

## Group-level concerns

1. **H31 silently substitutes generic exponential growth + uniform angular sampling** for the documented `φ^(θ/(π/2))` φ-growth + golden-angle phyllotaxis. This is the only G4 hypothesis whose `tests/test_*.py` does NOT verify the load-bearing claim in its own design doc (the Q&A section promises a `test_phi_growth` that does not exist). Any positive smoke result for `sg_only_golden_spiral_init` is interpretable as **structured-mask-on-He**, not as evidence for the φ-spiral prior.
2. **H35 `chladni_modes_banded` cycles literal duplicates** when `n_modes > k²`. Triggered by real conv shapes (`out_c=32, k=3` ⇒ `k² = 9`) but not exercised by any test. Silent orthonormality violation.
3. **H38 even-kernel crop introduces a half-pixel offset** between the k=3/5 paths and the k=8 path. The "fractal alignment across Fibonacci scales" framing is silently broken on the largest path.
4. **H37 cyclic-shift test is shift-magnitude-restricted** (`shift = L // 5`). General translation equivariance is plausible but un-tested; adversarial reviewer can construct a counterexample only outside the cyclic group of length `L`.
5. **Test rigor is materially stronger than G1** (per the doctrine). H32, H34, H36, H39, H40 all carry mechanism-verifying tests (`numerical match against reference`, `norm preservation`, `relative-equivariance under shift`, `state_dict roundtrip`, `coprime-dilation lcm density`). The shape-only-test antipattern is absent except in H31.
6. **Naming drift:** H33 "Vesica Piscis Filter" is closer to a 1-D row of overlapping discs than to the canonical 2-disc vesica lens. The mechanism is fine; the label overreaches.

## Follow-ups for the human PI

1. **(MAJOR)** Fix `inits.py:golden_spiral_mask` to implement the documented `r ∝ φ^(θ/(π/2))` growth AND golden-angle stepping; add `test_phi_growth` + `test_golden_angle_step`. Re-run any H31 smoke rows that were based on the broken implementation. If φ-growth was already tested in any branch, link the SHA.
2. **(MINOR)** Add a guard in `chladni_modes_banded` that raises on `n_modes > k²`, or sign-flips duplicates; add a regression test.
3. **(MINOR)** Decide between `(3, 5, 7)` odd-only kernels in `FractalGoldenFilter` or asymmetric padding for `k=8`. Either way, document the choice.
4. **(MINOR)** Extend the H37 cyclic-shift test to non-`L/5` shifts to confirm general translation equivariance (or document the cyclic-only property).
5. **(COSMETIC)** H33: rename to `OverlappingDiscFilter` or fix the disc layout to satisfy `|c₁ − c₂| = r` (canonical vesica).
6. **(COSMETIC)** H39 module text could cite Ramachandran 2017 Swish + Hendrycks 2016 GELU explicitly.

— end G4 audit —
