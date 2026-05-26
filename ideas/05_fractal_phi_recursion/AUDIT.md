# AUDIT — H05

> Adversarial self-critique BEFORE first experiment. Treat the
> implementation as if reviewing it for a hostile NeurIPS reviewer.

## Weaknesses found in v1

1. **No explicit 1/phi shrink rule yet.** The hypothesis title says
   "phi-recursion" but the shared `_FractalPath` in
   `src/nature_inspired_networks/blocks.py` does NOT scale c_out by
   1/phi at each recursion level -- the sub-block keeps c_out constant
   across all depths. Concretely, at depth=2 both branches use the
   same c_out as the parent block. The T1.5 single-seed result
   (+2.35 pp top-1) is therefore attributable to fractal recursion
   PER SE, not to phi-shrinkage. The H05 claim is currently a partial
   claim -- only the FractalNet part is implemented, and the phi part
   is queued for the shared-infra extension.

2. **Param-budget cost cancels the composite gain.** At depth=2 the
   block roughly doubles its param count (predicted factor 2.0,
   empirical 2.04 in T1.5). The composite penalty `-0.05 *
   log10(params_M)` is small (~0.015 per 10x params) but combined with
   the latency penalty (T1.5: +67 %) the net composite delta was
   -0.0031 despite the +2.35 pp top-1 lift. Any honest deployment must
   either (a) accept the composite drop or (b) reduce c_out by 1/sqrt(2)
   to maintain iso-params -- which has not been tested.

3. **Branch averaging is unweighted.** `_FractalPath.forward` returns
   `0.5 * (a + b)` -- equal weights on the two branches. This is the
   FractalNet (Larsson 2017) convention but the H05 phi-recursion
   ideal would weight branches by 1/phi^k where k is the recursion
   depth, so the deeper branch contributes less. The current code does
   NOT do this and the tests do not check for it.

4. **Drop-path regularisation is absent.** FractalNet's full method
   includes per-branch drop-path; without it the fractal ensemble is
   reduced to a "wider, deeper" block whose only benefit over a vanilla
   2-conv block is the path-diversity at training time. The shared
   `_FractalPath` does NOT implement drop-path. This is a known gap
   tracked under H52.

## Bugs caught by tests (good)

- `_FractalPath` does forward-pass cleanly at depth=3 without producing
  NaNs (`test_block_forward_finite_no_nans`).
- The predicted-vs-empirical param factor at depth=2 is within 50 %,
  catching any future regression where someone "optimises" the
  branching topology and silently changes the cost model.
- depth=0 is rejected at the wrapper level.

## Bugs NOT caught by tests but suspected

- The fractal recursion at depth=3 may be numerically unstable at
  high learning rates because the four-branch sum has effective
  variance scaling that diverges from He-init assumptions. Not
  observed in CPU smoke; needs bf16-on-GPU verification.
- BatchNorm momentum is shared across fractal branches but the
  branches see correlated activations -- this may bias the running
  statistics. Not tested.

## Performance / numerical-stability concerns

- Latency at depth=2 is +67 % (T1.5: 7.42 ms vs 4.43 ms) on the RTX 4090
  Laptop at batch 256. At depth=3 we predict +150 % from the recursion
  count alone, not measured.
- Memory: depth=2 doubles the activations stored for backward.
  Gradient checkpointing on the b2 branch would cut peak memory by
  ~30 % at no accuracy cost; not implemented.

## Mitigations queued for IMPROVEMENTS.md

- Add a `phi_shrink` argument to `build_fractal_block` that scales
  c_out by 1/phi at each recursion level (requires a shared-infra
  extension to `_FractalPath`).
- Add weighted branch averaging with weights [phi/(phi+1), 1/(phi+1)].

## Sign-off

- 2026-05-27 — Code-Agent-X
