# exp001_sg_full_fib_seed0 — MIGRATED stub for H50 (full hybrid)

> **MIGRATED FROM** `experiments/cifar10/sg_full_fib_seed0/`
> (legacy single-seed CIFAR-10 sweep, T1.9). This is **the headline
> negative result of the entire research program**.

This archive sub-directory is a **pointer** to the canonical
`sg_full_fib` run. The full per-seed artifacts (`metrics.json`,
`history.json`, `config.yaml`, `best.pt`, `reasoning.json` if present)
live at the legacy path.

## Headline numbers (verbatim from the migrated source)

| metric | observed | reference (`sg_chan_fib`) | Δ |
|---|---|---|---|
| top-1 | **73.24 %** | 80.11 % | **-6.87 pp** |
| top-5 | 98.24 % | — | — |
| params | 259 443 | 127 146 | +132 297 (+103 %) |
| FLOPs | 219.2 M | 27.8 M | +7.9× |
| latency (ms, b=1) | **20.02** | 4.43 | **+15.59 ms (5.0×)** |
| rot-eq err | 0.6827 | — | **-0.045 (only positive — equivariance is real)** |
| **composite** | **0.6966** | 0.8135 | **-0.1169** |
| composite fingerprint | `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` | (same) | — |
| epochs_to_target | -1 (never reached) | — | — |
| epochs | 12 | 12 | — |
| seed | 0 | 0 | — |
| train_seconds | 478.9 | — | — |
| generalization_gap | 0.019 | — | — |

## Verdict at this scale

**FALSIFIED.** The IDEA.md falsifier (Δ <= -0.005) is hit 23× over.
The originally pre-registered prediction was top-1 +2.0 to +6.0 pp
versus `sg_chan_fib`; the observed delta is -6.87 pp. Composite
0.6966 ties this row with `sg_only_group` (0.6937) for the bottom of
the 11-run ranking (`FINDINGS.md` rank 10 vs 11).

**The single positive prediction that held:** rotation-equivariance
error dropped 0.045, confirming the C4 prior IS providing equivariance —
the prior just isn't useful on canonical CIFAR-10 where test images
are not rotated.

## Why the compound failed (from FINDINGS.md)

1. **Multiplicative latency stack** — each prior adds 1.7–2.2×; six
   priors compound to 5.0× total latency.
2. **C4 max-pool destroys 75 % of orbit signal** — dominant negative
   single-prior at -10.27 pp.
3. **Toroidal pad on non-wrap data** — -2.06 pp single-prior; the
   wrap is structurally wrong for natural images.
4. **Cymatic init missed orthonormalisation + DC band** — -2.67 pp
   single-prior; root-caused in `ideas/35_cymatic_wavelet/AUDIT.md`.
5. **Non-orthogonal compound** — naive additive prediction was -13.74 pp;
   observed -6.87 pp implies some priors do partially recover when
   combined, but the net is still strongly negative.

## Reproduce the migrated run

```powershell
$env:SSL_CERT_FILE = "..\..\..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\..\..\.venv\Scripts\python ..\..\..\..\scripts\run_sweep.py `
   --config ..\..\..\..\configs\cifar10_quick.yaml `
   --only sg_full_fib --seeds 0 --skip-existing
```

Wall-clock: ~8 min on RTX 4090 Laptop, 16 GB.

## See also

- Legacy archive: `../../../../experiments/cifar10/sg_full_fib_seed0/`
- Hypothesis design doc: `../../../../hypotheses/g5_optimization_init_reg_nas/H50_full_sacred_hybrid.md`
- Headline campaign verdict: `../../../../FINDINGS.md`
- Refined follow-ups:
  - `../../../58_group_avg_pool/` (H58 DISCARDED — avg-pool was worse)
  - H45 Sacred NAS (queued in EXPERIMENT_LOG.md)
  - H60 3-seed reproduction (queued)
  - H67 curated LLM-track hybrid
