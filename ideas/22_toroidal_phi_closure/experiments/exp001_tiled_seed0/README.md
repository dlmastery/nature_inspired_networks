# exp001_tiled_seed0 — MIGRATED stub for H22 (toroidal phi-closure)

> **MIGRATED FROM** `experiments/cifar10/sg_only_toroidal_seed0/`
> (legacy single-seed CIFAR-10 sweep, T1.6).

This archive sub-directory is a **pointer** to the canonical single-prior
toroidal-padding run that was already completed in the legacy 11-row
CIFAR-10 sweep before the `ideas/` taxonomy existed. The full per-seed
artifacts (`metrics.json`, `history.json`, `config.yaml`, `best.pt`,
`reasoning.json` if present) live at the legacy path.

## Headline numbers (from the migrated source)

| metric | value | reference (`sg_chan_fib`) | Δ |
|---|---|---|---|
| top-1 | **78.05 %** | 80.11 % | -2.06 pp |
| top-5 | 98.56 % | — | — |
| params | 127 146 | 127 146 | 0 |
| latency (ms, b=1) | 9.34 | 4.43 | +4.91 (≈ 2.1×) |
| rot-eq err | 0.8936 | — | -0.025 (vs reference) |
| **composite** | **0.7768** | 0.8135 | **-0.0367** |
| composite fingerprint | `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` | (same) | — |
| epochs | 12 | 12 | — |
| seed | 0 | 0 | — |
| train_seconds | 395.4 | — | — |

## Verdict at this scale

**Predicted-negative.** Toroidal closure on upright CIFAR-10 is data-misaligned
(CIFAR images do not wrap), so the -2.06 pp top-1 shortfall is a confirmation,
not a refutation, of the wrap-awareness mechanism stated in
`hypotheses/H22_toroidal_phi_closure.md` § 11. The next archived run under this
idea will be the **wrap-aware tiled-CIFAR + φ-scaled wrap distance** experiment
(planned T2.6 in `EXPERIMENT_LOG.md`); CLAUDE.md rule #1 forbids changing more
than one knob at a time, so the data shift and the φ-scaling will be staged
across two consecutive runs.

## Reproduce the migrated run

```powershell
$env:SSL_CERT_FILE = "..\..\..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\..\..\.venv\Scripts\python ..\..\..\..\scripts\run_sweep.py `
   --config ..\..\..\..\configs\cifar10_quick.yaml `
   --only sg_only_toroidal --seeds 0 --skip-existing
```

## See also

- Legacy archive: `../../../../experiments/cifar10/sg_only_toroidal_seed0/`
- Hypothesis design doc: `../../../../hypotheses/H22_toroidal_phi_closure.md`
- Headline campaign verdict: `../../../../FINDINGS.md` row "toroidal -2.06 pp"
