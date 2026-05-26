# exp001_audio_seed0 — MIGRATED stub for H35 (cymatic wavelet init)

> **MIGRATED FROM** `experiments/cifar10/sg_only_cymatic_init_seed0/`
> (legacy single-seed CIFAR-10 sweep, T1.7).

This archive sub-directory is a **pointer** to the buggy-init / wrong-data
single-prior cymatic run from the legacy 11-row CIFAR-10 sweep. The full
per-seed artifacts (`metrics.json`, `history.json`, `config.yaml`,
`best.pt` if present) live at the legacy path.

The "audio_seed0" tag is forward-looking — the NEXT archive under this
idea will be the corrected `cymatic_init_ortho_` run on AudioMNIST.
Calling this stub `exp001_audio_seed0` rather than `exp001_cifar_seed0`
records that intent.

## Headline numbers from the migrated source

| metric | value | reference (`sg_chan_fib`) | Δ |
|---|---|---|---|
| top-1 | **77.44 %** | 80.11 % | -2.67 pp |
| top-5 | 98.53 % | — | — |
| params | 127 146 | 127 146 | 0 |
| latency (ms, b=1) | 4.14 | 4.43 | ~no-op (init only) |
| rot-eq err | 0.9006 | — | -0.011 |
| **composite** | **0.7883** | 0.8135 | **-0.0252** |
| composite fingerprint | `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893` | (same) | — |
| epochs | 12 | 12 | — |
| seed | 0 | 0 | — |
| train_seconds | 358.2 | — | — |

## Verdict at this scale

**Unexpected negative on CIFAR-10.** Latency is effectively unchanged
(init only) but top-1 dropped 2.67 pp - the second-worst single-prior
delta after `sg_only_group`. Root-cause analysis in
`hypotheses/H35_cymatic_wavelet.md` § 11 attributes this to (a) no
Gram-Schmidt across output channels, (b) (1,1) DC-dominated band, and
(c) data misalignment.

The corrected `cymatic_init_ortho_(band=(2,5))` in `implementation.py`
fixes (a) and (b); (c) requires shifting the primary benchmark from
CIFAR-10 to AudioMNIST, where the harmonic-vibration prior is data-aligned.

## Reproduce the migrated run

```powershell
$env:SSL_CERT_FILE = "..\..\..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\..\..\.venv\Scripts\python ..\..\..\..\scripts\run_sweep.py `
   --config ..\..\..\..\configs\cifar10_quick.yaml `
   --only sg_only_cymatic_init --seeds 0 --skip-existing
```

## See also

- Legacy archive: `../../../../experiments/cifar10/sg_only_cymatic_init_seed0/`
- Hypothesis design doc: `../../../../hypotheses/H35_cymatic_wavelet.md`
- Headline campaign verdict: `../../../../FINDINGS.md` row "cymatic_init -2.67 pp"
