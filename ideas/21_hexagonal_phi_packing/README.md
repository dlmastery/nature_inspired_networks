# H21 — Hexagonal phi-Packing (idea sub-project)

> See the full design document at `../../hypotheses/H21_hexagonal_phi_packing.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

H21 replaces the standard 3x3 square convolution with a 7-tap hexagonal
stencil whose six peripheral weights are scaled by 1/phi relative to
the centre tap (phi-radial energy distribution). The shared infra
already exposes `HexConv2d` (uniform 7-tap mask) and `hex_kernel_mask`
in `nature_inspired_networks.priors`; this sub-project adds the
phi-radial scale as a small buffer on top, exposed via a thin
`PhiRadialHexConv2d` wrapper. The previous CIFAR-10 sweep T1.3
(`sg_only_hex`) used UNIFORM weights and was mildly negative
(-0.79 pp top-1, -0.022 rot-eq err). H21 predicts the phi-radial
version recovers the top-1 on upright CIFAR-10 and significantly
improves rot-eq err on rotated-CIFAR-10.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` -> all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [ ] |
| Row added to `../../EXPERIMENT_LOG.md` | [ ] (T1.3 partial-variant logged) |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | `PhiRadialHexConv2d` (composes `HexConv2d` + phi-radial scale) |
| `tests.py` | unit tests covering shape, corner-zero, phi ratio, equivariance |
| `AUDIT.md` | self-critique |
| `IMPROVEMENTS.md` | fixes |
| `VERIFY.md` | dated sign-off |
| `experiment.py` | experiment driver |
| `configs/` | YAML configs |
| `experiments/exp001_<short>/` | per-experiment archive |
| `results.md` | rollup |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --config configs\h21_hex_phi.yaml --tag exp001_hex_phi --seed 0
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/H21_hexagonal_phi_packing.md`
- Master experiment list: `../../EXPERIMENT_LOG.md` row T1.3 (uniform variant)
- Manifesto: `../../MANIFESTO.md`
- IDEA table row: `../../IDEA_TABLE.md` H21
- Shared primitive: `src/nature_inspired_networks/priors.py::HexConv2d`,
  `hex_kernel_mask`, `PHI`
