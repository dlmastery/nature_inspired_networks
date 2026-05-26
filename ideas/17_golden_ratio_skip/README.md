# H17 — Golden Ratio Skip Connections (idea sub-project)

> See the full design document at `../../hypotheses/H17_golden_ratio_skip_connections.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

H17 scales the residual skip path by 1/phi (= 0.618) instead of the
standard 1.0, with three variants: (a) fixed 1/phi skip with branch
weight 1.0; (b) sum-to-one variant with skip=1/phi and branch=1/phi^2;
(c) learnable scalars initialised at 1/phi. The shared infra exposes
the golden-angle channel-modulate primitive (used in T1.8
`sg_only_golden_modulate`) but does NOT yet expose pure phi-skip
scaling -- this sub-project provides the thin composition wrapper
`PhiSkipWrapper` that takes any residual block and scales its skip
output by `PHI` from `nature_inspired_networks.priors`.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` -> all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [ ] |
| Row added to `../../EXPERIMENT_LOG.md` | [ ] (T1.8 partial-variant logged) |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | `PhiSkipWrapper` (uses `PHI` from shared infra) |
| `tests.py` | unit tests covering shape, modes, and the identity-branch math |
| `AUDIT.md` | self-critique: weaknesses found before first run |
| `IMPROVEMENTS.md` | what was fixed in response to the audit |
| `VERIFY.md` | dated sign-off that tests + smoke pass |
| `experiment.py` | idea-specific experiment driver |
| `configs/` | YAML configs |
| `experiments/exp001_<short>/` | per-experiment archive |
| `results.md` | rollup |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --config configs\h17_phi_skip.yaml --tag exp001_phi_skip --seed 0
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/H17_golden_ratio_skip_connections.md`
- Master experiment list: `../../EXPERIMENT_LOG.md` row T1.8 (variant)
- Manifesto: `../../MANIFESTO.md`
- IDEA table row: `../../IDEA_TABLE.md` H17
- Shared primitive: `src/nature_inspired_networks/priors.py::PHI`,
  `golden_angle_phases`
