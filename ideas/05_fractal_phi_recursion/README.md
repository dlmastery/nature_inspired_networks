# H05 — Fractal phi-Recursion (sub-block depth=2) (idea sub-project)

> See the full design document at `../../hypotheses/H05_fractal_phi_recursion.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

H05 wraps each conv-2 in a residual block with a FractalNet-style
recursive sub-block (`_FractalPath`) where the depth/width of the
recursion follows a 1/phi shrink rule. The shared primitive
`nature_inspired_networks.blocks._FractalPath` already implements
depth-N recursion; this sub-project re-exports it via `NaturePriorBlock`
with `flags.fractal=True` and tests the depth=2 case that the previous
sweep (T1.5 `sg_only_fractal`, single seed) showed lifts top-1 by
+2.35 pp at the cost of +104 % params and +67 % latency.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` -> all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [ ] |
| Row added to `../../EXPERIMENT_LOG.md` | [ ] (T1.5 already logged) |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | thin wrapper composing `NaturePriorBlock` with `flags.fractal=True` |
| `tests.py` | unit tests covering depth=1/2, flag interactions, T1.5 param-budget regression |
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
..\..\.venv\Scripts\python experiment.py --config configs\h05_fractal_d2.yaml --tag exp001_fractal_d2 --seed 0
```

Or via the global sweep:

```powershell
..\..\.venv\Scripts\python ..\..\scripts\run_sweep.py `
   --config configs\h05_fractal_d2.yaml --only sg_only_fractal --seeds 0 1 2
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/H05_fractal_phi_recursion.md`
- Master experiment list: `../../EXPERIMENT_LOG.md` row T1.5
- Manifesto: `../../MANIFESTO.md`
- IDEA table row: `../../IDEA_TABLE.md` H05
- Shared primitive: `src/nature_inspired_networks/blocks.py::_FractalPath`
