# H04 — phi / Fibonacci channel scaling (idea sub-project)

> See the full design document at `../../hypotheses/H04_phi_self_similar_width.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

H04 grows per-stage channel widths by successive multiplication by phi
(or by Fibonacci ratios) instead of linear doubling, rounded to
multiples of 8 for tensor-core throughput. The shared primitive
`nature_inspired_networks.priors.fibonacci_channels` implements all
three schedules ("fib", "phi", "linear"); this sub-project is the thin
wrapper that exposes them to the runner and pins the regression test
that catches the **mod-8 collapse** observed in T1.1/T1.2 (where
fib and phi both rounded to [16, 24, 32] / [16, 24, 40] at c0=16,
n_stages=3 and so produced identical top-1 80.11 %).

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [x] |
| `tests.py` green (`python tests.py` → all N tests passed) | [x] |
| `AUDIT.md` filed with >= 3 self-found weaknesses | [x] |
| `IMPROVEMENTS.md` records the fixes | [x] |
| `VERIFY.md` signed with date | [x] |
| First experiment archived under `experiments/exp001_*/` | [ ] |
| Row added to `../../EXPERIMENT_LOG.md` | [ ] (T1.1/T1.2 already logged) |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | thin wrapper around `fibonacci_channels` + a separation guard |
| `tests.py` | unit tests for `implementation.py` (incl. mod-8 collapse regression) |
| `AUDIT.md` | self-critique: weaknesses found before first run |
| `IMPROVEMENTS.md` | what was fixed in response to the audit |
| `VERIFY.md` | dated sign-off that tests + smoke pass |
| `experiment.py` | idea-specific experiment driver (wraps the project runner) |
| `configs/` | YAML configs |
| `experiments/exp001_<short>/` | per-experiment archive |
| `results.md` | auto-generated summary across all archived experiments |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --config configs\h04_c0_32_n4.yaml --tag exp001_c0_32_n4 --seed 0
```

Or via the global sweep:

```powershell
..\..\.venv\Scripts\python ..\..\scripts\run_sweep.py `
   --config configs\h04_c0_32_n4.yaml --only sg_chan_phi --seeds 0 1 2
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/H04_phi_self_similar_width.md`
- Master experiment list: `../../EXPERIMENT_LOG.md` rows T1.1, T1.2
- Manifesto: `../../MANIFESTO.md`
- IDEA table row: `../../IDEA_TABLE.md` H04
- Shared primitive: `src/nature_inspired_networks/priors.py::fibonacci_channels`
