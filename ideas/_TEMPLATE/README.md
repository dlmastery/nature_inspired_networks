# H{{NN}} — {{Short Title}} (idea sub-project)

> See the full design document at `hypotheses/H{{NN}}_<short>.md`.
> This README is the OPERATOR-FACING summary: implementation status,
> file layout, how to test it, how to run an experiment.

## TL;DR

{{2-3 sentences with the hypothesis + status + headline result if any}}.

## Status

| stage | done? |
|---|---|
| `implementation.py` written | [ ] |
| `tests.py` green (`python tests.py` → all N tests passed) | [ ] |
| `AUDIT.md` filed with ≥ 3 self-found weaknesses | [ ] |
| `IMPROVEMENTS.md` records the fixes | [ ] |
| `VERIFY.md` signed with date | [ ] |
| First experiment archived under `experiments/exp001_*/` | [ ] |
| Row added to `../../experiments/EXPERIMENT_LOG.md` | [ ] |
| Dashboard refreshed | [ ] |

## Files in this directory

| file | purpose |
|---|---|
| `README.md` | this file |
| `IDEA.md` | formal claim + falsifier + pre-registered prediction |
| `implementation.py` | the primitive(s) this idea introduces (re-exported from `nature_inspired_networks` where possible) |
| `tests.py` | unit tests for `implementation.py` |
| `AUDIT.md` | self-critique: weaknesses found before first run |
| `IMPROVEMENTS.md` | what was fixed in response to the audit |
| `VERIFY.md` | dated sign-off that tests + smoke pass |
| `experiment.py` | idea-specific experiment driver (wraps the project runner with idea-local config) |
| `configs/` | YAML configs |
| `experiments/exp001_<short>/` | per-experiment archive (see contract in `skills/autoresearch-experiment-archive/SKILL.md`) |
| `results.md` | auto-generated summary across all archived experiments |
| `dashboard/` | idea-level dashboard |

## How to run the experiment

```powershell
$env:SSL_CERT_FILE = "..\..\.venv\Lib\site-packages\certifi\cacert.pem"
..\..\.venv\Scripts\python tests.py        # must end 'All N tests passed.'
..\..\.venv\Scripts\python experiment.py --seeds 0      # first run
```

Or via the global sweep:

```powershell
..\..\.venv\Scripts\python ..\..\scripts\run_sweep.py `
   --config configs\<config>.yaml --only <tag> --seeds 0
```

## Cross-references

- Full hypothesis design doc: `../../hypotheses/g<G>_<group>/H{{NN}}_<short>.md`
- Master experiment list: `../../experiments/EXPERIMENT_LOG.md`
- Manifesto: `../../paper/MANIFESTO.md`
- IDEA table row: `../../hypotheses/IDEA_TABLE.md` H{{NN}}
- Headline campaign verdict: [`paper/FINDINGS.md`](https://github.com/dlmastery/nature_inspired_networks/blob/main/paper/FINDINGS.md)
