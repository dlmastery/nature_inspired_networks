# AUDIT — H{{NN}}

> Adversarial self-critique BEFORE first experiment. Treat the
> implementation as if reviewing it for a hostile NeurIPS reviewer.
> If you cannot list at least three weaknesses, the audit was lazy —
> re-read the implementation harder.

## Weaknesses found in v1

1. **{{one-line summary}}** — {{why it matters, where in the code,
   how it could pollute downstream metrics}}
2. **{{...}}** — {{...}}
3. **{{...}}** — {{...}}

## Bugs caught by tests (good)

- {{e.g., shape mismatch at stride=2 in C4 orbit reduction}}

## Bugs NOT caught by tests but suspected

- {{e.g., the cymatic init uses a fixed PRNG seed → reproducibility
   gain BUT every conv layer in every block gets the same modes →
   may cause spurious shared structure across layers}}

## Performance / numerical-stability concerns

- {{e.g., orbit reconstruction in GroupConv2d is recomputed every
   forward → measurable latency hit at batch=1}}

## Mitigations queued for IMPROVEMENTS.md

- {{...}}

## Sign-off

- {{YYYY-MM-DD}} — {{author}}
