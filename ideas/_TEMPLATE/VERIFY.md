# VERIFY — H{{NN}}

> Sign-off that the implementation is ready for its first archived
> experiment. All checkboxes must be green BEFORE
> `experiments/exp001_<short>/` is created.

## Tests

- [ ] `python tests.py` runs and ends "All N tests passed."
- [ ] Linter (ruff / flake8) has no new warnings on `implementation.py`
- [ ] Type checker (mypy) has no new errors on `implementation.py`

## Sanity

- [ ] Default flag combo produces output of expected shape + dtype
- [ ] Every Boolean-flag combination forward-passes (smoke test)
- [ ] Parameter count is within ±10 % of the value predicted in
      `IDEA.md` § Pre-registered prediction
- [ ] GPU smoke run on RTX 4090 (batch=2, 1 forward pass) succeeds
      with `bf16` autocast

## Documentation

- [ ] `README.md` filled in
- [ ] `IDEA.md` claim + falsifier + prediction + composite fingerprint
      recorded
- [ ] `AUDIT.md` has ≥ 3 weaknesses listed
- [ ] `IMPROVEMENTS.md` addresses every blocking weakness
- [ ] Cross-reference to `hypotheses/H<NN>_<short>.md` in place

## Sign-off

- {{YYYY-MM-DD}} — {{author}} — "ready for expNNN"
