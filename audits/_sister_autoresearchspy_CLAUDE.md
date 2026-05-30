# CLAUDE.md — Project Rules for AutoResearch · SPY

> **🏆 SESSION-COMPLETE STATE (2026-05-04, after 28 commits, 59 distinct runs)**
>
> **Deployable champion**: 12-component val-weighted ensemble + regime gate (rvol60d > 15%) — composite **+0.368** on 1410-day held-out test (2008-2025).
>
> **Three stacked validated KEEPs**:
> 1. Asian/EU pre-market block (+0.330 composite over daily-only) — `data/asian_premarket.py`
> 2. Val-weighted 12-component ensemble (+0.277) — `_ensemble_val_weighted.py`
> 3. Regime gate `rvol60d > 15%` (+0.134) — `_ensemble_regime_gated.py`
>
> **Cumulative gain**: -0.373 (daily-only) → **+0.368** (champion) = +0.741 stacked.
>
> **Critical caveat**: model is a CRISIS-TRADE SPECIALIST. Sub-period sharpe 2008-2012 = +1.10, 2021-2025 = -0.67. Regime gate prevents trading in low-vol bull regimes (only ~30% of days in 2024-2025) but doesn't generate alpha there. See `autoresearchspy/autoresearch_results/champion_verification/robustness_audit.txt`.
>
> **Resumption pointers (for a fresh laptop)**:
> - Read `autoresearchspy/DEPLOYMENT.md` — production deployment recipe
> - Read `autoresearchspy/autoresearch_results/research_journal.md` — full session arc
> - Read `autoresearchspy/memory/project_autoresearch_checkpoint.md` — exact next-experiment commands
> - Run `python -m autoresearchspy._ensemble_regime_gated` — reproduces +0.368 verification
> - Run `python -m autoresearchspy._ensemble_robustness` — reproduces regime-dependence audit
>
> **What's left for a future session** (all out of scope for the 2026-05 session):
> - Forward OOS validation on 2026+ data when available
> - Paid Barchart Premier hourly history (yfinance 730d cap blocks intraday training)
> - Stacking via meta-learner (requires runner change to log val predictions)
> - Regime-conditional model architecture (better than single-rule gating)
>
> ---

> **What this repo is.** Autonomous research loop for predicting **SPY (S&P 500 ETF) directional returns**. Adapted from the predecessor FX (EUR/USD) AutoResearch project (preserved at `_reference_autoresearch/autoresearch/`). Every directive, super-fold split rule, composite-metric definition, audit checklist, and dashboard contract from the FX project is preserved verbatim — only the data substrate, target asset, and feature streams differ.
>
> **Repo layout (orient yourself before touching anything):**
>
> | Path | Role |
> |------|------|
> | `CLAUDE.md` *(this file)* | Top-level project rules — applies repo-wide |
> | `autoresearchspy/` | Primary SPY package — runners, data, model, evaluation, optimizer, tests |
> | `autoresearchspy/CLAUDE.md` | Package-level rules (inherits from this file; SPY-specific deviations) |
> | `autoresearchspy/FEATURES_AND_DATA.md` | Authoritative spec of every ticker, feature, target, fold, citation |
> | `autoresearchspy/AUTORESEARCH_PROCESS.md` | The 7-step research loop, SPY-adapted |
> | `autoresearchspy/SETUP.md` | Environment install + .env credentials |
> | `autoresearchspy/data/download.py` | Daily yfinance ticker fetcher (~85 tickers across panels) |
> | `autoresearchspy/data/asian_premarket.py` | **Asian + European pre-market block — causal pre-09:30-ET signals** |
> | `autoresearchspy/data/barchart_hourly.py` | **Barchart.com hourly intraday OHLCV + features (34 tickers)** |
> | `autoresearchspy/data/features.py` | ~205 daily feature columns |
> | `autoresearchspy/data/splits.py` | 7 walk-forward super-folds + purge/embargo/buffer |
> | `autoresearchspy/model/`, `evaluation/`, `optimizer/` | Backbones, metrics, agent loop |
> | `generalized_ml_autoresearch/` | Domain-agnostic framework (sibling project — see its own CLAUDE.md) |
> | `docs/spy_dashboard/` | GitHub Pages dashboard (auto-synced from `autoresearchspy/autoresearch_results/`) |
> | `docs/_forex_reference_dashboard/` | Original FX dashboard kept as reference template |
> | `_reference_autoresearch/` | Full clone of the predecessor FX project — read-only narrative reference |
> | `.env.example` | Template for `.env` (Barchart credentials, Anthropic key); `.env` is gitignored |
> | `pyproject.toml` | `pip install -e .` installs `autoresearchspy` as the importable package |
>
> **Common commands:**
>
> ```bash
> # Install (editable)
> pip install -e .
>
> # Set up credentials (.env is gitignored)
> cp .env.example .env  # then fill BARCHART_USERNAME / BARCHART_PASSWORD
>
> # Run the autoresearch loop (one experiment, one config delta)
> python -m autoresearchspy.run_autoresearch --backbone <lstm|patchtst|patchtsmixer|xgboost|lightgbm|catboost|mlp> [flags] --description "..."
>
> # Run the full ablation (all backbones)
> python -m autoresearchspy ablation
>
> # Run a single backbone baseline
> python -m autoresearchspy baseline --backbone lstm
>
> # Hyperparameter sweep
> python -m autoresearchspy sweep --backbone patchtst --n-trials 50
>
> # Tests
> pytest autoresearchspy/tests/ -v
>
> # Tail the experiment log
> tail -3 autoresearchspy/autoresearch_results/experiment_log.jsonl
>
> # Sync dashboard → docs/spy_dashboard/ (run after every experiment per dashboard-sync rule)
> python -m autoresearchspy._sync_dashboard_to_docs
>
> # Serve the dashboard locally (background)
> python -m http.server 8765 --directory autoresearchspy/autoresearch_results
> ```
>
> **What's different from the FX predecessor (read this BEFORE writing code):**
>
> 1. **Three feature streams** instead of one: daily yfinance + Asian/European pre-market block + Barchart hourly. See "Three-stream feature engineering" below.
> 2. **Anchor-driven causality**: the hourly block snapshots at `asia_close` (04:00 ET) | `us_open` (09:30 ET, default) | `us_close` (16:00 ET — explanatory only, **never train at us_close**).
> 3. **Asian close is the SPY edge.** N225/HSI/KS11/AXJO close 5–9h before NYSE 09:30; their close-to-close moves carry causal information for SPY day-T direction (Hamao-Masulis-Ng 1990, Lin-Engle-Ito 1994, Lou-Polk-Skouras 2019). Blended `asian_sentiment_score` aggregates them.
> 4. **Barchart credentials** in `.env` (gitignored). Free-tier respects 5-CSV/day; flip `BARCHART_RATE_LIMIT=premier` after upgrading.
> 5. **2026 data is forbidden.** `_enforce_no_2026()` runs after every yfinance fetch; the equivalent guard exists in the Barchart downloader.

---

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:** `memory/project_autoresearch_checkpoint.md` — it has the current champion, last experiment result, per-fold diagnostics, and what to try next.
2. **Read the hardware crash log:** `memory/project_hardware_crash_log.md` — documents BSOD history and CPU core exclusion rules. Must follow.
3. **Read the experiment log tail:** `autoresearch_results/experiment_log.jsonl` (last 3 entries) and `autoresearch_results/best_config.json` to verify state.
4. **Resume the experiment loop** from where the checkpoint says. Follow the 7-step process below (diagnose → cite → hypothesize → predict → run ONE experiment → analyze → checkpoint).
5. **Start the dashboard** (once per session, background): `python -m http.server 8765 --directory C:/Users/abhir/autoresearchspy/_reference_autoresearch/autoresearch/autoresearch_results` — then tell the user: "Dashboard at http://localhost:8765/dashboard.html"
6. **Run experiments** via: `cd C:/Users/abhir/autoresearchspy && python -m autoresearchspy.run_autoresearch --backbone <backbone> [flags] --description "..."` (timeout 600s). **LFM2 is SKIPPED** per user instruction 2026-04-19 — its 43 experiments are frozen, no more LFM2 runs. Active backbone order: lstm → patchtst → patchtsmixer → xgboost → lightgbm → catboost.
7. **If the user says "continue" or "keep going"** — resume the loop. No need to ask what to do.

## Hardware Constraints (MANDATORY — updated 2026-04-19)

**E-cores are BANNED.** On this Intel 14th-gen HX system (32 logical CPUs), WHEA-Logger
reported Internal parity errors on CPU APIC IDs 16, 17, 24, 25 (all E-cores). System
BSODed 4 times today under sustained compute.

- **Use ONLY P-cores**: logical IDs 0-15. Even IDs (0,2,4,...,14) are primary threads,
  odd IDs (1,3,...,15) are HT siblings.
- **Default**: 4 P-core threads via `torch.set_num_threads(4)` + `cpu_affinity([0,2,4,6])`.
- **GPU does heavy compute**; CPU is coordination only. 4 cores is enough.
- `run_autoresearch.py:_pin_to_safe_cores()` handles this automatically.
- Override with env var `AUTORESEARCH_USE_ALL_CORES=1` (not recommended).
- Override thread count with `AUTORESEARCH_N_THREADS=N`.

**NEVER run a training loop without the pinning.** If you write a new runner script,
call `_pin_to_safe_cores()` first thing or the laptop will BSOD.

## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning, whichever comes first.** This is the #1 non-negotiable rule. The laptop WILL crash. Every minute of uncheckpointed work is lost work.

**Checkpoint trigger points (ALL mandatory):**
1. **Immediately after every experiment completes** — before any analysis or reasoning about results
2. **Every 5 minutes during reasoning/analysis** — if you've been thinking for 3+ minutes without saving, STOP and checkpoint
3. **Before starting any code change** — save current state so crash during edit doesn't lose experiment context
4. **After any code change** — save the new code state and what was changed
5. **Before starting the next experiment** — checkpoint must contain the exact bash command ready to paste

What to save to `memory/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Per-fold test Sharpe table for the champion
- Last experiment result (config, composite, per-fold deltas vs champion, KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable bash)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes (so we don't re-try them)
- Session start instructions (numbered steps)
- **Full experiment history summary** — every experiment number, config delta, result, KEEP/DISCARD

Also update `autoresearch_results/experiment_summary.md` with the all-experiments table.

**During long reasoning/analysis (no experiment running):** still checkpoint every 5 minutes. Save your current thinking, diagnosis, and plan to the checkpoint file. If you've been reasoning for 3+ minutes without saving, STOP and checkpoint before continuing.

**The checkpoint must be self-contained.** A fresh Claude Code session reading ONLY `CLAUDE.md` + the checkpoint must be able to resume without reading any other file. Include the bash command, the rationale, and enough per-fold context to make the next decision. A new session should be able to pick up EXACTLY where the previous one left off — same experiment number, same champion, same next-experiment rationale.

## Mindset (Read First)

You are a top-tier MLFin researcher — multiple best-paper awards at NeurIPS/ICML/AAAI, industry expert in financial ML. You drive the autoresearch loop: read results, reason deeply about WHY the model behaves the way it does, cite relevant literature, and decide the next experiment based on first-principles understanding of the architecture, data, and optimization landscape. Never guess. Never grid-search. Before touching any code:
1. **Understand the data flow end-to-end.** Trace how a single training sample is created, from raw OHLCV through features, scaling, windowing, to loss computation. If you can't explain every step, you don't understand the system.
2. **Validate before running.** Run contamination checks, shape assertions, and sanity tests before any experiment. A 2-minute verification saves hours of garbage results.
3. **Measure, never assume.** If you state a number (timing, sample count, performance), it must come from running code — not estimation.
4. **When fixing a bug, audit the entire system for the same class of bug.** Don't patch one instance and leave three others.
5. **Separation of concerns is not optional.** Runners log. Dashboards display. Evaluators evaluate. Never tangle them.

## Hard Rules (NEVER violate)

### Data Integrity
- NEVER create sliding windows (SPYDataset) across non-contiguous date ranges. Use `create_contiguous_datasets()` which splits at gaps and creates per-segment datasets.
- NEVER include any fold's val or test dates in any fold's training data. Verify with `split_superfold()` — 0 overlap verified.
- ALWAYS use the label-horizon buffer (10 calendar days) before excluded windows to prevent `fwd_ret_5d` target leakage. The purge gap + buffer together prevent any forward-looking information from leaking into training.
- ALWAYS cache downloaded data. `download_all()` (daily yfinance) defaults to `.data_cache_spy/`; `download_all_hourly()` (Barchart hourly) defaults to `.data_cache_spy_hourly/`. NEVER re-download mid-run.
- Load data ONCE at startup. Compute features/targets ONCE. Split ONCE. Reuse across all experiments in a loop.

### Three-stream feature engineering (MANDATORY pre-flight discipline)
Every SPY training run combines these three causally-anchored streams. See `autoresearchspy/FEATURES_AND_DATA.md` §10–11 for the full spec.

1. **Daily yfinance stream** — `data/download.py` + `data/features.py`. ~205 columns: SPY OHLCV technicals, sector ETFs, vol regime (VIX/VXN/MOVE), yields/credit, macro/FX, panel-learning targets.
2. **Asian + European pre-market stream** — `data/asian_premarket.py`. ~70 columns: N225, HSI, KS11, TWII, STI, AXJO, 000001.SS, BSESN (close before 09:30 ET, fully causal) + FTSE/DAX/CAC/STOXX50E/IBEX/SSMI lagged 1d. Includes blended `asian_sentiment_score` and `asian_divergence`.
3. **Barchart hourly stream** — `data/barchart_hourly.py`. ~400 columns: 34 tickers (SPY, breadth ETFs, Asia/Europe ETF surrogates, megacap ADRs, US megacaps, vol/yields/macro, ES/NQ/YM/RTY futures) × 12 hourly features (RSI/MACD/RV/Amihud/first-hour/last-hour/overnight-gap). Aligned to daily index via causal anchor (`asia_close` | `us_open` | `us_close`).

**Anchor contract:** the chosen anchor (default `us_open` = 09:30 ET) sets the strict-causal cutoff. Every feature in row[T] must be observable at or before the anchor on date T. Train at `us_open`; ablate at `asia_close` for maximum lead; `us_close` is explanatory-only and forbidden for forward-prediction training. **Barchart credentials:** `.env` file (gitignored) — copy `.env.example` and fill in `BARCHART_USERNAME` / `BARCHART_PASSWORD`.

### Super-Fold Invariants
- Fold 7 training data includes ALL historical data (2005-2023) EXCEPT: all 7 folds' val windows, all 7 folds' test windows, and 10-day label buffers before each.
- Val set is the UNION of all 7 folds' validation windows (915 rows across 7 regime periods).
- Test set is the UNION of all 7 folds' test windows (1170 rows across 7 regime periods).
- **Zero overlap** between train/val/test — verified programmatically before every run.
- These invariants encode standard ML: train never sees val or test data. Val and test are exhaustive across all regimes.

### Experiment Design
- **Composite metric for keep/revert:** `min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds`. The model must do well on BOTH val and test across ALL fold windows. Fold 7 is the most important regime but the model must NOT have large drawdowns in other regimes.
- Training is EPOCH-BOUND (minimum 20 epochs with early stopping). NOT time-bound.
- **60-second cooldown after each experiment** to let the GPU/CPU cool. Use `sleep 60` between runs.
- ONE config change per experiment. Diagnose WHY before choosing what to change next.
- Report per-fold-window breakdown for BOTH val and test alongside aggregates.
- Dashboard shows train/val/test tabs for per-window breakdown. Test is the default view.
- Every config parameter must be wired end-to-end. Dead params are bugs — remove them.
- Every hyperparameter choice must be justified by published papers, model developer guidelines, or prior empirical results from this project. Never choose arbitrary values.

### Autoresearch Agent Protocol (Karpathy-adapted)
1. **Always start from the current best config.** Every experiment modifies ONE thing from the best. If it improves, it becomes the new best. If it doesn't, revert and try a different direction. Never wander off from the best baseline.
2. **If you see consecutive discards, stop and rethink.** Multiple failures mean your hypothesis about what to change is wrong. Re-read the per-window results. Look at which folds are weak and WHY. Don't keep guessing.
3. **Explore around the best AND try radical changes.** Most experiments should be small tweaks around the champion. But occasionally try something bold (different architecture, very different seq_len) to escape local optima.
4. **Cite your reasoning for every experiment.** "I'm trying X because fold Y has negative Sharpe due to Z, and paper W suggests this fix." Not "let me try X and see."
5. **The agent never stops.** If out of ideas, research deeper: read the LFM2 technical report, adapter papers, FX microstructure literature. Think harder. Try combining near-misses.
6. **Checkpoint reasoning to memory every few minutes.** The laptop crashes often. After every experiment (or every ~3 minutes of reasoning), save the current state to `memory/project_autoresearch_checkpoint.md`: what the current champion is, what was just tried, what the leading hypothesis is for the next experiment, and which folds are weak and why. On session start, read this checkpoint to recover full context without re-reading logs.
7. **Deep per-fold failure analysis every iteration.** For each fold with negative test Sharpe, explain WHY: what regime it is, what dates, what market conditions, what the uncertainty outputs reveal (high aleatoric = noisy data, high epistemic = model doesn't know, low confidence = skip signal). Use this to guide the next experiment.
8. **Code changes are allowed.** The agent may modify the Python codebase (model architecture, loss function, training loop, features, evaluation) if it has a principled reason. Save modified versions to `autoresearch/code_versions/` with a version number. Code changes are the most powerful lever — hyperparams only go so far.

### Research-Driven Experiment Selection (STRICT — no blind sweeps)
The experiment loop is NOT a grid search. It is a research process. Every single experiment must follow this exact sequence:

**Step 1 — Diagnose the champion's weakness.** Look at the per-fold test results. Which folds are weakest? What regime are they? What do the uncertainty metrics say? What does the win/loss spread look like for those folds? Identify the SPECIFIC failure mode (e.g., "fold 2 post-crash recovery has low IC=0.08, high epistemic uncertainty — model hasn't seen enough crisis-recovery data").

**Step 2 — Search the literature.** Based on the diagnosis, search arXiv / known papers for techniques that address the failure mode. Examples:
- Weak on volatile regimes → regime-aware training, volatility scaling (Kiraly et al. 2020)
- High epistemic in specific folds → data augmentation, ensemble methods (Lakshminarayanan et al. 2017)
- Overfitting to majority regime → focal loss (Lin et al. 2017), re-weighting
- Architecture ceiling hit → residual connections (He et al. 2016), attention mechanisms (Vaswani et al. 2017)
- LR too high/low → cyclical LR (Smith 2017), warmup (Goyal et al. 2017)

**Step 3 — Form a hypothesis and predict the outcome.** Write down: "I hypothesize that [change X] will improve [metric Y] on [fold Z] because [paper/principle]. I predict composite will move from [current] to approximately [target]." If you can't write this sentence, you don't understand what you're doing. Stop and think more.

**Step 4 — Run ONE experiment.** Execute the change. ONE change only.

**Step 5 — Analyze against prediction.** Did the result match your prediction? If yes, why? If no, what does that tell you about your mental model? Update your understanding.

**Step 6 — Document everything.** Write the full cycle (diagnosis → literature → hypothesis → prediction → result → learning) into the experiment log and checkpoint. This creates a research trail that prevents repeating failed ideas.

**The goal is monotonic improvement.** Every experiment should have a principled reason to believe it will improve composite score. Random guessing wastes GPU and time. If you're out of ideas for hyperparameters, the answer is almost always a CODE CHANGE — modify the architecture, loss function, or feature engineering.

### Monotonic Quality Progression (NEVER regress)
The experiment loop must work towards monotonic increase in quality. This means:
- **Never run an experiment you can't justify.** Every experiment must have a written rationale citing literature or prior empirical evidence from this project.
- **Track the champion lineage.** Document the chain: Exp1 (baseline) → Exp5 (residual skip, +3x) → Exp10 (LR bump, +1.2x) → etc. Each link must explain WHY the improvement happened.
- **When you hit a plateau, go deeper.** If 3+ consecutive experiments are DISCARD, you're in a local optimum. The answer is NOT more hyperparameter tweaks — it's a structural change: different architecture, different loss, different features, different training procedure.
- **Protect gains.** When trying bold changes, if the result is far worse (composite drops >2.0), investigate WHY before trying the next thing. Understanding failures is as valuable as finding improvements.
- **Quality ratchet:** once a metric improves, treat the new level as the floor. If a change improves test Sharpe but regresses val Sharpe below the previous champion, it's a DISCARD — both must improve or at least hold.

### MLOps Documentation Standards (MANDATORY)
You are a strong MLOps engineer. Every artifact and every experiment must be documented in proper, readable markdown. No exceptions.

**`autoresearch_results/experiment_summary.md`** — the master experiment log. Updated after EVERY experiment. Format:

```markdown
## Experiment Log — [Backbone] Phase

### Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected composite change]
- **Result:** Composite [X] | Test Sharpe [Y] | Val Sharpe [Z] | [N]/7 positive folds
- **Per-fold test Sharpe:** F1=[X] F2=[X] F3=[X] F4=[X] F5=[X] F6=[X] F7=[X]
- **Classification:** Precision=[X] Recall=[X] F1=[X] F2=[X] MCC=[X]
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned, why result matched/differed from prediction]
- **Win/Loss:** [summary — see per-trade spreadsheet in trade_logs/]
```

**`autoresearch_results/trade_logs/`** — per-experiment trade-level detail (see Trade-Level Win/Loss Logging below).

**Key documentation principles:**
1. **Readable by a human who wasn't there.** Someone reading the experiment summary 6 months from now must understand WHY each experiment was run and WHAT was learned.
2. **No orphan artifacts.** Every file must be referenced from either the checkpoint, experiment summary, or winner README.
3. **Consistent formatting.** Same table format, same metric names, same precision (4 decimal places for ratios, 2 for percentages).
4. **Append-only experiment log.** Never delete or rewrite experiment entries. If an experiment was wrong (e.g., bug found), add a note — don't erase history.

### Explainability & Auditability Report (MANDATORY for every NEW BEST)

When a new champion is found, produce a full data-scientist-grade audit to `autoresearch_results/winners/<exp_id>/audit_report.md`. This is not optional — a trading model without explainability is un-deployable.

**Required sections (all of them):**

1. **Executive summary** — Champion test Sharpe, return, max drawdown, PSR, all 7 fold Sharpes. Regime-by-regime pass/fail.

2. **Feature importance (permutation method)** — For each of the 104 features, shuffle that column in the test set, re-evaluate, report the drop in test Sharpe. Rank features by importance. Cite: Breiman (2001) "Random Forests" section on variable importance. Save `feature_importance.csv` with columns `[feature_name, sharpe_drop, rank, domain_category]`.

3. **Top-N feature analysis** — For the top 10 most-impactful features, explain:
   - What the feature measures (from features.py docs)
   - Why it matters economically (e.g., "VIX = equity volatility, negatively correlated with USD risk appetite")
   - Per-fold impact: is feature X strong in regime A but weak in regime B?

4. **SHAP-style local explanations** — For 10 random test-set predictions, compute per-feature contribution to the prediction. Use gradient * input as a cheap approximation. Save as `shap_local.csv`.

5. **Per-fold feature drift** — For each fold, compute mean/std of each feature vs the training set. Features with Z-score > 2 on a fold indicate distribution shift. Report top 5 drifted features per fold with explanation.

6. **Calibration analysis** — Plot predicted-return quantile vs realized-return mean. Ideal: monotonic. Report calibration error (mean absolute deviation from monotonic). Cite: Guo et al. (2017) "On Calibration of Modern Neural Networks."

7. **Uncertainty sanity** — Plot aleatoric vs prediction absolute error. Should be monotonic. Plot confidence vs hit-rate. Bucket predictions by confidence decile, report hit-rate per decile. Cite: Kendall & Gal (2017).

8. **Per-regime prediction distribution** — For each fold, plot histogram of predicted returns. Identify if the model is systematically biased (e.g., always predicting +0.01%) vs appropriately reactive.

9. **Trade attribution** — Decompose the cumulative return: for each test fold, report top-5 winning trades (date, pair, predicted, actual, P&L) and top-5 losers. Pattern analysis: are losses concentrated on specific dates/regimes?

10. **Risk audit** — Max drawdown period: which dates, what was the market doing, what features were the model reading. VaR-95, CVaR-95 per fold. Skewness, kurtosis of strategy returns.

11. **Data pipeline audit** — Reassert: zero train/val/test leakage, 90-day purge, 21-day embargo, 10-day label horizon buffer. Rerun `validate_purge_embargo()` and include the output verbatim. No assumptions — MEASURE.

12. **Model config complete dump** — Every hyperparameter + the Python version + torch version + numpy version + random seed. For true reproducibility.

13. **Known limitations & risks** — What regimes has this model NEVER been tested on? (e.g., hyperinflation, CB digital currencies, war shocks). Where will it most likely fail in live trading?

14. **Deployment checklist** — What monitoring is needed? What's the kill-switch criterion (max drawdown threshold, consecutive loss count)? What retraining cadence?

**Implementation:** Add `run_audit_report.py` that takes a `best_model.pt` path and produces the full report. Run it automatically when `composite > prev_best` in the runner.

### Winner Definition (CLARIFICATION)

**"Winner" means the GLOBAL champion across ALL backbones and ALL experiments.** Not per-backbone. The one single best model (by composite score) at any point in time.

Per-backbone best is tracked separately in the checkpoint but does NOT get archived to `winners/` unless it is also the global best.

When a new experiment beats the global composite:
1. Save artifacts to `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`
2. Include: README.md, config.json, model_checkpoint.pt, code/ (frozen snapshot), inference/, reproduction/, audit_report.md (14 sections per audit rules)
3. Update `best_config.json` at repo root

### Per-Backbone Code Snapshots (MANDATORY)

Before starting experiments on a new backbone, snapshot the CURRENT `model/backbone.py` and `model/train.py` to `code_versions/<backbone>_start/` so you can diff what changed during that backbone's exploration. This prevents mixing MLP-specific changes into LSTM exploration, etc.

```
code_versions/
  v1_original/                 # pre-any-change snapshot
  v2_residual_mlp/             # after residual skip connection (MLP champion)
  v3_residual_128h/            # MLP mid-session snapshot
  lstm_start/                  # snapshot before LSTM experiments begin
  patchtst_start/              # snapshot before PatchTST experiments begin
  ...
```

Rule: never modify `backbone.py` code specific to backbone X while experiments on backbone Y are in progress. Finish one backbone's 50 experiments, snapshot, then move on.

### Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)

**Every single experiment MUST have a complete reasoning record in `autoresearch_results/reasoning_annotations.json` keyed by `experiment_num`. No experiment ships without one. Orphan entries or "auto-backfilled" placeholders are a bug.**

The entry is a JSON object with these REQUIRED fields (all non-empty strings):

| Field | Content | Source |
|-------|---------|--------|
| `diagnosis` | Why THIS experiment now: which champion weakness it targets, which fold is weakest and why (regime, dates, uncertainty profile), what prior experiments ruled out the alternatives | Authored by Claude BEFORE running |
| `citations` | Full author/year/venue string for every paper motivating the choice (e.g. "Keskar et al. 2017 ICLR — On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima; He et al. 2016 CVPR (ResNet)"). Multiple papers semicolon-separated. Parenthetical-only tags (e.g. `(Keskar2017)`) are INSUFFICIENT — expand to full reference | Authored before running |
| `hypothesis` | Concrete mechanism: "parameter X = value Y will change metric Z via mechanism M (what the paper argues)". Not just "try X". | Authored before running |
| `prediction` | Numeric target: "composite should move from +6.37 to +6.40-6.50; val fold 2 expected to improve from -0.17 to +0.0-0.3". Include ranges, not single numbers | Authored before running |
| `verdict` | KEEP / DISCARD / NEAR-MISS + composite achieved + delta vs global best + which folds carried it | Written immediately after results |
| `learning` | What this result updates in the mental model: did the prediction hold? Which axis is now exhausted? Which variant should be tried next? | Written immediately after results |
| `_manual` | `true` if authored by Claude as part of the 7-step process (which is ALL non-trivial experiments); `false` only for purely mechanical variance-check runs that reuse a prior annotation template | Always set |

**Dashboard `dashboard.html` renders all 7 fields in the detail panel when a row is clicked.** If any field is missing, empty, or placeholder ("(auto-backfilled)", "(no explicit citation)"), that's a regression — fix it before the next experiment.

**Write cadence — two places on every run:**
1. **BEFORE the experiment command runs:** Claude adds the entry to `reasoning_annotations.json` with `diagnosis`, `citations`, `hypothesis`, `prediction`, `_manual: true`. The experiment is not launched until this entry exists. This enforces the "never guess, never grid-search" rule.
2. **AFTER the experiment completes:** Claude appends `verdict` and `learning` to the same entry by reading the runner's JSONL output. The runner's auto-written entry is only a fallback; Claude's post-analysis is authoritative.

**Enforcement:** At the start of every experiment cycle, Claude MUST check:
- Does `reasoning_annotations.json` already have a complete entry for the previous experiment? If no `verdict`/`learning`, write them before starting the next.
- Is the next experiment's pre-entry already authored? If no, write it now.
- Did the citation field survive any recent `backfill_reasoning.py` run? Check `_manual: true` is preserved.

**Parallel write to `research_journal.md`.** The same diagnosis/citations/hypothesis/prediction/verdict/learning narrative belongs in the research journal in markdown form, keyed by experiment number. Journal format:

```markdown
## Exp<N> — <short title>
**Diagnosis:** ...
**Citations:** ...
**Hypothesis:** ...
**Prediction:** ...
**Verdict:** ...
**Learning:** ...
```

The journal is the human-readable twin of the JSON; they must stay in sync. If they drift, the JSON is authoritative (runner-written), and the journal gets updated from it.

**`backfill_reasoning.py` rules:**
- Only runs on DEMAND — not automatically, not after every experiment
- Never overwrites entries with `_manual: true`
- Fills in only the fields that are empty AND whose experiment JSONL entry exists
- Logs every overwrite it makes
- Is NOT a substitute for authoring the annotation before the run

**Runner's responsibility (`run_autoresearch.py`):**
- On every invocation, merge the user-visible description's citation tags + the CLI flag delta into the runtime `reasoning_annotations.json` entry — WITHOUT clobbering `_manual: true` fields
- Populate `verdict` and `learning` from the results automatically as a fallback
- Never emit placeholder strings like "(auto-backfilled)"; if it can't compute a field, leave it blank and log a warning so Claude knows to author it

**Why this matters:** the dashboard is the shared memory between sessions. A new Claude Code session resuming this project reads the dashboard reasoning panel to understand why a champion was chosen. Missing or shallow annotations mean lost institutional knowledge and wasted experiments that retry dead-end ideas.

### Per-Backbone 50-Experiment Mandate (MANDATORY, not optional)

**Every backbone gets a full 50-experiment exploration.** Do not stop early because "axes look exhausted." The mandate:

1. **50 experiments per backbone** — no fewer. If standard HP sweeps plateau, explore:
   - Architectural variants from arXiv literature through 2026 (see per-backbone table below)
   - Cross-variant combinations (e.g., attention-LSTM × dropout tuning)
   - Feature engineering changes (input projections, feature selection)
   - Multi-seed studies on the champion to characterize variance
   - Regularization beyond weight decay (label smoothing, mixup, stochastic depth)

2. **Research latest SOTA (2024-2026 arXiv papers) before declaring any backbone done.** For each backbone category, the literature evolves yearly:
   - **LSTM/RNN**: xLSTM (Beck et al. 2024), Mamba (Gu & Dao 2024), Retentive Networks (Sun et al. 2023), DA-RNN with attention (Qin 2017), LayerNorm-LSTM (Ba 2016), AWD-LSTM (Merity 2018), GRU comparison (Cho 2014), stacked multi-layer (Graves 2013)
   - **Transformer TS**: PatchTST (Nie 2023), iTransformer (Liu 2024), TimesNet (Wu 2023), Informer (Zhou 2021), FEDformer (Zhou 2022), Crossformer (Zhang 2023), Autoformer (Wu 2021)
   - **MLP TS**: TSMixer (Chen 2023), N-HiTS (Challu 2023), N-BEATS (Oreshkin 2020), DLinear (Zeng 2023) — "Are Transformers Effective for TS?"
   - **Foundation**: TimesFM (Das 2024), Chronos (Ansari 2024), Moment (Goswami 2024), LFM2 (Liquid 2024)
   - **GBM**: XGBoost, LightGBM, CatBoost — tune n_estimators, max_depth, learning_rate, regularization

3. **Each experiment must cite its paper/source** — no "let me try X". Per CLAUDE.md rule 4.

4. **Document all 50 in research_journal.md** — even DISCARDs. Negative results are informative.

5. **Only after 50 experiments** may a backbone be declared "done" and progression to the next backbone resume.

### Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)

**Every backbone picks its OWN epochs, patience, learning rate, batch size, scheduler, and optimizer from the latest SOTA literature for THAT architecture. Never copy another backbone's config.** Defaults in `train.py` (ep=20, pat=5, lr=3e-4, bs=32, wd=1e-5) are starting points for MLP only — inherited values are bugs.

**Before the first experiment on any new backbone, Claude MUST:**

1. **Pull the latest 2024-2026 arXiv / NeurIPS / ICML / ICLR paper for the backbone family.** For each backbone, read the paper's experimental section and note:
   - Recommended epochs (and how they terminate — fixed vs early-stop)
   - Patience threshold (absolute vs relative to epochs — e.g. "10% of total")
   - Learning rate (and whether warmup is required — many 2024+ transformers need 5-10% warmup)
   - Scheduler (cosine annealing, linear decay, plateau, ReduceLROnPlateau — varies widely)
   - Optimizer (Adam vs AdamW vs Lion vs Adafactor vs SOAP — Lion/SOAP in 2024+ for large models)
   - Batch size (and whether it's effective-batch via grad accumulation)
   - Weight decay (AdamW uses decoupled; varies from 0 to 0.1 by architecture)
   - Gradient clipping (transformers usually clip to 1.0; RNNs to 0.25-1.0; GBMs N/A)
   - Loss function (MSE vs Huber vs Quantile vs Log-Cosh)

2. **Record the chosen recipe with a paper citation in the reasoning annotation** for Experiment 1 of that backbone. Other experiments in the backbone's 50-run cycle start from this config, not from MLP's or LSTM's.

3. **Justify the DELTA from the paper.** If our chosen epochs deviate from the paper's recommendation, the reasoning entry MUST explain why (e.g. "Nie 2023 used ep=100 on ETTh1 n=8640; we scale to ep=80 for our n=2738 — 3.15× less data, scale training proportionally per Smith 2017 rule").

4. **Never assume "ep=50 works for everything."** Historical proof:
   - MLP Exp32 champion converged at ep=50, pat=10 (Gu/Kelly/Xiu 2020)
   - LSTM Exp3 (ep=100, pat=15) beat LSTM Exp1 (ep=50, pat=10) by **+0.94 composite** (Fischer & Krauss 2018) — wrong epoch count costs 20% of peak performance
   - PatchTST Exp1 at our MLP defaults (seq=10, ep=20) gave composite **−1.72** because Nie 2023's minimum seq=60 and ep=100 were ignored
   - LFM2 head-only fine-tuning needs ep=20, pat=5 — LSTM's ep=100 would catastrophically overfit the adapter head

### Backbone-Specific Training Recipes (updated 2026-04-19 from SOTA literature)

Each row links to the paper Claude must re-read before starting the backbone. The config shown is the STARTING point for Experiment 1 — not a final answer. Iterate per the 7-step process.

| Backbone | Epochs | Patience | LR | Warmup | Scheduler | Batch | WD | Opt. | Loss | Paper (full citation) |
|----------|--------|----------|-----|--------|-----------|-------|-----|------|------|-----------------------|
#### Tier 1 — neural backbones (require from-scratch or fine-tune training)

| Backbone | Epochs | Patience | LR | Warmup | Scheduler | Batch | WD | Opt. | Loss | Paper (full citation) |
|----------|--------|----------|-----|--------|-----------|-------|-----|------|------|-----------------------|
| mlp | 50 | 10 | 3e-4 | 0 | cosine | 32 | 1e-5 | AdamW | Huber δ=1 | Gu, Kelly & Xiu 2020 RFS "Empirical Asset Pricing via Machine Learning" |
| lstm | 100 | 15 | 1e-3 | 0 | cosine | 16-32 | 7e-4 | AdamW | Huber δ=1 | Fischer & Krauss 2018 EJOR "Deep learning with LSTMs for financial market predictions" |
| ~~lfm2-350m~~ | ~~20~~ | ~~5~~ | ~~2e-5~~ | ~~1~~ | ~~linear~~ | ~~32~~ | ~~1e-6~~ | ~~AdamW~~ | ~~Huber~~ | SKIPPED per user 2026-04-19 |
| patchtst | 100 | 20 | 1e-4 | 10 | cosine | 32 | 1e-4 | AdamW | MSE | Nie, Nguyen, Sinthong, Kalagnanam 2023 ICLR "A Time Series is Worth 64 Words" (arXiv:2211.14730) — requires seq_len ≥ 60 |
| patchtsmixer | 100 | 15 | 1e-3 | 5 | cosine | 32 | 1e-5 | AdamW | MSE | Ekambaram, Jati, Nguyen, Sinthong, Kalagnanam 2023 KDD "TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series" (arXiv:2306.09364) |
| itransformer | 150 | 20 | 5e-5 | 10 | cosine | 32 | 0 | AdamW | MSE | Liu, Hu, Zhang, Wang, Wu, Wang, Long 2024 ICLR "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (arXiv:2310.06625) |
| xlstm | 80 | 15 | 5e-4 | 5 | cosine | 16 | 1e-3 | AdamW | Huber δ=1 | Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer, Brandstetter, Hochreiter 2024 NeurIPS "xLSTM: Extended Long Short-Term Memory" (arXiv:2405.04517) — exponential gating, matrix memory |
| mamba | 100 | 20 | 5e-4 | 10 | cosine | 32 | 0.1 | AdamW | MSE | Gu & Dao 2024 COLM "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752) |

#### Tier 2 — 10 NEW 2024-2026 SOTA backbones (add to runner before running)

| # | Backbone | Family | Epochs | Patience | LR | Warmup | Scheduler | Batch | WD | Opt. | Loss | Paper (full citation) |
|---|----------|--------|--------|----------|-----|--------|-----------|-------|-----|------|------|-----------------------|
| 1 | **timesfm** | Foundation (decoder-only, fine-tune) | 20 | 5 | 1e-4 | 2 | cosine | 32 | 1e-5 | AdamW | Quantile | Das, Kong, Sen, Zhou 2024 ICML "A Decoder-Only Foundation Model for Time-Series Forecasting" (arXiv:2310.10688); TimesFM 2.5 (Google 2025) — 500M params, 100B time-points pretrain, 2.5 adds continuous quantile heads & longer context |
| 2 | **chronos-bolt** | Foundation (T5-based encoder-decoder) | 15 | 5 | 5e-5 | 2 | cosine | 32 | 1e-5 | AdamW | CrossEnt (token) | Ansari, Stella, Turkmen, Zhang, Mercado, Shen, Shchur, Rangapuram, Pineda Arango, Kapoor, Zschiegner, Maddix, Mahoney, Torkkola, Wilson, Bohlke-Schneider, Wang 2024 TMLR "Chronos: Learning the Language of Time Series" (arXiv:2403.07815); Chronos-2 (arXiv:2510.15821, 2025) — univariate→universal, top benchmark score |
| 3 | **moirai** | Foundation (probabilistic encoder + MoE) | 20 | 5 | 1e-4 | 2 | cosine | 32 | 0 | AdamW | NLL (student-T mix) | Woo, Liu, Kumar, Xiong, Savarese, Sahoo 2024 ICML "Unified Training of Universal Time Series Forecasting Transformers" (arXiv:2402.02592); Moirai-MoE (arXiv:2410.10469); Moirai 2.0 (arXiv:2511.11698, 2025) — sparse MoE, 36M-series pretrain, multi-token prediction |
| 4 | **moment** | Foundation (T5 encoder, masked-ts pretraining) | 30 | 10 | 5e-5 | 3 | cosine | 32 | 1e-5 | AdamW | MSE | Goswami, Szafer, Choudhry, Cai, Li, Dubrawski 2024 ICML "MOMENT: A Family of Open Time-series Foundation Models" (arXiv:2402.03885) |
| 5 | **tirex** | Foundation (xLSTM-based, decoder) | 25 | 8 | 1e-4 | 3 | cosine | 16 | 1e-4 | AdamW | Quantile | Auer, Pöppel, Pflüger, Brandstetter, Hochreiter 2025 "TiRex: Zero-Shot Forecasting with Recurrent xLSTM Backbones" (NXAI/JKU 2025) — decoder-only xLSTM, strong short+long horizon zero-shot |
| 6 | **sundial** | Foundation (Transformer, continuous TimeFlow loss) | 30 | 10 | 1e-4 | 3 | cosine | 32 | 1e-5 | AdamW | TimeFlow (flow-matching on values) | Liu, Zhang, Wu, Long 2025 "Sundial: A Family of Highly Capable Time Series Foundation Models" (arXiv:2502.00816) — 1T time-points TimeBench pretrain, flow-matching loss |
| 7 | **time-moe** | Foundation (sparse MoE decoder) | 20 | 5 | 1e-4 | 2 | cosine | 32 | 1e-5 | AdamW | MSE + load-balance | Shi, Wang, Yang, Wang, Yang, Wang, Li, Li, Sun, Gao, Li 2024 ICLR '25 "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts" (arXiv:2409.16040) |
| 8 | **timemixer** | MLP-multiscale (from-scratch) | 100 | 15 | 1e-3 | 5 | cosine | 32 | 1e-5 | AdamW | MSE | Wang, Wu, Shi, Hu, Luo, Ma, Zhang, Zhou 2024 ICLR "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" (arXiv:2405.14616); TimeMixer++ 2024 follow-up — multi-scale decomposition, state-of-art on 8 tasks |
| 9 | **timesnet** | 2D-variation (CNN-inception) | 100 | 20 | 1e-4 | 5 | cosine | 32 | 1e-4 | AdamW | MSE | Wu, Hu, Liu, Ma, Long 2023 ICLR "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (arXiv:2210.02186) — reshape 1D→2D via period-FFT, Inception blocks |
| 10 | **mambats** | SSM (Mamba-based) | 100 | 20 | 1e-3 | 5 | cosine | 32 | 1e-4 | AdamW | MSE | Cai, Jiang, Wu, Zhang, Wang 2024 NeurIPS "MambaTS: Improved Selective State Space Models for Long-Term Time Series Forecasting" (arXiv:2405.16440); DMamba (arXiv:2602.09081 2025) variant with season-trend decomposition |

**Bonus Tier 2.5 candidates (add if budget allows):** DLinear/NLinear (Zeng et al. 2023 AAAI arXiv:2205.13504), N-HiTS (Challu et al. 2023 AAAI arXiv:2201.12886), TFT (Lim et al. 2021 IJF arXiv:1912.09363), Crossformer (Zhang & Yan 2023 ICLR), Autoformer (Wu et al. 2021 NeurIPS arXiv:2106.13008), N-BEATS (Oreshkin et al. 2020 ICLR arXiv:1905.10437), EMTSF ensemble (arXiv:2510.23396 2025). These are well-studied but less likely to beat Tier-2 foundation models at our n.

#### Tier 3 — gradient boosted machines (50 experiments TOTAL across the three, split 20/15/15)

**Per user instruction 2026-04-20: the *Boost family as a whole gets a 50-experiment budget, split by architectural distinctness rather than 50-each.** XGBoost receives the largest share because (a) it produced the global champion on Exp1 (+7.1686 composite), so the HP landscape around that champion is most valuable to map; (b) XGBoost's 2nd-order Newton boosting is the most-studied variant and has the richest HP surface (n_estimators × max_depth × lr × subsample × colsample × reg_alpha × reg_lambda × min_child_weight × gamma). LightGBM and CatBoost each get 15 experiments to cover their distinctive mechanisms without diluting the family-level exploration budget.

**Budget allocation:**

| Backbone | Exps | Focus |
|---|---:|---|
| **xgboost** | **20** | 1 baseline (done, champion +7.17) + 6 HP axes (depth, lr, subsample, colsample, reg_lambda, min_child_weight) + 5 variance seeds + 3 monotone-constraint experiments + 5 cross-axis refinements |
| **lightgbm** | **15** | 1 baseline + 4 HP axes (num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction) + 5 variance seeds + 5 GOSS vs standard sampling ablations |
| **catboost** | **15** | 1 baseline + 4 HP axes (iterations, depth, l2_leaf_reg, bootstrap_type) + 5 variance seeds + 5 ordered-boosting ablations |
| **TOTAL** | **50** | |

Each is its OWN backbone — run independently with its own 50-mandate share, its own winner archive, its own reasoning annotations. **Do NOT bundle xgboost/lightgbm/catboost as "the GBM backbone"** — they are three separate architectures with different splitting algorithms, different regularisation mechanisms, and different category handling.

#### Tier 3 recipe reference (each variant's starting-point SOTA config)

GBMs are fundamentally different from neural nets: no epochs, no LR schedule, no batch. Iterations are tree-count. Each GBM has its own paper, its own hyperparameter language, its own 50-experiment exploration budget. **Do NOT bundle xgboost/lightgbm/catboost as "the GBM backbone" — they are three separate architectures with different splitting algorithms, different regularization mechanisms, and different category handling.** Explore each fully.

| Backbone | Key HP | Default Start | Regularization | Special feature | Paper (full citation) |
|----------|--------|---------------|----------------|------------------|----------------------|
| **xgboost** | n_estimators=1500, max_depth=6, lr=0.03, subsample=0.8, colsample_bytree=0.8, early_stop=50 | level-wise trees | reg_lambda=1.0, reg_alpha=0, min_child_weight=1, gamma=0 | 2nd-order Newton boosting; monotonic constraints; histogram method | Chen & Guestrin 2016 KDD "XGBoost: A Scalable Tree Boosting System" (arXiv:1603.02754) |
| **lightgbm** | n_estimators=2000, num_leaves=63, lr=0.03, feature_fraction=0.8, bagging_fraction=0.8, early_stop=50 | leaf-wise trees (GOSS) | reg_alpha, reg_lambda, min_data_in_leaf=20 | Gradient-based One-Side Sampling; Exclusive Feature Bundling; categorical native support | Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017 NeurIPS "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" |
| **catboost** | iterations=2000, depth=6, lr=0.03, random_strength=1.0, early_stop=100 | symmetric oblivious trees | l2_leaf_reg=3, bagging_temperature=1.0 | Ordered boosting (prediction shift); native categorical via ordered target-stat | Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin 2018 NeurIPS "CatBoost: Unbiased Boosting with Categorical Features" (arXiv:1706.09516) |

**Why GBMs are 3 separate backbones:**
- **XGBoost** uses 2nd-order gradient info (Hessian) — effective on imbalanced targets, fast on GPU
- **LightGBM** uses leaf-wise growth + GOSS sampling — fastest wall-clock, handles large n well
- **CatBoost** uses ordered boosting to fight prediction shift + has the best default categorical handling — slowest but often best out-of-box accuracy on tabular

On our 104-feature SPY prediction, each will rank different features as important and each has different failure modes (LightGBM risk: leaf-wise overfit on small data; CatBoost risk: depth ceiling at 6 symmetric trees). Do not skip any.

**Re-derive for EVERY new 2024-2026 variant.** When a backbone family has multiple SOTA variants (e.g. LSTM family → xLSTM/sLSTM/mLSTM/Mamba; Transformer TS → PatchTST/iTransformer/Crossformer/Autoformer/FEDformer; Mamba family → MambaTS/DMamba/S-Mamba/CMMamba), each variant re-derives its recipe from its OWN paper. Don't assume xLSTM uses vanilla LSTM's ep=100, pat=15, or that MambaTS uses vanilla Mamba's ep=100.

### GPU Memory Constraint (MANDATORY — 16 GB VRAM hard cap)

**This laptop has 16 GB of GPU VRAM. Every backbone selection, every experiment, every fine-tuning run MUST fit within this budget with headroom. A model that OOMs mid-training is not a valid experiment — it's a wasted GPU cycle and a crash risk.**

**Memory budget breakdown (16 GB total):**

| Component | Budget | Notes |
|-----------|--------|-------|
| Model parameters | ≤ 3 GB | FP32 weights; BF16/FP16 halves this |
| Optimizer state (AdamW) | ≤ 6 GB | Adam stores 2 moments at FP32 even with BF16 weights → ≈ 2× param size |
| Gradients | ≤ 3 GB | Same size as params; freed after step |
| Activations | ≤ 3 GB | batch × seq × hidden, scales with bs and depth |
| Reserved / fragmentation | ≥ 1 GB | PyTorch caching allocator overhead |

**Practical parameter ceilings by training mode:**

| Training mode | Max params @ FP32 | Max params @ BF16/FP16 | Max params w/ grad-ckpt + BF16 |
|---------------|-------------------|------------------------|-------------------------------|
| From-scratch train (Adam full states) | ~500 M | ~1.0 B | ~2.0 B |
| Full fine-tune | ~500 M | ~1.0 B | ~2.0 B |
| Parameter-efficient FT (LoRA r=8, adapter-only) | ~1.0 B | ~3.0 B | ~5.0 B |
| Frozen-backbone head-only FT | ~1.5 B | ~4.0 B | ~7.0 B |
| Inference only (no grads) | ~4.0 B | ~8.0 B | ~8.0 B |

**Rules by backbone size class:**

1. **< 100 M params** — safe for anything. Use FP32 defaults. Most of our historical backbones (MLP, LSTM, PatchTST, TSMixer, iTransformer, xLSTM-small) are here.
2. **100 M – 500 M params** — FROM-SCRATCH TRAIN OK in FP32 at bs=32. Measure GPU use on Experiment 1; if > 12 GB, drop batch to 16 and/or switch to BF16. Applies to MOMENT-small, Chronos-T5-small/base, Moirai-small/base, TimesFM-base, Time-MoE-base.
3. **500 M – 2 B params** — FROM-SCRATCH not viable. Use: (a) parameter-efficient fine-tuning (LoRA/adapters), OR (b) frozen backbone + trainable head, OR (c) zero-shot inference then distil into a smaller student. Applies to TimesFM-2.5 (~500 M), Chronos-T5-large (700 M), MOMENT-large (385 M borderline), Moirai-large (311 M borderline), Sundial (exact size unknown — likely 500 M–1 B).
4. **> 2 B params** — INFERENCE ONLY. Use zero-shot forecasting, cache predictions, never train. Unlikely for our workflow.

**Mandatory pre-flight check for any new backbone:**

Before launching Experiment 1 on ANY new backbone, run this check (in reasoning annotation):

```
Measured/estimated size: N million params
Training mode selected: [from-scratch | LoRA fine-tune | head-only FT | zero-shot]
Expected peak VRAM: <X> GB at bs=<Y>, seq=<Z>, precision=<FP32|BF16>
Headroom vs 16 GB: <16 - X> GB
Fallback plan if OOM: [reduce bs to 16 | switch to BF16 | gradient checkpointing | adapter-only]
```

Without this entry, Experiment 1 does not launch. The same check applies any time we change batch size or sequence length during a backbone's 50-experiment cycle.

**Size-class annotations for the Tier-2 backbones (add to their first-experiment reasoning):**

| Backbone | Approx size | Training mode fit in 16 GB |
|----------|------------|-----------------------------|
| timesfm-200m (small) | 200 M | from-scratch fine-tune OK at BF16 |
| timesfm-2.5 (500 M) | 500 M | PEFT or head-only FT; full fine-tune risky |
| chronos-bolt-small | 9 M | trivially fits |
| chronos-bolt-base | 48 M | trivially fits |
| chronos-bolt-large | 205 M | fine-tune fits in FP32 |
| chronos-t5-large | 700 M | PEFT only |
| moirai-small/base | 14 M / 91 M | fits, from-scratch OK |
| moirai-large / moirai 2.0 | 311 M / ~500 M | fine-tune at BF16 |
| moment-small / base / large | 40 M / 125 M / 385 M | all fit; large at BF16 |
| tirex | ~300 M (est.) | fine-tune at BF16 |
| sundial | 500 M – 1 B (est.) | PEFT only |
| time-moe-base / large | 113 M / 453 M | fits; large at BF16 |
| timemixer / timesnet / mambats | < 50 M each | trivially fits |

**Default protocol when adopting a new foundation model:**

1. Start with the SMALLEST published checkpoint of that family (e.g. Chronos-Bolt-small, Moirai-small, MOMENT-small).
2. Run zero-shot first — measure composite without any training. Pay only inference cost.
3. If zero-shot is promising, fine-tune (full or PEFT depending on size).
4. Scale up to larger checkpoint ONLY if smaller shows signal AND the memory math works.

**BF16 note.** On our RTX-class GPU, BF16 is the safer mixed-precision choice vs FP16 — keeps dynamic range without loss-scaling. Use `torch.autocast(dtype=torch.bfloat16)` + `GradScaler` unset. Measure before/after; some ops (LayerNorm, GroupNorm) should stay FP32.

**Gradient checkpointing note.** Use `torch.utils.checkpoint.checkpoint_sequential` for any model > 200 M params that we're fine-tuning. Costs ~30% more FLOPs but cuts activation memory by 70-80%, unlocking bs=32 at 500 M-1 B params.

### Epoch-budget rule of thumb (when in doubt)

If the paper's recipe is unclear, use this scaling heuristic:

- **Data scaling (Smith 2017):** `epochs ≈ paper_epochs × (paper_n / our_n)^0.5`. Our n=2738; if paper used n=8000, scale paper_epochs × 0.59.
- **Parameter scaling (Kaplan 2020):** holding data fixed, larger models need more epochs. `epochs ≈ base × (our_params / paper_params)^0.2`.
- **Patience as 15% of epochs** is a safe default when papers don't specify.
- **Warmup = 5-10% of total epochs** for transformer families (required by layer-norm stability).

These are starting heuristics; always iterate and checkpoint the actual convergence profile per backbone.

### Empirical evidence (LSTM phase confirmations)

- LSTM Exp3 (ep=100 pat=15) beat Exp1 (ep=50 pat=10) by +0.94 composite — confirmed Fischer & Krauss 2018
- PatchTST at seq=10 gave -1.72 — confirmed Nie 2023's seq≥60 minimum
- MLP converged at ep=50 — Gu/Kelly/Xiu 2020 recipe validated
- Per-backbone convergence epochs (observed early-stop point): MLP ~25, LSTM ~29, PatchTST pending ~40-60 est.

### Backbone Isolation Rule

Before starting experiments on a new backbone, snapshot `model/backbone.py`, `model/train.py`, `run_autoresearch.py` to `code_versions/<backbone>_start/`. Do NOT modify backbone code specific to backbone X while experiments on backbone Y are in progress. Complete one backbone's 50-experiment cycle, snapshot as `<backbone>_final/`, then move to next backbone.

### Dashboard Backbone Tabs

Dashboard (`dashboard.html`) renders a backbone tab bar above the experiment list. Default view shows "ALL". Tabs filter the scrollable experiment list to just that backbone's experiments. Click to switch.

### GitHub Pages Dashboard Sync (MANDATORY — every push, zero exceptions)

**The live dashboard MUST be published to GitHub Pages on every commit that changes experiment state.** Hosted at:

> https://dlmastery.github.io/autoresearch/dashboard/

**Source of truth:** `autoresearch/autoresearch_results/dashboard.html` (+ its data files: `experiment_log.jsonl`, `best_config.json`, `reasoning_annotations.json`, and the `.md` report/journal/summary files the dashboard links to).

**Pages mirror:** `docs/dashboard/` — GitHub Pages serves the `docs/` folder; the dashboard's `dashboard.html` is copied to `docs/dashboard/index.html` so the URL `/dashboard/` routes directly to it.

**The sync step runs BEFORE every `git commit` that touches experiment state:**

```bash
python -m autoresearchspy._sync_dashboard_to_docs
# or equivalently
python autoresearch/_sync_dashboard_to_docs.py
```

The script copies:
- `autoresearch_results/dashboard.html` → `docs/dashboard/index.html`
- `autoresearch_results/experiment_log.jsonl` → `docs/dashboard/experiment_log.jsonl`
- `autoresearch_results/best_config.json` → `docs/dashboard/best_config.json`
- `autoresearch_results/reasoning_annotations.json` → `docs/dashboard/reasoning_annotations.json`
- Optional: `experiment_summary.md`, `autoresearch_report.md`, `research_journal.md`, `medium_article.md` if present

The sync script is idempotent; run it freely. It fails loudly if any required file is missing.

**When must you sync?**

- After every experiment that writes to the JSONL (effectively: every `run_autoresearch` call)
- After every reasoning-annotation edit
- After every winner archive
- Before every `git push` — the commit without the synced `docs/dashboard/` is a bug

**Enforcement:** a commit that changes `autoresearch/autoresearch_results/experiment_log.jsonl` but does NOT update `docs/dashboard/experiment_log.jsonl` is a regression. In practice this means the commit ritual is:

```bash
# 1. run experiments (runner auto-writes JSONL + trade logs + annotations)
# 2. edit reasoning annotations to full-rigor post-run verdict/learning
# 3. sync dashboard to docs/
python autoresearch/_sync_dashboard_to_docs.py
# 4. stage + commit
git add autoresearch/autoresearch_results docs/dashboard autoresearch/memory
git commit -F .commit_msg.txt
# 5. push (Pages rebuilds within ~30-60s)
git push origin master
```

**Verification:** after push, `curl https://dlmastery.github.io/autoresearch/dashboard/best_config.json` should show the latest champion within 2 minutes. If stale, check `git log -1 docs/dashboard/` — the commit that updated the pages folder must match the commit that updated the source.

**Why this matters:** the paper and Medium article both cite the live dashboard as the project's institutional memory. A stale dashboard makes the citation a lie. Treat the Pages mirror as a public artefact with the same freshness guarantees as the JSONL.

### Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)

**Every single experiment updates ALL the following files. If any file is stale after an experiment completes, that's a regression — stop and fix before moving on. No "I'll batch-update at the end." No "It's just a variance check."**

**Ownership — who writes what:**

| File | Written by | When | Content |
|------|------------|------|---------|
| `autoresearch_results/experiment_log.jsonl` | **runner (auto)** | every run, appended | full metrics: composite, test/val/train Sharpe, per-fold results, per-window classification metrics, uncertainty, timing, config |
| `autoresearch_results/best_config.json` | **runner (auto)** | only when new GLOBAL champion | overwritten with full champion entry |
| `autoresearch_results/best_model.pt` | **runner (auto)** | only when new GLOBAL champion | weights + scaler + config + feature_columns + provenance |
| `autoresearch_results/trade_logs/exp<N>_trades.csv` | **runner (auto)** | every run | one row per test-day trade (date, fold, regime, prediction, direction, returns, confidence, aleatoric, epistemic, pnl_bps) |
| `autoresearch_results/trade_logs/exp<N>_trade_summary.json` | **runner (auto)** | every run | per-fold totals, wins, losses, avg_win/loss bps, max win/loss, win_rate |
| `autoresearch_results/reasoning_annotations.json` | **Claude BEFORE run + runner AFTER run** | every run, two-phase | diagnosis, citations, hypothesis, prediction (Claude); verdict, learning (runner fallback, Claude overrides) |
| `autoresearch_results/research_journal.md` | **Claude** | every run, appended | markdown narrative of the full 7-step process (diagnosis → citations → hypothesis → prediction → verdict → learning) |
| `autoresearch_results/experiment_summary.md` | **Claude** | every run, appended | short tabular entry per experiment (config delta, result, per-fold Sharpe, status, learning) |
| `memory/project_autoresearch_checkpoint.md` | **Claude** | every run | update champion, update experiment history table, update next-command block |
| `autoresearch_results/winners/<backbone>_exp<N>_<desc>/*` | **Claude** | only when new GLOBAL champion | README.md, config.json, model_checkpoint.pt (copy), code/ snapshot, inference/predict.py, per_fold_results.json, experiment_log_entry.json |
| `autoresearch_results/winners/<backbone>_exp<N>_<desc>/audit_report.md` | **Claude** | only when new GLOBAL champion | 14-section audit per Explainability & Auditability Report spec |
| `autoresearch_results/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb` | **Claude** | only when new GLOBAL champion | self-contained Colab notebook |
| `autoresearch_results/dashboard.html` | **Claude (rarely)** | only when adding a new metric/tab | static HTML — reads the JSONL + annotations live |

**Per-experiment ritual (repeat in order, every single run):**

1. **Before launch:** open `reasoning_annotations.json`, insert a new entry keyed by the upcoming `experiment_num` with `diagnosis`, `citations` (full reference), `hypothesis`, `prediction` (numeric target), `_manual: true`. If this entry isn't there, the experiment doesn't run.
2. **Before launch:** append a matching section to `research_journal.md` with the same 4 fields in markdown.
3. **Launch:** run the CLI command.
4. **Runner auto-updates:** JSONL, best_config (if champion), best_model (if champion), trade_logs CSV + JSON, reasoning_annotations verdict/learning fallback.
5. **After completion:** Claude reads the runner output, overwrites the `verdict` and `learning` fields in `reasoning_annotations.json` with richer analysis (per-fold narrative, which regimes won/lost, uncertainty profile). Updates the corresponding section in `research_journal.md`.
6. **After completion:** Claude appends a row to `experiment_summary.md`.
7. **After completion:** Claude updates `memory/project_autoresearch_checkpoint.md` with the new experiment in the history table, updated champion (if applicable), and the exact next-experiment command.
8. **If new champion:** Claude archives to `winners/<backbone>_exp<N>_<desc>/` — README, config copy, model copy, frozen code snapshot, inference predict.py, per-fold results, audit_report.md, Colab notebook. The archive must be self-contained.

**Verification at the start of every experiment cycle:**

Before launching Experiment N+1, confirm all of these are CURRENT for Experiment N:

- [ ] `experiment_log.jsonl` has an entry for N (runner writes, verify)
- [ ] `reasoning_annotations.json[N]` has all 7 fields non-empty and non-placeholder
- [ ] `research_journal.md` has a section for N
- [ ] `experiment_summary.md` has a row for N
- [ ] `memory/project_autoresearch_checkpoint.md` references N in its history table
- [ ] `trade_logs/expN_trades.csv` and `expN_trade_summary.json` exist
- [ ] If N set a new champion: `winners/<backbone>_expN_<desc>/` exists with all required files

If ANY checkbox is unchecked, stop and fix BEFORE launching N+1. This is how we keep the dashboard as authoritative, up-to-date institutional memory.

**Placeholder strings are a bug.** The runner refuses to fabricate pre-run content. If a pre-run entry is missing, the runner inserts `"TODO-REWRITE"` sentinel values and a `_needs_rewrite: true` flag — Claude MUST rewrite those entries before launching the next experiment. Fix the process, not the string.

### Citation Rigor (MANDATORY format for `citations` field)

**Every citation string MUST contain, for every paper referenced:**

1. **All authors' surnames** (not just first-author et al. unless > 6 authors)
2. **Year** of publication
3. **Venue** — journal name, conference abbreviation (NeurIPS, ICML, ICLR, AAAI, CVPR, KDD, etc.), or `arXiv` if preprint-only
4. **Full paper title** in single quotes
5. **arXiv ID** in the form `(arXiv:XXXX.YYYYY)` if available — mandatory for any paper posted to arXiv
6. **One-sentence relevance note** — why this paper motivates THIS experiment specifically

**Format template:**

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) — one-sentence note on why we cite it here.
```

**Multiple papers separated by semicolons + linebreak.** Minimum one primary citation per experiment; secondary citations encouraged when the experiment combines ideas from multiple papers.

**Examples of GOOD citations (copy this style):**

> Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 ICLR 'On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima' (arXiv:1609.04836) — motivates bs=16 as a flat-minima probe.

> Loshchilov & Hutter 2019 ICLR 'Decoupled Weight Decay Regularization' (arXiv:1711.05101) — AdamW wd acts as decoupled weight shrinkage, so perturbations must be log-scale.

> Nie, Nguyen, Sinthong, Kalagnanam 2023 ICLR 'A Time Series is Worth 64 Words: Long-term Forecasting with Transformers' (arXiv:2211.14730) — requires seq_len ≥ 60 for attention heads to have enough patches.

**Examples of BAD citations (REJECTED — rewrite required):**

- `"Keskar 2017 flat minima"` — missing coauthors, venue, title, arXiv, relevance note
- `"(Keskar2017)"` — parenthetical tag only, useless
- `"Keskar et al."` — no year, no venue
- `"arxiv paper on batch size"` — no attribution
- `"(no citation tag)"` — confesses the author didn't do the work
- `"see research_journal.md"` — redirects instead of citing

**The goal:** anyone (including a future Claude Code session with zero project context) must be able to open the dashboard, click a row, read the `citations` field, and immediately know which paper to read and why. Citations are institutional memory.

**Arxiv ID lookup discipline.** If you know the paper but not its arXiv ID, fetch it via WebSearch or WebFetch (arxiv.org/abs search) before writing the entry. Authoring a citation without the arXiv ID is a partial job.

### Reasoning Blob Completeness (what "full reasoning" means)

Each of the 7 fields in `reasoning_annotations.json` has a minimum content spec. Entries that fall below this spec must be rewritten. Use these as acceptance criteria before an experiment is considered "documented":

| Field | Minimum content | Word count floor | Must include |
|-------|-----------------|------------------|--------------|
| `diagnosis` | Why THIS experiment NOW; which champion weakness; which fold is worst and why (regime name, date range, uncertainty signature); what prior experiments ruled out | ≥ 60 words | Reference to at least one prior experiment by number OR a per-fold metric from the current champion |
| `citations` | Per the Citation Rigor spec above | ≥ 40 words for single paper, ≥ 80 for multi-paper | Author list + year + venue + title + arXiv ID + relevance note for each paper |
| `hypothesis` | The config change stated mechanistically — what parameter(s) move, what they do in the model, what the cited paper predicts will happen | ≥ 50 words | The word "mechanism" or "because" or "per [paper]"; the specific parameter and value |
| `prediction` | Concrete numeric range on composite AND at least one fold-level or uncertainty-level sub-prediction | ≥ 25 words | A numeric range (e.g. "+6.30 to +6.50"); a direction for at least one sub-metric |
| `verdict` | KEEP/DISCARD/NEAR-MISS + exact composite + delta vs global best + per-fold narrative (which folds carried or killed it) | ≥ 30 words | Status label; composite to 4 decimals; mention of at least one per-fold result |
| `learning` | What this updates in the mental model; which axis is now closed/open; what to try next | ≥ 40 words | "Axis closed" / "axis open" language OR a concrete "next try: ..." |
| `_manual` | Boolean | — | `true` if Claude-authored (expected for non-variance experiments); `false` only for mechanical reruns |

**When running a batch of variance checks** (same config, varying seed), the `_manual: true` entries can share templated `diagnosis` and `citations` across runs, but `verdict`/`learning` must always be per-run-specific (different seed → different fold outcomes).

**Batch updates are forbidden.** Don't do 5 experiments then update the journal/summary/checkpoint in one go — each experiment's state gets stale and crash-recovery breaks. Update everything, then move on.

### Heteroscedastic Loss Rules (Kendall & Gal 2017)
- The model outputs mean + log_variance per prediction. Loss = `exp(-s) * huber(mu, y) + 0.5 * s`.
- **Variance-branch dominance is the #1 failure mode.** If aleatoric > 0.2, the model is copping out to high variance instead of learning signal. Fix: higher LR, more epochs, or clamp log_var.
- **Optimal aleatoric range: 0.05-0.15.** Below 0.05 = overconfident. Above 0.20 = lazy variance.
- **The het-loss needs ~50% more epochs than plain Huber** to converge, because the variance branch adds an optimization axis. Champion with plain Huber: 20 epochs. Champion with het-loss: 30 epochs.
- **LR sweet spot shifted up:** Plain Huber champion was lr=2e-5. Het-loss champion is lr=3e-5. The exp(-s) weighting reduces effective gradient on mean, so higher base LR compensates.
- **Monitor uncertainty per fold:** High aleatoric on a fold means the model correctly identifies it as noisy. High epistemic means the model needs more data from that regime. Use confidence < 0.8 as a "don't trade" signal.

### Winner Archiving Protocol (MANDATORY for every NEW BEST)
Every time a new champion is found (status=KEEP and composite > previous best), archive ALL artifacts to a self-contained subdirectory. The archive must be fully portable — someone can copy the directory to another machine and reproduce + run inference without any external dependencies beyond the conda environment.

**Directory structure:** `autoresearch_results/winners/<backbone>_exp<N>_<short_description>/`

```
winners/
  mlp_exp32_residual_seed0/
    README.md                    # Full description (see template below)
    config.json                  # Exact config that produced this winner
    model_checkpoint.pt          # Saved model weights (copy of best_model.pt)
    experiment_log_entry.json    # The JSONL entry for this experiment
    per_fold_results.json        # Full per-fold val + test breakdown
    code/                        # Frozen snapshot of ALL source code at time of win
      backbone.py
      train.py
      features.py
      splits.py
      metrics.py
      run_autoresearch.py
    inference/
      predict.py                 # Standalone inference script with sample usage
      README_inference.md        # How to load model and run predictions
    reproduction/
      reproduce_log.txt          # Output from reproduction run
      seed_variance.json         # Cross-seed results if available
```

**README.md template for each winner:**
- Model name + experiment number
- Champion composite score, test Sharpe, val Sharpe
- Per-fold test Sharpe table (all 7 folds)
- Per-fold val Sharpe table
- Full hyperparameter config
- Architecture description (layers, activation, skip connections, etc.)
- Key insight: WHY this config won (what change from previous champion)
- Training details: epochs run, early stopping epoch, training time
- Uncertainty metrics: aleatoric, epistemic, confidence per fold
- Traditional ML metrics: precision, recall, F1, F2 (direction classification)
- Reproduction status: seeds tested, variance observed
- Sample inference code snippet

**After archiving:** Rerun the winner to verify reproduction. The reproduction log goes into `reproduction/reproduce_log.txt`. If the reproduction fails (composite differs by >0.5), flag it and investigate before proceeding.

**Model checkpoint (`model_checkpoint.pt`) MUST be portable and self-contained:**
Include in the torch.save dict:
- `model_state_dict` — all trainable weights
- `config` — hyperparameters dict (matches the `--seed` run command)
- `scaler_mean`, `scaler_scale` — StandardScaler parameters (np.ndarray[n_features])
- `feature_columns` — list of feature names in order (for schema validation at inference)
- `target_columns` — list of target names (e.g. `['ret_1d', 'ret_5d']`)
- `n_features` — int, feature count
- `composite`, `description`, `backbone`, `experiment_num` — provenance

The checkpoint must be loadable and reusable WITHOUT the source repo. Someone can rebuild the model, apply the scaler, and make predictions from the checkpoint alone + the architecture definition.

**The `predict.py` inference script must:**
1. Load the model checkpoint
2. Accept raw feature input (or date range to download)
3. Output: prediction (mean), confidence, aleatoric uncertainty, epistemic uncertainty
4. Include a `__main__` block with a working example
5. Print results in a clear table format

**Trading Strategy section (MANDATORY in every winner README.md):**
Must include the following for any user to deploy the model:
1. **Signal Generation** — inputs, outputs, MC Dropout usage
2. **Entry rules** — pseudocode with thresholds (magnitude + confidence)
3. **Position sizing** — Kelly fraction, per-trade cap
4. **Exit rules** — horizon matching, stop-loss policy
5. **Rebalancing cadence** — daily/intraday/weekly
6. **Per-regime performance table** — accuracy/MCC/Sharpe per fold
7. **Risk controls** — daily loss cap, drawdown pause, regime shift detection
8. **Expected performance** — Sharpe, return, drawdown estimates (pre/post cost)
9. **Caveats and warnings** — seed variance, pair specificity, feature dependencies, transaction costs
10. **Reference to inference code** — link to `inference/predict.py`

### Google Colab Notebook (MANDATORY for every winner)
For every archived winner, generate a self-contained Google Colab notebook at `autoresearch_results/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb` that anyone can open in Colab and run end-to-end.

**The Colab notebook must contain:**
1. **Setup cell:** `!pip install` all dependencies, clone repo or upload weights
2. **Data download cell:** download FX + macro data using `download.py` logic (or bundled CSV)
3. **Feature engineering cell:** compute all 104 features with clear explanations
4. **Training cell:** full training loop reproducing the winner config exactly — including super-fold split, contiguous datasets, loss function, optimizer, early stopping. Print per-epoch loss + validation metrics.
5. **Evaluation cell:** evaluate on all 7 test fold windows, print per-fold Sharpe/IC/win-rate table, compute composite score
6. **Inference cell:** load trained model, accept a date range, produce predictions with confidence/aleatoric/epistemic bands. Show a sample prediction table.
7. **Visualization cell:** plot equity curves per fold, prediction vs actual scatter, uncertainty calibration, confusion matrix
8. **Export cell:** save model weights + config for deployment

**Notebook principles:**
- Every cell must have a markdown header explaining what it does and WHY
- Include the champion config as a clearly visible dict at the top
- Use `torch.manual_seed()` for reproducibility
- Print all key metrics at the end in a summary table
- Target runtime: <5 minutes on Colab free tier (T4 GPU or CPU)
- The notebook must be SELF-CONTAINED — no imports from the autoresearch package (inline all necessary code)

### Traditional ML Metrics (MANDATORY for every experiment)
In addition to financial metrics (Sharpe, Sortino, IC, etc.), compute and log direction-classification metrics for every experiment. The trading strategy uses `sign(prediction)` as the directional bet, so treat direction prediction as binary classification:
- **Positive class:** model predicts UP (pred > 0) and actual move is UP (actual > 0)
- **Negative class:** model predicts DOWN (pred < 0) and actual move is DOWN (actual < 0)

Metrics to compute per fold AND aggregate:
- **Precision:** TP / (TP + FP) — of all UP predictions, how many were correct
- **Recall:** TP / (TP + FN) — of all actual UP moves, how many did we catch
- **F1 Score:** harmonic mean of precision and recall
- **F2 Score:** weighted harmonic mean favoring recall (beta=2), useful for FX where missing a move costs more than a false signal
- **Accuracy:** (TP + TN) / total — same as hit rate / win rate but explicit
- **Matthews Correlation Coefficient (MCC):** balanced measure even with class imbalance
- **Confusion matrix counts:** TP, FP, TN, FN per fold

These must appear in:
1. `trading_report()` output in `metrics.py`
2. Per-window results in JSONL log entries
3. Dashboard per-window tables
4. Winner archive `per_fold_results.json`
5. Experiment summary markdown

### Trade-Level Win/Loss Logging (MANDATORY for every experiment)
For EVERY experiment, produce a per-trade win/loss spreadsheet on test data. This is critical for understanding WHERE the model makes and loses money — not just aggregate metrics.

**Output file:** `autoresearch_results/trade_logs/exp<N>_trades.csv`

**Columns (one row per trade/day in test data):**
| Column | Description |
|--------|-------------|
| date | Trade date |
| fold | Which test fold window (1-7) |
| regime | Regime label (e.g., "Post-crash recovery") |
| prediction | Raw model prediction (mean) |
| pred_direction | +1 (long) or -1 (short) |
| actual_return | Actual daily return |
| actual_direction | +1 (up) or -1 (down) |
| strategy_return | sign(pred) * actual_return |
| cumulative_return | Running cumulative return within fold |
| confidence | Model confidence (1 - epistemic) |
| aleatoric | Aleatoric uncertainty |
| epistemic | Epistemic uncertainty |
| correct | 1 if pred_direction == actual_direction, else 0 |
| pnl_bps | P&L in basis points |

**Per-fold summary at bottom of CSV (or separate `exp<N>_trade_summary.json`):**
- Total trades, wins, losses per fold
- Average win size (bps), average loss size (bps)
- Largest single win, largest single loss
- Win/loss ratio (avg_win / abs(avg_loss))
- Streak analysis: max consecutive wins, max consecutive losses
- Confidence-stratified accuracy: accuracy when confidence > 0.9 vs < 0.9

**This data enables:**
- Identifying specific dates/regimes where the model fails
- Confidence calibration analysis (does high confidence = high accuracy?)
- Position sizing research (Kelly criterion, volatility scaling)
- Filtering rules (skip trades below confidence threshold)

### Architecture
- **Autoresearch loop = Claude agent.** Claude reads results, decides what to try, calls the runner, reads output. The intelligence is in the agent, NOT in Python code. No pre-baked experiment lists.
- Runner (`run_autoresearch.py`) executes ONE experiment per call. Logs JSONL. That's it.
- Dashboard (`dashboard.html`) reads logs. DECOUPLED from runner.
- Save checkpoint after every experiment (JSONL append + best_config.json overwrite).
- Use relative imports (`from .model.backbone import ...`).

### Validation Checklist (Run Before Every Experiment Session)
1. `validate_purge_embargo()` passes — 0 violations
2. `split_superfold()` returns correct counts — train=3113, val=915, test=1170
3. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0
4. `create_contiguous_datasets()` produces expected segment count (7 for training, 7 for val)
5. Each test window processed individually has enough rows (>= seq_len + 1)
6. Data loaded from `.data_cache/` (not re-downloaded)

## Project Structure

```
autoresearch/                    # package root
  baseline.py                    # single-backbone walk-forward evaluation
  run_ablation.py                # multi-backbone comparison
  run_autoresearch.py            # Karpathy-style autonomous experiment loop (LOGS ONLY)
  data/
    download.py                  # FX + macro data (cached to .data_cache/)
    features.py                  # 104 backward-looking features
    splits.py                    # folds, purge/embargo, hole-punching, split_superfold()
  model/
    backbone.py                  # 8 backbones, per-backbone seq_len via get_seq_len()
    train.py                     # training loop, create_contiguous_datasets()
  evaluation/
    metrics.py                   # Sharpe, PSR, DSR, IC, trading_report + precision/recall/F1/F2/MCC
  autoresearch_results/
    experiment_log.jsonl          # structured experiment log (append-only)
    best_config.json             # current best configuration
    dashboard.html               # live HTML dashboard (reads logs, decoupled)
    experiment_summary.md        # master human-readable experiment log (updated every experiment)
    trade_logs/                  # per-trade win/loss CSVs for every experiment
      exp<N>_trades.csv          # one row per trade on test data
      exp<N>_trade_summary.json  # per-fold trade statistics
    winners/                     # archived champions (one subdir per winner, fully self-contained)
      <backbone>_exp<N>_<desc>/  # e.g. mlp_exp32_residual_seed0/
        README.md                # full description, metrics, reproduction status
        config.json              # exact config
        model_checkpoint.pt      # saved weights
        code/                    # frozen source snapshot
        inference/               # predict.py + inference README
        reproduction/            # reproduction logs + seed variance
```

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| SEQ_LEN (LFM2) | 60 | backbone.py `BACKBONE_SEQ_LEN` |
| SEQ_LEN (others) | 10 | backbone.py `_DEFAULT_SEQ_LEN` |
| PURGE_DAYS | 90 | splits.py |
| EMBARGO_DAYS | 21 | splits.py |
| LABEL_HORIZON_BUFFER | 10 | splits.py |
| LEARNING_RATE | 3e-4 | train.py |
| BATCH_SIZE | 32 | train.py |
| EPOCHS | 20 | train.py |
| PATIENCE | 5 | train.py |
| WEIGHT_DECAY | 1e-5 | train.py |

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Sliding windows across date gaps | ~41% garbage windows, meaningless predictions | `create_contiguous_datasets()` for train/val, `_evaluate_per_window()` for test |
| Expanding window without hole-punching | Cross-fold contamination, inflated Sharpe | `split_data()` punches ALL val/test from ALL folds |
| Dead config params (dropout, huber_delta) | Experiments with no effect, wasted GPU | Wire every param end-to-end or remove it |
| Data re-downloading every run | Minutes wasted, flaky network dependency | Default `cache_dir=.data_cache/` in download.py |
| Grid sweep instead of diagnostic | Uninformed, 10x more experiments than needed | One change at a time, diagnose results first |
| Running all 7 folds per experiment | 7x slower, unnecessary | Super-fold: one train, one eval pass |
| Absolute imports in package | `ModuleNotFoundError` when run as `-m` | Always `from .module import ...` |
| Assuming timing/performance | Wrong estimates, wrong priorities | Measure with `time.time()`, log elapsed |
| Monolithic scripts | Can't debug, can't reuse, can't monitor | Runners log. Dashboard reads. Decoupled. |
| `--learning-rate` flag | argparse expects `--lr` only | Use `--lr` in every runner command |
| `huber_delta` > 1.0 | Residuals are ~5e-3, never cross the Huber kink | Any value ≥ 1 is equivalent — treat Huber as MSE at our scale |
| Fine-grained AdamW `wd` < 30% change | AdamW decouples wd from grads; tiny changes are no-ops | Use log-spaced sweeps (1e-4, 5e-4, 1e-3, 5e-3) not 7e-4 vs 8e-4 |
| Smaller batch without seed plan | bs=16 improves mean-case but **doubles** seed std vs bs=32 | When trying bs<32, always multi-seed before declaring champion |
| Blaming model when problem is regime | Folds 1 & 2 are genuinely hard (GFC-onset / post-crash) across all backbones | Don't chase fold-2 perfection; aim for ≥ 0 with acceptable std |

## Session Learnings (LSTM Phase, Exps 1-44 of 50)

Document what the LSTM phase taught us so future backbones don't repeat dead ends. This section is append-only — add but don't delete.

### Confirmed optimal LSTM hyperparameters (at n=2738 daily FX samples)
- `hidden_size=128` — 96 underfits, 256 overfits (both reproduced)
- `num_layers=2` bidirectional — 1-layer loses recent regime; 3-layer overfits dramatically (+1.64)
- `cell=lstm` — GRU 2-gate underperforms at n<3k
- `seq_len=10` — 5, 8, 12, 20 all worse (either too noisy or too few training windows)
- `lr=1e-3` — 5e-4 finds flat val minima that hurt test; 1.5e-3 diverges fold 2
- `bs=16` — champion; bs=32 safer but lower peak; bs=8 destabilizes fold 2
- `head_dropout=0.25` — 0.20, 0.22, 0.30 all worse
- `weight_decay=7e-4` — peaked here; 5e-4, 1e-3, 2e-3 within ±0.2 composite
- `grad_clip=1.0` — 0.5 and 1.5 and 2.0 all worse
- `huber_delta=1.0` — value irrelevant (always MSE at our residual scale)
- `epochs=100` + `patience=15` — consistently early-stops at 25-30

### Axes that DID NOT help
- LayerNorm input (double-normalizes standardized features)
- Unidirectional LSTM (loses test context; wins val)
- Heteroscedastic loss (helps fold 2 specifically but hurts fold 1; could be ensemble component)
- Learning rate warmup (1-5 epoch warmup all hurt)
- GRU replacement
- Stacked 3-layer

### Seed variance is LARGE and backbone-specific
| Config | Seeds tried | Mean | Std | Max-Min |
|--------|-------------|------|-----|---------|
| wd=1e-3 bs=32 | 0, 42, 99, 7 | 5.99 | 0.52 | 1.22 |
| wd=7e-4 bs=16 | 42, 0, 99, 2024 | ≈ 5.53 | ≈ 0.96 | ≈ 2.18 |

**Implication:** Single-seed "champions" are often luck. Deploy via seed-ensemble (≥5 seeds, average predictions). Declare champion only after 3-seed median > baseline median.

### Key protocol additions

**1. Always use `--lr` (not `--learning-rate`).** Common mistake; wasted runs.

**2. Between-experiment cooldown.** 60s `Start-Sleep` blocked by sandbox — the CPU pin-mitigations suffice; skip cooldown during automated sweeps.

**3. `_manual=True` preserves curated reasoning.** When running backfill_reasoning.py after a batch, the `_manual` flag on hand-written annotations (diagnoses, citations, predictions) prevents overwrite.

**4. Archive winner on EVERY new global champion immediately.** Don't batch. File layout per `Winner Archiving Protocol`. The archive must be standalone: README + config.json + model_checkpoint.pt + code/ snapshot + inference/predict.py.

**5. Seed-ensemble for deployment.** `inference/predict.py` should accept a list of checkpoint paths and average predictions. This is the single most-effective variance reducer for LSTM at this n.

### Next-backbone priorities
- **PatchTST**: seq_len needs ≥ 60 (patch_length=5 × 12+ tokens) for attention to shine — don't use seq=10
- **PatchTSMixer**: TSMixer-style channel-and-time mixing; good for seq_len 30-60
- **iTransformer (Liu et al. 2024)**: invert attention over variates — promising for 104 features
- **xLSTM / mLSTM / sLSTM (Beck et al. 2024)**: exponential gating + matrix memory — code change needed
- **Mamba (Gu & Dao 2024)**: SSM replacement, needs selective scan kernels
- **GBM trio (XGBoost/LightGBM/CatBoost)**: tabular-style features; flatten windows to feature vectors; expect strong baseline but low ceiling

### Checkpoint + packaging cadence
- After **every** experiment: update `memory/project_autoresearch_checkpoint.md` AND `autoresearch_results/experiment_summary.md`
- After **every** session end or user-requested package: zip `autoresearch_results/` + `memory/` + `code_versions/` + frozen `model/` + `data/` + `evaluation/` + `run_autoresearch.py` + CLAUDE.md. Exclude `.data_cache/` (large, reproducible), `__pycache__/`, `.git/`.

