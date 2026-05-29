"""Compute formal statistical tests addressing area-chair reviewer pass items.

Reads:  experiments/cifar10/<tag>_seed<s>/metrics.json
        experiments/cifar100/<tag>_seed<s>/metrics.json

Outputs: structured report on stdout, intended to be folded into
         STATISTICAL_TESTS.md at the repo root.

Tests:
  1. Paired Wilcoxon signed-rank on per-seed top-1 deltas (n=3).
  2. One-sided sign test (the project's Phase-5 ordinal gate, α=(1/2)^3=0.125).
  3. Bootstrap 95% pivotal CI on top-1 difference (10000 resamples).
  4. Per-seed reproducibility std for leader and baseline.
  5. Holm-Bonferroni adjustment of family-wise α=0.05.
  6. CIFAR-10 single-seed Δ distribution (35-row screening sweep).
  7. CIFAR-10 3-seed coverage tags: empirical seed-noise std.

This script is read-only with respect to metrics.json.
"""

from __future__ import annotations

import io
import json
import math
import pathlib
import statistics
import sys
from typing import Iterable

import numpy as np
from scipy import stats as sps

# Force UTF-8 stdout on Windows so we can print Greek letters / unicode.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


REPO = pathlib.Path(__file__).resolve().parent.parent
CIFAR10 = REPO / "experiments" / "cifar10"
CIFAR100 = REPO / "experiments" / "cifar100"

RNG = np.random.default_rng(20260529)


def load_metric(path: pathlib.Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def get_seed_top1s(root: pathlib.Path, tag: str, seeds: Iterable[int] = (0, 1, 2)) -> list[float]:
    out: list[float] = []
    for s in seeds:
        m = load_metric(root / f"{tag}_seed{s}" / "metrics.json")
        if m is not None and "top1" in m:
            out.append(float(m["top1"]))
    return out


def bootstrap_ci_diff(
    leader: list[float], baseline: list[float], n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Naive pivotal bootstrap 95% CI on (mean leader) - (mean baseline).

    With n=3 per arm, this is genuinely a low-power estimator; reporting it
    explicitly is the point of the reviewer correction.
    """
    leader_a = np.asarray(leader, dtype=float)
    baseline_a = np.asarray(baseline, dtype=float)
    obs_diff = leader_a.mean() - baseline_a.mean()
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        l_resample = RNG.choice(leader_a, size=leader_a.size, replace=True)
        b_resample = RNG.choice(baseline_a, size=baseline_a.size, replace=True)
        diffs[i] = l_resample.mean() - b_resample.mean()
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return obs_diff, lo, hi


def paired_wilcoxon(leader: list[float], baseline: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test on n=3 seed-matched deltas.

    With n=3 the exact null distribution has 2^3 = 8 sign patterns. The
    smallest one-sided p achievable is 1/8 = 0.125 (all 3 deltas positive).
    The two-sided minimum is 0.25.
    Scipy's exact method handles this correctly; we report what it returns
    AND the theoretical floor.
    """
    if len(leader) != len(baseline) or len(leader) == 0:
        return {"W": float("nan"), "p_two": float("nan"), "p_one": float("nan"), "n": 0}
    diffs = np.array(leader) - np.array(baseline)
    n_nonzero = int(np.count_nonzero(diffs))
    if n_nonzero == 0:
        return {"W": 0.0, "p_two": 1.0, "p_one": 0.5, "n": len(diffs), "n_nonzero": 0, "diffs": diffs.tolist()}
    # Suppress scipy small-sample warning by using exact method.
    res_two = sps.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox", method="exact")
    res_one = sps.wilcoxon(diffs, alternative="greater", zero_method="wilcox", method="exact")
    return {
        "W": float(res_two.statistic),
        "p_two": float(res_two.pvalue),
        "p_one": float(res_one.pvalue),
        "n": int(len(diffs)),
        "n_nonzero": n_nonzero,
        "diffs": diffs.tolist(),
    }


def sign_test_one_sided(leader: list[float], baseline: list[float]) -> dict:
    """Project's Phase-5 ordinal gate: min(leader) > max(baseline).

    Equivalent under exchangeable-null sampling to a one-sided sign test on
    paired matchups. For n=3 with seed-matched pairs, P(all 3 leader_s >
    baseline_s | H0) = (1/2)^3 = 0.125 = α achievable.
    """
    if not leader or not baseline:
        return {"pass": False, "alpha": float("nan"), "min_lead": float("nan"), "max_base": float("nan")}
    return {
        "pass": min(leader) > max(baseline),
        "alpha": 0.5 ** len(leader),
        "min_lead": min(leader),
        "max_base": max(baseline),
    }


def fmt_pp(x: float) -> str:
    return f"{x * 100:+.2f} pp"


def section_1_phase8_winners() -> str:
    out = ["## Section 1 — Phase-8 winner formal tests (CIFAR-100, n=3)\n"]
    baseline = get_seed_top1s(CIFAR100, "baseline_resnet20")
    leaders = [
        ("pair_gm_pdw", "+1.34 pp claim"),
        ("slot_act_sine", "+1.32 pp claim"),
        ("sg_only_phi_budget", "+0.89 pp post-fix claim"),
    ]
    out.append(f"Baseline CIFAR-100 seeds {baseline}, median={statistics.median(baseline):.4f}, "
               f"mean={statistics.mean(baseline):.4f}, std={statistics.stdev(baseline):.4f}\n")
    out.append("\n| Claim | Leader top1 (s0,s1,s2) | Leader median | Δmedian | Δmean | "
               "Wilcoxon W | p_one-sided | p_two-sided | 95% bootstrap CI on Δmean | "
               "Ordinal gate α | Pass at α=0.05? | Pass at Holm α'=0.05/3=0.0167? |\n")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows: list[dict] = []
    for tag, label in leaders:
        leader = get_seed_top1s(CIFAR100, tag)
        wilc = paired_wilcoxon(leader, baseline)
        sign = sign_test_one_sided(leader, baseline)
        obs, lo, hi = bootstrap_ci_diff(leader, baseline)
        d_median = statistics.median(leader) - statistics.median(baseline)
        d_mean = statistics.mean(leader) - statistics.mean(baseline)
        leader_str = ",".join(f"{v:.4f}" for v in leader)
        out.append(
            f"| {tag} ({label}) | {leader_str} | {statistics.median(leader):.4f} | "
            f"{fmt_pp(d_median)} | {fmt_pp(d_mean)} | "
            f"{wilc['W']:.2f} | {wilc['p_one']:.4f} | {wilc['p_two']:.4f} | "
            f"[{fmt_pp(lo)}, {fmt_pp(hi)}] | "
            f"{sign['alpha']:.3f} | "
            f"{'YES' if wilc['p_one'] < 0.05 else 'NO'} | "
            f"{'YES' if wilc['p_one'] < 0.0167 else 'NO'} |\n"
        )
        rows.append(
            {
                "tag": tag, "leader": leader, "baseline": baseline,
                "d_median": d_median, "d_mean": d_mean,
                "wilc": wilc, "sign": sign, "boot_ci": (obs, lo, hi),
                "leader_std": statistics.stdev(leader),
            }
        )
    out.append("\n### Per-claim verdict\n")
    for r in rows:
        in_ci = r["boot_ci"][1] <= 0.0 <= r["boot_ci"][2]
        out.append(
            f"- **{r['tag']}**: Δmedian={fmt_pp(r['d_median'])}, leader std={r['leader_std']:.4f}; "
            f"paired Wilcoxon one-sided p={r['wilc']['p_one']:.4f} (theoretical floor at n=3 is 0.125); "
            f"95% bootstrap CI on Δmean = [{fmt_pp(r['boot_ci'][1])}, {fmt_pp(r['boot_ci'][2])}], "
            f"contains 0 = {in_ci}; Phase-5 ordinal-gate pass = {r['sign']['pass']} (α={r['sign']['alpha']}).\n"
        )
    return "".join(out), rows


def section_2_ordinal_gate_derivation() -> str:
    return (
        "## Section 2 — The Phase-5 worst-leader-seed > best-baseline-seed gate\n\n"
        "The project's Phase-5 ordinal gate accepts a candidate as a winner when, on "
        "n=3 seeds, the worst leader seed strictly beats the best baseline seed:\n\n"
        "    pass_5 := min({leader_s}) > max({baseline_s}), |leaders|=|baselines|=3.\n\n"
        "Under the seed-exchangeability null, conditional on the 6 observed values "
        "being a random labeling of 3 leader and 3 baseline draws from a common "
        "distribution, the probability that all 3 leader values exceed all 3 baseline "
        "values is C(6,3)^{-1} = 1/20 = 0.05 ONLY IF we use a permutation-of-rank "
        "argument. Under the SIGN-TEST variant (seeds are matched pairs, sign of "
        "delta), the probability that all 3 paired deltas are positive is\n\n"
        "    P(all 3 sgn(d_s) = +) = (1/2)^3 = 0.125 = α_gate.\n\n"
        "The project uses unpaired min/max wording but the seeds are seed-aligned, "
        "so the matched-pair sign test is the appropriate characterization. **The "
        "Phase-5 ordinal gate is therefore a one-sided sign test at α≈0.125, NOT at "
        "α=0.05.** This is 2.5× looser than the NeurIPS-standard α=0.05.\n\n"
        "To achieve a one-sided sign-test α=0.05 we need n ≥ 5 (P(all 5 +) = 1/32 = "
        "0.03125). For paired Wilcoxon, n ≥ 6 (min two-sided p at n=6 is 2/64=0.031). "
        "**The project's claimed Phase-8 winners cannot clear α=0.05 at n=3, by any "
        "test, even when every seed favors the leader.**\n\n"
    )


def section_3_holm_bonferroni() -> str:
    return (
        "## Section 3 — Multiple-comparisons correction (Holm-Bonferroni)\n\n"
        "**CIFAR-10 screening sweep (35 rows, n=1 each).** Family-wise α=0.05 under "
        "Bonferroni → per-test α'_Bonf = 0.05/35 ≈ 0.00143. At n=1 seed per tag, the "
        "smallest paired p-value achievable is 0.5 (one paired sample, two-sided). "
        "**No CIFAR-10 screening row can clear ANY α' at n=1.** The 35-row sweep is "
        "exploratory by mathematical necessity; the paper should not present any row "
        "of it as a confirmed positive or confirmed negative.\n\n"
        "**Phase-8 CIFAR-100 family (3 simultaneous tests).** Family-wise α=0.05 "
        "under Bonferroni → per-test α'_Bonf = 0.05/3 ≈ 0.0167. Under Holm step-down, "
        "sort p-values ascending: smallest test must clear α/3, second must clear "
        "α/2, third must clear α/1. **At n=3 the theoretical minimum p_one-sided is "
        "0.125 (all 3 deltas same sign), which exceeds 0.0167. None of the three "
        "Phase-8 winners can clear Holm-Bonferroni at α=0.05 regardless of effect "
        "size.**\n\n"
        "**Sample-size requirement.** To clear Holm-Bonferroni with k=3 at α=0.05 we "
        "need each p ≤ 0.05/k = 0.0167. For a paired sign test that requires n ≥ 6 "
        "(P=1/64=0.0156). For a paired Wilcoxon with all positive deltas n ≥ 7 "
        "(one-sided exact p at n=7 with W=28 is 1/128=0.0078). **The Phase-8 family "
        "must be re-run with n ≥ 7 seeds before any Holm-corrected claim is "
        "achievable.**\n\n"
    )


def section_4_seed_noise(rows: list[dict]) -> str:
    out = ["## Section 4 — Seed-noise floor estimates\n\n"]
    # CIFAR-100 baseline std
    base100 = get_seed_top1s(CIFAR100, "baseline_resnet20")
    s_b = statistics.stdev(base100)
    out.append(
        f"**CIFAR-100 baseline_resnet20 (n=3):** seeds={base100}, "
        f"mean={statistics.mean(base100):.4f}, σ={s_b:.4f} ({s_b*100:.3f} pp). "
        f"2σ ≈ {2*s_b*100:.2f} pp. A single-seed Δ smaller than 2σ is indistinguishable "
        f"from null at the 95% confidence level under a Gaussian approximation.\n\n"
    )
    # CIFAR-10 multi-seed tags
    multi_tags = [
        "baseline_resnet20", "baseline_sg_vanilla", "sg_chan_fib", "sg_chan_phi",
        "sg_only_cymatic_init", "sg_only_fractal", "sg_only_golden_modulate",
        "sg_only_group", "sg_only_hex", "sg_only_phi_budget", "sg_only_toroidal",
    ]
    stds: list[tuple[str, float, list[float]]] = []
    out.append("**CIFAR-10 12-ep multi-seed coverage (tags with seeds 0/1/2):**\n\n")
    out.append("| Tag | seed0 | seed1 | seed2 | mean | std (pp) |\n")
    out.append("|---|---|---|---|---|---|\n")
    for t in multi_tags:
        vals = get_seed_top1s(CIFAR10, t)
        if len(vals) == 3:
            s = statistics.stdev(vals)
            stds.append((t, s, vals))
            out.append(
                f"| {t} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
                f"{statistics.mean(vals):.4f} | {s*100:.3f} |\n"
            )
    pooled = math.sqrt(sum(s*s for _, s, _ in stds) / len(stds))
    out.append(
        f"\n**Pooled CIFAR-10 12-ep seed σ across {len(stds)} multi-seed tags = "
        f"{pooled*100:.3f} pp** (RMS of per-tag std). 2σ_pooled ≈ "
        f"{2*pooled*100:.2f} pp. This is the empirical CIFAR-10 12-ep noise floor "
        f"per row. The paper's stated 'within ±0.5 pp is seed noise' rule of thumb "
        f"is {'CONSERVATIVE' if 0.5 > 2*pooled*100 else 'OPTIMISTIC'} relative to "
        f"this estimate.\n\n"
    )
    # CIFAR-100 multi-seed leader stds
    out.append("**CIFAR-100 30-ep 3-seed coverage — leader stds:**\n\n")
    out.append("| Tag | seed0 | seed1 | seed2 | mean | std (pp) |\n|---|---|---|---|---|---|\n")
    for r in rows:
        v = r["leader"]
        out.append(
            f"| {r['tag']} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | "
            f"{statistics.mean(v):.4f} | {r['leader_std']*100:.3f} |\n"
        )
    return "".join(out), pooled


def section_5_single_seed_distribution(pooled_std10: float) -> str:
    out = ["## Section 5 — CIFAR-10 single-seed Δ distribution (35-row screen)\n\n"]
    base0 = get_seed_top1s(CIFAR10, "baseline_resnet20", seeds=(0,))
    if not base0:
        out.append("Baseline seed0 missing; skipping.\n")
        return "".join(out)
    b0 = base0[0]
    deltas: list[tuple[str, float]] = []
    for p in sorted(CIFAR10.glob("*_seed0")):
        tag = p.name.removesuffix("_seed0")
        if tag == "baseline_resnet20":
            continue
        m = load_metric(p / "metrics.json")
        if m and "top1" in m:
            deltas.append((tag, float(m["top1"]) - b0))
    vals = [d for _, d in deltas]
    n = len(vals)
    arr = np.array(vals)
    mean = float(arr.mean())
    median = float(np.median(arr))
    p90 = float(np.quantile(arr, 0.90))
    p95 = float(np.quantile(arr, 0.95))
    p99 = float(np.quantile(arr, 0.99))
    abs_arr = np.abs(arr)
    out.append(
        f"Baseline seed-0 CIFAR-10 12-ep top1 = {b0:.4f}. Comparing all "
        f"{n} non-baseline seed-0 tags:\n\n"
        f"- Δtop1 mean = {fmt_pp(mean)}\n"
        f"- Δtop1 median = {fmt_pp(median)}\n"
        f"- Δtop1 90th percentile = {fmt_pp(p90)}\n"
        f"- Δtop1 95th percentile = {fmt_pp(p95)}\n"
        f"- Δtop1 99th percentile = {fmt_pp(p99)}\n"
        f"- mean |Δtop1| = {fmt_pp(float(abs_arr.mean()))}\n"
        f"- max |Δtop1| = {fmt_pp(float(abs_arr.max()))}\n\n"
        f"Pooled multi-seed σ on baseline-class tags = {pooled_std10*100:.3f} pp. "
        f"2σ band = ±{2*pooled_std10*100:.2f} pp. The fraction of single-seed |Δ| "
        f"observations that EXCEED 2σ pooled = "
        f"{int((abs_arr > 2*pooled_std10).sum())}/{n} = "
        f"{(abs_arr > 2*pooled_std10).mean()*100:.1f}%. At n=1 per row, only |Δ| "
        f"greater than ~2σ_pooled has any prima-facie credibility, and even then "
        f"is not statistically tested.\n\n"
    )
    # H09 phi_budget at seed 0
    pb = get_seed_top1s(CIFAR10, "sg_only_phi_budget", seeds=(0,))
    if pb:
        out.append(
            f"**H09 phi_budget CIFAR-10 12-ep seed-0:** top1={pb[0]:.4f}, "
            f"Δ vs baseline_seed0 = {fmt_pp(pb[0]-b0)}. Compared to 2σ_pooled = "
            f"{2*pooled_std10*100:.2f} pp, this is "
            f"{'OUTSIDE' if abs(pb[0]-b0) > 2*pooled_std10 else 'INSIDE'} the noise "
            f"band.\n\n"
        )
    # 3-seed phi_budget
    pb3 = get_seed_top1s(CIFAR10, "sg_only_phi_budget")
    b3 = get_seed_top1s(CIFAR10, "baseline_resnet20")
    if len(pb3) == 3 and len(b3) == 3:
        d3 = statistics.mean(pb3) - statistics.mean(b3)
        w = paired_wilcoxon(pb3, b3)
        out.append(
            f"**H09 phi_budget CIFAR-10 3-seed paired test:** Δmean = {fmt_pp(d3)}, "
            f"paired Wilcoxon one-sided p={w['p_one']:.4f}, two-sided p={w['p_two']:.4f}. "
            f"At theoretical floor p_one_min(n=3)=0.125, this {'achieves' if w['p_one']<=0.125 else 'does NOT achieve'} "
            f"the floor.\n\n"
        )
    return "".join(out)


def section_6_phi_budget_ci_check() -> str:
    """The reviewer specifically flagged phi_budget +0.25 pp claim."""
    out = ["## Section 6 — phi_budget +0.25 pp lead claim, bootstrap CI check\n\n"]
    leader = get_seed_top1s(CIFAR100, "sg_only_phi_budget")
    baseline = get_seed_top1s(CIFAR100, "baseline_resnet20")
    obs, lo, hi = bootstrap_ci_diff(leader, baseline)
    out.append(
        f"phi_budget CIFAR-100 seeds = {leader}, baseline seeds = {baseline}.\n"
        f"Δmean = {fmt_pp(obs)}, 95% bootstrap CI = [{fmt_pp(lo)}, {fmt_pp(hi)}].\n"
        f"The +0.25 pp lead floor (PAPER.md:138) corresponds to leader_worst - "
        f"baseline_best = {fmt_pp(min(leader) - max(baseline))}. "
        f"0 is {'INSIDE' if lo <= 0 <= hi else 'OUTSIDE'} the bootstrap CI. "
        f"The +0.25 pp claim is therefore {'NOT statistically distinguishable' if lo <= 0 <= hi else 'distinguishable'} "
        f"from 0 at 95% confidence.\n\n"
        f"The CIFAR-100 baseline 3-seed σ = "
        f"{statistics.stdev(baseline)*100:.3f} pp. The leader 3-seed σ = "
        f"{statistics.stdev(leader)*100:.3f} pp. Pooled σ on Δmean = "
        f"{math.sqrt(statistics.variance(leader)/3 + statistics.variance(baseline)/3)*100:.3f} pp. "
        f"|Δmean|/σ_Δmean ratio = "
        f"{abs(obs)/math.sqrt(statistics.variance(leader)/3 + statistics.variance(baseline)/3):.2f}.\n\n"
    )
    return "".join(out)


def main() -> None:
    section1, rows = section_1_phase8_winners()
    section2 = section_2_ordinal_gate_derivation()
    section3 = section_3_holm_bonferroni()
    section4, pooled10 = section_4_seed_noise(rows)
    section5 = section_5_single_seed_distribution(pooled10)
    section6 = section_6_phi_budget_ci_check()
    print(section1)
    print(section2)
    print(section3)
    print(section4)
    print(section5)
    print(section6)


if __name__ == "__main__":
    main()
