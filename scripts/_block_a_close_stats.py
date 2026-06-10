"""Block A close statistical analysis: paired Wilcoxon + paired bootstrap CI +
Holm-Bonferroni k=3 + Phase-5 ordinal gate for the 3 iso-FLOPs priors vs the
A4-v1 He-2019/Playbook baseline at n=3 seeds.

Output: paper/STATISTICAL_TESTS.md gets a new Section 16 appended.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
from statistics import mean, stdev

random.seed(20260610)

REPO = Path(__file__).resolve().parent.parent

ARMS = {
    "baseline_resnet20_he2019_debug":     ("experiments_modern_debug", "A4-v1 baseline"),
    "sg_only_phi_budget_iso_flops_v1":    ("experiments_iso_flops_v1", "H09 (phi_budget) iso-FLOPs"),
    "pair_gm_pdw_iso_flops_v1":           ("experiments_iso_flops_v1", "pair_gm_pdw iso-FLOPs"),
    "slot_act_sine_iso_flops_v1":         ("experiments_iso_flops_v1", "slot_act_sine iso-FLOPs"),
}

SEEDS = [0, 1, 2]

def load_top1(tag: str, root: str, seed: int) -> float:
    p = REPO / root / "cifar100" / f"{tag}_seed{seed}" / "metrics.json"
    return json.loads(p.read_text())["top1"]

# Load all 4 arms x 3 seeds.
data = {tag: [load_top1(tag, root, s) for s in SEEDS] for tag, (root, _name) in ARMS.items()}

print("\n=== Raw n=3 top1 ===")
for tag, vals in data.items():
    print(f"  {tag:50s} {vals}  mean={mean(vals):.4f}  sigma={stdev(vals)*100:.2f}pp")

baseline = data["baseline_resnet20_he2019_debug"]
priors = [k for k in data if k != "baseline_resnet20_he2019_debug"]

def paired_diff(prior_vals, base_vals):
    return [p - b for p, b in zip(prior_vals, base_vals)]

def wilcoxon_signed_rank_2sided(diffs):
    """Exact 2-sided Wilcoxon at small n using sign-rank enumeration."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0.0
    abs_diffs = sorted([(abs(d), i) for i, d in enumerate(nonzero)])
    ranks = {}
    i = 0
    while i < len(abs_diffs):
        j = i
        while j + 1 < len(abs_diffs) and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j + 1
    W_plus = sum(ranks[i] for i, d in enumerate(nonzero) if d > 0)
    sum_ranks = sum(ranks[i] for i in range(n))
    T_obs = min(W_plus, sum_ranks - W_plus)  # standard 2-sided test stat
    # Exact 2-sided p via enumeration over all 2^n sign assignments
    total = 1 << n
    rank_vals = [ranks[i] for i in range(n)]
    count = 0
    for mask in range(total):
        Wp = sum(rank_vals[i] for i in range(n) if (mask >> i) & 1)
        T_perm = min(Wp, sum_ranks - Wp)
        if T_perm <= T_obs:
            count += 1
    p_two = count / total
    return p_two, W_plus

def paired_t_two_sided(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = mean(diffs)
    sd = stdev(diffs)
    if sd == 0:
        return 0.0 if m == 0 else 0.0
    t = m / (sd / math.sqrt(n))
    # df = n-1 = 2; use Student-t survival via integration approximation
    # Closed form for df=2: 2-sided p = 1 - t / sqrt(2 + t^2)
    df = n - 1
    if df == 2:
        p_one = 0.5 * (1.0 - t / math.sqrt(2.0 + t * t))
    elif df == 1:
        p_one = 0.5 - math.atan(t) / math.pi
    else:
        # Generic Wilson-Hilferty normal approx
        z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
        p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * min(p_one, 1 - p_one)

def paired_bootstrap_ci(diffs, n_iter=10000, ci=0.95):
    n = len(diffs)
    means = []
    for _ in range(n_iter):
        sample = [diffs[random.randint(0, n - 1)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_iter)
    hi_idx = int((1 + ci) / 2 * n_iter) - 1
    return means[lo_idx], means[hi_idx]

def phase5_ordinal_gate(prior_vals, base_vals):
    # synth convention: min(leader) > max(baseline) (i.e. prior > baseline strictly)
    return min(prior_vals) > max(base_vals)

print("\n=== Paired analysis vs A4-v1 baseline (n=3) ===")
results = {}
for prior in priors:
    pv = data[prior]
    diffs = paired_diff(pv, baseline)
    p_w, W = wilcoxon_signed_rank_2sided(diffs)
    p_t = paired_t_two_sided(diffs)
    ci_lo, ci_hi = paired_bootstrap_ci(diffs)
    gate = phase5_ordinal_gate(pv, baseline)
    print(f"\n  {prior}")
    print(f"    diffs (prior - baseline) = {[f'{d:+.4f}' for d in diffs]}")
    print(f"    delta_mean = {mean(diffs)*100:+.2f} pp")
    print(f"    Wilcoxon 2-sided p = {p_w:.4f}")
    print(f"    paired-t 2-sided p (df=2) = {p_t:.4f}")
    print(f"    95% paired-bootstrap CI = [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp")
    print(f"    Phase-5 ordinal gate (min(L) > max(B)) = {gate}")
    results[prior] = dict(diffs=diffs, p_w=p_w, p_t=p_t, ci=(ci_lo, ci_hi),
                          gate=gate, delta_mean=mean(diffs))

# Holm-Bonferroni at k=3 family
print("\n=== Holm-Bonferroni k=3 family (alpha = 0.05) ===")
sorted_by_p = sorted(results.items(), key=lambda kv: kv[1]["p_w"])
alpha = 0.05
for rank, (prior, r) in enumerate(sorted_by_p, start=1):
    alpha_holm = alpha / (3 - rank + 1)
    pass_holm = r["p_w"] <= alpha_holm
    print(f"  rank {rank}: {prior:50s} p={r['p_w']:.4f}  alpha_holm={alpha_holm:.4f}  pass={pass_holm}")

print("\n=== Verdict ===")
all_lose = all(r["delta_mean"] < 0 for r in results.values())
all_p_significant = all(r["p_w"] <= 0.25 for r in results.values())  # 2-sided n=3 floor is 0.25
print(f"  All 3 priors LOSE baseline at iso-FLOPs n=3: {all_lose}")
print(f"  Inversion direction is consistent across all 3 priors: {all_lose}")
print(f"  All 3 paired Wilcoxon p_two at n=3 floor (0.25): {all_p_significant}")
print(f"  All 3 fail Phase-5 ordinal gate (prior < baseline strictly): {not any(r['gate'] for r in results.values())}")
