"""H60 — Three-Seed Uncertainty Quantification.

Design doc: ``hypotheses/g6_topological_bridging/H60_three_seed_uncertainty.md``.

Standard practice in deep learning is 3-seed median + IQR or 5-seed
mean + 95% CI. The 11-row CIFAR sweep that already lives in this repo
was run with single seed=0 and therefore has no error bars; H60
provides the post-hoc UQ machinery: given a list of per-seed
``metrics.json`` dicts, aggregate per-key mean / std / min / max +
bootstrap 95% CI and render the result as a markdown table for
inclusion in dashboards and the FINDINGS log.

Public surface
--------------
- :func:`aggregate_seeds`     -- list[dict] -> dict[key -> stats].
- :func:`bootstrap_ci`        -- bootstrap 95% CI on a 1-D sample.
- :func:`format_seeds_table`  -- markdown table renderer for an
                                  aggregated stats dict.

References (Citation Rigor)
---------------------------
    Henderson, Peter, Islam, Riashat, Bachman, Philip, Pineau, Joelle,
    Precup, Doina, Meger, David 2018 AAAI 'Deep Reinforcement Learning
    that Matters' (arXiv:1709.06560) -- exposes the single-seed
    reproducibility crisis; mandates 3+ seeds.
    He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian 2016 CVPR
    'Deep Residual Learning for Image Recognition' (arXiv:1512.03385)
    -- canonical 5-seed ResNet reporting practice.
    Loshchilov, Ilya, Hutter, Frank 2019 ICLR 'Decoupled Weight Decay
    Regularization' (arXiv:1711.05101) -- canonical 3-seed median
    reporting precedent.
    Bouthillier, Xavier, Laurent, Cesar, Vincent, Pascal 2019 NeurIPS
    workshop 'Unreproducible Research is Reproducible' -- seed
    variance + IQR vs CI reporting.
"""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


_Numeric = (int, float)


def _is_numeric(v: object) -> bool:
    """``True`` for finite int / float values (excludes bool and NaN)."""
    if isinstance(v, bool):
        return False
    if not isinstance(v, _Numeric):
        return False
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return True


def aggregate_seeds(
    metrics_list: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per-seed metrics dicts into mean / std / min / max / n.

    Per-key behaviour:

    - keys present in *all* provided dicts AND with *all-numeric*
      values across dicts are aggregated into
      ``{"mean", "std", "min", "max", "n"}`` (sample std, ddof=1).
    - keys that are not numeric in *any* dict are silently skipped
      (so string-valued metadata like ``"timestamp"`` does not
      surface in the output).
    - missing keys (present in some seeds but not others) are also
      skipped to avoid silently averaging over partial samples.

    Parameters
    ----------
    metrics_list : sequence of dict
        per-seed ``metrics.json`` payloads.

    Returns
    -------
    dict[str, dict[str, float | int]]
        per-key aggregated statistics. Empty if there are zero seeds.
    """
    if len(metrics_list) == 0:
        return {}
    # Collect the intersection of keys -- only aggregate metrics
    # present in every seed.
    common_keys = set(metrics_list[0].keys())
    for m in metrics_list[1:]:
        common_keys &= set(m.keys())
    out: dict[str, dict[str, float | int]] = {}
    for key in sorted(common_keys):
        values: list[float] = []
        all_numeric = True
        for m in metrics_list:
            v = m[key]
            if not _is_numeric(v):
                all_numeric = False
                break
            values.append(float(v))
        if not all_numeric or len(values) == 0:
            continue
        n = len(values)
        mean = sum(values) / n
        if n >= 2:
            var = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out[key] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "n": n,
        }
    return out


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap ``(1-alpha)`` confidence interval for the sample mean.

    Resamples ``values`` with replacement ``n_boot`` times and returns
    the empirical ``[alpha/2, 1-alpha/2]`` quantiles of the resampled
    means. Pure-python implementation (no numpy dependency) so the
    routine is import-safe under the minimal toolchain.

    Parameters
    ----------
    values : sequence of float
        the observed sample.
    n_boot : int, default 1000
        number of bootstrap resamples.
    alpha : float, default 0.05
        two-sided significance level. ``alpha=0.05`` -> 95% CI.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    tuple (lower, upper)
        ordered such that ``lower <= upper``. Returns ``(NaN, NaN)``
        for an empty input.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    rng = random.Random(seed)
    vals = [float(v) for v in values]
    means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_boot))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    lo_idx = max(0, min(n_boot - 1, lo_idx))
    hi_idx = max(0, min(n_boot - 1, hi_idx))
    lower = means[lo_idx]
    upper = means[hi_idx]
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)


def format_seeds_table(
    aggregated: Mapping[str, Mapping[str, float | int]],
) -> str:
    """Render an aggregated stats dict as a GitHub-flavoured markdown table.

    The table columns are ``metric | mean | std | min | max | n``.
    Floats are formatted with 4 decimal places; the ``n`` column is
    rendered as an integer. Returns ``""`` if the input is empty.

    Parameters
    ----------
    aggregated : mapping
        output of :func:`aggregate_seeds`.

    Returns
    -------
    str
        markdown table (header + separator + one row per metric).
    """
    if len(aggregated) == 0:
        return ""
    lines = [
        "| metric | mean | std | min | max | n |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(aggregated.keys()):
        s = aggregated[key]
        lines.append(
            f"| {key} | {float(s['mean']):.4f} | {float(s['std']):.4f} | "
            f"{float(s['min']):.4f} | {float(s['max']):.4f} | {int(s['n'])} |"
        )
    return "\n".join(lines)
