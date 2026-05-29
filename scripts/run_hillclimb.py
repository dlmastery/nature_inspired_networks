"""Per-hypothesis hill-climb runner (skill: autoresearch-per-hypothesis-hillclimb).

Implements coordinate-descent / random / grid search over the 5-axis cube
(lr × wd × batch × seed × optimizer) on top of a screening tag's base
overrides drawn from ``scripts/run_sweep.py:build_matrix()``.

Outputs:
- ``experiments/<dataset>/<tag>__hc_lr<LR>_wd<WD>_bs<BS>_opt<OPT>_seed<SEED>/``
  per-cell archive with ``metrics.json`` (+ history / best.pt via run_one).
- ``ideas/<NN>/hillclimb_results.json`` — summary.
- Optional per-cell scoped git commit + push (Rule 11 / 20).

Algorithms:
- ``coordinate`` (default): sequential axis sweep, strict-> champion rule.
- ``random``: ``--budget`` uniform draws from the cube (replace=False).
- ``grid``: full Cartesian over lr × wd × batch × opt at the given seed(s).

End-to-end test:
    python scripts/run_hillclimb.py --tag baseline_resnet20 \
        --idea _TEMPLATE --algorithm grid --budget 4 --lr 3e-4 1e-3 \
        --wd 5e-4 --batch 256 --optimizer AdamW --seeds 0 --epochs 2 \
        --config configs/cifar10_quick.yaml --no-commit

The script is importable (``from scripts.run_hillclimb import hillclimb``)
and runnable standalone.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from nature_inspired_networks.runner import run_one  # noqa: E402
from run_sweep import build_matrix  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults — keep in sync with skills/autoresearch-per-hypothesis-hillclimb.
# ---------------------------------------------------------------------------
DEFAULT_LRS = [3.0e-4, 1.0e-3, 3.0e-3]
DEFAULT_WDS = [1.0e-4, 5.0e-4, 2.0e-3]
DEFAULT_BATCHES = [128, 256]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_OPTIMIZERS = ["AdamW", "SGD"]

# Coordinate-descent axis order; ``seed`` is reserved for the final 3-seed
# confirmation step and is NOT swept during HC.
HC_AXIS_ORDER = ("lr", "weight_decay", "optimizer", "batch_size")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    """Compact, filesystem-safe string for a hyperparameter value."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        exp = int(math.floor(math.log10(abs(v))))
        m = v / (10 ** exp)
        if abs(m - round(m)) < 1e-6:
            return f"{int(round(m))}e{exp:+d}".replace("+", "p").replace("-", "m")
        return f"{m:.2g}e{exp:+d}".replace("+", "p").replace("-", "m")
    return str(v).replace(".", "p").replace("+", "").replace("-", "m")


def _config_id(cell: dict) -> str:
    """Deterministic hash-free id, e.g. ``lr1em3_wd5em4_bs256_optAdamW``."""
    return (
        f"lr{_fmt(cell['lr'])}_wd{_fmt(cell['weight_decay'])}"
        f"_bs{cell['batch_size']}_opt{cell['optimizer']}"
    )


def _cell_tag(base_tag: str, cell: dict, seed: int) -> str:
    return f"{base_tag}__hc_{_config_id(cell)}_seed{seed}"


def _base_overrides(tag: str) -> dict:
    """Pull the row in build_matrix(curated=False) whose tag == ``tag``."""
    matrix = build_matrix(curated=False)
    for row in matrix:
        if row["tag"] == tag:
            return copy.deepcopy(row["overrides"])
    raise SystemExit(
        f"[hc] tag {tag!r} not found in scripts/run_sweep.py:build_matrix(). "
        f"Available examples: " + ", ".join(r["tag"] for r in matrix[:6]) + " ..."
    )


def _materialise_cfg(base_cfg: dict, tag_overrides: dict, cell: dict,
                     epochs: int | None) -> dict:
    """Compose the full cfg dict run_one() expects."""
    cfg = copy.deepcopy(base_cfg)
    cfg.update(tag_overrides)
    cfg["lr"] = float(cell["lr"])
    cfg["weight_decay"] = float(cell["weight_decay"])
    cfg["batch_size"] = int(cell["batch_size"])
    # The runner already understands ``optimizer`` (Phase C). Map our
    # display names to its expected literals.
    cfg["optimizer"] = {"AdamW": "adamw", "SGD": "sgd_momentum"}.get(
        cell["optimizer"], cell["optimizer"].lower()
    )
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    return cfg


def _commit_checkpoint(repo: Path, tag: str, cell_id: str,
                       extra_paths: Iterable[Path] = ()) -> None:
    """Retry-wrapped scoped commit + push (Rule 11 / 20)."""
    paths = [
        f"experiments/{tag}__hc_*",
        "ideas",
    ]
    for p in extra_paths:
        paths.append(str(p.relative_to(repo)) if p.is_absolute() else str(p))
    msg = f"hill-climb cell: {tag} / {cell_id}"
    for attempt in range(5):
        try:
            for pat in paths:
                subprocess.run(
                    ["git", "add", "--", pat], cwd=repo,
                    check=False, capture_output=True,
                )
            r = subprocess.run(
                ["git", "-c", "user.name=dlmastery",
                 "-c", "user.email=eranti@gmail.com",
                 "commit", "-m", msg],
                cwd=repo, capture_output=True, text=True,
            )
            if r.returncode != 0 and "nothing to commit" in (r.stdout + r.stderr):
                return
            if r.returncode != 0:
                # Likely index.lock contention; back off + retry.
                time.sleep(2 + attempt)
                continue
            subprocess.run(
                ["git", "push"], cwd=repo, capture_output=True, check=False,
            )
            return
        except Exception:
            time.sleep(2 + attempt)
    # 5 attempts exhausted; intentionally non-fatal — the auto-checkpoint
    # loop (autoresearch-auto-checkpoint-loop) will sweep the artifacts.


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
def _grid_cells(cube: dict, seeds: list[int]) -> list[tuple[dict, int]]:
    axes = ("lr", "weight_decay", "batch_size", "optimizer")
    out: list[tuple[dict, int]] = []
    for combo in itertools.product(*[cube[a] for a in axes]):
        cell = dict(zip(axes, combo))
        for s in seeds:
            out.append((cell, s))
    return out


def _random_cells(cube: dict, seeds: list[int], budget: int,
                  rng_seed: int = 0) -> list[tuple[dict, int]]:
    rng = random.Random(rng_seed)
    all_cells = _grid_cells(cube, seeds)
    rng.shuffle(all_cells)
    return all_cells[: budget]


def _coordinate_plan(cube: dict, base_cell: dict) -> list[dict]:
    """Build the ordered list of cells to visit. ``run_cell`` is responsible
    for evaluating each, updating the champion, and DECIDING which cells in
    the plan still need to run (best-config movement may skip values
    already evaluated)."""
    plan: list[dict] = [copy.deepcopy(base_cell)]
    for axis in HC_AXIS_ORDER:
        for v in cube[axis]:
            if v == base_cell[axis]:
                continue
            cand = copy.deepcopy(base_cell); cand[axis] = v
            plan.append(cand)
    return plan


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def hillclimb(
    tag: str,
    idea: str,
    config_path: str | Path,
    *,
    cube: dict | None = None,
    seeds: list[int] | None = None,
    algorithm: str = "coordinate",
    budget: int = 25,
    root: str = "experiments",
    epochs: int | None = None,
    skip_existing: bool = True,
    do_commit: bool = True,
    rng_seed: int = 0,
    out_path: str | Path | None = None,
) -> dict:
    """Run the hill-climb. Returns the summary dict (also written to disk).

    ``tag`` must be a row in ``scripts/run_sweep.py:build_matrix(curated=False)``.
    ``idea`` is the ``ideas/<idea>/`` subdir; the summary lands at
    ``ideas/<idea>/hillclimb_results.json``.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    tag_overrides = _base_overrides(tag)
    cube = cube or dict(lr=DEFAULT_LRS, weight_decay=DEFAULT_WDS,
                        batch_size=DEFAULT_BATCHES, optimizer=DEFAULT_OPTIMIZERS)
    seeds = seeds or DEFAULT_SEEDS
    dataset = base_cfg["dataset"]
    repo = _REPO

    # Resolve the project-default base cell from the YAML (the screening row
    # uses these unless its own overrides say otherwise — but
    # build_matrix rows do NOT override lr/wd/batch/opt today, so the cfg
    # values are the screening defaults).
    base_cell = dict(
        lr=float(base_cfg.get("lr", 1.0e-3)),
        weight_decay=float(base_cfg.get("weight_decay", 5.0e-4)),
        batch_size=int(base_cfg.get("batch_size", 256)),
        optimizer="AdamW",  # the project default; SGD is the alternative
    )

    cells_visited: list[dict] = []
    best: dict | None = None
    t_campaign = time.perf_counter()

    def run_cell(cell: dict, seed: int) -> dict | None:
        """Train one cell. Returns the metrics dict (or None on failure)."""
        cell_tag = _cell_tag(tag, cell, seed)
        out_dir = Path(root) / dataset / cell_tag.split("__hc_", 1)[0]
        # Each cell lives in its own dir keyed by the FULL cell_tag so
        # restarts skip cleanly.
        cell_dir = Path(root) / dataset / cell_tag
        metrics_path = cell_dir / "metrics.json"

        if skip_existing and metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
            print(f"  [hc skip] {cell_tag} top1={m.get('top1')}")
        else:
            cfg = _materialise_cfg(base_cfg, tag_overrides, cell, epochs)
            t1 = time.perf_counter()
            try:
                # run_one writes to <root>/<dataset>/<tag_arg>_seed<seed>/
                # We use cell_tag (with no trailing _seed) and pass seed
                # explicitly so the directory comes out
                # <root>/<dataset>/<cell_tag>_seed<seed>/.
                # Then we rename to drop the trailing _seed<seed>
                # (the cell_tag already encodes the seed).
                tmp_dir = run_one(
                    cfg, tag=cell_tag, seed=seed, root=root,
                )
                if tmp_dir != cell_dir and tmp_dir.exists():
                    # run_one created e.g. <root>/<dataset>/<cell_tag>_seed<seed>/.
                    # Move its contents into our cell_dir (idempotent).
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    for child in tmp_dir.iterdir():
                        target = cell_dir / child.name
                        if not target.exists():
                            child.rename(target)
                    try:
                        tmp_dir.rmdir()
                    except OSError:
                        pass
            except Exception as exc:
                cell_dir.mkdir(parents=True, exist_ok=True)
                (cell_dir / "error.txt").write_text(repr(exc))
                print(f"  [hc FAIL] {cell_tag}: {exc!r}")
                return None
            dt = time.perf_counter() - t1
            print(f"  [hc run ] {cell_tag} {dt:.1f}s")

            # Annotate metrics.json with config_id + hc bookkeeping.
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as fh:
                    m = json.load(fh)
                m["config_id"] = _config_id(cell)
                m["hc_cell"] = cell
                m["hc_seed"] = seed
                with metrics_path.open("w", encoding="utf-8") as fh:
                    json.dump(m, fh, indent=2)
            if do_commit:
                _commit_checkpoint(repo, tag, _config_id(cell) + f"_seed{seed}")

        with metrics_path.open("r", encoding="utf-8") as fh:
            m = json.load(fh)
        return m

    def record(cell: dict, seed: int, m: dict) -> None:
        nonlocal best
        entry = dict(
            config=cell, seed=seed,
            config_id=_config_id(cell),
            top1=float(m.get("top1", m.get("top1_acc", 0.0))),
            composite=float(m.get("composite", 0.0)),
            wallclock_s=float(m.get("wallclock_s", m.get("elapsed_s", 0.0)) or 0.0),
        )
        cells_visited.append(entry)
        if best is None or entry["top1"] > best["top1"]:  # strict-> rule
            best = entry

    # ---------------------------------------------------------------
    # Execute the chosen algorithm.
    # ---------------------------------------------------------------
    if algorithm == "coordinate":
        plan = _coordinate_plan(cube, base_cell)
        budget_left = budget
        current = copy.deepcopy(base_cell)

        # First, evaluate the base cell at seed 0.
        cells_done: set[str] = set()
        m = run_cell(current, seeds[0]); budget_left -= 1
        if m is None:
            return _finalise(tag, idea, algorithm, budget, cube, base_cell,
                             cells_visited, best, t_campaign, out_path)
        cells_done.add((_config_id(current), seeds[0]).__repr__())
        record(current, seeds[0], m)

        # Per-axis sweep at the (current) champion's frozen others.
        for axis in HC_AXIS_ORDER:
            for v in cube[axis]:
                if budget_left <= 0:
                    break
                if v == current[axis]:
                    continue
                cand = copy.deepcopy(current); cand[axis] = v
                key = (_config_id(cand), seeds[0]).__repr__()
                if key in cells_done:
                    continue
                cells_done.add(key)
                m = run_cell(cand, seeds[0]); budget_left -= 1
                if m is None:
                    continue
                record(cand, seeds[0], m)
                if best is not None and best["config_id"] == _config_id(cand):
                    current = cand  # axis champion advanced
            if budget_left <= 0:
                break

        # Final 3-seed confirmation at the hill-climbed best.
        if best is not None and budget_left > 0:
            for s in seeds:
                if budget_left <= 0:
                    break
                key = (best["config_id"], s).__repr__()
                if key in cells_done:
                    continue
                cells_done.add(key)
                m = run_cell(best["config"], s); budget_left -= 1
                if m is None:
                    continue
                record(best["config"], s, m)

    elif algorithm == "random":
        cells = _random_cells(cube, [seeds[0]], budget, rng_seed=rng_seed)
        for cell, s in cells:
            m = run_cell(cell, s)
            if m is None:
                continue
            record(cell, s, m)

    elif algorithm == "grid":
        cells = _grid_cells(cube, seeds)[: budget]
        for cell, s in cells:
            m = run_cell(cell, s)
            if m is None:
                continue
            record(cell, s, m)

    else:
        raise SystemExit(f"[hc] unknown algorithm: {algorithm!r}")

    return _finalise(tag, idea, algorithm, budget, cube, base_cell,
                     cells_visited, best, t_campaign, out_path)


def _finalise(tag, idea, algorithm, budget, cube, base_cell, cells,
              best, t_campaign, out_path) -> dict:
    # 3-seed median / min / range at the best config.
    best_seeds = [c for c in cells if best is not None
                  and c["config_id"] == best["config_id"]]
    top1s = sorted(c["top1"] for c in best_seeds)
    seeds_confirmed = sorted({c["seed"] for c in best_seeds})

    def _median(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    summary = dict(
        tag=tag, idea=idea, algorithm=algorithm, budget=budget,
        cube=cube, base_config=base_cell,
        best_config=best["config"] if best else None,
        best_top1_median=_median(top1s) if top1s else None,
        best_top1_min=min(top1s) if top1s else None,
        best_top1_range=(max(top1s) - min(top1s)) if top1s else None,
        seeds_confirmed=seeds_confirmed,
        cells=cells,
        wallclock_total_s=time.perf_counter() - t_campaign,
    )
    out_path = Path(out_path) if out_path else (
        _REPO / "ideas" / idea / "hillclimb_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[hc] summary -> {out_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True,
                    help="Screening tag (must exist in run_sweep.build_matrix)")
    ap.add_argument("--idea", required=True,
                    help="ideas/<idea>/ subdir name (e.g. 04_phi_fib_width)")
    ap.add_argument("--lr", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--wd", type=float, nargs="+", default=DEFAULT_WDS)
    ap.add_argument("--batch", type=int, nargs="+", default=DEFAULT_BATCHES)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--optimizer", nargs="+", default=DEFAULT_OPTIMIZERS,
                    choices=["AdamW", "SGD"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override cfg.epochs (default: from --config)")
    ap.add_argument("--config", default="configs/cifar10_quick.yaml")
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--algorithm", choices=["coordinate", "random", "grid"],
                    default="coordinate")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--out", default=None,
                    help="Override summary path (default: ideas/<idea>/hillclimb_results.json)")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="random.Random seed for --algorithm random")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false", default=True)
    ap.add_argument("--no-commit", dest="do_commit",
                    action="store_false", default=True,
                    help="Disable per-cell git commit+push (testing)")
    args = ap.parse_args(argv)

    cube = dict(
        lr=args.lr, weight_decay=args.wd,
        batch_size=args.batch, optimizer=args.optimizer,
    )
    summary = hillclimb(
        tag=args.tag, idea=args.idea, config_path=args.config,
        cube=cube, seeds=args.seeds,
        algorithm=args.algorithm, budget=args.budget,
        root=args.root, epochs=args.epochs,
        skip_existing=args.skip_existing,
        do_commit=args.do_commit, rng_seed=args.rng_seed,
        out_path=args.out,
    )
    if summary["best_config"] is None:
        print("[hc] WARNING: no successful cells; nothing to confirm.")
        return 1
    bc = summary["best_config"]
    print(
        f"[hc] best: lr={bc['lr']} wd={bc['weight_decay']} "
        f"bs={bc['batch_size']} opt={bc['optimizer']} "
        f"top1_median={summary['best_top1_median']} "
        f"seeds={summary['seeds_confirmed']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
