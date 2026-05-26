"""H50 - idea-specific experiment driver."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one experiment for H50 (full sacred hybrid).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    from nature_inspired_networks.runner import run_one  # noqa: E402
    import yaml  # noqa: E402

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    root = (args.root
            or str(Path(__file__).resolve().parent / "experiments" / args.tag / "run"))
    out = run_one(cfg, tag=args.tag, seed=args.seed, root=root)
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
