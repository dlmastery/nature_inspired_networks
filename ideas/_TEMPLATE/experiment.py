"""H{{NN}} — idea-specific experiment driver.

Thin wrapper that loads a YAML config from ``configs/`` and calls the
project runner. The experiment archive is written under
``experiments/exp001_<short>/run_seed<S>/`` so each archived
sub-directory is self-contained per CLAUDE.md rule #8.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap the project's src/ onto sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nature_inspired_networks.runner import run_one  # noqa: E402
import yaml  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one experiment for H{{NN}}.")
    ap.add_argument("--config", required=True,
                    help="Path to config YAML in this idea's configs/ dir.")
    ap.add_argument("--tag", required=True,
                    help="Run tag, also used as archive sub-dir name "
                         "(e.g., exp001_<short>).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None,
                    help="Archive root; defaults to "
                         "ideas/<NN_short>/experiments/<tag>/run")
    args = ap.parse_args(argv)

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
