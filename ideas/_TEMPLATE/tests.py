"""H{{NN}} — unit tests for the idea's implementation.

Run with:
    python ideas/<NN_short>/tests.py

Output must end with "All N tests passed." or fail loudly. No pytest
required.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project src/ importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_idea_signature_present():
    """The idea must expose a signature dict so the runner can log it."""
    from implementation import idea_signature
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert "hypothesis_id" in sig and sig["hypothesis_id"].startswith("H")


# Add idea-specific tests below. Mandatory test coverage per CLAUDE.md
# rule 12:
# - canonical forward path (shape + dtype assert)
# - every Boolean-flag combination
# - every reduction / branching option
# - one regression test named after the bug class this idea addresses


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
