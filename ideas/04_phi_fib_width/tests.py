"""H04 — unit tests for the idea's implementation.

Run with:
    python ideas/04_phi_fib_width/tests.py

Output must end with "All N tests passed." or fail loudly. No pytest
required.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project src/ AND this idea dir importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_idea_signature_present():
    """The idea must expose a signature dict so the runner can log it."""
    from implementation import idea_signature
    sig = idea_signature()
    assert isinstance(sig, dict)
    assert sig["hypothesis_id"] == "H04"
    assert sig["short"] == "phi_fib_width"
    assert "fibonacci_channels" in sig["primitives_touched"]


def test_phi_widths_strictly_increasing_at_c0_32_n4():
    """The recommended H04 config c0=32, n=4 must produce a strictly
    monotone schedule for all three modes."""
    from implementation import phi_fib_widths
    for mode in ("phi", "fib", "linear"):
        w = phi_fib_widths(32, 4, mode=mode)
        assert len(w) == 4, (mode, w)
        assert w == sorted(w), (mode, w)
        # No duplicates -- mod-8 collapse to constant must not happen here
        assert len(set(w)) == 4, (mode, w)


def test_schedules_distinct_at_recommended_config():
    """Regression test for the T1.1/T1.2 mod-8 collapse.

    At the H04-recommended config (c0=32, n_stages=4) phi and fib must
    produce STRICTLY DIFFERENT integer schedules. If this regresses,
    every future phi-vs-fib ablation degenerates into a null study.
    """
    from implementation import phi_fib_widths, schedules_are_distinct
    assert schedules_are_distinct(32, 4)
    fib = phi_fib_widths(32, 4, mode="fib")
    phi = phi_fib_widths(32, 4, mode="phi")
    # Concrete expected values (locked-in golden numbers):
    # fib: 32 * [1, 2, 3, 5] / 1 = [32, 64, 96, 160] mod-8 -> [32, 64, 96, 160]
    # phi: 32 * phi^[0..3] = [32, 51.78, 83.75, 135.45] mod-8 -> [32, 48, 80, 136]
    assert fib == [32, 64, 96, 160], fib
    assert phi == [32, 48, 80, 136], phi
    assert fib != phi


def test_mod8_collapse_caught_at_c0_16_n3():
    """At the legacy T1.1/T1.2 config (c0=16, n_stages=3) the mod-8
    rounding makes phi and fib NEARLY collapse. The first two stages
    are identical; only stage 3 differs (32 vs 40). The
    ``schedules_are_distinct`` returns True technically, but we lock in
    the historical numbers so a future regression of the primitive is
    visible."""
    from implementation import phi_fib_widths
    fib_16_3 = phi_fib_widths(16, 3, mode="fib")
    phi_16_3 = phi_fib_widths(16, 3, mode="phi")
    # Historical T1.1/T1.2 actual integer widths
    assert fib_16_3 == [16, 24, 32], fib_16_3
    assert phi_16_3 == [16, 24, 40], phi_16_3
    # First two stages match -- this is the "near collapse" that
    # made T1.1 and T1.2 produce identical top-1 at single seed.
    assert fib_16_3[0:2] == phi_16_3[0:2]


def test_rejects_c0_below_8():
    """c0 < 8 makes every stage mod-8-round to 8, collapsing every
    schedule. The wrapper must refuse rather than silently degenerate."""
    from implementation import phi_fib_widths
    try:
        phi_fib_widths(4, 3, mode="phi")
    except ValueError as exc:
        assert "below 8" in str(exc)
    else:
        raise AssertionError("expected ValueError for c0=4")


def test_phi_growth_ratio_close_to_phi():
    """Successive widths in phi mode should grow by roughly phi=1.618,
    modulo mod-8 quantisation. We test the ratio b/a for c0=32,n=4."""
    from implementation import phi_fib_widths
    from nature_inspired_networks.priors import PHI
    w = phi_fib_widths(32, 4, mode="phi")
    for a, b in zip(w[:-1], w[1:]):
        ratio = b / a
        # tolerate +/- 25 % deviation from PHI due to mod-8 quantisation
        assert PHI * 0.75 < ratio < PHI * 1.25, (a, b, ratio)


def test_linear_mode_is_strict_doubling_proxy():
    """The 'linear' mode is the control baseline: c0 * (k+1). Should be
    monotone and contain no surprises."""
    from implementation import phi_fib_widths
    w = phi_fib_widths(16, 3, mode="linear")
    # 16, 32, 48 -- arithmetic growth
    assert w == [16, 32, 48], w


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in globals().items()
           if k.startswith("test_") and inspect.isfunction(v)]
    for f in fns:
        f()
        print(f"  ok {f.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
