# AUDIT — H50

> Adversarial self-critique of the falsified six-prior composition.
> This is **the headline negative result of the program**, so the
> audit doubles as a post-mortem.

## Weaknesses found in v1 (i.e., why the composition failed)

1. **Multiplicative latency stack.** Each prior individually adds
   1.5–2.5× per-block latency (hex +1.7×, group +2.2×, toroidal
   +2.1×, fractal +1.7×). At the block level these multiply rather
   than add: observed 20.02 ms vs 4.43 ms reference = **5.0× latency**.
   The composite metric penalises latency directly with weight ~0.15,
   subtracting ≈ 0.05 from the composite before considering top-1.
   This was foreseeable from the per-prior FINDINGS.md table and
   **should have been caught at design time** rather than after the
   12-min training run.
2. **C4 max-pool over the orbit destroys 75 % of the signal.**
   `GroupConv2d` (`priors.py:152`) emits a `(G, O, k, k)` orbit then
   collapses with `amax(dim=1)` over the 4-rotation orbit. Max-pool
   over 4 channels at every spatial location preserves only the
   strongest of 4 features — at every layer. This is the dominant
   single-prior negative (T1.4: -10.27 pp top-1 in isolation) and
   it propagates straight into the full hybrid. The H58 follow-up
   tried `reduce='mean'` and **made things worse** — see
   `ideas/58_group_avg_pool/FINDINGS-quoted`.
3. **Priors are not orthogonal in the composition.** Hex conv changes
   the spatial neighborhood; group conv then operates on a tensor
   whose receptive field is no longer rotation-equivariant in the
   sense the C4 prior assumes; fractal recursion adds two more layers
   that compound the deviation. The implicit assumption in the source
   PDF — that each prior "adds" a property independently — does not
   hold under composition. This is **the central scientific lesson**
   of the campaign and is recorded in `FINDINGS.md`.

## Bugs caught by tests (good)

- `tests/test_blocks.py::test_block_forward_shape_keeps_h_w` exercises
  every flag combination and would have caught any forward-shape
  regression introduced while wiring the six priors together. No such
  regression occurred; the failure is purely scientific, not a bug.

## Bugs NOT caught by tests but suspected

- The cymatic init is skipped silently for the group-conv path (see
  `blocks.py:_GenericConv.__init__`). In the full hybrid, the
  group-conv branch dominates the conv1 path, so the "cymatic init"
  prior is **partially inert** in the falsified configuration. The
  ablation row labeled `sg_only_cymatic_init` tested the init on a
  vanilla conv where it DID apply; the `sg_full_fib` row tested it on
  a conv where it largely DID NOT. The numbers in FINDINGS.md are
  reported faithfully, but the **interpretation** "all six priors are
  on" overstates what was actually composed.

## Performance / numerical-stability concerns

- 5× latency at b=1 is a deal-breaker for any deployment context. The
  composite metric correctly punishes it. The next refined hypothesis
  (H45, Sacred NAS) is explicitly tasked with finding a 2–3-prior
  subset that achieves composite >= 0.82 at reasonable latency.

## Mitigations queued for IMPROVEMENTS.md

- H58 (group avg-pool fix) — DONE, DISCARDED, see `ideas/58_group_avg_pool/`.
- H45 (Sacred NAS) — queued, expected to drop hex+toroidal+cymatic.
- H60 (3-seed error bars for the negative result) — queued.
- Leave-one-out from full hybrid (`sg_loo_no_*`) — queued; will
  attribute the single largest negative contributor in the
  composition.

## Sign-off

- 2026-04-12 — original audit at falsifier hit
- 2026-05-26 — Code-Agent-Y — re-signed for the `ideas/` taxonomy
