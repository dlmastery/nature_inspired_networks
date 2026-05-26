---
name: autoresearch-modular-block
description: Use when designing a toggleable, ablation-friendly composable neural building block. Each design choice maps to a single Boolean flag that can be flipped in isolation, enabling clean leave-one-out analyses.
---

# Skill — Design an ablation-friendly modular block

## When to use

Any time you propose a "novel block" that combines multiple ideas.
Without modular toggles you cannot attribute headline effects to
specific design choices — the block becomes a black box.

## The flags dataclass

```python
@dataclass
class BlockFlags:
    """One Boolean per orthogonal design choice. Adding a flag is
    cheap; removing one mid-campaign breaks ablation comparisons."""
    flag_a: bool = True
    flag_b: bool = True
    flag_c: bool = True
    # ...

    def tag(self) -> str:
        on = [k for k, v in self.__dict__.items() if v]
        return "+".join(on) if on else "vanilla"
```

The block consumes the flags in `__init__` and dispatches to
sub-modules. **Never** branch on a flag inside `forward()` if you can
avoid it — pre-dispatch at construction time so the forward graph is
static.

## The block skeleton

```python
class MyBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1,
                 flags: BlockFlags | None = None):
        super().__init__()
        flags = flags or BlockFlags()
        self.flags = flags

        # Pre-dispatch sub-modules from flags. Each component is its
        # own class so the unit test for that component lives next
        # to it.
        if flags.flag_a:
            self.op1 = OptionAImpl(c_in, c_out, stride)
        else:
            self.op1 = OptionBImpl(c_in, c_out, stride)
        # ...

    def forward(self, x):
        y = self.op1(x)
        # ...
        return y
```

## Hard rules

1. **One flag = one orthogonal design choice.** If flipping flag_a
   only makes sense when flag_b is on, you have one design choice
   (the pair), not two. Merge them.
2. **Vanilla = all flags False = the literature baseline scaffold.**
   The block with everything off should be functionally identical to
   the lit-anchored reference (ResNet-20 / EfficientNet-B0 / ViT-Tiny
   / whatever).
3. **Component classes live next to their flag.** `OptionAImpl` and
   `OptionBImpl` should be defined within ~50 lines of the flag that
   selects them.
4. **Each component has its own unit test.** `tests/test_components.py`
   exercises `OptionAImpl` in isolation; the block-level smoke test
   just confirms shape and gradient flow.
5. **No silent compounding.** Don't tie flag_a's behaviour to flag_b
   via a global mutable. If two components must interact, surface
   that as a third explicit flag.

## Constructor smoke test (mandatory)

```python
import itertools
def test_all_flag_combos_forward():
    for combo in itertools.product([False, True], repeat=N_FLAGS):
        flags = BlockFlags(*combo)
        m = MyBlock(c_in=16, c_out=32, stride=2, flags=flags)
        y = m(torch.randn(2, 16, 8, 8))
        assert y.shape == (2, 32, 4, 4), (flags.tag(), y.shape)
```

If this test fails on any combo, the block is not actually modular.

## Output: the ablation matrix is automatic

Once the flags are clean, the sweep matrix writes itself:

| row | description |
|---|---|
| `baseline_lit` | reference architecture from the literature |
| `baseline_vanilla` | your block, all flags False |
| `only_<flag>` × N | each flag flipped on alone |
| `loo_no_<flag>` × N | each flag flipped off from the all-on state |
| `full` | all flags True |

That's `2 + 2N + 1 = 2N + 3` rows; pair with 3 seeds.

## Anti-patterns

- Flags whose default values are not the "vanilla" state — confusing.
- Flags that secretly multiply your parameter count (e.g., `fractal`
  silently turns one conv into two). Document the cost in the block's
  docstring AND in the ablation README.
- Configuration objects with non-Boolean knobs labeled as "flags"
  (e.g., `kernel_size`, `dropout_p`). Those are hyperparameters,
  not flags; they belong in a separate `BlockConfig`.

## Where this pays off

A modular block lets you cleanly answer "which one design choice did
the headline number come from?" — which is the question reviewers
always ask. Without it, the paper claims "our block is 30% more
efficient"; with it, the paper claims "the fractal sub-block
contributes X pp, the hex mask contributes Y pp, the group conv
contributes Z pp."
