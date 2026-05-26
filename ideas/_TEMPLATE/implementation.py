"""H{{NN}} — {{Short Title}} — implementation module.

Re-exports the primitive(s) this idea relies on from the shared
infrastructure at ``nature_inspired_networks``. Idea-specific glue (a
wrapper, a config-bound helper, a new init function) lives here. Do
NOT duplicate primitives that already exist in
``nature_inspired_networks.priors`` — import them.
"""
from __future__ import annotations

# Example imports (delete / adjust per hypothesis):
# from nature_inspired_networks.priors import GroupConv2d
# from nature_inspired_networks.blocks import NaturePriorBlock, NaturePriorFlags


def idea_signature() -> dict:
    """Return a dict identifying this idea for the experiment log."""
    return dict(
        hypothesis_id="H{{NN}}",
        short="<short>",
        primitives_touched=[],
        flags_touched=[],
    )
