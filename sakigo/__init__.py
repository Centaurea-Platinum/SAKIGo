"""SAKIGo rebuilt Python stack.

Modular packages:
- sakigo.model: D4-equivariant network (group_size-parameterized).
- sakigo.data: record schema, datasets, augmentation.
- sakigo.train: trainer, losses, metrics, checkpoints.
- sakigo.generate: KataGo phase-1 teacher data generation.
- sakigo.eval: self-play evaluation.
- sakigo.engine: Rust engine bindings.

See sakigo/CONTRACTS.md for the frozen cross-module contracts.
"""

from sakigo.constants import (
    ACTION_HEADS,
    BOARD_PLANE_COUNT,
    HEADS,
    RULE_FEATURE_COUNT,
    SCHEMA_VERSION,
    WDL_LABELS,
)

__all__ = [
    "ACTION_HEADS",
    "BOARD_PLANE_COUNT",
    "HEADS",
    "RULE_FEATURE_COUNT",
    "SCHEMA_VERSION",
    "WDL_LABELS",
]
