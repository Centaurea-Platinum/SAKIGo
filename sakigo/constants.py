"""Frozen cross-module constants (see sakigo/CONTRACTS.md).

Ported unchanged from Training/common.py (values are part of the data contract).
"""

from __future__ import annotations

SCHEMA_VERSION = 1
DISTILLATION_SCHEMA_VERSION = 3
BOARD_PLANE_COUNT = 6
RULE_FEATURE_COUNT = 10
WDL_LABELS = ("win", "draw", "loss", "no_result")
HEADS = ("wdl", "score", "policy", "budget")
ACTION_HEADS = ("policy", "budget")

# Board planes, in order (perspective = side to move):
# 0 MyStones, 1 OpponentStones, 2 Empty, 3 BoundaryCorner, 4 BoundaryEdge, 5 NonTrivialIllegal.
BOARD_PLANE_NAMES = (
    "my_stones",
    "opponent_stones",
    "empty",
    "boundary_corner",
    "boundary_edge",
    "non_trivial_illegal",
)

# Action vectors (policy/budget/legal_mask) have length N*N + 1; the pass logit is LAST.
PASS_IS_LAST = True
