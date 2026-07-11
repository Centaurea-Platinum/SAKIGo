"""Teacher-record construction from KataGo analysis responses.

Ported behavior-identically from Training/generate_katago_phase1.py —
the perspective-flip block and budget/policy derivation are correctness
invariants (CONTRACTS.md).
"""

from __future__ import annotations

import math
import random
from typing import Any

from sakigo.constants import SCHEMA_VERSION
from sakigo.generate.game import GeneratorGame
from sakigo.rulesets import BLACK


def normalize_policy(raw_policy: list[float], legal_mask: list[bool]) -> list[float]:
    values = [
        max(0.0, float(value)) if legal_mask[index] else 0.0
        for index, value in enumerate(raw_policy)
    ]
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total):
        values = [0.0] * len(legal_mask)
        values[-1] = 1.0
        return values
    return [value / total for value in values]


def one_hot_top1(distribution: list[float]) -> list[float]:
    target = [0.0] * len(distribution)
    target[max(range(len(distribution)), key=distribution.__getitem__)] = 1.0
    return target


def sample_action(distribution: list[float], rng: random.Random, temperature: float) -> int:
    if temperature <= 0.0:
        return max(range(len(distribution)), key=distribution.__getitem__)
    if temperature != 1.0:
        adjusted = [value ** (1.0 / temperature) if value > 0.0 else 0.0 for value in distribution]
        total = sum(adjusted)
        distribution = [value / total for value in adjusted] if total > 0.0 else distribution
    threshold = rng.random()
    cumulative = 0.0
    for index, value in enumerate(distribution):
        cumulative += value
        if threshold <= cumulative:
            return index
    return len(distribution) - 1


def record_from_response(
    game: GeneratorGame, response: dict[str, Any]
) -> tuple[dict[str, Any], list[float]]:
    raw_policy = response.get("policy")
    ownership = response.get("ownership")
    root_info = response.get("rootInfo", {})
    if not isinstance(raw_policy, list) or len(raw_policy) != game.action_count:
        raise ValueError(f"bad policy in response {response.get('id')}")
    if not isinstance(ownership, list) or len(ownership) != game.area:
        raise ValueError(f"bad ownership in response {response.get('id')}")

    board_planes, rule_features, legal_mask = game.model_inputs()
    budget = normalize_policy([float(value) for value in raw_policy], legal_mask)
    policy = one_hot_top1(budget)

    black_win = float(root_info["rawWinrate"])
    draw = float(root_info.get("rawDrawProb", root_info.get("drawProb", 0.0)))
    no_result = float(root_info.get("rawNoResultProb", 0.0))
    black_loss = max(0.0, 1.0 - black_win - draw - no_result)
    if game.to_move == BLACK:
        wdl = [black_win, draw, black_loss, no_result]
        score = float(root_info["rawLead"]) / game.area
        ownership_target = [float(value) for value in ownership]
    else:
        wdl = [black_loss, draw, black_win, no_result]
        score = -float(root_info["rawLead"]) / game.area
        ownership_target = [-float(value) for value in ownership]

    record = {
        "schema_version": SCHEMA_VERSION,
        "board_size": game.board_size,
        "ply": game.ply,
        "position_key": game.position_key(),
        "ruleset": game.ruleset.metadata(),
        "board_planes": board_planes,
        "rule_features": rule_features,
        "wdl": wdl,
        "score": score,
        "ownership": ownership_target,
        "policy": policy,
        "budget": budget,
        "legal_mask": legal_mask,
        "source": {
            "teacher": "katago",
            "katago_id": response.get("id"),
            "katago_rules": game.ruleset.katago_rules,
            "katago_play": {
                "ko": game.ruleset.katago_ko,
                "suicide": game.ruleset.katago_suicide,
            },
            "saki_rules": {
                "scoring": game.ruleset.scoring,
                "ko": game.ruleset.ko,
                "suicide": game.ruleset.suicide,
            },
            "reportAnalysisWinratesAs": "BLACK",
            "rawLead": root_info.get("rawLead"),
            "rawWinrate": root_info.get("rawWinrate"),
            "rawDrawProb": root_info.get("rawDrawProb", root_info.get("drawProb")),
            "rawNoResultProb": root_info.get("rawNoResultProb"),
        },
    }
    return record, budget
