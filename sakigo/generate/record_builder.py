"""Build schema-valid records directly from KataGo book nodes."""

from __future__ import annotations

import random
from hashlib import blake2b
from typing import Any

import numpy as np

from sakigo.constants import DISTILLATION_SCHEMA_VERSION
from sakigo.generate.book import row_actions
from sakigo.generate.d4 import transform_action
from sakigo.generate.game import GeneratorGame, index_from_coord
from sakigo.generate.targets import (
    ConcreteBookMove,
    book_budget,
    book_policy,
    book_wdl,
    round_half_point,
)
from sakigo.rulesets import BLACK, RulesetSpec


def replay(history: list[list[str]], ruleset: RulesetSpec) -> GeneratorGame:
    game = GeneratorGame(0, random.Random(0), 9, ruleset)
    for color, coord in history:
        expected = "B" if game.to_move == BLACK else "W"
        if color.upper() != expected:
            raise ValueError(f"history color mismatch: expected {expected}, got {color}")
        game.play(index_from_coord(coord, 9))
    return game


def model_visible_position_key(game: GeneratorGame) -> str:
    board, rules, _ = game.model_inputs()
    digest = blake2b(digest_size=20)
    digest.update(np.asarray(board, dtype="<f4").tobytes())
    digest.update(np.asarray(rules, dtype="<f4").tobytes())
    return digest.hexdigest()


def _row_number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            return float(value)
    return None


def concrete_book_moves(task: dict[str, Any]) -> list[ConcreteBookMove]:
    symmetry = int(task.get("page_to_history_symmetry", 0))
    output: list[ConcreteBookMove] = []
    for row in task["moves"]:
        label = str(row.get("move", "")).lower()
        actions = row_actions(row)
        if symmetry:
            actions = tuple(transform_action(action, 9, symmetry) for action in actions)
        ss_m = _row_number(row, "ssM", "scoreMean", "score")
        output.append(
            ConcreteBookMove(
                actions=actions,
                score_lead=0.0 if ss_m is None else -ss_m,
                a_visits=_row_number(row, "av", "AVisits", "aVisits") or 0.0,
                wl=_row_number(row, "wl", "winLossValue"),
                is_other=label == "other" or bool(row.get("isOther", False)),
            )
        )
    return output


def build_book_training_record(
    task: dict[str, Any], *, ruleset: RulesetSpec
) -> dict[str, Any]:
    game = replay([list(move) for move in task["history"]], ruleset)
    board_planes, rule_features, legal_mask = game.model_inputs()
    moves = concrete_book_moves(task)
    policy, rounded_black_score = book_policy(
        moves, to_move=game.to_move, action_count=82
    )
    budget = book_budget(moves, action_count=82)
    optimal_wl = [
        move.wl
        for move in moves
        if not move.is_other
        and move.actions
        and round_half_point(move.score_lead) == rounded_black_score
        and move.wl is not None
    ]
    if not optimal_wl:
        raise ValueError(f"book node {task['node_id']} has no W/L for an optimal move")
    mover_score = rounded_black_score if game.to_move == BLACK else -rounded_black_score
    return {
        "schema_version": DISTILLATION_SCHEMA_VERSION,
        "board_size": 9,
        "ply": game.ply,
        "position_key": model_visible_position_key(game),
        "ruleset": ruleset.metadata(),
        "board_planes": board_planes,
        "rule_features": rule_features,
        "wdl": book_wdl(
            rounded_black_score,
            to_move=game.to_move,
            book_wl=sum(optimal_wl) / len(optimal_wl),
        ),
        "score": mover_score / 81,
        "policy": policy,
        "budget": budget,
        "legal_mask": legal_mask,
        "source": {
            "kind": "book",
            "split": task["split"],
            "task_id": task["task_id"],
            "task_index": task["task_index"],
            "node_id": task["node_id"],
            "book": "book9x9tt-20260226",
        },
    }
