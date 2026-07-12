"""Book-derived policy, budget, score, and WDL targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from sakigo.rulesets import BLACK


def round_half_point(value: float) -> float:
    scaled = float(value) * 2.0
    if scaled >= 0:
        return math.floor(scaled + 0.5) / 2.0
    return math.ceil(scaled - 0.5) / 2.0


def normalize(values: Sequence[float]) -> list[float]:
    clean = [
        max(0.0, float(value)) if math.isfinite(float(value)) else 0.0
        for value in values
    ]
    total = sum(clean)
    if total <= 0:
        raise ValueError("target distribution has no positive mass")
    return [value / total for value in clean]


@dataclass(frozen=True)
class ConcreteBookMove:
    actions: tuple[int, ...]
    score_lead: float
    a_visits: float
    wl: float | None = None
    is_other: bool = False


def _concrete(moves: Iterable[ConcreteBookMove]) -> list[ConcreteBookMove]:
    return [move for move in moves if not move.is_other and move.actions]


def book_policy(
    moves: Iterable[ConcreteBookMove], *, to_move: int, action_count: int
) -> tuple[list[float], float]:
    concrete = _concrete(moves)
    if not concrete:
        raise ValueError("book node has no concrete moves")
    rounded = [round_half_point(move.score_lead) for move in concrete]
    optimal = max(rounded) if to_move == BLACK else min(rounded)
    actions = sorted(
        action
        for move, score in zip(concrete, rounded, strict=True)
        if score == optimal
        for action in move.actions
    )
    policy = [0.0] * action_count
    mass = 1.0 / len(actions)
    for action in actions:
        policy[action] = mass
    return policy, optimal


def book_budget(
    moves: Iterable[ConcreteBookMove], *, action_count: int
) -> list[float]:
    budget = [0.0] * action_count
    for move in _concrete(moves):
        if move.a_visits < 0:
            raise ValueError("AVisits must be non-negative")
        share = move.a_visits / len(move.actions)
        for action in move.actions:
            budget[action] += share
    return normalize(budget)


def book_wdl(
    rounded_black_score: float,
    *,
    to_move: int,
    book_wl: float,
) -> list[float]:
    mover_score = rounded_black_score if to_move == BLACK else -rounded_black_score
    if mover_score == 0.0:
        return [0.0, 1.0, 0.0, 0.0]
    black_win = min(1.0, max(0.0, 0.5 * (1.0 - float(book_wl))))
    mover_win = black_win if to_move == BLACK else 1.0 - black_win
    return [mover_win, 0.0, 1.0 - mover_win, 0.0]
