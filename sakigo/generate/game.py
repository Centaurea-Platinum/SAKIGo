"""Generator game state: the Rust engine plus the per-game bookkeeping the
KataGo analysis loop needs (move list for queries, ply/pass tracking,
position keys).

Legality, board state, captures, and the model encoding all come from
sakigo.engine (single rules implementation); the preset rulesets project
exactly onto KataGo's play rules (enforced by RulesetSpec), and the Rust
engine's agreement with the legacy generator Game is pinned by
    tests/test_engine_binding.py.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from sakigo.engine import Game as EngineGame
from sakigo.rulesets import BLACK, RulesetSpec

LETTERS = "ABCDEFGHJKLMNOPQRST"


def coord_from_index(index: int, board_size: int) -> str:
    row, col = divmod(index, board_size)
    return f"{LETTERS[col]}{board_size - row}"


def index_from_coord(coord: str, board_size: int) -> int:
    if coord.lower() == "pass":
        return board_size * board_size
    col = LETTERS.index(coord[0].upper())
    row = board_size - int(coord[1:])
    return row * board_size + col


class GeneratorGame:
    """One in-flight generation game."""

    def __init__(
        self,
        game_id: int,
        rng: random.Random,
        board_size: int,
        ruleset: RulesetSpec,
    ) -> None:
        self.rng = rng
        self.reset(game_id, board_size=board_size, ruleset=ruleset)

    def reset(
        self,
        game_id: int,
        *,
        board_size: int | None = None,
        ruleset: RulesetSpec | None = None,
    ) -> None:
        self.game_id = game_id
        if board_size is not None:
            self.board_size = board_size
        if ruleset is not None:
            self.ruleset = ruleset
        self.engine = EngineGame(
            self.board_size,
            self.ruleset.scoring,
            self.ruleset.ko,
            self.ruleset.suicide,
            self.ruleset.komi,
        )
        self.moves: list[list[str]] = []
        self.passes = 0
        self.ply = 0

    @property
    def area(self) -> int:
        return self.board_size * self.board_size

    @property
    def action_count(self) -> int:
        return self.area + 1

    @property
    def to_move(self) -> int:
        return self.engine.to_move

    def legal_mask(self) -> list[bool]:
        return self.engine.legal_mask()

    def board_planes(self) -> list[float]:
        return self.engine.board_planes()

    def rule_features(self) -> list[float]:
        return self.engine.rule_features()

    def model_inputs(self) -> tuple[list[float], list[float], list[bool]]:
        if hasattr(self.engine, "model_inputs"):
            board, rules, legal = self.engine.model_inputs()
            return list(board), list(rules), list(legal)
        return self.board_planes(), self.rule_features(), self.legal_mask()

    def play(self, action: int) -> None:
        color = "B" if self.engine.to_move == BLACK else "W"
        self.engine.play(action)
        if action == self.area:
            self.moves.append([color, "pass"])
            self.passes += 1
        else:
            self.moves.append([color, coord_from_index(action, self.board_size)])
            self.passes = 0
        self.ply += 1

    def should_reset(self, max_plies: int) -> bool:
        return self.passes >= 2 or self.ply >= max_plies

    def position_key(self) -> str:
        payload = json.dumps([self.moves, self.engine.to_move], separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:20]

    def query(self, query_id: str) -> str:
        payload: dict[str, Any] = {
            "id": query_id,
            "moves": self.moves,
            **self.ruleset.query_fields(),
            "boardXSize": self.board_size,
            "boardYSize": self.board_size,
            "maxVisits": 1,
            "includePolicy": True,
            "includeOwnership": True,
            "includeNoResultValue": True,
            "analysisPVLen": 1,
            "overrideSettings": {
                "rootNumSymmetriesToSample": 1,
                "wideRootNoise": 0.0,
            },
        }
        return json.dumps(payload, separators=(",", ":"))
