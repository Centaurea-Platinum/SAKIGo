from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Training.common import BOARD_PLANE_COUNT, ROOT, SCHEMA_VERSION
from Training.data import is_zstd_jsonl_path, open_jsonl_writer
from Training.rulesets import (
    BLACK,
    WHITE,
    RulesetSpec,
    available_rulesets,
    ruleset_from_name,
    ruleset_from_overrides,
)


BOARD_SIZE = 19
AREA = BOARD_SIZE * BOARD_SIZE
ACTION_COUNT = AREA + 1
KOMI = ruleset_from_name("tromp-taylor").komi
LETTERS = "ABCDEFGHJKLMNOPQRST"
NEIGHBOR_CACHE: dict[int, list[tuple[int, ...]]] = {}
DEFAULT_BOARD_SIZES = "13,16,19"
DEFAULT_KOMIS = ",".join(f"{index * 0.5:.1f}" for index in range(27))


def katago_executable_names(platform_name: str | None = None) -> tuple[str, ...]:
    platform_key = (platform_name or sys.platform).lower()
    if platform_key.startswith("win"):
        return ("katago.exe", "katago")
    return ("katago", "katago.exe")


def find_katago_path(engine_root: Path, platform_name: str | None = None) -> Path:
    candidates: list[Path] = []
    for executable_name in katago_executable_names(platform_name):
        candidates.extend(sorted(engine_root.glob(f"*/{executable_name}")))
    if not candidates:
        names = ", ".join(katago_executable_names(platform_name))
        raise FileNotFoundError(f"could not find Distillation/engine/*/{{{names}}}")
    return candidates[0]


def default_katago_path() -> Path:
    return find_katago_path(ROOT / "Distillation" / "engine")


def default_config_path(katago_path: Path) -> Path:
    return katago_path.parent / "analysis_example.cfg"


def default_model_path() -> Path:
    candidates = sorted((ROOT / "Distillation" / "models").glob("*.bin.gz"))
    if not candidates:
        raise FileNotFoundError("could not find Distillation/models/*.bin.gz")
    return candidates[0]


def coord_from_index(index: int, board_size: int = BOARD_SIZE) -> str:
    row, col = divmod(index, board_size)
    return f"{LETTERS[col]}{board_size - row}"


def index_from_coord(coord: str, board_size: int = BOARD_SIZE) -> int:
    if coord.lower() == "pass":
        return board_size * board_size
    col = LETTERS.index(coord[0].upper())
    row = board_size - int(coord[1:])
    return row * board_size + col


def neighbors_for_board(board_size: int) -> list[tuple[int, ...]]:
    cached = NEIGHBOR_CACHE.get(board_size)
    if cached is not None:
        return cached
    area = board_size * board_size
    neighbors_by_point: list[tuple[int, ...]] = []
    for point in range(area):
        row, col = divmod(point, board_size)
        neighbors: list[int] = []
        if row > 0:
            neighbors.append(point - board_size)
        if row + 1 < board_size:
            neighbors.append(point + board_size)
        if col > 0:
            neighbors.append(point - 1)
        if col + 1 < board_size:
            neighbors.append(point + 1)
        neighbors_by_point.append(tuple(neighbors))
    NEIGHBOR_CACHE[board_size] = neighbors_by_point
    return neighbors_by_point


NEIGHBORS = neighbors_for_board(BOARD_SIZE)
DEFAULT_RULESETS = ",".join(available_rulesets())
DEFAULT_SAMPLES_PER_COMBINATION = 2**13
DEFAULT_SAMPLES_PER_FILE = 2**16
DEFAULT_ZSTD_LEVEL = 3
PROGRESS_WIDTH = 24
PROGRESS_MIN_INTERVAL_SECONDS = 0.25


def _duration_text(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


class _TerminalPalette:
    """Small prettyterm adapter with an ANSI fallback.

    The optional prettyterm package is not required for non-interactive or CI
    runs. When it is available, this tries a few common style-call shapes and
    falls back to direct ANSI escape codes otherwise.
    """

    _ANSI = {
        "cyan": "36",
        "green": "32",
        "yellow": "33",
        "dim": "2",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        try:
            import prettyterm  # type: ignore[import-not-found]
        except Exception:
            prettyterm = None
        self._prettyterm = prettyterm

    def style(self, text: str, color: str, *, bold: bool = False) -> str:
        if not self.enabled:
            return text
        styled = self._prettyterm_style(text, color, bold=bold)
        if styled is not None:
            return styled
        codes: list[str] = []
        if bold:
            codes.append("1")
        ansi = self._ANSI.get(color)
        if ansi is not None:
            codes.append(ansi)
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def _prettyterm_style(self, text: str, color: str, *, bold: bool) -> str | None:
        prettyterm = self._prettyterm
        if prettyterm is None:
            return None
        for name in ("style", "color", "paint"):
            function = getattr(prettyterm, name, None)
            if not callable(function):
                continue
            for kwargs in ({"fg": color, "bold": bold}, {"color": color, "bold": bold}):
                try:
                    return str(function(text, **kwargs))
                except TypeError:
                    continue
        function = getattr(prettyterm, color, None)
        if callable(function):
            try:
                return str(function(text))
            except TypeError:
                return None
        return None


class GenerationProgressBar:
    """Brief single-line progress display for interactive generation runs."""

    def __init__(
        self,
        target_samples: int,
        *,
        enabled: bool,
        width: int = PROGRESS_WIDTH,
        color: bool = True,
    ) -> None:
        self.target = max(1, target_samples)
        self.enabled = enabled
        self.width = max(4, width)
        self.started_at = time.monotonic()
        self._last_render = 0.0
        self._line_len = 0
        self._palette = _TerminalPalette(enabled and color)

    def render(
        self,
        *,
        samples: int,
        samples_per_second: float,
        eta_seconds: float,
        completed_combinations: int,
        combination_count: int,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render < PROGRESS_MIN_INTERVAL_SECONDS:
            return
        self._last_render = now
        ratio = min(1.0, max(0.0, samples / self.target))
        filled = min(self.width, int(self.width * ratio))
        remaining = self.width - filled
        bar = (
            self._palette.style("#" * filled, "green")
            + self._palette.style("-" * remaining, "dim")
        )
        label = self._palette.style("generate", "cyan", bold=True)
        pct = self._palette.style(f"{ratio * 100.0:5.1f}%", "yellow", bold=True)
        text = (
            f"{label} [{bar}] {pct} {samples:,}/{self.target:,} visits "
            f"{samples_per_second:,.1f}/s eta {_duration_text(eta_seconds)} "
            f"combos {completed_combinations}/{combination_count}"
        )
        padding = max(self._line_len - len(text), 0)
        print("\r" + text + " " * padding, end="", flush=True)
        self._line_len = len(text)

    def clear(self) -> None:
        if not self.enabled or self._line_len == 0:
            return
        print("\r" + " " * self._line_len + "\r", end="", flush=True)
        self._line_len = 0


@dataclass(frozen=True)
class GenerationVariant:
    board_size: int
    ruleset: RulesetSpec

    def key(self) -> str:
        return f"{self.board_size}|{self.ruleset.key()}"

    def metadata(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "ruleset": self.ruleset.metadata(),
        }


@dataclass(frozen=True)
class GenerationPlan:
    variants: list[GenerationVariant]
    quotas: dict[str, int]
    quota_mode: str
    samples_per_combination: int | None

    @property
    def target_samples(self) -> int:
        return sum(self.quotas.values())


class GenerationSchedule:
    def __init__(self, plan: GenerationPlan, rng: random.Random) -> None:
        self.plan = plan
        self.rng = rng
        self.written = {variant.key(): 0 for variant in plan.variants}
        self.reserved = {variant.key(): 0 for variant in plan.variants}
        self.total_written = 0

    def can_reserve(self, variant: GenerationVariant) -> bool:
        key = variant.key()
        return self.written[key] + self.reserved[key] < self.plan.quotas[key]

    def choose_variant(self) -> GenerationVariant | None:
        available = [variant for variant in self.plan.variants if self.can_reserve(variant)]
        if not available:
            return None
        return self.rng.choice(available)

    def reserve(self, variant: GenerationVariant) -> bool:
        if not self.can_reserve(variant):
            return False
        self.reserved[variant.key()] += 1
        return True

    def complete(self, variant: GenerationVariant, *, success: bool) -> bool:
        key = variant.key()
        if self.reserved[key] <= 0:
            raise RuntimeError(f"generation schedule reservation underflow for {key}")
        self.reserved[key] -= 1
        if not success:
            return False
        if self.written[key] >= self.plan.quotas[key]:
            raise RuntimeError(f"generation schedule quota overflow for {key}")
        self.written[key] += 1
        self.total_written += 1
        return True


def group_and_liberties(
    board: list[int],
    start: int,
    board_size: int = BOARD_SIZE,
) -> tuple[set[int], set[int]]:
    color = board[start]
    stack = [start]
    stones = {start}
    liberties: set[int] = set()
    neighbors_by_point = neighbors_for_board(board_size)
    while stack:
        point = stack.pop()
        for neighbor in neighbors_by_point[point]:
            value = board[neighbor]
            if value == 0:
                liberties.add(neighbor)
            elif value == color and neighbor not in stones:
                stones.add(neighbor)
                stack.append(neighbor)
    return stones, liberties


@dataclass
class AnalyzedMove:
    board: list[int]
    captured_opponent: int
    captured_self: int


@dataclass
class Game:
    game_id: int
    rng: random.Random
    board_size: int = BOARD_SIZE
    ruleset: RulesetSpec = field(default_factory=lambda: ruleset_from_name("tromp-taylor"))
    board: list[int] = field(init=False)
    to_move: int = field(init=False)
    captures: list[int] = field(init=False)
    moves: list[list[str]] = field(init=False)
    seen: set[tuple[int, ...]] = field(init=False)
    seen_states: set[tuple[tuple[int, ...], int]] = field(init=False)
    simple_ko: int | None = field(init=False)
    passes: int = field(init=False)
    ply: int = field(init=False)

    def __post_init__(self) -> None:
        self.reset(self.game_id, board_size=self.board_size, ruleset=self.ruleset)

    @property
    def area(self) -> int:
        return self.board_size * self.board_size

    @property
    def action_count(self) -> int:
        return self.area + 1

    @property
    def neighbors(self) -> list[tuple[int, ...]]:
        return neighbors_for_board(self.board_size)

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
        self.board = [0] * self.area
        self.to_move = BLACK
        self.captures = [0, 0]
        self.moves = []
        self.seen = {tuple(self.board)}
        self.seen_states = {(tuple(self.board), BLACK)}
        self.simple_ko = None
        self.passes = 0
        self.ply = 0

    def analyze_play(self, point: int) -> AnalyzedMove | None:
        if point < 0 or point >= self.area or self.board[point] != 0:
            return None
        if self.ruleset.katago_ko == "simple_ko" and self.simple_ko == point:
            return None
        color = self.to_move
        opponent = -color
        board = self.board.copy()
        board[point] = color
        captured_points: set[int] = set()

        for neighbor in self.neighbors[point]:
            if board[neighbor] != opponent:
                continue
            stones, liberties = group_and_liberties(board, neighbor, self.board_size)
            if not liberties:
                captured_points.update(stones)
        for captured in captured_points:
            board[captured] = 0

        captured_self_points: set[int] = set()
        if board[point] == color:
            own_stones, own_liberties = group_and_liberties(board, point, self.board_size)
            if not own_liberties:
                if own_stones == {point}:
                    return None
                if not self.ruleset.allows_suicide:
                    return None
                captured_self_points.update(own_stones)
                for captured in captured_self_points:
                    board[captured] = 0

        board_key = tuple(board)
        if self.ruleset.uses_positional_superko and board_key in self.seen:
            return None
        if self.ruleset.uses_situational_superko and (board_key, -color) in self.seen_states:
            return None
        return AnalyzedMove(board, len(captured_points), len(captured_self_points))

    def legal_mask(self) -> list[bool]:
        mask = [self.analyze_play(index) is not None for index in range(self.area)]
        mask.append(True)
        return mask

    def play(self, action: int) -> None:
        color = self.to_move
        if action == self.area:
            self.moves.append(["B" if color == BLACK else "W", "pass"])
            self.to_move = -self.to_move
            self.seen_states.add((tuple(self.board), self.to_move))
            self.simple_ko = None
            self.passes += 1
            self.ply += 1
            return

        analysis = self.analyze_play(action)
        if analysis is None:
            # This should not happen if selection respected the legal mask.
            self.play(self.area)
            return
        self.board = analysis.board
        if color == BLACK:
            self.captures[0] += analysis.captured_opponent
            self.captures[1] += analysis.captured_self
        else:
            self.captures[1] += analysis.captured_opponent
            self.captures[0] += analysis.captured_self
        board_key = tuple(self.board)
        next_to_move = -self.to_move
        self.seen.add(board_key)
        self.seen_states.add((board_key, next_to_move))
        self.moves.append(["B" if color == BLACK else "W", coord_from_index(action, self.board_size)])
        self.simple_ko = self._next_simple_ko(action, analysis)
        self.to_move = next_to_move
        self.passes = 0
        self.ply += 1

    def _next_simple_ko(self, action: int, analysis: AnalyzedMove) -> int | None:
        if (
            self.ruleset.katago_ko != "simple_ko"
            or analysis.captured_opponent != 1
            or analysis.captured_self != 0
            or analysis.board[action] != self.to_move
        ):
            return None
        stones, liberties = group_and_liberties(analysis.board, action, self.board_size)
        if len(stones) != 1 or len(liberties) != 1:
            return None
        return next(iter(liberties))

    def should_reset(self, max_plies: int) -> bool:
        return self.passes >= 2 or self.ply >= max_plies

    def query(self, query_id: str) -> str:
        payload = {
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
        return json.dumps(
            payload,
            separators=(",", ":"),
        )

    def board_planes(self, legal_mask: list[bool]) -> list[float]:
        area = self.area
        planes = [0.0] * (BOARD_PLANE_COUNT * area)
        for index, value in enumerate(self.board):
            row, col = divmod(index, self.board_size)
            if value == self.to_move:
                planes[index] = 1.0
            elif value == -self.to_move:
                planes[area + index] = 1.0
            else:
                planes[2 * area + index] = 1.0
                if not legal_mask[index]:
                    planes[5 * area + index] = 1.0
            if (row == 0 or row == self.board_size - 1) and (col == 0 or col == self.board_size - 1):
                planes[3 * area + index] = 1.0
            elif row == 0 or row == self.board_size - 1 or col == 0 or col == self.board_size - 1:
                planes[4 * area + index] = 1.0
        return planes

    def rule_features(self) -> list[float]:
        return self.ruleset.rule_features(
            to_move=self.to_move,
            captures=self.captures,
            board_area=self.area,
        )

    def position_key(self) -> str:
        payload = json.dumps([self.moves, self.to_move], separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:20]


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


def record_from_response(game: Game, response: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    raw_policy = response.get("policy")
    ownership = response.get("ownership")
    root_info = response.get("rootInfo", {})
    if not isinstance(raw_policy, list) or len(raw_policy) != game.action_count:
        raise ValueError(f"bad policy in response {response.get('id')}")
    if not isinstance(ownership, list) or len(ownership) != game.area:
        raise ValueError(f"bad ownership in response {response.get('id')}")

    legal_mask = game.legal_mask()
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
        "board_planes": game.board_planes(legal_mask),
        "rule_features": game.rule_features(),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 SAKIGo records with KataGo.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Total records to generate. Overrides the per-combination default when set.",
    )
    parser.add_argument(
        "--samples-per-combination",
        type=int,
        default=None,
        help=f"Records per board-size/ruleset/komi combination. Default: {DEFAULT_SAMPLES_PER_COMBINATION}.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=DEFAULT_SAMPLES_PER_FILE,
        help=(
            "Records per numbered .jsonl.zst shard. "
            f"Default: {DEFAULT_SAMPLES_PER_FILE}. Set 0 to write one legacy output file."
        ),
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=DEFAULT_ZSTD_LEVEL,
        help=f"Zstandard compression level for .jsonl.zst output. Default: {DEFAULT_ZSTD_LEVEL}.",
    )
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--katago", type=Path, default=default_katago_path())
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--nn-batch-size", type=int, default=20)
    parser.add_argument("--analysis-threads", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1024)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render a brief colorful progress bar. Defaults to on when stdout is a terminal.",
    )
    parser.add_argument(
        "--board-sizes",
        default=DEFAULT_BOARD_SIZES,
        help=f"Comma-separated board sizes. Default: {DEFAULT_BOARD_SIZES}.",
    )
    parser.add_argument(
        "--ruleset",
        default=None,
        help="Legacy single named rule mapping. Mutually exclusive with --rulesets.",
    )
    parser.add_argument(
        "--rulesets",
        default=None,
        help=f"Comma-separated named rule mappings. Default: {DEFAULT_RULESETS}.",
    )
    parser.add_argument(
        "--katago-rules",
        default="",
        help="Override the KataGo analysis 'rules' field. Accepts a preset string or JSON object.",
    )
    parser.add_argument("--katago-ko", default="", help="Override local KataGo ko legality mapping.")
    parser.add_argument("--katago-suicide", default="", help="Override local KataGo suicide legality mapping.")
    parser.add_argument("--komi", type=float, default=None, help="Override komi for KataGo and SAKIGo features.")
    parser.add_argument(
        "--komis",
        default=None,
        help=f"Comma-separated komi values sampled per game. Default: {DEFAULT_KOMIS}.",
    )
    parser.add_argument("--saki-scoring", default="", help="Override SAKIGo scoring feature mapping.")
    parser.add_argument("--saki-ko", default="", help="Override SAKIGo ko feature mapping.")
    parser.add_argument("--saki-suicide", default="", help="Override SAKIGo suicide feature mapping.")
    return parser.parse_args(argv)


def progress_enabled(args: argparse.Namespace) -> bool:
    if args.progress is not None:
        return bool(args.progress)
    return sys.stdout.isatty()


def parse_board_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--board-sizes must include at least one size")
    max_size = len(LETTERS)
    if any(size <= 0 or size > max_size for size in sizes):
        raise ValueError(f"--board-sizes entries must be in [1, {max_size}]")
    if len(set(sizes)) != len(sizes):
        raise ValueError("--board-sizes must not contain duplicates")
    return sizes


def parse_komis(raw: str, default: float) -> list[float]:
    if not raw.strip():
        return [default]
    komis = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not komis:
        raise ValueError("--komis must include at least one value")
    if any(not math.isfinite(komi) for komi in komis):
        raise ValueError("--komis must contain only finite values")
    if len(set(komis)) != len(komis):
        raise ValueError("--komis must not contain duplicates")
    return komis


def parse_ruleset_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--rulesets must include at least one ruleset")
    canonical = [ruleset_from_name(name).name for name in names]
    if len(set(canonical)) != len(canonical):
        raise ValueError("--rulesets must not contain duplicates")
    return canonical


def build_generation_plan(
    board_sizes: list[int],
    base_rulesets: list[RulesetSpec],
    komis: list[float],
    *,
    samples: int | None,
    samples_per_combination: int | None,
) -> GenerationPlan:
    variants = [
        GenerationVariant(board_size=board_size, ruleset=ruleset.with_komi(komi))
        for board_size in board_sizes
        for ruleset in base_rulesets
        for komi in komis
    ]
    if not variants:
        raise ValueError("generation plan must include at least one combination")
    if samples is not None:
        if samples <= 0:
            raise ValueError("--samples must be positive")
        if samples_per_combination is not None:
            raise ValueError("use either --samples or --samples-per-combination, not both")
        base_quota, remainder = divmod(samples, len(variants))
        quotas = {
            variant.key(): base_quota + (1 if index < remainder else 0)
            for index, variant in enumerate(variants)
        }
        return GenerationPlan(
            variants=variants,
            quotas=quotas,
            quota_mode="total",
            samples_per_combination=None,
        )

    per_combination = (
        DEFAULT_SAMPLES_PER_COMBINATION
        if samples_per_combination is None
        else samples_per_combination
    )
    if per_combination <= 0:
        raise ValueError("--samples-per-combination must be positive")
    return GenerationPlan(
        variants=variants,
        quotas={variant.key(): per_combination for variant in variants},
        quota_mode="per_combination",
        samples_per_combination=per_combination,
    )


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _strip_jsonl_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.zstd", ".jsonl.zst", ".jsonl"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem if path.suffix else "samples"


class GenerationOutputWriter:
    def __init__(
        self,
        output: Path,
        *,
        samples_per_file: int,
        zstd_level: int,
    ) -> None:
        if samples_per_file < 0:
            raise ValueError("--samples-per-file must be non-negative")
        if zstd_level < 1 or zstd_level > 22:
            raise ValueError("--zstd-level must be in [1, 22]")
        self.output = output
        self.samples_per_file = samples_per_file
        self.zstd_level = zstd_level
        self.paths: list[Path] = []
        self._handle: Any = None
        self._shard_index = 0
        self._samples_in_file = 0
        self._single_file = samples_per_file == 0

        if self._single_file:
            self.directory = output.parent
            self.prefix = output.stem
        elif output.suffix:
            self.directory = output.parent
            self.prefix = _strip_jsonl_suffix(output)
        else:
            self.directory = output
            self.prefix = "samples"
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def data_format(self) -> str:
        if self._single_file:
            return "single_jsonl_zstd" if is_zstd_jsonl_path(self.output) else "legacy_single_jsonl_deprecated"
        return "jsonl_zstd_shards"

    def __enter__(self) -> GenerationOutputWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write_record(self, record: dict[str, Any]) -> None:
        if self._handle is None or (
            not self._single_file and self._samples_in_file >= self.samples_per_file
        ):
            self._rotate()
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._samples_in_file += 1

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _rotate(self) -> None:
        self.close()
        if self._single_file:
            path = self.output
        else:
            path = self.directory / f"{self.prefix}_{self._shard_index:06d}.jsonl.zst"
            self._shard_index += 1
        self.paths.append(path)
        self._samples_in_file = 0
        self._handle = open_jsonl_writer(path, compression_level=self.zstd_level)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.komi is not None and args.komis is not None and args.komis.strip():
        raise ValueError("use either --komi or --komis, not both")
    if args.ruleset and args.rulesets:
        raise ValueError("use either --ruleset or --rulesets, not both")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    board_sizes = parse_board_sizes(args.board_sizes)
    ruleset_names = parse_ruleset_names(args.rulesets or args.ruleset or DEFAULT_RULESETS)
    has_ruleset_specific_override = any(
        value
        for value in (
            args.katago_rules,
            args.katago_ko,
            args.katago_suicide,
            args.saki_scoring,
            args.saki_ko,
            args.saki_suicide,
        )
    )
    if len(ruleset_names) > 1 and has_ruleset_specific_override:
        raise ValueError("KataGo/SAKIGo rule overrides require a single --ruleset")
    base_rulesets = [
        ruleset_from_overrides(
            ruleset=ruleset_name,
            katago_rules=args.katago_rules or None,
            katago_ko=args.katago_ko or None,
            katago_suicide=args.katago_suicide or None,
            saki_scoring=args.saki_scoring or None,
            saki_ko=args.saki_ko or None,
            saki_suicide=args.saki_suicide or None,
            komi=args.komi,
        )
        for ruleset_name in ruleset_names
    ]
    komi_values = "" if args.komi is not None else args.komis if args.komis is not None else DEFAULT_KOMIS
    komis = parse_komis(komi_values, base_rulesets[0].komi)
    plan = build_generation_plan(
        board_sizes,
        base_rulesets,
        komis,
        samples=args.samples,
        samples_per_combination=args.samples_per_combination,
    )
    ruleset_variants = [
        ruleset.with_komi(komi)
        for ruleset in base_rulesets
        for komi in komis
    ]
    config = args.config or default_config_path(args.katago)
    if args.samples_per_file < 0:
        raise ValueError("--samples-per-file must be non-negative")
    if args.zstd_level < 1 or args.zstd_level > 22:
        raise ValueError("--zstd-level must be in [1, 22]")
    if args.samples_per_file == 0 and not is_zstd_jsonl_path(args.output):
        print(
            "warning: single plain JSONL generation is deprecated; prefer numbered .jsonl.zst shards",
            file=sys.stderr,
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    stderr_log = args.run_dir / "katago_stderr.log"
    analysis_log_dir = args.run_dir / "analysis_logs"
    override = (
        f"numAnalysisThreads={args.analysis_threads},"
        "numSearchThreadsPerAnalysisThread=1,"
        f"nnMaxBatchSize={args.nn_batch_size},"
        "nnCacheSizePowerOfTwo=12,"
        f"logDir={analysis_log_dir},"
        "logToStderr=true"
    )
    proc = subprocess.Popen(
        [
            str(args.katago),
            "analysis",
            "-model",
            str(args.model),
            "-config",
            str(config),
            "-override-config",
            override,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        raise RuntimeError("failed to open KataGo pipes")

    ready = threading.Event()
    responses: queue.Queue[dict[str, Any]] = queue.Queue()
    stderr_tail: list[str] = []

    def stderr_reader() -> None:
        with stderr_log.open("w", encoding="utf-8") as handle:
            for line in proc.stderr:
                handle.write(line)
                handle.flush()
                text = line.rstrip("\n")
                stderr_tail.append(text)
                del stderr_tail[:-20]
                if "Started, ready" in text:
                    ready.set()

    def stdout_reader() -> None:
        for line in proc.stdout:
            if line.startswith("{"):
                try:
                    responses.put(json.loads(line))
                except json.JSONDecodeError:
                    continue

    threading.Thread(target=stderr_reader, daemon=True).start()
    threading.Thread(target=stdout_reader, daemon=True).start()
    if not ready.wait(300):
        proc.kill()
        raise RuntimeError("KataGo did not become ready: " + " | ".join(stderr_tail))

    rng = random.Random(args.seed)
    schedule = GenerationSchedule(plan, rng)

    games: list[Game] = []
    for index in range(args.concurrency):
        variant = schedule.choose_variant()
        if variant is None:
            break
        games.append(
            Game(
                game_id=index,
                rng=random.Random(args.seed + index + 1),
                board_size=variant.board_size,
                ruleset=variant.ruleset,
            )
        )
    if not games:
        raise RuntimeError("generation plan has no schedulable combinations")
    pending: dict[str, Game] = {}
    next_query = 0
    next_game_id = args.concurrency
    completed_games = 0
    started_at = time.time()
    progress = GenerationProgressBar(plan.target_samples, enabled=progress_enabled(args))

    def variant_for_game(game: Game) -> GenerationVariant:
        return GenerationVariant(board_size=game.board_size, ruleset=game.ruleset)

    def send(game: Game) -> bool:
        nonlocal next_query
        variant = variant_for_game(game)
        if not schedule.reserve(variant):
            return False
        query_id = f"g{game.game_id}-q{next_query}"
        next_query += 1
        pending[query_id] = game
        proc.stdin.write(game.query(query_id) + "\n")
        return True

    def reset_game(game: Game) -> bool:
        nonlocal next_game_id
        variant = schedule.choose_variant()
        if variant is None:
            return False
        completed_id = next_game_id
        next_game_id += 1
        game.reset(completed_id, board_size=variant.board_size, ruleset=variant.ruleset)
        return True

    def send_or_reschedule(game: Game) -> None:
        if send(game):
            return
        if reset_game(game):
            send(game)

    for game in games:
        send_or_reschedule(game)
    proc.stdin.flush()

    try:
        with GenerationOutputWriter(
            args.output,
            samples_per_file=args.samples_per_file,
            zstd_level=args.zstd_level,
        ) as output:
            while schedule.total_written < plan.target_samples:
                if not pending:
                    raise RuntimeError("generation schedule exhausted before reaching target samples")
                response = responses.get()
                query_id = response.get("id")
                game = pending.pop(str(query_id), None)
                if game is None:
                    continue
                variant = variant_for_game(game)
                if "error" in response:
                    schedule.complete(variant, success=False)
                    proc.kill()
                    raise RuntimeError(
                        "KataGo analysis error for "
                        f"board_size={game.board_size}, ruleset={game.ruleset.metadata()}: "
                        f"{response.get('error')}"
                    )

                record, budget = record_from_response(game, response)
                schedule.complete(variant, success=True)
                output.write_record(record)

                action = sample_action(budget, game.rng, args.temperature)
                game.play(action)
                can_continue = True
                if game.should_reset(args.max_plies):
                    completed_games += 1
                    can_continue = reset_game(game)

                if can_continue and schedule.total_written < plan.target_samples:
                    send_or_reschedule(game)

                elapsed = max(1e-9, time.time() - started_at)
                samples_per_second = schedule.total_written / elapsed
                eta_seconds = max(0.0, (plan.target_samples - schedule.total_written) / samples_per_second)
                completed_combinations = sum(
                    1
                    for key, quota in plan.quotas.items()
                    if schedule.written[key] >= quota
                )
                progress.render(
                    samples=schedule.total_written,
                    samples_per_second=samples_per_second,
                    eta_seconds=eta_seconds,
                    completed_combinations=completed_combinations,
                    combination_count=len(plan.variants),
                )
                if schedule.total_written % args.log_interval == 0 or schedule.total_written == plan.target_samples:
                    output.flush()
                    status = {
                        "state": "running" if schedule.total_written < plan.target_samples else "complete",
                        "samples": schedule.total_written,
                        "target_samples": plan.target_samples,
                        "samples_per_second": samples_per_second,
                        "eta_seconds": eta_seconds,
                        "active_games": len(pending),
                        "completed_games": completed_games,
                        "output": str(args.output),
                        "output_files": [str(path) for path in output.paths],
                        "data_format": output.data_format,
                        "samples_per_file": args.samples_per_file,
                        "run_dir": str(args.run_dir),
                        "quota_mode": plan.quota_mode,
                        "samples_per_combination": plan.samples_per_combination,
                        "combination_count": len(plan.variants),
                        "rulesets": [ruleset.metadata() for ruleset in base_rulesets],
                        "ruleset_variants": [variant.metadata() for variant in ruleset_variants],
                        "board_sizes": board_sizes,
                        "komis": komis,
                        "completed_combinations": completed_combinations,
                        "updated_at": now_iso(),
                    }
                    write_status(args.status, status)
                    if progress.enabled:
                        progress.render(
                            samples=schedule.total_written,
                            samples_per_second=samples_per_second,
                            eta_seconds=eta_seconds,
                            completed_combinations=completed_combinations,
                            combination_count=len(plan.variants),
                            force=schedule.total_written == plan.target_samples,
                        )
                    else:
                        print(
                            f"samples={schedule.total_written}/{plan.target_samples} "
                            f"sps={samples_per_second:.2f} eta={eta_seconds / 60.0:.1f}m",
                            flush=True,
                        )
                proc.stdin.flush()
    finally:
        if progress.enabled and schedule.total_written >= plan.target_samples:
            print()
        else:
            progress.clear()
        # Engine shutdown must run on every exit path (including writer
        # errors and Ctrl+C), or the KataGo process is orphaned.
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def main() -> None:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"generator failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
