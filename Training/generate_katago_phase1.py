from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Training.common import BOARD_PLANE_COUNT, RULE_FEATURE_COUNT, ROOT, SCHEMA_VERSION


BLACK = 1
WHITE = -1
BOARD_SIZE = 19
AREA = BOARD_SIZE * BOARD_SIZE
ACTION_COUNT = AREA + 1
KOMI = 7.5
LETTERS = "ABCDEFGHJKLMNOPQRST"


def default_katago_path() -> Path:
    candidates = sorted((ROOT / "Distillation" / "engine").glob("*/katago.exe"))
    if not candidates:
        raise FileNotFoundError("could not find Distillation/engine/*/katago.exe")
    return candidates[0]


def default_config_path(katago_path: Path) -> Path:
    return katago_path.parent / "analysis_example.cfg"


def default_model_path() -> Path:
    candidates = sorted((ROOT / "Distillation" / "models").glob("*.bin.gz"))
    if not candidates:
        raise FileNotFoundError("could not find Distillation/models/*.bin.gz")
    return candidates[0]


def coord_from_index(index: int) -> str:
    row, col = divmod(index, BOARD_SIZE)
    return f"{LETTERS[col]}{BOARD_SIZE - row}"


def index_from_coord(coord: str) -> int:
    if coord.lower() == "pass":
        return AREA
    col = LETTERS.index(coord[0].upper())
    row = BOARD_SIZE - int(coord[1:])
    return row * BOARD_SIZE + col


NEIGHBORS: list[tuple[int, ...]] = []
for point in range(AREA):
    row, col = divmod(point, BOARD_SIZE)
    neighbors: list[int] = []
    if row > 0:
        neighbors.append(point - BOARD_SIZE)
    if row + 1 < BOARD_SIZE:
        neighbors.append(point + BOARD_SIZE)
    if col > 0:
        neighbors.append(point - 1)
    if col + 1 < BOARD_SIZE:
        neighbors.append(point + 1)
    NEIGHBORS.append(tuple(neighbors))


def group_and_liberties(board: list[int], start: int) -> tuple[set[int], set[int]]:
    color = board[start]
    stack = [start]
    stones = {start}
    liberties: set[int] = set()
    while stack:
        point = stack.pop()
        for neighbor in NEIGHBORS[point]:
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
    board: list[int] = field(default_factory=lambda: [0] * AREA)
    to_move: int = BLACK
    captures: list[int] = field(default_factory=lambda: [0, 0])
    moves: list[list[str]] = field(default_factory=list)
    seen: set[tuple[int, ...]] = field(default_factory=lambda: {tuple([0] * AREA)})
    passes: int = 0
    ply: int = 0

    def reset(self, game_id: int) -> None:
        self.game_id = game_id
        self.board = [0] * AREA
        self.to_move = BLACK
        self.captures = [0, 0]
        self.moves = []
        self.seen = {tuple(self.board)}
        self.passes = 0
        self.ply = 0

    def analyze_play(self, point: int) -> AnalyzedMove | None:
        if point < 0 or point >= AREA or self.board[point] != 0:
            return None
        color = self.to_move
        opponent = -color
        board = self.board.copy()
        board[point] = color
        captured_points: set[int] = set()

        for neighbor in NEIGHBORS[point]:
            if board[neighbor] != opponent:
                continue
            stones, liberties = group_and_liberties(board, neighbor)
            if not liberties:
                captured_points.update(stones)
        for captured in captured_points:
            board[captured] = 0

        captured_self_points: set[int] = set()
        if board[point] == color:
            own_stones, own_liberties = group_and_liberties(board, point)
            if not own_liberties:
                # Tromp-Taylor permits suicide; remove the self-captured group.
                captured_self_points.update(own_stones)
                for captured in captured_self_points:
                    board[captured] = 0

        if tuple(board) in self.seen:
            return None
        return AnalyzedMove(board, len(captured_points), len(captured_self_points))

    def legal_mask(self) -> list[bool]:
        mask = [self.analyze_play(index) is not None for index in range(AREA)]
        mask.append(True)
        return mask

    def play(self, action: int) -> None:
        color = self.to_move
        if action == AREA:
            self.moves.append(["B" if color == BLACK else "W", "pass"])
            self.to_move = -self.to_move
            self.passes += 1
            self.ply += 1
            return

        analysis = self.analyze_play(action)
        if analysis is None:
            # This should not happen if selection respected the legal mask.
            self.play(AREA)
            return
        self.board = analysis.board
        if color == BLACK:
            self.captures[0] += analysis.captured_opponent
            self.captures[1] += analysis.captured_self
        else:
            self.captures[1] += analysis.captured_opponent
            self.captures[0] += analysis.captured_self
        self.seen.add(tuple(self.board))
        self.moves.append(["B" if color == BLACK else "W", coord_from_index(action)])
        self.to_move = -self.to_move
        self.passes = 0
        self.ply += 1

    def should_reset(self, max_plies: int) -> bool:
        return self.passes >= 2 or self.ply >= max_plies

    def query(self, query_id: str) -> str:
        return json.dumps(
            {
                "id": query_id,
                "moves": self.moves,
                "rules": "tromp-taylor",
                "komi": KOMI,
                "boardXSize": BOARD_SIZE,
                "boardYSize": BOARD_SIZE,
                "maxVisits": 1,
                "includePolicy": True,
                "includeOwnership": True,
                "includeNoResultValue": True,
                "analysisPVLen": 1,
                "overrideSettings": {
                    "rootNumSymmetriesToSample": 1,
                    "wideRootNoise": 0.0,
                },
            },
            separators=(",", ":"),
        )

    def board_planes(self, legal_mask: list[bool]) -> list[float]:
        planes = [0.0] * (BOARD_PLANE_COUNT * AREA)
        for index, value in enumerate(self.board):
            row, col = divmod(index, BOARD_SIZE)
            if value == self.to_move:
                planes[index] = 1.0
            elif value == -self.to_move:
                planes[AREA + index] = 1.0
            else:
                planes[2 * AREA + index] = 1.0
                if not legal_mask[index]:
                    planes[5 * AREA + index] = 1.0
            if (row == 0 or row == BOARD_SIZE - 1) and (col == 0 or col == BOARD_SIZE - 1):
                planes[3 * AREA + index] = 1.0
            elif row == 0 or row == BOARD_SIZE - 1 or col == 0 or col == BOARD_SIZE - 1:
                planes[4 * AREA + index] = 1.0
        return planes

    def rule_features(self) -> list[float]:
        features = [0.0] * RULE_FEATURE_COUNT
        features[0] = 1.0  # Area scoring.
        features[5] = 1.0  # Positional superko.
        features[6] = 1.0  # Suicide allowed under Tromp-Taylor.
        signed_komi = -KOMI if self.to_move == BLACK else KOMI
        if self.to_move == BLACK:
            capture_diff = self.captures[0] - self.captures[1]
        else:
            capture_diff = self.captures[1] - self.captures[0]
        features[8] = max(-1.0, min(1.0, signed_komi / AREA))
        features[9] = max(-1.0, min(1.0, capture_diff / AREA))
        return features

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
        values = [0.0] * ACTION_COUNT
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
    if not isinstance(raw_policy, list) or len(raw_policy) != ACTION_COUNT:
        raise ValueError(f"bad policy in response {response.get('id')}")
    if not isinstance(ownership, list) or len(ownership) != AREA:
        raise ValueError(f"bad ownership in response {response.get('id')}")

    legal_mask = game.legal_mask()
    budget = normalize_policy([float(value) for value in raw_policy], legal_mask)
    policy = one_hot_top1(budget)

    black_win = float(root_info["rawWinrate"])
    draw = float(root_info.get("rawNoResultProb", 0.0))
    black_loss = max(0.0, 1.0 - black_win - draw)
    if game.to_move == BLACK:
        wdl = [black_win, draw, black_loss]
        score = float(root_info["rawLead"]) / AREA
        ownership_target = [float(value) for value in ownership]
    else:
        wdl = [black_loss, draw, black_win]
        score = -float(root_info["rawLead"]) / AREA
        ownership_target = [-float(value) for value in ownership]

    record = {
        "schema_version": SCHEMA_VERSION,
        "board_size": BOARD_SIZE,
        "ply": game.ply,
        "position_key": game.position_key(),
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
            "rules": "tromp-taylor",
            "reportAnalysisWinratesAs": "BLACK",
            "rawLead": root_info.get("rawLead"),
            "rawWinrate": root_info.get("rawWinrate"),
            "rawNoResultProb": root_info.get("rawNoResultProb"),
        },
    }
    return record, budget


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 SAKIGo records with KataGo.")
    parser.add_argument("--samples", type=int, default=2**18)
    parser.add_argument("--output", type=Path, required=True)
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
    return parser.parse_args(argv)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = args.config or default_config_path(args.katago)
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
    games = [Game(game_id=index, rng=random.Random(args.seed + index + 1)) for index in range(args.concurrency)]
    pending: dict[str, Game] = {}
    next_query = 0
    next_game_id = args.concurrency
    written = 0
    completed_games = 0
    started_at = time.time()

    def send(game: Game) -> None:
        nonlocal next_query
        query_id = f"g{game.game_id}-q{next_query}"
        next_query += 1
        pending[query_id] = game
        proc.stdin.write(game.query(query_id) + "\n")

    for game in games:
        send(game)
    proc.stdin.flush()

    with args.output.open("w", encoding="utf-8") as output:
        while written < args.samples:
            response = responses.get()
            query_id = response.get("id")
            game = pending.pop(str(query_id), None)
            if game is None:
                continue
            if "error" in response:
                completed_games += 1
                game.reset(next_game_id)
                next_game_id += 1
                send(game)
                proc.stdin.flush()
                continue

            record, budget = record_from_response(game, response)
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            written += 1

            action = sample_action(budget, game.rng, args.temperature)
            game.play(action)
            if game.should_reset(args.max_plies):
                completed_games += 1
                game.reset(next_game_id)
                next_game_id += 1

            if written < args.samples:
                send(game)
            if written % args.log_interval == 0 or written == args.samples:
                output.flush()
                elapsed = max(1e-9, time.time() - started_at)
                samples_per_second = written / elapsed
                eta_seconds = max(0.0, (args.samples - written) / samples_per_second)
                status = {
                    "state": "running" if written < args.samples else "complete",
                    "samples": written,
                    "target_samples": args.samples,
                    "samples_per_second": samples_per_second,
                    "eta_seconds": eta_seconds,
                    "active_games": len(pending),
                    "completed_games": completed_games,
                    "output": str(args.output),
                    "run_dir": str(args.run_dir),
                    "updated_at": now_iso(),
                }
                write_status(args.status, status)
                print(
                    f"samples={written}/{args.samples} "
                    f"sps={samples_per_second:.2f} eta={eta_seconds / 60.0:.1f}m",
                    flush=True,
                )
            proc.stdin.flush()

    proc.stdin.close()
    proc.wait(timeout=120)


def main() -> None:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"generator failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
