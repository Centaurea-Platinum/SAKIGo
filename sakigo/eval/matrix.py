"""Ordered multi-model self-play matrix.

Runs all requested directed pairings in one process. Active games are grouped by
the model to move, so each model sees one batched inference call per ply wave.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch

from sakigo.engine import ENGINE_AVAILABLE, Game
from sakigo.eval.selfplay import (
    PolicyPlayer,
    SGF_LETTERS,
    elo_from_p,
    engine_final_score,
    paired_mean_interval,
    wilson_interval,
)
from sakigo.generate.book import iter_frozen_sample
from sakigo.generate.game import index_from_coord
from sakigo.rulesets import RulesetSpec, ruleset_from_name


DEFAULT_PAIRINGS = ("AB", "AC", "BC", "BA", "CA", "CB")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched ordered self-play matrix.")
    parser.add_argument("--player", action="append", required=True, help="Label=checkpoint path.")
    parser.add_argument("--pairings", default=",".join(DEFAULT_PAIRINGS))
    parser.add_argument("--games-per-pairing", type=int, default=6)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument(
        "--book-index",
        default="",
        help="Optional indexed book database supplying canonical paired openings.",
    )
    parser.add_argument(
        "--book-split",
        default="validation",
        choices=("train", "validation"),
        help="Frozen book sample split used when --book-index is set.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--ruleset", default="tromp-taylor")
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--max-plies", type=int, default=0, help="0 = no ply cap.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--sgf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unsafe-legacy-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _new_game(board_size: int, ruleset: RulesetSpec) -> Game:
    return Game(board_size, ruleset.scoring, ruleset.ko, ruleset.suicide, ruleset.komi)


def _random_opening(
    rng: random.Random,
    plies: int,
    board_size: int,
    ruleset: RulesetSpec,
) -> list[int]:
    game = _new_game(board_size, ruleset)
    area = board_size * board_size
    opening: list[int] = []
    for _ in range(max(0, plies)):
        legal = game.legal_mask()
        moves = [index for index in range(area) if legal[index]]
        if not moves:
            break
        action = rng.choice(moves)
        game.play(action)
        opening.append(action)
    return opening


def _book_openings(
    database: Path,
    split: str,
    count: int,
    seed: int,
    board_size: int,
    opening_plies: int | None = None,
) -> list[tuple[str, list[int]]]:
    """Select deterministic canonical histories from a frozen book sample."""
    tasks = list(iter_frozen_sample(database, split=split))
    if count > len(tasks):
        raise ValueError(
            f"requested {count} book openings but split {split!r} contains only {len(tasks)}"
        )
    selected = random.Random(seed).sample(tasks, count)
    openings: list[tuple[str, list[int]]] = []
    for task in selected:
        actions: list[int] = []
        history = task["history"]
        if opening_plies is not None:
            history = history[:opening_plies]
        for ply, (color, coord) in enumerate(history):
            expected = "B" if ply % 2 == 0 else "W"
            if str(color).upper() != expected:
                raise ValueError(
                    f"book node {task['node_id']} history color mismatch at ply {ply}: "
                    f"expected {expected}, got {color}"
                )
            actions.append(index_from_coord(str(coord), board_size))
        openings.append((str(task["node_id"]), actions))
    return openings


def _opening_pool(
    args: argparse.Namespace,
    ruleset: RulesetSpec,
) -> tuple[str, list[tuple[str, list[int]]]]:
    """Build one opening pool shared by every directed pairing."""
    if args.book_index:
        database = Path(args.book_index)
        if not database.is_file():
            raise FileNotFoundError(f"book index does not exist: {database}")
        return (
            f"book:{args.book_split}",
            _book_openings(
                database,
                args.book_split,
                args.games_per_pairing,
                args.seed,
                args.board_size,
                args.opening_plies,
            ),
        )
    rng = random.Random(args.seed)
    return (
        "random",
        [
            (
                f"random-{index:06d}",
                _random_opening(
                    random.Random(rng.randrange(1 << 30)),
                    args.opening_plies,
                    args.board_size,
                    ruleset,
                ),
            )
            for index in range(args.games_per_pairing)
        ],
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_players(raw_players: list[str]) -> dict[str, Path]:
    players: dict[str, Path] = {}
    for raw in raw_players:
        if "=" not in raw:
            raise ValueError(f"player must be Label=checkpoint, got {raw!r}")
        label, path = raw.split("=", 1)
        label = label.strip()
        if len(label) != 1:
            raise ValueError(f"player label must be one character, got {label!r}")
        if label in players:
            raise ValueError(f"duplicate player label {label!r}")
        players[label] = Path(path)
    return players


def _parse_pairings(raw: str, players: dict[str, Path]) -> tuple[str, ...]:
    pairings = tuple(part.strip() for part in raw.split(",") if part.strip())
    for pairing in pairings:
        if len(pairing) != 2:
            raise ValueError(f"pairing must have two labels, got {pairing!r}")
        first, second = pairing
        if first == second:
            raise ValueError(f"self-pairing is not supported: {pairing!r}")
        if first not in players or second not in players:
            raise ValueError(f"pairing {pairing!r} references unknown player")
    return pairings


class MatrixGame:
    def __init__(
        self,
        game_index: int,
        pairing: str,
        opening: list[int],
        max_plies: int | None,
        board_size: int,
        ruleset: RulesetSpec,
        opening_id: str = "",
    ) -> None:
        self.game_index = game_index
        self.pairing = pairing
        self.black_label = pairing[0]
        self.white_label = pairing[1]
        self.max_plies = max_plies
        self.board_size = board_size
        self.ruleset = ruleset
        self.komi = ruleset.komi
        self.opening_id = opening_id
        self.opening_plies = len(opening)
        self.game = _new_game(board_size, ruleset)
        self.actions: list[int] = []
        self.passes = 0
        self.ply = 0
        self.done = False
        self.score = 0.0
        self.ended_by = ""
        for action in opening:
            if self.done:
                break
            self.play(action)

    def label_to_move(self) -> str:
        return self.black_label if self.game.to_move == 1 else self.white_label

    def play(self, action: int) -> None:
        area = self.board_size * self.board_size
        self.game.play(action)
        self.actions.append(action)
        self.passes = self.passes + 1 if action == area else 0
        self.ply += 1
        if self.passes >= 2:
            self.done = True
            self.ended_by = "passes"
        elif self.max_plies is not None and self.ply >= self.max_plies:
            self.done = True
            self.ended_by = "max_plies"
        if self.done:
            if self.ruleset.scoring != "area":
                raise ValueError(
                    f"matrix scoring supports Tromp-Taylor area scoring only, got {self.ruleset.scoring!r}"
                )
            self.score = engine_final_score(self.game, self.board_size, self.komi)

    def winner(self) -> str | None:
        if self.ended_by == "max_plies" or self.score == 0.0:
            return None
        return self.black_label if self.score > 0 else self.white_label


def _write_sgf(path: Path, game: MatrixGame, player_names: dict[str, str]) -> None:
    area = game.board_size * game.board_size
    if game.ended_by == "max_plies":
        result = "Void"
    elif game.score == 0.0:
        result = "0"
    else:
        result = ("B+" if game.score > 0 else "W+") + f"{abs(game.score):g}"
    nodes: list[str] = []
    color = "B"
    for action in game.actions:
        if action == area:
            nodes.append(f";{color}[]")
        else:
            row, col = divmod(action, game.board_size)
            nodes.append(f";{color}[{SGF_LETTERS[col]}{SGF_LETTERS[row]}]")
        color = "W" if color == "B" else "B"
    content = (
        f"(;GM[1]FF[4]SZ[{game.board_size}]KM[{game.komi}]RU[{game.ruleset.name}]"
        f"PB[{game.black_label}:{player_names[game.black_label]}]"
        f"PW[{game.white_label}:{player_names[game.white_label]}]"
        f"RE[{result}]" + "".join(nodes) + ")"
    )
    path.write_text(content, encoding="utf-8")


def _summary(
    games: list[MatrixGame],
    players: dict[str, Path],
    pairings: tuple[str, ...],
    args: argparse.Namespace,
    elapsed: float,
) -> dict:
    by_pairing: dict[str, dict[str, object]] = {}
    for pairing in pairings:
        group = [game for game in games if game.pairing == pairing]
        first, second = pairing
        wins = Counter(game.winner() for game in group)
        first_wins = wins[first]
        draws = sum(1 for game in group if game.ended_by != "max_plies" and game.score == 0.0)
        voids = sum(1 for game in group if game.ended_by == "max_plies")
        scored = len(group) - voids
        first_points = first_wins + 0.5 * draws
        low, high = wilson_interval(first_points, scored)
        by_pairing[pairing] = {
            "games": len(group),
            "first": first,
            "second": second,
            "wins_first": first_wins,
            "wins_second": wins[second],
            "draws": draws,
            "void_games": voids,
            "score_first": first_points / scored if scored else 0.0,
            "winrate_first": first_points / scored if scored else 0.0,
            "winrate_first_ci95": [low, high],
            "elo_first_minus_second": elo_from_p(first_points / scored) if scored else 0.0,
            "elo_ci95": [elo_from_p(low), elo_from_p(high)],
            "avg_plies": sum(game.ply for game in group) / len(group) if group else 0.0,
            "max_plies_seen": max((game.ply for game in group), default=0),
            "ended_by": dict(Counter(game.ended_by for game in group)),
        }

    unordered: dict[str, dict[str, object]] = {}
    buckets: dict[str, list[MatrixGame]] = defaultdict(list)
    for game in games:
        buckets["".join(sorted(game.pairing))].append(game)
    for key, group in sorted(buckets.items()):
        first, second = key
        wins = Counter(game.winner() for game in group)
        first_wins = wins[first]
        draws = sum(1 for game in group if game.ended_by != "max_plies" and game.score == 0.0)
        voids = sum(1 for game in group if game.ended_by == "max_plies")
        scored = len(group) - voids
        first_points = first_wins + 0.5 * draws
        low, high = wilson_interval(first_points, scored)
        opening_groups: dict[str, list[MatrixGame]] = defaultdict(list)
        for game in group:
            opening_groups[game.opening_id].append(game)
        paired_scores: list[float] = []
        paired_margins: list[float] = []
        void_opening_pairs = 0
        for opening_games in opening_groups.values():
            if len(opening_games) != 2 or {game.pairing for game in opening_games} != {
                first + second,
                second + first,
            }:
                continue
            if any(game.ended_by == "max_plies" for game in opening_games):
                void_opening_pairs += 1
                continue
            first_score = 0.0
            first_margin = 0.0
            for game in opening_games:
                winner = game.winner()
                first_score += 1.0 if winner == first else 0.5 if winner is None else 0.0
                first_margin += game.score if game.black_label == first else -game.score
            paired_scores.append(first_score / 2.0)
            paired_margins.append(first_margin / 2.0)
        paired_low, paired_high = paired_mean_interval(paired_scores)
        unordered[key] = {
            "games": len(group),
            "labels": [first, second],
            "wins": {first: wins[first], second: wins[second]},
            "draws": draws,
            "void_games": voids,
            "score_first_label": first_points / scored if scored else 0.0,
            "winrate_first_label": first_points / scored if scored else 0.0,
            "winrate_first_label_ci95": [low, high],
            "elo_first_label_minus_second": elo_from_p(first_points / scored) if scored else 0.0,
            "elo_ci95": [elo_from_p(low), elo_from_p(high)],
            "paired_openings": len(paired_scores),
            "void_opening_pairs": void_opening_pairs,
            "paired_score_first_label": (
                sum(paired_scores) / len(paired_scores) if paired_scores else 0.0
            ),
            "paired_score_first_label_ci95": [paired_low, paired_high],
            "paired_mean_margin_first_minus_second": (
                sum(paired_margins) / len(paired_margins) if paired_margins else 0.0
            ),
        }

    return {
        "players": {label: str(path) for label, path in sorted(players.items())},
        "ruleset": ruleset_from_name(args.ruleset).with_komi(args.komi).metadata(),
        "pairings": list(pairings),
        "games_per_pairing": args.games_per_pairing,
        "games": len(games),
        "opening_plies": args.opening_plies,
        "opening_source": "book:" + args.book_split if args.book_index else "random",
        "book_index": args.book_index or None,
        "temperature": args.temperature,
        "board_size": args.board_size,
        "komi": args.komi,
        "max_plies": args.max_plies,
        "max_plies_effective": None if args.max_plies <= 0 else args.max_plies,
        "seed": args.seed,
        "device": args.device,
        "elapsed_seconds": elapsed,
        "by_pairing": by_pairing,
        "unordered": unordered,
    }


def main(argv: list[str] | None = None) -> None:
    if not ENGINE_AVAILABLE:
        raise RuntimeError("sakigo_engine wheel is not installed; see sakigo/engine/__init__.py")
    args = parse_args(argv)
    if args.games_per_pairing <= 0:
        raise ValueError("games-per-pairing must be positive")
    if args.opening_plies < 0:
        raise ValueError("opening-plies must be non-negative")
    if args.board_size <= 0 or args.batch_size <= 0:
        raise ValueError("board-size and batch-size must be positive")
    if not math.isfinite(args.komi):
        raise ValueError("komi must be finite")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")

    checkpoint_paths = _parse_players(args.player)
    pairings = _parse_pairings(args.pairings, checkpoint_paths)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    torch.manual_seed(args.seed)
    ruleset = ruleset_from_name(args.ruleset).with_komi(args.komi)
    if ruleset.scoring != "area":
        raise ValueError(
            f"matrix scoring supports Tromp-Taylor area scoring only, got {ruleset.scoring!r}"
        )
    max_plies = None if args.max_plies <= 0 else args.max_plies
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"matrix_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir = run_dir / "sgf"
    if args.sgf:
        sgf_dir.mkdir(parents=True, exist_ok=True)

    players = {
        label: PolicyPlayer(
            path,
            device,
            args.batch_size,
            args.board_size,
            allow_unsafe_legacy=args.allow_unsafe_legacy_checkpoint,
        )
        for label, path in sorted(checkpoint_paths.items())
    }
    player_names = {label: path.parent.parent.name for label, path in checkpoint_paths.items()}

    opening_source, opening_pool = _opening_pool(args, ruleset)

    games: list[MatrixGame] = []
    for pairing in pairings:
        for opening_id, opening in opening_pool:
            games.append(
                MatrixGame(
                    len(games),
                    pairing,
                    opening,
                    max_plies,
                    args.board_size,
                    ruleset,
                    opening_id,
                )
            )

    _atomic_write_json(
        run_dir / "status.json",
        {
            "state": "running",
            "games": len(games),
            "done": 0,
            "pairings": list(pairings),
            "ruleset": ruleset.metadata(),
            "max_plies_effective": max_plies,
            "opening_source": opening_source,
            "run_dir": str(run_dir),
        },
    )

    start = time.monotonic()
    last_status = 0.0
    while True:
        active = [game for game in games if not game.done]
        if not active:
            break
        buckets: dict[str, list[MatrixGame]] = defaultdict(list)
        for game in active:
            buckets[game.label_to_move()].append(game)
        for label, bucket in sorted(buckets.items()):
            actions = players[label].select_actions(bucket, args.temperature)
            for game, action in zip(bucket, actions):
                game.play(action)
        done_count = sum(1 for game in games if game.done)
        elapsed = time.monotonic() - start
        if elapsed - last_status >= 5.0:
            last_status = elapsed
            _atomic_write_json(
                run_dir / "status.json",
                {
                    "state": "running",
                    "games": len(games),
                    "done": done_count,
                    "active": len(games) - done_count,
                    "plies": sum(game.ply for game in games),
                    "max_game_plies": max(game.ply for game in games),
                    "elapsed_seconds": elapsed,
                    "pairings": list(pairings),
                    "ruleset": ruleset.metadata(),
                    "max_plies_effective": max_plies,
                    "opening_source": opening_source,
                    "run_dir": str(run_dir),
                    "updated": datetime.now().isoformat(timespec="seconds"),
                },
            )
        print(
            f"\rgames {done_count}/{len(games)}  plies {sum(g.ply for g in games)}  "
            f"max {max(g.ply for g in games)}  {elapsed:,.0f}s",
            end="",
            flush=True,
        )
    print()

    with (run_dir / "games.jsonl").open("w", encoding="utf-8") as handle:
        for game in games:
            if game.ended_by == "max_plies":
                result = "Void"
                winner = "void"
            elif game.score == 0.0:
                result = "0"
                winner = "draw"
            else:
                result = ("B+" if game.score > 0 else "W+") + f"{abs(game.score):g}"
                winner = game.winner()
            if args.sgf:
                _write_sgf(sgf_dir / f"{game.game_index:04d}_{game.pairing}.sgf", game, player_names)
            handle.write(
                json.dumps(
                    {
                        "game_index": game.game_index,
                        "pairing": game.pairing,
                        "black": game.black_label,
                        "white": game.white_label,
                        "winner": winner,
                        "result": result,
                        "score_black_minus_white": game.score,
                        "plies": game.ply,
                        "ended_by": game.ended_by,
                        "opening_id": game.opening_id,
                        "opening_plies": game.opening_plies,
                        "actions": game.actions,
                    }
                )
                + "\n"
            )

    elapsed = time.monotonic() - start
    summary = _summary(games, checkpoint_paths, pairings, args, elapsed)
    _atomic_write_json(run_dir / "summary.json", summary)
    _atomic_write_json(
        run_dir / "status.json",
        {
            "state": "finished",
            "games": len(games),
            "done": len(games),
            "plies": sum(game.ply for game in games),
            "max_game_plies": max(game.ply for game in games),
            "elapsed_seconds": elapsed,
            "run_dir": str(run_dir),
            "updated": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"run_dir={run_dir}")
    print(json.dumps(summary["unordered"], indent=2))


if __name__ == "__main__":
    main()
