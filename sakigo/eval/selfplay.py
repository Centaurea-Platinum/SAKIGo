"""Self-play evaluation: paired color-reversed matches between policy checkpoints.

Ported from Training/selfplay_eval.py with the pure-Python Game replaced by
the Rust engine (sakigo.engine.Game) — same paired-opening design,
Tromp-Taylor adjudication, Wilson CI + Elo, JSONL + SGF outputs.

Examples:
    uv run python -m sakigo.eval --player-a runs/<run>/checkpoints/step_002048.pt --player-b random --pairs 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch

from sakigo.engine import ENGINE_AVAILABLE, Game
from sakigo.model import CHECKPOINT_SCHEMA_VERSION, SakiGoNet, config_from_dict
from sakigo.rulesets import ruleset_from_name

BLACK = 1
WHITE = -1
SGF_LETTERS = "abcdefghijklmnopqrs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired self-play evaluation between two players.")
    parser.add_argument("--player-a", required=True, help="Checkpoint path or 'random'.")
    parser.add_argument("--player-b", required=True, help="Checkpoint path or 'random'.")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--max-plies", type=int, default=0, help="0 = 2 * board area.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--sgf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unsafe-legacy-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _neighbors(board_size: int) -> list[tuple[int, ...]]:
    result = []
    for point in range(board_size * board_size):
        row, col = divmod(point, board_size)
        near = []
        if row > 0:
            near.append(point - board_size)
        if row < board_size - 1:
            near.append(point + board_size)
        if col > 0:
            near.append(point - 1)
        if col < board_size - 1:
            near.append(point + 1)
        result.append(tuple(near))
    return result


def tromp_taylor_score(board: list[int], board_size: int, komi: float) -> float:
    """Black minus White area, komi included (positive = Black leads)."""
    area = board_size * board_size
    neighbors = _neighbors(board_size)
    black_area = 0
    white_area = 0
    visited = [False] * area
    for point in range(area):
        value = board[point]
        if value == BLACK:
            black_area += 1
        elif value == WHITE:
            white_area += 1
        elif not visited[point]:
            region = [point]
            visited[point] = True
            stack = [point]
            touches_black = False
            touches_white = False
            while stack:
                current = stack.pop()
                for neighbor in neighbors[current]:
                    neighbor_value = board[neighbor]
                    if neighbor_value == BLACK:
                        touches_black = True
                    elif neighbor_value == WHITE:
                        touches_white = True
                    elif not visited[neighbor]:
                        visited[neighbor] = True
                        region.append(neighbor)
                        stack.append(neighbor)
            if touches_black and not touches_white:
                black_area += len(region)
            elif touches_white and not touches_black:
                white_area += len(region)
    return black_area - white_area - komi


def _new_game(board_size: int, komi: float) -> Game:
    spec = ruleset_from_name("tromp-taylor")
    return Game(board_size, spec.scoring, spec.ko, spec.suicide, komi)


def engine_final_score(game: Game, board_size: int, komi: float) -> float:
    if hasattr(game, "final_score"):
        return float(game.final_score())
    return tromp_taylor_score(game.board(), board_size, komi)


class RandomPlayer:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def select_actions(self, games: list["MatchGame"], temperature: float) -> list[int]:
        actions = []
        for match in games:
            legal = match.game.legal_mask()
            board_moves = [index for index in range(len(legal) - 1) if legal[index]]
            actions.append(self.rng.choice(board_moves) if board_moves else len(legal) - 1)
        return actions


def load_policy_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    allow_unsafe_legacy: bool = False,
) -> SakiGoNet:
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as error:
        if not allow_unsafe_legacy:
            raise RuntimeError(
                "checkpoint is not safe-loadable; convert the trusted legacy file or pass "
                "--allow-unsafe-legacy-checkpoint"
            ) from error
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    version = payload.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema {version!r} is incompatible; expected "
            f"{CHECKPOINT_SCHEMA_VERSION} for the book-only no-ownership model"
        )
    config = config_from_dict(payload["model_config"])
    state = payload.get("model_state", payload.get("model"))
    if state is None:
        raise ValueError(f"{checkpoint_path} has no model weights")
    model = SakiGoNet(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


class PolicyPlayer:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        batch_size: int,
        board_size: int,
        *,
        allow_unsafe_legacy: bool = False,
    ) -> None:
        self.name = str(checkpoint_path)
        self.device = device
        self.batch_size = max(1, batch_size)
        self.board_size = board_size
        self.model = load_policy_model(
            checkpoint_path, device, allow_unsafe_legacy=allow_unsafe_legacy
        )
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    @torch.inference_mode()
    def select_actions(self, games: list["MatchGame"], temperature: float) -> list[int]:
        actions: list[int] = []
        size = self.board_size
        for start in range(0, len(games), self.batch_size):
            chunk = games[start : start + self.batch_size]
            model_inputs = []
            for match in chunk:
                if hasattr(match.game, "model_inputs"):
                    model_inputs.append(match.game.model_inputs())
                else:
                    model_inputs.append(
                        (
                            match.game.board_planes(),
                            match.game.rule_features(),
                            match.game.legal_mask(),
                        )
                    )
            legal_masks = [inputs[2] for inputs in model_inputs]
            board = torch.tensor(
                [inputs[0] for inputs in model_inputs], dtype=torch.float32
            ).reshape(len(chunk), -1, size, size)
            rules = torch.tensor(
                [inputs[1] for inputs in model_inputs], dtype=torch.float32
            )
            board = board.to(self.device)
            rules = rules.to(self.device)
            if self.amp_dtype is not None:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    output = self.model(board, rules)
            else:
                output = self.model(board, rules)
            logits = output["policy_logits"].float().cpu()
            legal = torch.tensor(legal_masks, dtype=torch.bool)
            logits = logits.masked_fill(~legal, float("-inf"))
            if temperature <= 0.0:
                actions.extend(int(action) for action in logits.argmax(dim=-1))
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                for row in probabilities:
                    actions.append(int(torch.multinomial(row, 1).item()))
        return actions


def make_player(
    spec: str,
    device: torch.device,
    batch_size: int,
    seed: int,
    board_size: int,
    *,
    allow_unsafe_legacy: bool = False,
):
    if spec.strip().lower() == "random":
        return RandomPlayer(seed)
    return PolicyPlayer(
        Path(spec),
        device,
        batch_size,
        board_size,
        allow_unsafe_legacy=allow_unsafe_legacy,
    )


class MatchGame:
    """One game: engine state plus color assignment, ply/pass tracking, opening."""

    def __init__(
        self,
        game_index: int,
        black_player: int,
        opening: list[int],
        max_plies: int,
        board_size: int,
        komi: float,
    ) -> None:
        self.game_index = game_index
        self.black_player = black_player  # 0 = player A, 1 = player B
        self.max_plies = max_plies
        self.board_size = board_size
        self.komi = komi
        self.game = _new_game(board_size, komi)
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

    def player_to_move(self) -> int:
        return self.black_player if self.game.to_move == BLACK else 1 - self.black_player

    def play(self, action: int) -> None:
        area = self.board_size * self.board_size
        self.game.play(action)
        self.actions.append(action)
        self.passes = self.passes + 1 if action == area else 0
        self.ply += 1
        if self.passes >= 2 or self.ply >= self.max_plies:
            self.done = True
            self.ended_by = "passes" if self.passes >= 2 else "max_plies"
            self.score = engine_final_score(self.game, self.board_size, self.komi)

    def winner(self) -> int | None:
        """0 = player A, 1 = player B; None is a draw or unadjudicated cap."""
        if self.ended_by == "max_plies" or self.score == 0.0:
            return None
        black_won = self.score > 0
        return self.black_player if black_won else 1 - self.black_player


def random_opening(rng: random.Random, plies: int, board_size: int, komi: float) -> list[int]:
    game = _new_game(board_size, komi)
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


def write_sgf(path: Path, match_game: MatchGame, player_names: tuple[str, str], result: str) -> None:
    size = match_game.board_size
    area = size * size
    black = player_names[match_game.black_player]
    white = player_names[1 - match_game.black_player]
    nodes = []
    color = "B"
    for action in match_game.actions:
        if action == area:
            nodes.append(f";{color}[]")
        else:
            row, col = divmod(action, size)
            nodes.append(f";{color}[{SGF_LETTERS[col]}{SGF_LETTERS[row]}]")
        color = "W" if color == "B" else "B"
    content = (
        f"(;GM[1]FF[4]SZ[{size}]KM[{match_game.komi}]RU[Tromp-Taylor]"
        f"PB[{black}]PW[{white}]RE[{result}]" + "".join(nodes) + ")"
    )
    path.write_text(content, encoding="utf-8")


def wilson_interval(wins: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def paired_mean_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = z * math.sqrt(variance / len(values))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def elo_from_p(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main(argv: list[str] | None = None) -> None:
    if not ENGINE_AVAILABLE:
        raise RuntimeError("sakigo_engine wheel is not installed; see sakigo/engine/__init__.py")
    args = parse_args(argv)
    if args.pairs <= 0 or args.board_size <= 0 or args.batch_size <= 0:
        raise ValueError("pairs, board-size, and batch-size must be positive")
    if not math.isfinite(args.komi):
        raise ValueError("komi must be finite")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    board_size = args.board_size
    max_plies = args.max_plies or 2 * board_size * board_size
    if max_plies <= 0:
        raise ValueError("max-plies must be positive")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    player_a = make_player(
        args.player_a,
        device,
        args.batch_size,
        args.seed + 101,
        board_size,
        allow_unsafe_legacy=args.allow_unsafe_legacy_checkpoint,
    )
    player_b = make_player(
        args.player_b,
        device,
        args.batch_size,
        args.seed + 202,
        board_size,
        allow_unsafe_legacy=args.allow_unsafe_legacy_checkpoint,
    )
    players = (player_a, player_b)
    player_names = (f"A:{Path(player_a.name).stem}", f"B:{Path(player_b.name).stem}")

    games: list[MatchGame] = []
    for pair in range(args.pairs):
        opening = random_opening(
            random.Random(rng.randrange(1 << 30)), args.opening_plies, board_size, args.komi
        )
        games.append(MatchGame(2 * pair, 0, opening, max_plies, board_size, args.komi))
        games.append(MatchGame(2 * pair + 1, 1, opening, max_plies, board_size, args.komi))

    total = len(games)
    start_time = time.monotonic()
    while True:
        active = [game for game in games if not game.done]
        if not active:
            break
        for player_index in (0, 1):
            moving = [g for g in active if not g.done and g.player_to_move() == player_index]
            if not moving:
                continue
            actions = players[player_index].select_actions(moving, args.temperature)
            for game, action in zip(moving, actions):
                game.play(action)
        done_count = sum(1 for game in games if game.done)
        elapsed = time.monotonic() - start_time
        print(f"\rgames {done_count}/{total}  plies {sum(g.ply for g in games)}  {elapsed:,.0f}s", end="", flush=True)
    print()

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"selfplay_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir = run_dir / "sgf"
    if args.sgf:
        sgf_dir.mkdir(parents=True, exist_ok=True)
    wins = [0, 0]
    draws = 0
    voids = 0
    pair_outcomes: dict[str, int] = {}
    with (run_dir / "games.jsonl").open("w", encoding="utf-8") as handle:
        for game in games:
            winner = game.winner()
            if winner is None:
                if game.ended_by == "max_plies":
                    voids += 1
                    result = "Void"
                    winner_label = "void"
                else:
                    draws += 1
                    result = "0"
                    winner_label = "draw"
            else:
                wins[winner] += 1
                margin = abs(game.score)
                result = ("B+" if game.score > 0 else "W+") + f"{margin:g}"
                winner_label = "A" if winner == 0 else "B"
            if args.sgf:
                write_sgf(sgf_dir / f"game_{game.game_index:04d}.sgf", game, player_names, result)
            handle.write(
                json.dumps(
                    {
                        "game_index": game.game_index,
                        "black_player": "A" if game.black_player == 0 else "B",
                        "winner": winner_label,
                        "result": result,
                        "score_black_minus_white": game.score,
                        "plies": game.ply,
                        "actions": game.actions,
                    }
                )
                + "\n"
            )
    pair_scores: list[float] = []
    for pair in range(args.pairs):
        pair_games = games[2 * pair : 2 * pair + 2]
        if any(game.ended_by == "max_plies" for game in pair_games):
            pair_outcomes["void"] = pair_outcomes.get("void", 0) + 1
            continue
        a_points = sum(
            1.0 if game.winner() == 0 else 0.5 if game.winner() is None else 0.0
            for game in pair_games
        )
        key = f"{a_points:g}-{2.0 - a_points:g}"
        pair_outcomes[key] = pair_outcomes.get(key, 0) + 1
        pair_scores.append(a_points / 2.0)

    scored_games = total - voids
    a_points = wins[0] + 0.5 * draws
    p = a_points / scored_games if scored_games else 0.0
    low, high = wilson_interval(a_points, scored_games)
    pair_low, pair_high = paired_mean_interval(pair_scores)
    summary = {
        "player_a": player_a.name,
        "player_b": player_b.name,
        "pairs": args.pairs,
        "games": total,
        "wins_a": wins[0],
        "wins_b": wins[1],
        "draws": draws,
        "void_games": voids,
        "winrate_a": p,
        "winrate_a_ci95": [low, high],
        "elo_a_minus_b": elo_from_p(p),
        "elo_ci95": [elo_from_p(low), elo_from_p(high)],
        "pair_outcomes_a": pair_outcomes,
        "paired_score_a_ci95": [pair_low, pair_high],
        "opening_plies": args.opening_plies,
        "temperature": args.temperature,
        "board_size": board_size,
        "komi": args.komi,
        "max_plies": max_plies,
        "seed": args.seed,
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"A wins {wins[0]}, B wins {wins[1]}, draws {draws}, void {voids} "
        f"(score={p:.3f}, paired 95% CI {pair_low:.3f}-{pair_high:.3f})  "
        f"Elo {elo_from_p(p):+.0f} (CI {elo_from_p(low):+.0f}..{elo_from_p(high):+.0f})  "
        f"pairs {pair_outcomes}"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
