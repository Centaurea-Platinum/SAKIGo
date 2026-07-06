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
from sakigo.model import SakiGoNet, config_from_dict, remap_legacy_scalar_state_dict
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


def load_policy_model(checkpoint_path: Path, device: torch.device) -> SakiGoNet:
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:  # legacy checkpoints embed argparse leftovers
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = config_from_dict(payload["model_config"])
    state = payload.get("model_state", payload.get("model"))
    if state is None:
        raise ValueError(f"{checkpoint_path} has no model weights")
    if config.group_size == 1 and any(
        value.ndim == 2 and key.endswith(".weight") and not key.startswith("rule_mlp.")
        for key, value in state.items()
    ):
        state = remap_legacy_scalar_state_dict(state)
    model = SakiGoNet(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


class PolicyPlayer:
    def __init__(self, checkpoint_path: Path, device: torch.device, batch_size: int, board_size: int) -> None:
        self.name = str(checkpoint_path)
        self.device = device
        self.batch_size = max(1, batch_size)
        self.board_size = board_size
        self.model = load_policy_model(checkpoint_path, device)
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    @torch.inference_mode()
    def select_actions(self, games: list["MatchGame"], temperature: float) -> list[int]:
        actions: list[int] = []
        size = self.board_size
        for start in range(0, len(games), self.batch_size):
            chunk = games[start : start + self.batch_size]
            legal_masks = [match.game.legal_mask() for match in chunk]
            board = torch.tensor(
                [match.game.board_planes() for match in chunk], dtype=torch.float32
            ).reshape(len(chunk), -1, size, size)
            rules = torch.tensor(
                [match.game.rule_features() for match in chunk], dtype=torch.float32
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


def make_player(spec: str, device: torch.device, batch_size: int, seed: int, board_size: int):
    if spec.strip().lower() == "random":
        return RandomPlayer(seed)
    return PolicyPlayer(Path(spec), device, batch_size, board_size)


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
        for action in opening:
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
            self.score = tromp_taylor_score(self.game.board(), self.board_size, self.komi)

    def winner(self) -> int:
        """0 = player A, 1 = player B (fractional komi makes ties impossible)."""
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


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def elo_from_p(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main(argv: list[str] | None = None) -> None:
    if not ENGINE_AVAILABLE:
        raise RuntimeError("sakigo_engine wheel is not installed; see sakigo/engine/__init__.py")
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    board_size = args.board_size
    max_plies = args.max_plies or 2 * board_size * board_size
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    player_a = make_player(args.player_a, device, args.batch_size, args.seed + 101, board_size)
    player_b = make_player(args.player_b, device, args.batch_size, args.seed + 202, board_size)
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
    pair_outcomes = {"2-0": 0, "1-1": 0, "0-2": 0}
    with (run_dir / "games.jsonl").open("w", encoding="utf-8") as handle:
        for game in games:
            winner = game.winner()
            wins[winner] += 1
            margin = abs(game.score)
            result = ("B+" if game.score > 0 else "W+") + f"{margin:g}"
            if args.sgf:
                write_sgf(sgf_dir / f"game_{game.game_index:04d}.sgf", game, player_names, result)
            handle.write(
                json.dumps(
                    {
                        "game_index": game.game_index,
                        "black_player": "A" if game.black_player == 0 else "B",
                        "winner": "A" if winner == 0 else "B",
                        "result": result,
                        "score_black_minus_white": game.score,
                        "plies": game.ply,
                        "actions": game.actions,
                    }
                )
                + "\n"
            )
    for pair in range(args.pairs):
        a_wins = sum(1 for game in games[2 * pair : 2 * pair + 2] if game.winner() == 0)
        pair_outcomes[("0-2", "1-1", "2-0")[a_wins]] += 1

    p = wins[0] / total if total else 0.0
    low, high = wilson_interval(wins[0], total)
    summary = {
        "player_a": player_a.name,
        "player_b": player_b.name,
        "pairs": args.pairs,
        "games": total,
        "wins_a": wins[0],
        "wins_b": wins[1],
        "winrate_a": p,
        "winrate_a_ci95": [low, high],
        "elo_a_minus_b": elo_from_p(p),
        "elo_ci95": [elo_from_p(low), elo_from_p(high)],
        "pair_outcomes_a": pair_outcomes,
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
        f"A wins {wins[0]}/{total} (p={p:.3f}, 95% CI {low:.3f}-{high:.3f})  "
        f"Elo {elo_from_p(p):+.0f} (CI {elo_from_p(low):+.0f}..{elo_from_p(high):+.0f})  "
        f"pairs 2-0/1-1/0-2: {pair_outcomes['2-0']}/{pair_outcomes['1-1']}/{pair_outcomes['0-2']}"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
