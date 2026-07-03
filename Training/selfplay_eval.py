"""Self-play evaluation: paired color-reversed matches between policy checkpoints.

Players are raw-policy agents (argmax or temperature-sampled) loaded from
Training checkpoints, or the uniform-random baseline. Games reuse the Phase 1
generator's Tromp-Taylor rules and board/rule encoding, so positions are
encoded exactly as in training data. Each pair shares one seeded random
opening and swaps colors; finished games are scored by Tromp-Taylor area
count (komi 7.5) at two consecutive passes or the ply cap.

Examples:
    uv run python -m Training.selfplay_eval --player-a Training/runs/phase1_model1/checkpoints/step_002048.pt --player-b random --pairs 50
    uv run python -m Training.selfplay_eval --player-a <ckpt A> --player-b <ckpt B> --pairs 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Training.checkpoints import load_checkpoint, restore_model_from_checkpoint  # noqa: E402
from Training.common import make_run_dir, resolve_root_path, training_device  # noqa: E402
from Training.generate_katago_phase1 import (  # noqa: E402
    AREA,
    BLACK,
    BOARD_SIZE,
    KOMI,
    WHITE,
    Game,
    NEIGHBORS,
)

SGF_LETTERS = "abcdefghijklmnopqrs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired self-play evaluation between two players.")
    parser.add_argument("--player-a", required=True, help="Checkpoint path or 'random'.")
    parser.add_argument("--player-b", required=True, help="Checkpoint path or 'random'.")
    parser.add_argument("--pairs", type=int, default=50, help="Color-reversed game pairs (2 games each).")
    parser.add_argument("--opening-plies", type=int, default=6, help="Shared uniform-random opening moves per pair.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Policy sampling temperature after the opening (0 = argmax).")
    parser.add_argument("--max-plies", type=int, default=2 * AREA, help="Adjudicate by area count at this ply cap.")
    parser.add_argument("--batch-size", type=int, default=256, help="Max positions per model forward.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-dir", default="", help="Output directory. Defaults to Training/runs/<timestamp>.")
    parser.add_argument("--sgf", action=argparse.BooleanOptionalAction, default=True, help="Write one SGF per game.")
    return parser.parse_args(argv)


def tromp_taylor_score(board: list[int]) -> float:
    """Black minus White area, komi included (positive = Black leads)."""
    black_area = 0
    white_area = 0
    visited = [False] * AREA
    for point in range(AREA):
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
                for neighbor in NEIGHBORS[current]:
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
    return black_area - white_area - KOMI


class RandomPlayer:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def select_actions(self, games: list[Game], temperature: float) -> list[int]:
        actions = []
        for game in games:
            legal = game.legal_mask()
            board_moves = [index for index in range(AREA) if legal[index]]
            actions.append(self.rng.choice(board_moves) if board_moves else AREA)
        return actions


class PolicyPlayer:
    def __init__(self, checkpoint_path: Path, device: torch.device, batch_size: int) -> None:
        self.name = str(checkpoint_path)
        self.device = device
        self.batch_size = max(1, batch_size)
        checkpoint = load_checkpoint(checkpoint_path, device)
        self.model = restore_model_from_checkpoint(checkpoint, device, minimum_board_size=BOARD_SIZE)
        self.model.eval()
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    @torch.inference_mode()
    def select_actions(self, games: list[Game], temperature: float) -> list[int]:
        actions: list[int] = []
        for start in range(0, len(games), self.batch_size):
            chunk = games[start : start + self.batch_size]
            legal_masks = [game.legal_mask() for game in chunk]
            board = torch.tensor(
                [game.board_planes(mask) for game, mask in zip(chunk, legal_masks)],
                dtype=torch.float32,
            ).reshape(len(chunk), -1, BOARD_SIZE, BOARD_SIZE)
            rules = torch.tensor([game.rule_features() for game in chunk], dtype=torch.float32)
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


def make_player(spec: str, device: torch.device, batch_size: int, seed: int):
    if spec.strip().lower() == "random":
        return RandomPlayer(seed)
    return PolicyPlayer(resolve_root_path(spec), device, batch_size)


class MatchGame:
    """One game: a Game plus color assignment and its shared opening."""

    def __init__(self, game_index: int, black_player: int, opening: list[int], max_plies: int) -> None:
        self.game_index = game_index
        self.black_player = black_player  # 0 = player A, 1 = player B
        self.max_plies = max_plies
        self.game = Game(game_id=game_index, rng=random.Random(game_index))
        self.actions: list[int] = []
        self.done = False
        self.score = 0.0
        for action in opening:
            self.play(action)

    def player_to_move(self) -> int:
        return self.black_player if self.game.to_move == BLACK else 1 - self.black_player

    def play(self, action: int) -> None:
        self.game.play(action)
        self.actions.append(action)
        if self.game.passes >= 2 or self.game.ply >= self.max_plies:
            self.done = True
            self.score = tromp_taylor_score(self.game.board)

    def winner(self) -> int:
        """0 = player A, 1 = player B (komi 7.5 makes ties impossible)."""
        black_won = self.score > 0
        return self.black_player if black_won else 1 - self.black_player


def random_opening(rng: random.Random, plies: int) -> list[int]:
    game = Game(game_id=0, rng=rng)
    opening: list[int] = []
    for _ in range(max(0, plies)):
        legal = game.legal_mask()
        moves = [index for index in range(AREA) if legal[index]]
        if not moves:
            break
        action = rng.choice(moves)
        game.play(action)
        opening.append(action)
    return opening


def write_sgf(path: Path, match_game: MatchGame, player_names: tuple[str, str], result: str) -> None:
    black = player_names[match_game.black_player]
    white = player_names[1 - match_game.black_player]
    nodes = []
    color = "B"
    for action in match_game.actions:
        if action == AREA:
            nodes.append(f";{color}[]")
        else:
            row, col = divmod(action, BOARD_SIZE)
            nodes.append(f";{color}[{SGF_LETTERS[col]}{SGF_LETTERS[row]}]")
        color = "W" if color == "B" else "B"
    content = (
        f"(;GM[1]FF[4]SZ[{BOARD_SIZE}]KM[{KOMI}]RU[Tromp-Taylor]"
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
    args = parse_args(argv)
    device = training_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    player_a = make_player(args.player_a, device, args.batch_size, args.seed + 101)
    player_b = make_player(args.player_b, device, args.batch_size, args.seed + 202)
    players = (player_a, player_b)
    player_names = (f"A:{Path(player_a.name).stem}", f"B:{Path(player_b.name).stem}")

    games: list[MatchGame] = []
    for pair in range(args.pairs):
        opening = random_opening(random.Random(rng.randrange(1 << 30)), args.opening_plies)
        games.append(MatchGame(2 * pair, black_player=0, opening=opening, max_plies=args.max_plies))
        games.append(MatchGame(2 * pair + 1, black_player=1, opening=opening, max_plies=args.max_plies))

    total = len(games)
    start_time = time.monotonic()
    while True:
        active = [game for game in games if not game.done]
        if not active:
            break
        for player_index in (0, 1):
            moving = [game for game in active if not game.done and game.player_to_move() == player_index]
            if not moving:
                continue
            actions = players[player_index].select_actions([game.game for game in moving], args.temperature)
            for game, action in zip(moving, actions):
                game.play(action)
        done_count = sum(1 for game in games if game.done)
        elapsed = time.monotonic() - start_time
        print(f"\rgames {done_count}/{total}  plies {sum(g.game.ply for g in games)}  {elapsed:,.0f}s", end="", flush=True)
    print()

    run_dir = make_run_dir(args.run_dir)
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
                        "plies": game.game.ply,
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
        "max_plies": args.max_plies,
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
