"""Phase-1 generation loop: schedule variants, drive KataGo, write shards.

Decomposed from the legacy 290-line run(): plan/schedule (plan.py), engine
game (game.py), record building (records.py), KataGo client (katago.py),
shard writer (writer.py). Progress = tqdm; heartbeat = status.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from sakigo.data.records import is_zstd_jsonl_path
from sakigo.generate.game import GeneratorGame
from sakigo.generate.katago import (
    KataGoAnalysisClient,
    default_config_path,
    default_katago_path,
    default_model_path,
)
from sakigo.generate.plan import (
    DEFAULT_SAMPLES_PER_COMBINATION,
    GenerationSchedule,
    GenerationVariant,
    build_generation_plan,
    parse_board_sizes,
    parse_komis,
    parse_ruleset_names,
)
from sakigo.generate.records import record_from_response, sample_action
from sakigo.generate.writer import GenerationOutputWriter
from sakigo.rulesets import available_rulesets, ruleset_from_overrides

DEFAULT_BOARD_SIZES = "13,16,19"
DEFAULT_KOMIS = ",".join(f"{index * 0.5:.1f}" for index in range(27))
DEFAULT_RULESETS = ",".join(available_rulesets())
DEFAULT_SAMPLES_PER_FILE = 2**16
DEFAULT_ZSTD_LEVEL = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 SAKIGo records with KataGo.")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--samples-per-combination",
        type=int,
        default=None,
        help=f"Default: {DEFAULT_SAMPLES_PER_COMBINATION}.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-file", type=int, default=DEFAULT_SAMPLES_PER_FILE)
    parser.add_argument("--zstd-level", type=int, default=DEFAULT_ZSTD_LEVEL)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--katago", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--nn-batch-size", type=int, default=20)
    parser.add_argument("--analysis-threads", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1024)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--board-sizes", default=DEFAULT_BOARD_SIZES)
    parser.add_argument("--rulesets", default=None, help=f"Default: {DEFAULT_RULESETS}.")
    parser.add_argument("--katago-rules", default="")
    parser.add_argument("--katago-ko", default="")
    parser.add_argument("--katago-suicide", default="")
    parser.add_argument("--komi", type=float, default=None)
    parser.add_argument("--komis", default=None, help=f"Default: {DEFAULT_KOMIS}.")
    parser.add_argument("--saki-scoring", default="")
    parser.add_argument("--saki-ko", default="")
    parser.add_argument("--saki-suicide", default="")
    return parser.parse_args(argv)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.komi is not None and args.komis is not None and args.komis.strip():
        raise ValueError("use either --komi or --komis, not both")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    board_sizes = parse_board_sizes(args.board_sizes)
    ruleset_names = parse_ruleset_names(args.rulesets or DEFAULT_RULESETS)
    has_override = any(
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
    if len(ruleset_names) > 1 and has_override:
        raise ValueError("KataGo/SAKIGo rule overrides require a single ruleset")
    base_rulesets = [
        ruleset_from_overrides(
            ruleset=name,
            katago_rules=args.katago_rules or None,
            katago_ko=args.katago_ko or None,
            katago_suicide=args.katago_suicide or None,
            saki_scoring=args.saki_scoring or None,
            saki_ko=args.saki_ko or None,
            saki_suicide=args.saki_suicide or None,
            komi=args.komi,
        )
        for name in ruleset_names
    ]
    komi_values = "" if args.komi is not None else (args.komis if args.komis is not None else DEFAULT_KOMIS)
    komis = parse_komis(komi_values, base_rulesets[0].komi)
    plan = build_generation_plan(
        board_sizes,
        base_rulesets,
        komis,
        samples=args.samples,
        samples_per_combination=args.samples_per_combination,
    )
    if args.samples_per_file == 0 and not is_zstd_jsonl_path(args.output):
        print("warning: single plain JSONL output is deprecated; prefer shards", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status or (args.run_dir / "status.json")

    katago_path = args.katago or default_katago_path()
    client = KataGoAnalysisClient(
        katago=katago_path,
        model=args.model or default_model_path(),
        config=args.config or default_config_path(katago_path),
        analysis_threads=args.analysis_threads,
        nn_batch_size=args.nn_batch_size,
        run_dir=args.run_dir,
    )

    rng = random.Random(args.seed)
    schedule = GenerationSchedule(plan, rng)

    games: list[GeneratorGame] = []
    for index in range(args.concurrency):
        variant = schedule.choose_variant()
        if variant is None:
            break
        games.append(
            GeneratorGame(
                game_id=index,
                rng=random.Random(args.seed + index + 1),
                board_size=variant.board_size,
                ruleset=variant.ruleset,
            )
        )
    if not games:
        raise RuntimeError("generation plan has no schedulable combinations")

    pending: dict[str, GeneratorGame] = {}
    next_query = 0
    next_game_id = args.concurrency
    completed_games = 0
    started_at = time.time()
    show_progress = args.progress if args.progress is not None else sys.stdout.isatty()
    progress = tqdm(
        total=plan.target_samples, disable=not show_progress, unit="rec", dynamic_ncols=True
    )

    def variant_for_game(game: GeneratorGame) -> GenerationVariant:
        return GenerationVariant(board_size=game.board_size, ruleset=game.ruleset)

    def send(game: GeneratorGame) -> bool:
        nonlocal next_query
        if not schedule.reserve(variant_for_game(game)):
            return False
        query_id = f"g{game.game_id}-q{next_query}"
        next_query += 1
        pending[query_id] = game
        client.send(game.query(query_id))
        return True

    def reset_game(game: GeneratorGame) -> bool:
        nonlocal next_game_id
        variant = schedule.choose_variant()
        if variant is None:
            return False
        game.reset(next_game_id, board_size=variant.board_size, ruleset=variant.ruleset)
        next_game_id += 1
        return True

    def send_or_reschedule(game: GeneratorGame) -> None:
        if send(game):
            return
        if reset_game(game):
            send(game)

    for game in games:
        send_or_reschedule(game)
    client.flush()

    try:
        with GenerationOutputWriter(
            args.output, samples_per_file=args.samples_per_file, zstd_level=args.zstd_level
        ) as output:
            while schedule.total_written < plan.target_samples:
                if not pending:
                    raise RuntimeError("generation schedule exhausted before reaching target samples")
                response = client.responses.get()
                if response.get("_engine_exited"):
                    tail = " | ".join(response.get("stderr_tail") or [])
                    raise RuntimeError(
                        f"KataGo exited unexpectedly (returncode={response.get('returncode')}): {tail}"
                    )
                game = pending.pop(str(response.get("id")), None)
                if game is None:
                    continue
                variant = variant_for_game(game)
                if "error" in response:
                    schedule.complete(variant, success=False)
                    client.kill()
                    raise RuntimeError(
                        f"KataGo analysis error for board_size={game.board_size}, "
                        f"ruleset={game.ruleset.metadata()}: {response.get('error')}"
                    )

                record, budget = record_from_response(game, response)
                schedule.complete(variant, success=True)
                output.write_record(record)
                progress.update(1)

                action = sample_action(budget, game.rng, args.temperature)
                game.play(action)
                can_continue = True
                if game.should_reset(args.max_plies):
                    completed_games += 1
                    can_continue = reset_game(game)
                if can_continue and schedule.total_written < plan.target_samples:
                    send_or_reschedule(game)

                if (
                    schedule.total_written % args.log_interval == 0
                    or schedule.total_written == plan.target_samples
                ):
                    output.flush()
                    elapsed = max(1e-9, time.time() - started_at)
                    samples_per_second = schedule.total_written / elapsed
                    completed_combinations = sum(
                        1 for key, quota in plan.quotas.items() if schedule.written[key] >= quota
                    )
                    _write_status(
                        status_path,
                        {
                            "state": "running"
                            if schedule.total_written < plan.target_samples
                            else "complete",
                            "samples": schedule.total_written,
                            "target_samples": plan.target_samples,
                            "samples_per_second": samples_per_second,
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
                            "board_sizes": board_sizes,
                            "komis": komis,
                            "completed_combinations": completed_combinations,
                            "updated_at": _now_iso(),
                        },
                    )
                client.flush()
    finally:
        progress.close()
        client.shutdown()


def main() -> None:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"generator failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
