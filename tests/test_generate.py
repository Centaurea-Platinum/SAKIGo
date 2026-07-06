"""Generator component invariants (self-contained; parity with the legacy
generator — queries, position keys, record construction — was verified before
the P6 cutover).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from sakigo.data.records import open_jsonl_text, record_from_json
from sakigo.engine import ENGINE_AVAILABLE
from sakigo.generate.plan import GenerationSchedule, build_generation_plan
from sakigo.generate.records import normalize_policy, record_from_response, sample_action
from sakigo.generate.writer import GenerationOutputWriter
from sakigo.rulesets import BLACK, WHITE, ruleset_from_name

pytestmark = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="sakigo_engine wheel not installed")


def _game(board_size: int = 7, ruleset_name: str = "tromp-taylor"):
    from sakigo.generate.game import GeneratorGame

    return GeneratorGame(
        game_id=3, rng=random.Random(1), board_size=board_size, ruleset=ruleset_from_name(ruleset_name)
    )


def _fake_response(game, rng: random.Random, query_id: str = "q0") -> dict:
    return {
        "id": query_id,
        "policy": [rng.random() for _ in range(game.action_count)],
        "ownership": [rng.uniform(-1, 1) for _ in range(game.area)],
        "rootInfo": {
            "rawWinrate": 0.61,
            "rawDrawProb": 0.02,
            "rawNoResultProb": 0.01,
            "rawLead": 3.5,
        },
    }


def test_query_payload_and_position_key() -> None:
    game = _game()
    payload = json.loads(game.query("qid"))
    assert payload["id"] == "qid"
    assert payload["boardXSize"] == payload["boardYSize"] == 7
    assert payload["maxVisits"] == 1 and payload["includePolicy"] and payload["includeOwnership"]
    key_empty = game.position_key()
    game.play(10)
    assert payload["moves"] == []
    assert json.loads(game.query("q2"))["moves"] == [["B", "D6"]]
    assert game.position_key() != key_empty
    assert len(game.position_key()) == 20


def test_record_perspective_flip_and_schema() -> None:
    rng = random.Random(7)
    # Black to move: targets are Black's values.
    black_game = _game()
    assert black_game.to_move == BLACK
    black_record, _ = record_from_response(black_game, _fake_response(black_game, random.Random(7)))
    assert black_record["wdl"] == pytest.approx([0.61, 0.02, 0.36, 0.01])
    assert black_record["score"] == pytest.approx(3.5 / 49)

    # White to move: wdl order flipped, score and ownership negated.
    white_game = _game()
    white_game.play(10)
    assert white_game.to_move == WHITE
    response = _fake_response(white_game, random.Random(7))
    white_record, budget = record_from_response(white_game, response)
    assert white_record["wdl"] == pytest.approx([0.36, 0.02, 0.61, 0.01])
    assert white_record["score"] == pytest.approx(-3.5 / 49)
    assert white_record["ownership"] == pytest.approx(
        [-float(value) for value in response["ownership"]]
    )
    assert white_record["policy"][max(range(len(budget)), key=budget.__getitem__)] == 1.0

    # Both records must satisfy the frozen schema contract.
    for raw in (black_record, white_record):
        record = record_from_json(raw)
        assert record.board_size == 7 and record.legal_mask is not None


def test_normalize_policy_degenerate_case() -> None:
    mask = [True, False, True, True]
    assert normalize_policy([0.5, 0.9, -0.1, 0.2], mask) == pytest.approx(
        [0.5 / 0.7, 0.0, 0.0, 0.2 / 0.7]
    )
    assert normalize_policy([-1.0, -1.0, -1.0, -1.0], mask) == [0.0, 0.0, 0.0, 1.0]


def test_sample_action_greedy_and_temperature() -> None:
    distribution = [0.1, 0.7, 0.2]
    assert sample_action(distribution, random.Random(0), 0.0) == 1
    counts = [0, 0, 0]
    rng = random.Random(0)
    for _ in range(500):
        counts[sample_action(distribution, rng, 1.0)] += 1
    assert counts[1] > counts[0] and counts[1] > counts[2]


def test_schedule_quota_accounting() -> None:
    plan = build_generation_plan(
        [5],
        [ruleset_from_name("tromp-taylor"), ruleset_from_name("chinese")],
        [7.5],
        samples=7,
        samples_per_combination=None,
    )
    assert plan.target_samples == 7
    schedule = GenerationSchedule(plan, random.Random(0))
    written = 0
    while True:
        variant = schedule.choose_variant()
        if variant is None:
            break
        assert schedule.reserve(variant)
        schedule.complete(variant, success=True)
        written += 1
    assert written == 7
    assert all(
        schedule.written[variant.key()] == plan.quotas[variant.key()] for variant in plan.variants
    )


def test_output_writer_shards(tmp_path: Path) -> None:
    output = tmp_path / "samples"
    with GenerationOutputWriter(output, samples_per_file=3, zstd_level=3) as writer:
        for index in range(8):
            writer.write_record({"index": index})
        paths = list(writer.paths)
    assert [path.name for path in paths] == [
        "samples_000000.jsonl.zst",
        "samples_000001.jsonl.zst",
        "samples_000002.jsonl.zst",
    ]
    seen = []
    for path in paths:
        with open_jsonl_text(path) as handle:
            seen.extend(json.loads(line)["index"] for line in handle if line.strip())
    assert seen == list(range(8))


def test_katago_client_signals_engine_exit() -> None:
    """If KataGo dies, the reader must enqueue a sentinel so run() fails loudly
    instead of blocking forever on responses.get()."""
    import io
    import queue
    import types

    from sakigo.generate.katago import KataGoAnalysisClient

    client = object.__new__(KataGoAnalysisClient)
    client.process = types.SimpleNamespace(
        stdout=io.StringIO('{"id":"q1"}\nnoise\n'), poll=lambda: 137
    )
    client.responses = queue.Queue()
    client._stderr_tail = ["boom"]
    client._stdout_reader()
    assert client.responses.get_nowait() == {"id": "q1"}
    sentinel = client.responses.get_nowait()
    assert sentinel["_engine_exited"] is True
    assert sentinel["returncode"] == 137
    assert sentinel["stderr_tail"] == ["boom"]
