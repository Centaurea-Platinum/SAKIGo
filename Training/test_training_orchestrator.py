from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import pytest
import numpy as np
import torch

from Model.sakigo_model import config_from_spec
from Training.checkpoints import model_from_config
from Training.data import (
    TRAIN_SPLIT,
    StreamingJsonlBuffer,
    augment_record_d4,
    collate,
    load_records,
    record_from_json,
    scan_jsonl_stream_metadata,
    split_records,
)
from Training.losses import (
    LossWeights,
    compute_head_losses,
    masked_soft_cross_entropy,
    weighted_total_loss,
)
from Training.train import _make_optimizer, main as train_main
from Training.generate_katago_phase1 import (
    AREA as GO_AREA,
    BLACK,
    BOARD_SIZE,
    Game,
    WHITE,
    DEFAULT_BOARD_SIZES,
    DEFAULT_KOMIS,
    DEFAULT_RULESETS,
    DEFAULT_SAMPLES_PER_COMBINATION,
    GenerationSchedule,
    build_generation_plan,
    find_katago_path,
    katago_executable_names,
    parse_board_sizes,
    parse_komis,
    parse_ruleset_names,
    record_from_response,
)
from Training.rulesets import ruleset_from_name, ruleset_from_overrides, validate_rule_features
from Training.selfplay_eval import main as eval_main, tromp_taylor_score


def test_tromp_taylor_score_counts_area_and_komi() -> None:
    empty = [0] * GO_AREA
    assert tromp_taylor_score(empty) == 0 - 7.5  # neutral empty board, komi to White

    black_only = [0] * GO_AREA
    black_only[0] = BLACK
    assert tromp_taylor_score(black_only) == 361 - 7.5  # all empty space touches only Black

    white_only = [0] * GO_AREA
    white_only[5] = WHITE
    assert tromp_taylor_score(white_only) == -361 - 7.5

    contested = [0] * GO_AREA
    contested[0] = BLACK
    contested[GO_AREA - 1] = WHITE
    assert tromp_taylor_score(contested) == 1 - 1 - 7.5  # shared empty region is neutral


def test_selfplay_eval_random_match_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval"
    eval_main(
        [
            "--player-a",
            "random",
            "--player-b",
            "random",
            "--pairs",
            "1",
            "--opening-plies",
            "4",
            "--max-plies",
            "24",
            "--device",
            "cpu",
            "--seed",
            "5",
            "--run-dir",
            str(run_dir),
        ]
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["games"] == 2
    assert summary["wins_a"] + summary["wins_b"] == 2
    games = [json.loads(line) for line in (run_dir / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(games) == 2
    assert {game["black_player"] for game in games} == {"A", "B"}
    assert games[0]["actions"][:4] == games[1]["actions"][:4]  # shared opening
    assert len(list((run_dir / "sgf").glob("*.sgf"))) == 2


def sample_record(position_key: str = "p0", board_size: int = 3) -> dict:
    area = board_size * board_size
    action_count = area + 1
    board_planes = [0.0 for _ in range(6 * area)]
    for cell in range(area):
        board_planes[2 * area + cell] = 1.0
    policy = [0.0 for _ in range(action_count)]
    policy[-1] = 1.0
    budget = [1.0 for _ in range(action_count)]
    legal_mask = [True for _ in range(action_count)]
    legal_mask[0] = False
    return {
        "schema_version": 1,
        "board_size": board_size,
        "ply": 0,
        "position_key": position_key,
        "board_planes": board_planes,
        "rule_features": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "wdl": [1.0, 0.0, 0.0, 0.0],
        "score": 0.25,
        "ownership": [1.0 if cell % 2 == 0 else -1.0 for cell in range(area)],
        "policy": policy,
        "budget": budget,
        "legal_mask": legal_mask,
    }


def sample_record_with_ruleset(position_key: str, ruleset_name: str, board_size: int = 3) -> dict:
    ruleset = ruleset_from_name(ruleset_name)
    raw = sample_record(position_key, board_size)
    raw["ruleset"] = ruleset.metadata()
    raw["rule_features"] = ruleset.rule_features(
        to_move=BLACK,
        captures=[0, 0],
        board_area=board_size * board_size,
    )
    return raw


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_katago_engine_lookup_prefers_host_executable_name(tmp_path: Path) -> None:
    engine_root = tmp_path / "engine"
    windows_engine = engine_root / "katago-v1.16.5-windows-x64"
    linux_engine = engine_root / "katago-v1.16.5-opencl-linux-x64"
    windows_engine.mkdir(parents=True)
    linux_engine.mkdir(parents=True)
    (windows_engine / "katago.exe").touch()
    (linux_engine / "katago").touch()

    assert katago_executable_names("win32")[0] == "katago.exe"
    assert katago_executable_names("linux")[0] == "katago"
    assert find_katago_path(engine_root, "win32") == windows_engine / "katago.exe"
    assert find_katago_path(engine_root, "linux") == linux_engine / "katago"


def test_record_validation_rejects_bad_lengths_and_missing_targets() -> None:
    bad_planes = sample_record()
    bad_planes["board_planes"] = bad_planes["board_planes"][:-1]
    with pytest.raises(ValueError, match="board_planes"):
        record_from_json(bad_planes)

    missing_targets = sample_record()
    for key in ("wdl", "score", "ownership", "policy", "budget"):
        missing_targets.pop(key)
    with pytest.raises(ValueError, match="at least one target"):
        record_from_json(missing_targets)

    bad_policy = sample_record()
    bad_policy["policy"] = bad_policy["policy"][:-1]
    with pytest.raises(ValueError, match="policy"):
        record_from_json(bad_policy)


def test_ruleset_presets_project_katago_rules_to_sakigo_features() -> None:
    with pytest.raises(ValueError, match="unknown ruleset"):
        ruleset_from_name("aga")
    with pytest.raises(ValueError, match="unknown ruleset"):
        ruleset_from_name("new-zealand")

    tromp = ruleset_from_name("tromp-taylor")
    assert tromp.query_fields() == {"rules": "tromp-taylor", "komi": 7.5}
    assert tromp.katago_ko == "positional_superko"
    assert tromp.rule_features(to_move=BLACK, captures=[0, 0], board_area=GO_AREA)[:8] == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]

    ancient = ruleset_from_name("ancient-chinese")
    assert ancient.query_fields()["rules"]["tax"] == "ALL"
    assert ancient.query_fields()["rules"]["whiteHandicapBonus"] == "0"
    ancient_features = ancient.rule_features(to_move=BLACK, captures=[0, 0], board_area=GO_AREA)
    assert ancient_features[:8] == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    validate_rule_features(ancient_features, ancient)

    chinese = ruleset_from_name("chinese")
    assert chinese.query_fields()["rules"] == "chinese"
    assert chinese.katago_ko == "simple_ko"
    assert chinese.ko == "simple_ko"
    validate_rule_features(chinese.rule_features(to_move=BLACK, captures=[0, 0], board_area=GO_AREA), chinese)

    chinese_ogs = ruleset_from_name("chinese-ogs")
    assert chinese_ogs.katago_ko == "positional_superko"
    assert chinese_ogs.ko == "positional_superko"

    japanese = ruleset_from_name("japanese")
    assert japanese.scoring == "territory"
    features = japanese.rule_features(to_move=BLACK, captures=[0, 0], board_area=GO_AREA)
    assert features[:8] == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    validate_rule_features(features, japanese)


def test_custom_katago_rules_can_override_local_and_sakigo_mapping() -> None:
    with pytest.raises(ValueError, match="does not exactly project"):
        ruleset_from_overrides(
            ruleset="tromp-taylor",
            katago_rules='{"koRule":"SITUATIONAL","multiStoneSuicideLegal":false}',
            saki_scoring="territory_with_seki_score",
            saki_ko="positional_superko",
            saki_suicide="forbidden",
            komi=6.5,
        )
    with pytest.raises(ValueError, match="no exact SAKIGo mapping"):
        ruleset_from_overrides(ruleset="tromp-taylor", katago_rules="aga")
    with pytest.raises(ValueError, match="no exact SAKIGo mapping"):
        ruleset_from_overrides(ruleset="tromp-taylor", katago_rules="new-zealand")

    ruleset = ruleset_from_overrides(
        ruleset="tromp-taylor",
        katago_rules='{"koRule":"POSITIONAL","scoringRule":"TERRITORY","taxRule":"NONE","multiStoneSuicideLegal":false}',
        saki_scoring="territory_with_seki_score",
        saki_ko="positional_superko",
        saki_suicide="forbidden",
        komi=6.5,
    )

    assert ruleset.query_fields()["rules"] == {
        "ko": "POSITIONAL",
        "scoring": "TERRITORY",
        "tax": "NONE",
        "suicide": False,
    }
    assert ruleset.katago_ko == "positional_superko"
    assert ruleset.katago_suicide == "forbidden"
    assert ruleset.scoring == "territory_with_seki_score"
    assert ruleset.komi == 6.5
    validate_rule_features(ruleset.rule_features(to_move=WHITE, captures=[2, 5], board_area=GO_AREA), ruleset)


def test_generator_query_and_record_include_ruleset_metadata() -> None:
    board_size = 9
    area = board_size * board_size
    game = Game(game_id=3, rng=random.Random(9), board_size=board_size, ruleset=ruleset_from_name("japanese"))
    query = json.loads(game.query("q0"))
    assert query["rules"] == "japanese"
    assert query["komi"] == 6.5
    assert query["boardXSize"] == board_size
    assert query["boardYSize"] == board_size

    response = {
        "id": "q0",
        "policy": [0.0 for _ in range(area)] + [1.0],
        "ownership": [0.0 for _ in range(area)],
        "rootInfo": {
            "rawWinrate": 0.5,
            "rawDrawProb": 0.1,
            "rawNoResultProb": 0.25,
            "rawLead": 0.0,
        },
    }
    record, budget = record_from_response(game, response)
    assert budget[-1] == 1.0
    assert record["board_size"] == board_size
    assert len(record["policy"]) == area + 1
    assert len(record["ownership"]) == area
    assert record["ruleset"]["name"] == "japanese"
    assert record["source"]["katago_play"] == {"ko": "simple_ko", "suicide": "forbidden"}
    assert record["rule_features"][:8] == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert record["wdl"] == pytest.approx([0.5, 0.1, 0.15, 0.25])
    parsed = record_from_json(record)
    assert parsed.ruleset is not None
    assert parsed.ruleset["name"] == "japanese"


def test_generator_board_size_and_komi_options_validate() -> None:
    assert parse_board_sizes(DEFAULT_BOARD_SIZES) == [13, 16, 19]
    assert parse_komis(DEFAULT_KOMIS, 7.5) == [index * 0.5 for index in range(27)]
    assert parse_ruleset_names(DEFAULT_RULESETS) == [
        "ancient-chinese",
        "chinese",
        "chinese-ogs",
        "japanese",
        "korean",
        "tromp-taylor",
    ]
    assert parse_board_sizes("9,13,19") == [9, 13, 19]
    assert parse_komis("5.5,6.5,7.5", 7.5) == [5.5, 6.5, 7.5]
    assert parse_komis("", 6.5) == [6.5]
    with pytest.raises(ValueError, match="board-sizes"):
        parse_board_sizes("")
    with pytest.raises(ValueError, match="board-sizes"):
        parse_board_sizes("0")
    with pytest.raises(ValueError, match="komis"):
        parse_komis("nan", 7.5)
    with pytest.raises(ValueError, match="duplicates"):
        parse_board_sizes("13,13")
    with pytest.raises(ValueError, match="duplicates"):
        parse_komis("6.5,6.5", 7.5)
    with pytest.raises(ValueError, match="duplicates"):
        parse_ruleset_names("chinese,chinese")


def test_generation_plan_defaults_to_full_factorial_quota() -> None:
    board_sizes = parse_board_sizes(DEFAULT_BOARD_SIZES)
    rulesets = [ruleset_from_name(name) for name in parse_ruleset_names(DEFAULT_RULESETS)]
    komis = parse_komis(DEFAULT_KOMIS, rulesets[0].komi)

    plan = build_generation_plan(
        board_sizes,
        rulesets,
        komis,
        samples=None,
        samples_per_combination=None,
    )

    assert len(plan.variants) == 3 * 6 * 27
    assert set(plan.quotas.values()) == {DEFAULT_SAMPLES_PER_COMBINATION}
    assert plan.target_samples == 3_981_312
    assert plan.samples_per_combination == 2**13


def test_generation_schedule_enforces_exact_combination_quotas() -> None:
    plan = build_generation_plan(
        [13],
        [ruleset_from_name("chinese"), ruleset_from_name("japanese")],
        [0.0, 0.5],
        samples=None,
        samples_per_combination=2,
    )
    schedule = GenerationSchedule(plan, random.Random(5))

    while schedule.total_written < plan.target_samples:
        variant = schedule.choose_variant()
        assert variant is not None
        assert schedule.reserve(variant)
        assert schedule.complete(variant, success=True)

    assert schedule.choose_variant() is None
    assert plan.target_samples == 8
    assert set(schedule.written.values()) == {2}


def test_game_legality_uses_katago_mapping_for_ko_and_suicide() -> None:
    simple = Game(game_id=0, rng=random.Random(1), ruleset=ruleset_from_name("chinese"))
    simple.simple_ko = 0
    assert simple.analyze_play(0) is None

    positional = Game(game_id=1, rng=random.Random(1), ruleset=ruleset_from_name("tromp-taylor"))
    positional.simple_ko = 0
    assert positional.analyze_play(0) is not None

    suicide = ruleset_from_overrides(
        ruleset="tromp-taylor",
        katago_rules='{"koRule":"SIMPLE","multiStoneSuicideLegal":true}',
        saki_ko="simple_ko",
        saki_suicide="allowed",
    )
    game = Game(game_id=2, rng=random.Random(1), ruleset=suicide)
    game.board[1] = WHITE
    game.board[BOARD_SIZE] = WHITE
    game.seen = {tuple(game.board)}
    game.seen_states = {(tuple(game.board), BLACK)}
    assert game.analyze_play(0) is None


def test_record_ruleset_metadata_validates_rule_feature_projection() -> None:
    raw = sample_record_with_ruleset("jp", "japanese")
    record = record_from_json(raw)
    assert record.ruleset is not None
    assert record.ruleset["name"] == "japanese"
    assert record.ruleset_key

    mismatched = sample_record()
    mismatched["ruleset"] = ruleset_from_name("japanese").metadata()
    with pytest.raises(ValueError, match="one-hots do not match ruleset"):
        record_from_json(mismatched)


def test_split_and_streaming_scan_keep_rulesets_separate(tmp_path: Path) -> None:
    rows = [
        sample_record_with_ruleset("same-position", "chinese"),
        sample_record_with_ruleset("same-position", "chinese-ogs"),
    ]
    records = [record_from_json(row) for row in rows]
    train_records, val_records = split_records(records, 0.5, random.Random(2))
    assert len(train_records) == 1
    assert len(val_records) == 1

    data_path = tmp_path / "rulesets.jsonl"
    write_jsonl(data_path, rows)
    metadata = scan_jsonl_stream_metadata([data_path], val_fraction=0.5, seed=2, use_cache=False)
    assert metadata.record_count == 2
    assert metadata.train_count + metadata.val_count == 2
    assert sorted(metadata.ruleset_counts.values()) == [1, 1]


def test_collate_keeps_pass_as_final_action_and_rejects_mixed_sizes(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    write_jsonl(data_path, [sample_record()])
    records = load_records([data_path])
    batch = collate(records, torch.device("cpu"))
    assert batch["board"].shape == (1, 6, 3, 3)
    assert batch["rules"].shape == (1, 10)
    assert batch["policy_target"].shape == (1, 10)
    assert batch["policy_target"][0, -1] == 1.0
    assert not bool(batch["legal_mask"][0, 0])

    with pytest.raises(ValueError, match="one board size"):
        collate([record_from_json(sample_record("a", 3)), record_from_json(sample_record("b", 4))], torch.device("cpu"))


def test_losses_match_known_cross_entropy_values() -> None:
    record = record_from_json(sample_record())
    batch = collate([record], torch.device("cpu"))
    output = {
        "wdl_logits": torch.zeros(1, 4),
        "score": torch.zeros(1, 1),
        "ownership_logits": torch.zeros(1, 9),
        "policy_logits": torch.zeros(1, 10),
        "budget_logits": torch.zeros(1, 10),
    }
    head_losses = compute_head_losses(output, batch)
    assert head_losses["wdl"].item() == pytest.approx(math.log(4.0))
    assert head_losses["policy"].item() == pytest.approx(math.log(10.0))
    assert head_losses["budget"].item() == pytest.approx(math.log(10.0))
    assert head_losses["ownership"].item() == pytest.approx(math.log(2.0))
    assert head_losses["score"].item() == pytest.approx(0.5 * 0.25 * 0.25)

    total = weighted_total_loss(head_losses, LossWeights())
    expected = sum(value.item() for value in head_losses.values())
    assert total.item() == pytest.approx(expected)

    empty_mask = torch.zeros(1, dtype=torch.bool)
    assert masked_soft_cross_entropy(torch.zeros(1, 2), torch.zeros(1, 2), empty_mask).item() == 0.0


def test_augment_record_d4_moves_spatial_fields_consistently() -> None:
    raw = sample_record()
    area = 9
    policy = [0.0] * (area + 1)
    policy[1] = 0.5  # cell (0, 1)
    policy[-1] = 0.5  # pass
    raw["policy"] = policy
    raw["ownership"] = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  # top row owned
    record = record_from_json(raw)
    size = record.board_size

    def reference(plane: np.ndarray, transform: int) -> np.ndarray:
        out = np.rot90(plane, k=transform % 4, axes=(-2, -1))
        if transform >= 4:
            out = out[..., ::-1]
        return out

    assert augment_record_d4(record, 0) is record
    for transform in range(8):
        augmented = augment_record_d4(record, transform)
        assert np.array_equal(augmented.board_planes, reference(record.board_planes, transform))
        assert np.array_equal(
            augmented.ownership.reshape(size, size),
            reference(record.ownership.reshape(size, size), transform),
        )
        assert np.array_equal(
            augmented.policy[:-1].reshape(size, size),
            reference(record.policy[:-1].reshape(size, size), transform),
        )
        assert augmented.policy[-1] == record.policy[-1]
        assert np.array_equal(
            augmented.legal_mask[:-1].reshape(size, size),
            reference(record.legal_mask[:-1].reshape(size, size), transform),
        )
        assert augmented.legal_mask[-1] == record.legal_mask[-1]
        assert np.array_equal(augmented.rule_features, record.rule_features)
        assert np.array_equal(augmented.wdl, record.wdl)
        assert augmented.score == record.score
        assert abs(float(augmented.policy.sum()) - 1.0) < 1e-6


def test_train_smoke_and_resume(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    rows = [sample_record(f"p{index}") for index in range(4)]
    write_jsonl(data_path, rows)
    run_dir = tmp_path / "run"

    base_args = [
        "--data",
        str(data_path),
        "--run-dir",
        str(run_dir),
        "--device",
        "cpu",
        "--model-board-size",
        "3",
        "--batch-size",
        "2",
        "--log-interval",
        "1",
        "--checkpoint-interval",
        "1",
        "--loss-eval-batches",
        "1",
        "--val-fraction",
        "0.25",
        "--seed",
        "3",
    ]
    train_main([*base_args, "--steps", "1"])
    first_checkpoint = run_dir / "checkpoints" / "step_000001.pt"
    assert first_checkpoint.exists()

    train_main([*base_args, "--steps", "2", "--resume", str(first_checkpoint)])
    assert (run_dir / "checkpoints" / "step_000002.pt").exists()

    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1", "2"]
    assert rows[-1]["val_policy_loss"]


def test_train_only_logs_on_log_steps(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    rows = [sample_record(f"p{index}") for index in range(4)]
    write_jsonl(data_path, rows)
    run_dir = tmp_path / "run"

    train_main(
        [
            "--data",
            str(data_path),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--model-board-size",
            "3",
            "--batch-size",
            "2",
            "--steps",
            "3",
            "--log-interval",
            "100",
            "--checkpoint-interval",
            "100",
            "--early-eval-steps",
            "",
            "--loss-eval-batches",
            "1",
            "--val-fraction",
            "0.25",
            "--seed",
            "3",
            "--prefetch-batches",
            "0",
        ]
    )

    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert [row["step"] for row in metrics_rows] == ["1", "3"]
    assert metrics_rows[-1]["train_wdl_target_count"] == "4"


def test_optimizer_excludes_offsets_norms_and_register_seed_from_weight_decay() -> None:
    model = model_from_config(config_from_spec("model1", board_size=3))
    optimizer = _make_optimizer(
        model,
        argparse.Namespace(lr=3e-4, weight_decay=1e-4, cuda_graphs=False),
        torch.device("cpu"),
    )

    decay_ids = {id(parameter) for group in optimizer.param_groups if group["weight_decay"] for parameter in group["params"]}
    no_decay_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if not group["weight_decay"]
        for parameter in group["params"]
    }
    params_by_name = dict(model.named_parameters())

    assert id(params_by_name["register_seed"]) in no_decay_ids
    assert any(parameter_id in decay_ids for parameter_id in (id(parameter) for parameter in params_by_name.values()))
    for name, parameter in params_by_name.items():
        if name.endswith(".bias") or "norm" in name.lower():
            assert id(parameter) in no_decay_ids
            assert id(parameter) not in decay_ids


def test_streaming_train_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "stream.jsonl"
    rows = [sample_record(f"p{index}") for index in range(24)]
    write_jsonl(data_path, rows)

    metadata = scan_jsonl_stream_metadata([data_path], val_fraction=0.5, seed=11)
    assert metadata.record_count == 24
    assert metadata.train_count + metadata.val_count == 24
    assert metadata.board_counts == {3: 24}

    run_dir = tmp_path / "stream_run"
    train_main(
        [
            "--data",
            str(data_path),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--model-board-size",
            "3",
            "--batch-size",
            "2",
            "--steps",
            "2",
            "--log-interval",
            "1",
            "--checkpoint-interval",
            "1",
            "--loss-eval-batches",
            "1",
            "--val-fraction",
            "0.5",
            "--seed",
            "11",
            "--stream-buffer-mb",
            "0.02",
        ]
    )

    assert (run_dir / "checkpoints" / "step_000002.pt").exists()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["data_loading"] == "streaming"
    assert config["stream_metadata"]["record_count"] == 24
    assert config["stream_buffer"]["max_buffer_bytes"] == int(0.02 * 1024 * 1024)

    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert [row["step"] for row in metrics_rows] == ["1", "2"]


def test_streaming_train_buffer_samples_without_replacement(tmp_path: Path) -> None:
    data_path = tmp_path / "stream_unique.jsonl"
    rows = [sample_record(f"p{index}") for index in range(10)]
    write_jsonl(data_path, rows)

    first_record = record_from_json(rows[0])
    record_bytes = first_record.array_nbytes() + 256
    metadata = scan_jsonl_stream_metadata([data_path], val_fraction=0.0, seed=7)

    with StreamingJsonlBuffer(
        paths=[data_path],
        boards=[3],
        val_fraction=0.0,
        seed=7,
        max_buffer_bytes=record_bytes * 4,
        metadata=metadata,
    ) as stream:
        stream.prime(minimum_records=2)
        rng = random.Random(123)
        seen: list[str] = []
        for _ in range(5):
            seen.extend(record.position_key for record in stream.sample_batch(TRAIN_SPLIT, 2, rng))

    assert len(seen) == 10
    assert sorted(seen) == [f"p{index}" for index in range(10)]
