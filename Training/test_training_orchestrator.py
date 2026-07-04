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
)
from Training.losses import (
    LossWeights,
    compute_head_losses,
    masked_soft_cross_entropy,
    weighted_total_loss,
)
from Training.train import _make_optimizer, main as train_main
from Training.generate_katago_phase1 import AREA as GO_AREA, BLACK, WHITE
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
        "wdl": [1.0, 0.0, 0.0],
        "score": 0.25,
        "ownership": [1.0 if cell % 2 == 0 else -1.0 for cell in range(area)],
        "policy": policy,
        "budget": budget,
        "legal_mask": legal_mask,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


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
        "wdl_logits": torch.zeros(1, 3),
        "score": torch.zeros(1, 1),
        "ownership_logits": torch.zeros(1, 9),
        "policy_logits": torch.zeros(1, 10),
        "budget_logits": torch.zeros(1, 10),
    }
    head_losses = compute_head_losses(output, batch)
    assert head_losses["wdl"].item() == pytest.approx(math.log(3.0))
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
