"""Data gates: prepared tensor shards preserve record data exactly, the
blake2b split is applied per position, and batches are single-board-size
with balanced rulesets and the contract layout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from sakigo.data import (
    PreparedDataset,
    RulesetBalancedBatchSampler,
    collate_prepared,
    iter_records,
    make_dataloader,
    prepare_tensor_shards,
)
from sakigo.data.prepare import load_manifest
from sakigo.data.records import augment_record_d4, record_from_json, split_for_position
from sakigo.rulesets import ruleset_from_name

SEED = 11
VAL_FRACTION = 0.2


def _make_raw_record(rng: random.Random, board_size: int, ruleset_name: str, ply: int) -> dict:
    ruleset = ruleset_from_name(ruleset_name)
    area = board_size * board_size
    action = area + 1
    captures = [rng.randrange(4), rng.randrange(4)]
    to_move = rng.choice((1, -1))
    policy = [rng.random() for _ in range(action)]
    budget = [rng.random() for _ in range(action)]
    wdl = [rng.random() for _ in range(4)]
    return {
        "schema_version": 1,
        "board_size": board_size,
        "ply": ply,
        "position_key": f"pos-{board_size}-{ruleset_name}-{ply}-{rng.randrange(10**9)}",
        "ruleset": ruleset.metadata(),
        "board_planes": [float(rng.random() > 0.7) for _ in range(6 * area)],
        "rule_features": ruleset.rule_features(
            to_move=to_move, captures=captures, board_area=area
        ),
        "wdl": [value / sum(wdl) for value in wdl],
        "score": rng.uniform(-0.5, 0.5),
        "ownership": [rng.uniform(-1.0, 1.0) for _ in range(area)],
        "policy": [value / sum(policy) for value in policy],
        "budget": [value / sum(budget) for value in budget],
        "legal_mask": [rng.random() > 0.2 for _ in range(area)] + [True],
        "source": {"generator": "test"},
    }


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    rng = random.Random(3)
    root = tmp_path_factory.mktemp("p2data")
    jsonl_path = root / "samples_000000.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index in range(160):
            board_size = 5 if index % 2 == 0 else 7
            ruleset_name = ("tromp-taylor", "chinese", "japanese")[index % 3]
            handle.write(json.dumps(_make_raw_record(rng, board_size, ruleset_name, index)) + "\n")
    out_dir = root / "prepared"
    prepare_tensor_shards([jsonl_path], out_dir, seed=SEED, val_fraction=VAL_FRACTION)
    return jsonl_path, out_dir


def test_split_membership_matches_hash_split(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    manifest = load_manifest(out_dir)
    expected_counts: dict[tuple[str, int], int] = {}
    for record in iter_records([jsonl_path]):
        split = split_for_position(
            record.board_size, record.position_key, VAL_FRACTION, SEED, record.ruleset_key
        )
        expected_counts[(split, record.board_size)] = (
            expected_counts.get((split, record.board_size), 0) + 1
        )
    actual_counts = {
        (group["split"], group["board_size"]): group["count"] for group in manifest["groups"]
    }
    assert actual_counts == expected_counts
    assert any(split == "val" for split, _ in actual_counts)


def test_prepared_arrays_round_trip(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    datasets = {
        split: PreparedDataset(out_dir, split) for split in ("train", "val")
    }
    cursors: dict[tuple[str, int], int] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            legacy = record_from_json(json.loads(line), jsonl_path, line_number)
            split = split_for_position(
                legacy.board_size, legacy.position_key, VAL_FRACTION, SEED, legacy.ruleset_key
            )
            dataset = datasets[split]
            group = next(g for g in dataset.groups if g.board_size == legacy.board_size)
            row = cursors.get((split, legacy.board_size), 0)
            cursors[(split, legacy.board_size)] = row + 1
            arrays = group.arrays
            np.testing.assert_array_equal(arrays["board_planes"][row], legacy.board_planes)
            np.testing.assert_array_equal(arrays["rule_features"][row], legacy.rule_features)
            np.testing.assert_array_equal(arrays["wdl"][row], legacy.wdl)
            assert arrays["score"][row] == np.float32(legacy.score)
            np.testing.assert_array_equal(arrays["ownership"][row], legacy.ownership)
            np.testing.assert_array_equal(arrays["policy"][row], legacy.policy)
            np.testing.assert_array_equal(arrays["budget"][row], legacy.budget)
            np.testing.assert_array_equal(arrays["legal_mask"][row], legacy.legal_mask)
            assert arrays["legal_available"][row]
            assert dataset.ruleset_keys[arrays["ruleset_code"][row]] == legacy.ruleset_key


def test_prepare_is_cached(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    manifest_before = load_manifest(out_dir)
    manifest_after = prepare_tensor_shards(
        [jsonl_path], out_dir, seed=SEED, val_fraction=VAL_FRACTION
    )
    assert manifest_after == manifest_before


def test_batches_are_single_size_and_ruleset_balanced(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    sampler = RulesetBalancedBatchSampler(dataset, batch_size=9, seed=5, length=40)
    ruleset_count = len({dataset.ruleset_code_of(i) for i in range(len(dataset))})
    assert ruleset_count == 3
    for batch_indices in sampler:
        assert len(batch_indices) == 9
        sizes = {dataset.board_size_of(index) for index in batch_indices}
        assert len(sizes) == 1
        code_counts: dict[int, int] = {}
        for index in batch_indices:
            code = dataset.ruleset_code_of(index)
            code_counts[code] = code_counts.get(code, 0) + 1
        assert max(code_counts.values()) - min(code_counts.values()) <= 1


def test_collate_layout_and_values(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    indices = [i for i in range(len(dataset)) if dataset.board_size_of(i) == 5][:4]
    batch = collate_prepared([dataset[i] for i in indices])

    records = [
        record
        for record in iter_records([jsonl_path])
        if record.board_size == 5
        and split_for_position(
            record.board_size, record.position_key, VAL_FRACTION, SEED, record.ruleset_key
        )
        == "train"
    ][:4]
    area = 25
    expected = {
        "board": ((4, 6, 5, 5), torch.float32),
        "rules": ((4, 10), torch.float32),
        "ply": ((4,), torch.int64),
        "wdl_target": ((4, 4), torch.float32),
        "wdl_mask": ((4,), torch.bool),
        "score_target": ((4, 1), torch.float32),
        "score_mask": ((4,), torch.bool),
        "ownership_target": ((4, area), torch.float32),
        "ownership_mask": ((4,), torch.bool),
        "policy_target": ((4, area + 1), torch.float32),
        "policy_mask": ((4,), torch.bool),
        "budget_target": ((4, area + 1), torch.float32),
        "budget_mask": ((4,), torch.bool),
        "legal_mask": ((4, area + 1), torch.bool),
        "legal_mask_available": ((4,), torch.bool),
    }
    assert set(batch) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert tuple(batch[key].shape) == shape, key
        assert batch[key].dtype == dtype, key
    for row, record in enumerate(records):
        np.testing.assert_array_equal(batch["board"][row].numpy(), record.board_planes)
        np.testing.assert_array_equal(batch["rules"][row].numpy(), record.rule_features)
        np.testing.assert_array_equal(batch["policy_target"][row].numpy(), record.policy)
        assert batch["score_target"][row, 0].item() == pytest.approx(record.score)
        assert bool(batch["wdl_mask"][row]) and bool(batch["legal_mask_available"][row])


def test_augmentation_matches_legacy(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train", augment_d4=True)

    class _FixedRng:
        def __init__(self, transform: int) -> None:
            self.transform = transform

        def randrange(self, _stop: int) -> int:
            return self.transform

    plain = PreparedDataset(out_dir, "train")
    index = 0
    size = plain.board_size_of(index)
    base = plain[index]
    for transform in range(8):
        dataset._rng = _FixedRng(transform)  # type: ignore[assignment]
        sample = dataset[index]
        legacy = augment_record_d4(
            _record_from_sample(base, size),
            transform,
        )
        np.testing.assert_array_equal(sample["board_planes"], legacy.board_planes)
        np.testing.assert_array_equal(sample["ownership"], legacy.ownership)
        np.testing.assert_array_equal(sample["policy"], legacy.policy)
        np.testing.assert_array_equal(sample["budget"], legacy.budget)
        np.testing.assert_array_equal(sample["legal_mask"], legacy.legal_mask)


def _record_from_sample(sample: dict, size: int):
    from sakigo.data.records import TrainingRecord

    return TrainingRecord(
        schema_version=1,
        board_size=size,
        ply=int(sample["ply"]),
        position_key="x",
        board_planes=np.asarray(sample["board_planes"]),
        rule_features=np.asarray(sample["rule_features"]),
        wdl=np.asarray(sample["wdl"]),
        score=float(sample["score"]),
        ownership=np.asarray(sample["ownership"]),
        policy=np.asarray(sample["policy"]),
        budget=np.asarray(sample["budget"]),
        legal_mask=np.asarray(sample["legal_mask"]),
    )


def test_dataloader_with_workers(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    sampler = RulesetBalancedBatchSampler(dataset, batch_size=6, seed=7, length=4)
    loader = make_dataloader(dataset, sampler, num_workers=2, pin_memory=False, seed=1)
    batches = list(loader)
    assert len(batches) == 4
    for batch in batches:
        assert batch["board"].shape[0] == 6
        assert batch["board"].dtype == torch.float32
        assert batch["legal_mask"].dtype == torch.bool
