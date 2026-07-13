"""Data gates: prepared tensor shards preserve record data exactly, the
blake2b split is applied per position, and batches are single-board-size
with randomly mixed rulesets and the contract layout.
"""

from __future__ import annotations

import copy
from collections import Counter
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from sakigo.data import (
    GroupedValidationBatchSampler,
    PreparedDataset,
    SizeGroupedBatchSampler,
    collate_prepared,
    iter_records,
    make_dataloader,
    prepare_tensor_shards,
)
from sakigo.data.prepare import load_manifest
from sakigo.data.records import augment_record_d4, record_from_json, split_for_record
from sakigo.rulesets import ruleset_from_name

SEED = 11
VAL_FRACTION = 0.2


def _make_raw_record(rng: random.Random, board_size: int, ruleset_name: str, ply: int) -> dict:
    ruleset = ruleset_from_name(ruleset_name)
    area = board_size * board_size
    action = area + 1
    captures = [rng.randrange(4), rng.randrange(4)]
    to_move = rng.choice((1, -1))
    legal_mask = [rng.random() > 0.2 for _ in range(area)] + [True]
    legal_actions = [index for index, legal in enumerate(legal_mask) if legal]
    policy = [0.0 for _ in range(action)]
    policy[rng.choice(legal_actions)] = 1.0
    budget = [rng.random() if legal else 0.0 for legal in legal_mask]
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
        "policy": policy,
        "budget": [value / sum(budget) for value in budget],
        "legal_mask": legal_mask,
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
    expected_validation: Counter[tuple[int, str]] = Counter()
    for record in iter_records([jsonl_path]):
        split = split_for_record(record, VAL_FRACTION, SEED)
        expected_counts[(split, record.board_size)] = (
            expected_counts.get((split, record.board_size), 0) + 1
        )
        if split == "val":
            expected_validation[(record.board_size, record.ruleset_key)] += 1
    actual_counts = {
        (group["split"], group["board_size"]): group["count"] for group in manifest["groups"]
    }
    assert actual_counts == expected_counts
    assert any(split == "val" for split, _ in actual_counts)
    actual_validation = {
        (group["board_size"], group["ruleset_key"]): group["count"]
        for group in manifest["validation_groups"]
    }
    assert actual_validation == expected_validation


def test_validation_batches_cover_each_size_and_ruleset_cohort(
    prepared: tuple[Path, Path],
) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "val")
    sampler = GroupedValidationBatchSampler(
        dataset, batch_size=4, seed=17, fixed=True
    )
    observed: set[tuple[int, int]] = set()
    for batch in sampler:
        cohorts = {
            (dataset.board_size_of(index), dataset.ruleset_code_of(index))
            for index in batch
        }
        assert len(cohorts) == 1
        observed.update(cohorts)
    expected = {
        (cohort.board_size, cohort.ruleset_code)
        for cohort in dataset.validation_cohorts()
    }
    assert observed == expected

    with pytest.raises(ValueError, match="cannot cover all"):
        GroupedValidationBatchSampler(
            dataset, batch_size=4, seed=17, length=len(expected) - 1
        )


def test_prepared_arrays_round_trip(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    datasets = {
        split: PreparedDataset(out_dir, split) for split in ("train", "val")
    }
    cursors: dict[tuple[str, int], int] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            legacy = record_from_json(json.loads(line), jsonl_path, line_number)
            split = split_for_record(legacy, VAL_FRACTION, SEED)
            dataset = datasets[split]
            group = next(g for g in dataset.groups if g.board_size == legacy.board_size)
            row = cursors.get((split, legacy.board_size), 0)
            cursors[(split, legacy.board_size)] = row + 1
            arrays = group.arrays
            np.testing.assert_array_equal(arrays["board_planes"][row], legacy.board_planes)
            np.testing.assert_array_equal(arrays["rule_features"][row], legacy.rule_features)
            np.testing.assert_array_equal(arrays["wdl"][row], legacy.wdl)
            assert arrays["score"][row] == np.float32(legacy.score)
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


def test_prepare_cache_normalizes_relative_source_paths(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path, out_dir = prepared
    manifest_before = load_manifest(out_dir)
    monkeypatch.chdir(jsonl_path.parent)
    manifest_after = prepare_tensor_shards(
        [Path(jsonl_path.name)], out_dir, seed=SEED, val_fraction=VAL_FRACTION
    )
    assert manifest_after == manifest_before


def test_prepare_cache_detects_same_size_same_mtime_content_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    raw = _make_raw_record(random.Random(43), 5, "tromp-taylor", 0)
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    prepared_dir = tmp_path / "prepared-digest"
    before = prepare_tensor_shards(
        [source], prepared_dir, seed=3, val_fraction=0.0
    )
    stat = source.stat()
    raw["position_key"] = "z" * len(raw["position_key"])
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    assert source.stat().st_size == stat.st_size
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    after = prepare_tensor_shards(
        [source], prepared_dir, seed=3, val_fraction=0.0
    )
    assert after["generation"] != before["generation"]
    assert after["sources"][0]["sha256"] != before["sources"][0]["sha256"]


def test_prepare_preserves_explicit_validation_sources(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    rng = random.Random(31)
    with train_path.open("w", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(json.dumps(_make_raw_record(rng, 5, "tromp-taylor", index)) + "\n")
    with validation_path.open("w", encoding="utf-8") as handle:
        for index in range(3):
            handle.write(json.dumps(_make_raw_record(rng, 5, "tromp-taylor", 100 + index)) + "\n")
    manifest = prepare_tensor_shards(
        [train_path],
        tmp_path / "prepared-explicit",
        validation_data=[validation_path],
        seed=99,
        val_fraction=0.99,
    )
    counts = {group["split"]: group["count"] for group in manifest["groups"]}
    assert counts == {"train": 5, "val": 3}
    assert manifest["split_mode"] == "explicit"


def test_prepare_rejects_duplicate_and_overlapping_sources(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate training"):
        prepare_tensor_shards(
            [source, source], tmp_path / "prepared-duplicate", seed=1, val_fraction=0.1
        )
    with pytest.raises(ValueError, match="overlap"):
        prepare_tensor_shards(
            [source],
            tmp_path / "prepared-overlap",
            validation_data=[source],
            seed=1,
            val_fraction=0.0,
        )


def test_force_prepare_atomically_switches_generation(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    before = load_manifest(out_dir)
    old_directories = [out_dir / group["directory"] for group in before["groups"]]
    after = prepare_tensor_shards(
        [jsonl_path], out_dir, seed=SEED, val_fraction=VAL_FRACTION, force=True
    )
    assert after["generation"] != before["generation"]
    assert all(path.is_dir() for path in old_directories)
    assert all((out_dir / group["directory"]).is_dir() for group in after["groups"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["legal_mask"].__setitem__(0, "false"), "JSON booleans"),
        (lambda raw: raw["legal_mask"].__setitem__(-1, False), "pass entry"),
        (
            lambda raw: raw.update(policy=[1.0 / len(raw["policy"])] * len(raw["policy"])),
            "one-hot",
        ),
        (
            lambda raw: (
                raw["legal_mask"].__setitem__(0, False),
                raw.update(budget=[1.0] + [0.0] * (len(raw["budget"]) - 1)),
            ),
            "illegal actions",
        ),
    ],
)
def test_record_contract_rejects_malformed_targets(mutate, message: str) -> None:
    raw = _make_raw_record(random.Random(13), 5, "tromp-taylor", 0)
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        record_from_json(raw)


@pytest.mark.parametrize("field", ["schema_version", "ruleset"])
def test_record_contract_requires_version_and_ruleset(field: str) -> None:
    raw = _make_raw_record(random.Random(19), 5, "tromp-taylor", 0)
    raw.pop(field)
    with pytest.raises(ValueError):
        record_from_json(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1.5, "schema_version must be an integer"),
        ("board_size", True, "board_size must be an integer"),
        ("ply", "3", "ply must be an integer"),
        ("position_key", 17, "position_key must be a string"),
    ],
)
def test_record_contract_rejects_coerced_identity_fields(
    field: str, value: object, message: str
) -> None:
    raw = _make_raw_record(random.Random(31), 5, "tromp-taylor", 0)
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        record_from_json(raw)


def test_canonical_split_ignores_move_path_identity() -> None:
    raw = _make_raw_record(random.Random(17), 5, "tromp-taylor", 0)
    transposed = copy.deepcopy(raw)
    transposed["position_key"] = "different-move-order"
    first = record_from_json(raw)
    second = record_from_json(transposed)
    assert split_for_record(first, VAL_FRACTION, SEED) == split_for_record(
        second, VAL_FRACTION, SEED
    )


def test_batches_are_single_size_with_unstratified_rulesets(
    prepared: tuple[Path, Path],
) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    sampler = SizeGroupedBatchSampler(dataset, batch_size=9, seed=5, length=40)
    ruleset_count = len({dataset.ruleset_code_of(i) for i in range(len(dataset))})
    assert ruleset_count == 3
    for batch_indices in sampler:
        assert len(batch_indices) == 9
        sizes = {dataset.board_size_of(index) for index in batch_indices}
        assert len(sizes) == 1
    # Sampling is deterministic for a seed, but neither rules nor consecutive
    # board sizes are constrained. At least one naturally uneven rules batch
    # distinguishes this from the removed fixed-ratio scheduler.
    replay = SizeGroupedBatchSampler(dataset, batch_size=9, seed=5, length=40)
    batches = list(replay)
    assert any(
        len({dataset.ruleset_code_of(index) for index in batch}) < ruleset_count
        for batch in batches
    )
    sizes = [dataset.board_size_of(batch[0]) for batch in batches]
    assert any(left == right for left, right in zip(sizes, sizes[1:]))


def test_default_size_schedule_preserves_uneven_group_frequency(
    prepared: tuple[Path, Path],
) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    batch_size = 9
    expected = {
        group.board_size: (group.count + batch_size - 1) // batch_size
        for group in dataset.groups
    }
    sampler = SizeGroupedBatchSampler(
        dataset,
        batch_size=batch_size,
        seed=41,
        length=sum(expected.values()),
    )
    observed = Counter(dataset.board_size_of(batch[0]) for batch in sampler)
    assert observed == expected


def test_sampler_state_round_trip(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    first = SizeGroupedBatchSampler(dataset, batch_size=9, seed=23)
    iterator = iter(first)
    for _ in range(7):
        next(iterator)
    state = first.state_dict()
    expected = [next(iterator) for _ in range(8)]

    resumed = SizeGroupedBatchSampler(dataset, batch_size=9, seed=999)
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    actual = [next(resumed_iterator) for _ in range(8)]
    assert actual == expected


def test_sampler_never_duplicates_a_record_when_a_pool_wraps(
    prepared: tuple[Path, Path],
) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    minimum_pool = min(
        sum(group.count for group in dataset.groups if group.board_size == size)
        for size in dataset.board_sizes
    )
    batch_size = max(2, minimum_pool // 2 + 1)
    sampler = SizeGroupedBatchSampler(
        dataset,
        batch_size=batch_size,
        seed=71,
        length=8,
    )
    for batch in sampler:
        assert len(batch) == len(set(batch))


def test_sampler_state_rejects_changed_batch_size(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    source = SizeGroupedBatchSampler(dataset, batch_size=9, seed=2)
    state = source.state_dict()
    changed = SizeGroupedBatchSampler(dataset, batch_size=8, seed=2)
    with pytest.raises(ValueError, match="batch size"):
        changed.load_state_dict(state)


def test_augmentation_rng_state_round_trip(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    first = PreparedDataset(out_dir, "train", augment_d4=True)
    indices = list(range(min(6, len(first))))
    for index in indices:
        first[index]
    state = first.augmentation_state_dict()
    expected = [first[index]["board_planes"] for index in indices]

    resumed = PreparedDataset(out_dir, "train", augment_d4=True)
    resumed.load_augmentation_state_dict(state)
    actual = [resumed[index]["board_planes"] for index in indices]
    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_collate_layout_and_values(prepared: tuple[Path, Path]) -> None:
    jsonl_path, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    indices = [i for i in range(len(dataset)) if dataset.board_size_of(i) == 5][:4]
    batch = collate_prepared([dataset[i] for i in indices])

    records = [
        record
        for record in iter_records([jsonl_path])
        if record.board_size == 5
        and split_for_record(record, VAL_FRACTION, SEED)
        == "train"
    ][:4]
    area = 25
    expected = {
        "board": ((4, 6, 5, 5), torch.float32),
        "board_size": ((4,), torch.int64),
        "ruleset_code": ((4,), torch.int32),
        "rules": ((4, 10), torch.float32),
        "ply": ((4,), torch.int64),
        "wdl_target": ((4, 4), torch.float32),
        "wdl_mask": ((4,), torch.bool),
        "score_target": ((4, 1), torch.float32),
        "score_mask": ((4,), torch.bool),
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


def test_fetch_batch_matches_scalar_collate(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    board_size = dataset.board_size_of(0)
    indices = [index for index in range(len(dataset)) if dataset.board_size_of(index) == board_size][:8]

    fast = dataset.fetch_batch(indices)
    slow = collate_prepared([dataset[index] for index in indices])

    assert isinstance(fast, dict)
    for key, value in slow.items():
        torch.testing.assert_close(fast[key], value)


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
        ownership=None,
        policy=np.asarray(sample["policy"]),
        budget=np.asarray(sample["budget"]),
        legal_mask=np.asarray(sample["legal_mask"]),
    )


def test_dataloader_with_workers(prepared: tuple[Path, Path]) -> None:
    _, out_dir = prepared
    dataset = PreparedDataset(out_dir, "train")
    sampler = SizeGroupedBatchSampler(dataset, batch_size=6, seed=7, length=4)
    loader = make_dataloader(dataset, sampler, num_workers=2, pin_memory=False, seed=1)
    batches = list(loader)
    assert len(batches) == 4
    for batch in batches:
        assert batch["board"].shape[0] == 6
        assert batch["board"].dtype == torch.float32
        assert batch["legal_mask"].dtype == torch.bool
