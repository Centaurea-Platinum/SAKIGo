from __future__ import annotations

import dataclasses
import json
import queue
import random
import threading
from dataclasses import dataclass
from hashlib import blake2b
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import (
    BOARD_PLANE_COUNT,
    RULE_FEATURE_COUNT,
    SCHEMA_VERSION,
    TrainingRecord,
)


TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


def _float_array(raw: Any, expected: int, label: str) -> np.ndarray:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    if len(raw) != expected:
        raise ValueError(f"{label} must have length {expected}")
    try:
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only numbers") from exc
    if values.ndim != 1:
        raise ValueError(f"{label} must be a flat list")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    return values


def _distribution(raw: Any, expected: int, label: str) -> np.ndarray:
    values = _float_array(raw, expected, label)
    if (values < 0.0).any():
        raise ValueError(f"{label} must be non-negative")
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError(f"{label} must have positive mass")
    return (values / total).astype(np.float32, copy=False)


def _optional_distribution(source: Mapping[str, Any], key: str, expected: int) -> np.ndarray | None:
    if key not in source or source[key] is None:
        return None
    return _distribution(source[key], expected, key)


def _optional_vector(source: Mapping[str, Any], key: str, expected: int) -> np.ndarray | None:
    if key not in source or source[key] is None:
        return None
    return _float_array(source[key], expected, key)


def _optional_score(source: Mapping[str, Any]) -> float | None:
    if "score" not in source or source["score"] is None:
        return None
    value = float(source["score"])
    if not np.isfinite(value):
        raise ValueError("score must be finite")
    return value


def _optional_legal_mask(source: Mapping[str, Any], expected: int) -> np.ndarray | None:
    if "legal_mask" not in source or source["legal_mask"] is None:
        return None
    raw = source["legal_mask"]
    if not isinstance(raw, list):
        raise ValueError("legal_mask must be a list")
    if len(raw) != expected:
        raise ValueError(f"legal_mask must have length {expected}")
    return np.asarray(raw, dtype=bool)


def record_from_json(raw: Mapping[str, Any], path: Path | None = None, line_number: int = 0) -> TrainingRecord:
    label = f"{path}:{line_number}" if path is not None and line_number else "record"
    try:
        board_size = int(raw["board_size"])
        ply = int(raw["ply"])
        position_key = str(raw["position_key"])
        schema_version = int(raw.get("schema_version", raw.get("version", SCHEMA_VERSION)))
    except KeyError as exc:
        raise ValueError(f"{label} is missing field {exc}") from exc
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{label} uses unsupported schema_version {schema_version}")
    if board_size <= 0:
        raise ValueError(f"{label} board_size must be positive")
    if ply < 0:
        raise ValueError(f"{label} ply must be non-negative")
    if not position_key:
        raise ValueError(f"{label} position_key must be non-empty")

    area = board_size * board_size
    action_count = area + 1
    targets = raw.get("targets", raw)
    if not isinstance(targets, Mapping):
        raise ValueError(f"{label} targets must be an object")
    board_planes = _float_array(raw.get("board_planes"), BOARD_PLANE_COUNT * area, "board_planes")
    board_planes = board_planes.reshape(BOARD_PLANE_COUNT, board_size, board_size)
    rule_features = _float_array(raw.get("rule_features"), RULE_FEATURE_COUNT, "rule_features")
    ownership = _optional_vector(targets, "ownership", area)
    if ownership is not None and ((ownership < -1.0) | (ownership > 1.0)).any():
        raise ValueError("ownership values must be in [-1, 1]")

    legal_source = targets if "legal_mask" in targets else raw
    record = TrainingRecord(
        schema_version=schema_version,
        board_size=board_size,
        ply=ply,
        position_key=position_key,
        board_planes=board_planes,
        rule_features=rule_features,
        wdl=_optional_distribution(targets, "wdl", 3),
        score=_optional_score(targets),
        ownership=ownership,
        policy=_optional_distribution(targets, "policy", action_count),
        budget=_optional_distribution(targets, "budget", action_count),
        legal_mask=_optional_legal_mask(legal_source, action_count),
    )
    if all(getattr(record, key) is None for key in ("wdl", "score", "ownership", "policy", "budget")):
        raise ValueError(f"{label} must provide at least one target")
    return record


def load_records(paths: Iterable[Path]) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing data file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(record_from_json(json.loads(stripped), path, line_number))
    if not records:
        raise ValueError("no training records loaded")
    return records


@dataclass(frozen=True)
class JsonlStreamMetadata:
    record_count: int
    train_count: int
    val_count: int
    board_counts: dict[int, int]

    @property
    def board_sizes(self) -> list[int]:
        return sorted(self.board_counts)

    def count_for_split(self, split: str) -> int:
        if split == TRAIN_SPLIT:
            return self.train_count
        if split == VAL_SPLIT:
            return self.val_count
        raise ValueError(f"unknown split: {split}")


@dataclass(frozen=True)
class BufferedJsonlRecord:
    """A decoded record held in the streaming buffer. Decoding happens once at insert."""

    record: TrainingRecord
    byte_size: int
    board_size: int
    split: str


def split_for_position(
    board_size: int,
    position_key: str,
    val_fraction: float,
    seed: int,
) -> str:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0.0:
        return TRAIN_SPLIT
    key = f"{seed}:{board_size}:{position_key}".encode("utf-8")
    value = int.from_bytes(blake2b(key, digest_size=8).digest(), byteorder="big")
    fraction = value / float(1 << 64)
    return VAL_SPLIT if fraction < val_fraction else TRAIN_SPLIT


def _line_board_and_position(line: str, path: Path, line_number: int) -> tuple[int, str]:
    label = f"{path}:{line_number}"
    try:
        raw = json.loads(line)
        board_size = int(raw["board_size"])
        position_key = str(raw["position_key"])
    except KeyError as exc:
        raise ValueError(f"{label} is missing field {exc}") from exc
    if board_size <= 0:
        raise ValueError(f"{label} board_size must be positive")
    if not position_key:
        raise ValueError(f"{label} position_key must be non-empty")
    return board_size, position_key


def scan_jsonl_stream_metadata(
    paths: Iterable[Path],
    val_fraction: float,
    seed: int,
    boards: Iterable[int] | None = None,
) -> JsonlStreamMetadata:
    board_filter = set(boards) if boards is not None else None
    board_counts: dict[int, int] = {}
    record_count = 0
    train_count = 0
    val_count = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing data file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                board_size, position_key = _line_board_and_position(stripped, path, line_number)
                if board_filter is not None and board_size not in board_filter:
                    continue
                split = split_for_position(board_size, position_key, val_fraction, seed)
                record_count += 1
                board_counts[board_size] = board_counts.get(board_size, 0) + 1
                if split == VAL_SPLIT:
                    val_count += 1
                else:
                    train_count += 1
    if record_count == 0:
        raise ValueError("no streaming records found")
    return JsonlStreamMetadata(
        record_count=record_count,
        train_count=train_count,
        val_count=val_count,
        board_counts=board_counts,
    )


class StreamingJsonlBuffer:
    def __init__(
        self,
        paths: Iterable[Path],
        boards: Iterable[int],
        val_fraction: float,
        seed: int,
        max_buffer_bytes: int,
        metadata: JsonlStreamMetadata,
    ) -> None:
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("at least one data path is required")
        self.boards = set(boards)
        if not self.boards:
            raise ValueError("at least one board size is required")
        self.val_fraction = val_fraction
        self.seed = seed
        self.max_buffer_bytes = max_buffer_bytes
        self.metadata = metadata
        self.entries: list[BufferedJsonlRecord] = []
        self.buffer_bytes = 0
        self.lock = threading.Lock()
        self._evict_rng = random.Random(seed + 99_001)
        self._path_index = 0
        self._handle = None
        self._line_number = 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> StreamingJsonlBuffer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def prime(self, minimum_records: int = 1) -> None:
        with self.lock:
            minimum_records = max(1, minimum_records)
            while len(self.entries) < minimum_records:
                entry = self._read_next_accepted()
                if self.entries and self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
                    break
                self._add_entry(entry)
            while True:
                entry = self._read_next_accepted()
                if self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
                    break
                self._add_entry(entry)
            if not self.entries:
                raise ValueError("streaming buffer could not load any records")

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        rng: random.Random,
        board_weights: Mapping[int, float] | None = None,
        advance: bool = True,
    ) -> list[TrainingRecord]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.metadata.count_for_split(split) <= 0:
            raise ValueError(f"no records are available for split {split}")
        with self.lock:
            self._ensure_split_available(split)
            sizes = sorted({entry.board_size for entry in self.entries if entry.split == split})
            if not sizes:
                raise ValueError(f"streaming buffer has no records for split {split}")
            if board_weights is None:
                board_size = rng.choice(sizes)
            else:
                weights = [board_weights.get(size, 1.0) for size in sizes]
                board_size = rng.choices(sizes, weights=weights, k=1)[0]
            pool = [entry for entry in self.entries if entry.split == split and entry.board_size == board_size]
            records = [rng.choice(pool).record for _ in range(batch_size)]
            if advance:
                for _ in range(batch_size):
                    self._add_entry(self._read_next_accepted())
        return records

    def advance(self, accepted_records: int) -> None:
        with self.lock:
            for _ in range(max(0, accepted_records)):
                self._add_entry(self._read_next_accepted())

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "buffer_bytes": self.buffer_bytes,
                "buffer_records": len(self.entries),
                "max_buffer_bytes": self.max_buffer_bytes,
            }

    def _ensure_split_available(self, split: str) -> None:
        while not any(entry.split == split for entry in self.entries):
            self._add_entry(self._read_next_accepted())

    def _add_entry(self, entry: BufferedJsonlRecord) -> None:
        if entry.byte_size > self.max_buffer_bytes:
            raise ValueError(
                "streaming buffer is too small to hold a single record; "
                "increase --stream-buffer-mb"
            )
        while self.entries and self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
            index = self._evict_rng.randrange(len(self.entries))
            removed = self.entries.pop(index)
            self.buffer_bytes -= removed.byte_size
        self.entries.append(entry)
        self.buffer_bytes += entry.byte_size

    def _read_next_accepted(self) -> BufferedJsonlRecord:
        while True:
            path, line_number, line = self._read_next_line()
            stripped = line.strip()
            if not stripped:
                continue
            record = record_from_json(json.loads(stripped), path, line_number)
            if record.board_size not in self.boards:
                continue
            split = split_for_position(record.board_size, record.position_key, self.val_fraction, self.seed)
            return BufferedJsonlRecord(
                record=record,
                byte_size=record.array_nbytes() + 256,
                board_size=record.board_size,
                split=split,
            )

    def _read_next_line(self) -> tuple[Path, int, str]:
        while True:
            if self._handle is None:
                path = self.paths[self._path_index]
                self._handle = path.open("r", encoding="utf-8")
                self._line_number = 0
            line = self._handle.readline()
            if line:
                self._line_number += 1
                return self.paths[self._path_index], self._line_number, line
            self._handle.close()
            self._handle = None
            self._path_index = (self._path_index + 1) % len(self.paths)


def _d4_transform_planes(array: np.ndarray, transform: int) -> np.ndarray:
    """Apply one of the 8 dihedral transforms to the last two (spatial) axes."""
    out = np.rot90(array, k=transform % 4, axes=(-2, -1))
    if transform >= 4:
        out = out[..., ::-1]
    return np.ascontiguousarray(out)


def augment_record_d4(record: TrainingRecord, transform: int) -> TrainingRecord:
    """Apply a D4 board symmetry (0..7) to all spatial fields; the pass entry stays last.

    Global fields (rules, wdl, score) are symmetry-invariant and unchanged.
    """
    if transform % 8 == 0:
        return record
    size = record.board_size

    def spatial(vector: np.ndarray) -> np.ndarray:
        return _d4_transform_planes(vector.reshape(size, size), transform).reshape(-1)

    def action(vector: np.ndarray) -> np.ndarray:
        out = vector.copy()
        out[:-1] = spatial(vector[:-1])
        return out

    return dataclasses.replace(
        record,
        board_planes=_d4_transform_planes(record.board_planes, transform),
        ownership=spatial(record.ownership) if record.ownership is not None else None,
        policy=action(record.policy) if record.policy is not None else None,
        budget=action(record.budget) if record.budget is not None else None,
        legal_mask=action(record.legal_mask) if record.legal_mask is not None else None,
    )


def infer_board_sizes(records: Iterable[TrainingRecord]) -> list[int]:
    return sorted({record.board_size for record in records})


def filter_records_by_boards(records: list[TrainingRecord], boards: Iterable[int]) -> list[TrainingRecord]:
    board_set = set(boards)
    filtered = [record for record in records if record.board_size in board_set]
    if not filtered:
        raise ValueError("no records remain after board-size filtering")
    return filtered


def split_records(
    records: list[TrainingRecord],
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    by_position: dict[tuple[int, str], list[TrainingRecord]] = {}
    for record in records:
        by_position.setdefault((record.board_size, record.position_key), []).append(record)
    positions = list(by_position)
    rng.shuffle(positions)
    val_count = int(len(positions) * val_fraction)
    if val_count == 0 and val_fraction > 0.0 and len(positions) > 1:
        val_count = 1
    val_positions = set(positions[:val_count])
    train_records: list[TrainingRecord] = []
    val_records: list[TrainingRecord] = []
    for position, position_records in by_position.items():
        (val_records if position in val_positions else train_records).extend(position_records)
    return train_records, val_records


def build_groups(records: list[TrainingRecord]) -> dict[int, list[list[TrainingRecord]]]:
    grouped: dict[int, dict[int, list[TrainingRecord]]] = {}
    for record in records:
        grouped.setdefault(record.board_size, {}).setdefault(record.ply, []).append(record)
    return {size: list(groups.values()) for size, groups in grouped.items() if groups}


def sample_batch(
    groups_by_size: dict[int, list[list[TrainingRecord]]],
    batch_size: int,
    rng: random.Random,
    board_weights: Mapping[int, float] | None = None,
) -> list[TrainingRecord]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sizes = list(groups_by_size)
    if not sizes:
        raise ValueError("cannot sample from empty groups")
    if board_weights is None:
        size = rng.choice(sizes)
    else:
        weights = [board_weights.get(size, 1.0) for size in sizes]
        size = rng.choices(sizes, weights=weights, k=1)[0]
    groups = groups_by_size[size]
    return [rng.choice(rng.choice(groups)) for _ in range(batch_size)]


def _stack_optional(
    records: list[TrainingRecord],
    key: str,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(records), width), dtype=np.float32)
    mask = np.zeros(len(records), dtype=bool)
    for index, record in enumerate(records):
        field = getattr(record, key)
        if field is not None:
            values[index] = field
            mask[index] = True
    return values, mask


def collate_cpu(records: list[TrainingRecord], pin_memory: bool = False) -> dict[str, torch.Tensor]:
    """Assemble a CPU batch from numpy-backed records (optionally pinned for async H2D copy)."""
    if not records:
        raise ValueError("cannot collate an empty batch")
    board_size = records[0].board_size
    if any(record.board_size != board_size for record in records):
        raise ValueError("batches must contain one board size")
    area = board_size * board_size
    action_count = area + 1

    batch: dict[str, torch.Tensor] = {
        "board": torch.from_numpy(np.stack([record.board_planes for record in records])),
        "rules": torch.from_numpy(np.stack([record.rule_features for record in records])),
        "ply": torch.tensor([record.ply for record in records], dtype=torch.long),
    }

    for key, width in (("wdl", 3), ("ownership", area), ("policy", action_count), ("budget", action_count)):
        values, mask = _stack_optional(records, key, width)
        batch[f"{key}_target"] = torch.from_numpy(values)
        batch[f"{key}_mask"] = torch.from_numpy(mask)

    score_values = np.zeros((len(records), 1), dtype=np.float32)
    score_mask = np.zeros(len(records), dtype=bool)
    for index, record in enumerate(records):
        if record.score is not None:
            score_values[index, 0] = record.score
            score_mask[index] = True
    batch["score_target"] = torch.from_numpy(score_values)
    batch["score_mask"] = torch.from_numpy(score_mask)

    legal_values = np.ones((len(records), action_count), dtype=bool)
    legal_available = np.zeros(len(records), dtype=bool)
    for index, record in enumerate(records):
        if record.legal_mask is not None:
            legal_values[index] = record.legal_mask
            legal_available[index] = True
    batch["legal_mask"] = torch.from_numpy(legal_values)
    batch["legal_mask_available"] = torch.from_numpy(legal_available)

    if pin_memory:
        batch = {key: value.pin_memory() for key, value in batch.items()}
    return batch


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    if device.type == "cpu":
        return batch
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def collate(records: list[TrainingRecord], device: torch.device) -> dict[str, torch.Tensor]:
    return batch_to_device(collate_cpu(records, pin_memory=device.type == "cuda"), device)


class BatchPrefetcher:
    """Assembles CPU batches on a background thread so ingest overlaps the GPU step.

    produce_fn must be thread-safe (the streaming buffer locks internally; the
    sampling rng is only ever touched from the producer thread).
    """

    def __init__(
        self,
        produce_fn: Callable[[], dict[str, torch.Tensor]],
        depth: int = 2,
    ) -> None:
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self._produce_fn = produce_fn
        # Held while producing; lets the trainer snapshot rng state between batches (checkpointing).
        self.lock = threading.Lock()
        self._queue: queue.Queue[dict[str, torch.Tensor] | None] = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="batch-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self.lock:
                    batch = self._produce_fn()
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # propagate to the consumer
            self._error = exc
            self._queue.put(None)

    def next(self) -> dict[str, torch.Tensor]:
        item = self._queue.get()
        if item is None:
            assert self._error is not None
            raise self._error
        return item

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)

    def __enter__(self) -> BatchPrefetcher:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
