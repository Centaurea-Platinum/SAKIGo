from __future__ import annotations

import dataclasses
import glob
import io
import json
import random
import threading
from dataclasses import dataclass, field
from hashlib import blake2b
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import zstandard as zstd
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .common import (
    BOARD_PLANE_COUNT,
    RULE_FEATURE_COUNT,
    SCHEMA_VERSION,
    TrainingRecord,
    WDL_LABELS,
)
from .rulesets import (
    ruleset_from_metadata,
    ruleset_key_from_raw,
    validate_rule_features,
)


TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
ZSTD_JSONL_SUFFIXES = (".jsonl.zst", ".jsonl.zstd")
LEGACY_JSONL_SUFFIX = ".jsonl"


def is_zstd_jsonl_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ZSTD_JSONL_SUFFIXES)


def is_legacy_jsonl_path(path: Path) -> bool:
    return path.name.lower().endswith(LEGACY_JSONL_SUFFIX) and not is_zstd_jsonl_path(path)


def data_format_label(paths: Iterable[Path]) -> str:
    paths = list(paths)
    if paths and all(is_zstd_jsonl_path(path) for path in paths):
        return "jsonl_zstd_shards"
    if paths and all(is_legacy_jsonl_path(path) for path in paths):
        return "legacy_jsonl_deprecated"
    return "mixed_jsonl"


def expand_data_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        raw = str(path)
        if any(char in raw for char in "*?[]"):
            expanded.extend(Path(match) for match in sorted(glob.glob(raw)))
            continue
        if path.is_dir():
            candidates = sorted(path.glob("*.jsonl.zst"))
            candidates.extend(sorted(path.glob("*.jsonl.zstd")))
            if not candidates:
                candidates = sorted(path.glob("*.jsonl"))
            expanded.extend(candidates)
            continue
        expanded.append(path)
    if not expanded:
        raise ValueError("no data files matched --data")
    return expanded


class _ZstdTextReader:
    def __init__(self, path: Path) -> None:
        self._raw = path.open("rb")
        self._reader = zstd.ZstdDecompressor().stream_reader(self._raw)
        self._text = io.TextIOWrapper(self._reader, encoding="utf-8")

    def readline(self) -> str:
        return self._text.readline()

    def __iter__(self):
        return iter(self._text)

    def close(self) -> None:
        self._text.close()
        self._reader.close()
        self._raw.close()

    def __enter__(self) -> _ZstdTextReader:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _ZstdTextWriter:
    def __init__(self, path: Path, compression_level: int = 3) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = path.open("wb")
        compressor = zstd.ZstdCompressor(level=compression_level, threads=-1)
        self._writer = compressor.stream_writer(self._raw, closefd=False)
        self._text = io.TextIOWrapper(self._writer, encoding="utf-8")

    def write(self, text: str) -> int:
        return self._text.write(text)

    def flush(self) -> None:
        self._text.flush()

    def close(self) -> None:
        self._text.close()
        self._raw.close()

    def __enter__(self) -> _ZstdTextWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def open_jsonl_text(path: Path) -> TextIO | _ZstdTextReader:
    if is_zstd_jsonl_path(path):
        return _ZstdTextReader(path)
    return path.open("r", encoding="utf-8")


def open_jsonl_writer(path: Path, compression_level: int = 3) -> TextIO | _ZstdTextWriter:
    if is_zstd_jsonl_path(path):
        return _ZstdTextWriter(path, compression_level=compression_level)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


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


def _record_ruleset(raw: Mapping[str, Any]) -> dict[str, object] | None:
    ruleset_raw = raw.get("ruleset")
    if ruleset_raw is None:
        source = raw.get("source")
        if isinstance(source, Mapping):
            legacy_rules = source.get("rules")
            if isinstance(legacy_rules, str):
                ruleset_raw = legacy_rules
    ruleset = ruleset_from_metadata(ruleset_raw)
    return ruleset.metadata() if ruleset is not None else None


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
    ruleset = _record_ruleset(raw)
    parsed_ruleset = ruleset_from_metadata(ruleset)
    validate_rule_features(rule_features, parsed_ruleset)
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
        ruleset_key=ruleset_key_from_raw(ruleset),
        ruleset=ruleset,
        wdl=_optional_distribution(targets, "wdl", len(WDL_LABELS)),
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
        with open_jsonl_text(path) as handle:
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
    ruleset_counts: dict[str, int]
    board_ruleset_counts: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def board_sizes(self) -> list[int]:
        return sorted(self.board_counts)

    def count_for_split(self, split: str) -> int:
        if split == TRAIN_SPLIT:
            return self.train_count
        if split == VAL_SPLIT:
            return self.val_count
        raise ValueError(f"unknown split: {split}")

    def rulesets_for_board(self, board_size: int) -> list[str]:
        counts = self.board_ruleset_counts.get(board_size)
        if counts:
            return sorted(counts)
        if len(self.board_counts) == 1 and board_size in self.board_counts:
            return sorted(self.ruleset_counts)
        return []


@dataclass(frozen=True)
class BufferedJsonlRecord:
    """A decoded record held in the streaming buffer. Decoding happens once at insert."""

    record: TrainingRecord
    byte_size: int
    board_size: int
    split: str
    ruleset_key: str


@dataclass(frozen=True)
class JsonlRecordOffset:
    path_index: int
    byte_offset: int
    line_number: int


def _choose_board_size(
    sizes: Iterable[int],
    rng: random.Random,
    board_weights: Mapping[int, float] | None = None,
) -> int:
    choices = list(sizes)
    if not choices:
        raise ValueError("cannot sample from empty board-size set")
    if board_weights is None:
        return rng.choice(choices)
    weights = [board_weights.get(size, 1.0) for size in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def _balanced_key_order(keys: Iterable[str], count: int, rng: random.Random) -> list[str]:
    choices = list(keys)
    if not choices:
        raise ValueError("cannot sample from empty ruleset set")
    order: list[str] = []
    while len(order) < count:
        cycle = choices.copy()
        rng.shuffle(cycle)
        order.extend(cycle[: count - len(order)])
    return order


def split_for_position(
    board_size: int,
    position_key: str,
    val_fraction: float,
    seed: int,
    ruleset_key: str = "",
) -> str:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0.0:
        return TRAIN_SPLIT
    key = f"{seed}:{board_size}:{ruleset_key}:{position_key}".encode("utf-8")
    value = int.from_bytes(blake2b(key, digest_size=8).digest(), byteorder="big")
    fraction = value / float(1 << 64)
    return VAL_SPLIT if fraction < val_fraction else TRAIN_SPLIT


def _line_board_position_and_ruleset(line: str, path: Path, line_number: int) -> tuple[int, str, str]:
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
    return board_size, position_key, ruleset_key_from_raw(_record_ruleset(raw))


def scan_jsonl_stream_metadata(
    paths: Iterable[Path],
    val_fraction: float,
    seed: int,
    boards: Iterable[int] | None = None,
    use_cache: bool = True,
) -> JsonlStreamMetadata:
    paths = list(paths)
    board_filter = set(boards) if boards is not None else None
    cache_path: Path | None = None
    cache_key: str | None = None
    if use_cache and len(paths) == 1:
        path = paths[0]
        if not path.exists():
            raise FileNotFoundError(f"missing data file: {path}")
        stat = path.stat()
        boards_key = ",".join(str(size) for size in sorted(board_filter)) if board_filter else "all"
        cache_key = f"{stat.st_size}:{stat.st_mtime_ns}:{val_fraction}:{seed}:{boards_key}"
        cache_path = path.with_name(path.name + ".scancache.json")
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = {}
            entry = cached.get(cache_key)
            if entry is not None:
                return JsonlStreamMetadata(
                    record_count=int(entry["record_count"]),
                    train_count=int(entry["train_count"]),
                    val_count=int(entry["val_count"]),
                    board_counts={int(size): int(count) for size, count in entry["board_counts"].items()},
                    ruleset_counts={
                        str(ruleset): int(count)
                        for ruleset, count in entry.get("ruleset_counts", {"": entry["record_count"]}).items()
                    },
                    board_ruleset_counts={
                        int(size): {str(ruleset): int(count) for ruleset, count in counts.items()}
                        for size, counts in entry.get("board_ruleset_counts", {}).items()
                    },
                )
    board_counts: dict[int, int] = {}
    ruleset_counts: dict[str, int] = {}
    board_ruleset_counts: dict[int, dict[str, int]] = {}
    record_count = 0
    train_count = 0
    val_count = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing data file: {path}")
        with open_jsonl_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                board_size, position_key, ruleset_key = _line_board_position_and_ruleset(
                    stripped,
                    path,
                    line_number,
                )
                if board_filter is not None and board_size not in board_filter:
                    continue
                split = split_for_position(
                    board_size,
                    position_key,
                    val_fraction,
                    seed,
                    ruleset_key,
                )
                record_count += 1
                board_counts[board_size] = board_counts.get(board_size, 0) + 1
                ruleset_counts[ruleset_key] = ruleset_counts.get(ruleset_key, 0) + 1
                board_rule_counts = board_ruleset_counts.setdefault(board_size, {})
                board_rule_counts[ruleset_key] = board_rule_counts.get(ruleset_key, 0) + 1
                if split == VAL_SPLIT:
                    val_count += 1
                else:
                    train_count += 1
    if record_count == 0:
        raise ValueError("no streaming records found")
    metadata = JsonlStreamMetadata(
        record_count=record_count,
        train_count=train_count,
        val_count=val_count,
        board_counts=board_counts,
        ruleset_counts=ruleset_counts,
        board_ruleset_counts=board_ruleset_counts,
    )
    if cache_path is not None and cache_key is not None:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        except (OSError, ValueError):
            cached = {}
        cached[cache_key] = {
            "record_count": metadata.record_count,
            "train_count": metadata.train_count,
            "val_count": metadata.val_count,
            "board_counts": {str(size): count for size, count in metadata.board_counts.items()},
            "ruleset_counts": metadata.ruleset_counts,
            "board_ruleset_counts": {
                str(size): counts for size, counts in metadata.board_ruleset_counts.items()
            },
        }
        try:
            tmp_path = cache_path.with_name(cache_path.name + ".tmp")
            tmp_path.write_text(json.dumps(cached), encoding="utf-8")
            tmp_path.replace(cache_path)
        except OSError:
            pass
    return metadata


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
        self.supports_offset_index = not any(is_zstd_jsonl_path(path) for path in self.paths)
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
        self._pending_entry: BufferedJsonlRecord | None = None
        self._reader_at_eof = False
        self._offset_index: dict[str, dict[int, dict[str, list[JsonlRecordOffset]]]] | None = None
        self._offset_bags: dict[tuple[str, int, str], list[JsonlRecordOffset]] = {}
        self._offset_handles: dict[int, Any] = {}

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        for handle in self._offset_handles.values():
            handle.close()
        self._offset_handles.clear()

    def __enter__(self) -> StreamingJsonlBuffer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def prime(self, minimum_records: int = 1) -> None:
        with self.lock:
            minimum_records = max(1, minimum_records)
            while len(self.entries) < minimum_records:
                entry = self._read_next_accepted()
                if entry is None:
                    break
                if self.entries and self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
                    self._pending_entry = entry
                    break
                self._add_entry(entry, allow_eviction=False)
            while True:
                entry = self._read_next_accepted()
                if entry is None:
                    break
                if self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
                    self._pending_entry = entry
                    break
                self._add_entry(entry, allow_eviction=False)
            if not self.entries:
                raise ValueError("streaming buffer could not load any records")

    def sample_ruleset_aware_batch(
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
            if self.supports_offset_index:
                self._ensure_offset_index()
                assert self._offset_index is not None
                indexed_sizes = sorted(self._offset_index.get(split, {}))
                if indexed_sizes:
                    board_size = _choose_board_size(indexed_sizes, rng, board_weights)
                    return self._sample_ruleset_batch_from_index(
                        split,
                        board_size,
                        batch_size,
                        rng,
                        advance,
                    )
            self._ensure_split_available(split)
            sizes = sorted({entry.board_size for entry in self.entries if entry.split == split})
            if not sizes:
                raise ValueError(f"streaming buffer has no records for split {split}")
            board_size = _choose_board_size(sizes, rng, board_weights)
            self._ensure_rulesets_available(split, board_size)
            if advance:
                records = self._pop_ruleset_batch_without_replacement(split, board_size, batch_size, rng)
                self._refill_split(split, len(records))
            else:
                records = self._sample_ruleset_batch_with_replacement(split, board_size, batch_size, rng)
        return records

    def advance(self, accepted_records: int) -> None:
        with self.lock:
            for _ in range(max(0, accepted_records)):
                entry = self._read_next_accepted()
                if entry is None:
                    self._reset_reader()
                    entry = self._read_next_accepted()
                if entry is None:
                    raise ValueError("streaming buffer could not advance")
                self._add_entry(entry)

    def build_ruleset_index(self) -> None:
        with self.lock:
            if not self.supports_offset_index:
                return
            self._ensure_offset_index()

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "buffer_bytes": self.buffer_bytes,
                "buffer_records": len(self.entries),
                "max_buffer_bytes": self.max_buffer_bytes,
                "buffer_rulesets": len({entry.ruleset_key for entry in self.entries}),
                "offset_index_records": self._offset_index_record_count(),
            }

    def _offset_index_record_count(self) -> int:
        if self._offset_index is None:
            return 0
        return sum(
            len(offsets)
            for split_groups in self._offset_index.values()
            for board_groups in split_groups.values()
            for offsets in board_groups.values()
        )

    def _ensure_offset_index(self) -> None:
        if self._offset_index is not None:
            return
        if not self.supports_offset_index:
            raise ValueError("offset indexing is not available for compressed JSONL streams")
        index: dict[str, dict[int, dict[str, list[JsonlRecordOffset]]]] = {
            TRAIN_SPLIT: {},
            VAL_SPLIT: {},
        }
        for path_index, path in enumerate(self.paths):
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    byte_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    stripped = line.strip()
                    if not stripped:
                        continue
                    board_size, position_key, ruleset_key = _line_board_position_and_ruleset(
                        stripped.decode("utf-8"),
                        path,
                        line_number,
                    )
                    if board_size not in self.boards:
                        continue
                    split = split_for_position(
                        board_size,
                        position_key,
                        self.val_fraction,
                        self.seed,
                        ruleset_key,
                    )
                    index.setdefault(split, {}).setdefault(board_size, {}).setdefault(ruleset_key, []).append(
                        JsonlRecordOffset(
                            path_index=path_index,
                            byte_offset=byte_offset,
                            line_number=line_number,
                        )
                    )
        self._offset_index = {
            split: {
                board_size: {
                    ruleset_key: offsets
                    for ruleset_key, offsets in ruleset_groups.items()
                    if offsets
                }
                for board_size, ruleset_groups in board_groups.items()
                if ruleset_groups
            }
            for split, board_groups in index.items()
        }
        if not any(self._offset_index.values()):
            raise ValueError("streaming ruleset index could not load any records")

    def _sample_ruleset_batch_from_index(
        self,
        split: str,
        board_size: int,
        batch_size: int,
        rng: random.Random,
        advance: bool,
    ) -> list[TrainingRecord]:
        assert self._offset_index is not None
        ruleset_groups = self._offset_index[split][board_size]
        records: list[TrainingRecord] = []
        for ruleset_key in _balanced_key_order(sorted(ruleset_groups), batch_size, rng):
            records.append(self._read_indexed_record(split, board_size, ruleset_key, rng, advance))
        return records

    def _read_indexed_record(
        self,
        split: str,
        board_size: int,
        ruleset_key: str,
        rng: random.Random,
        advance: bool,
    ) -> TrainingRecord:
        assert self._offset_index is not None
        offsets = self._offset_index[split][board_size][ruleset_key]
        if advance:
            bag_key = (split, board_size, ruleset_key)
            bag = self._offset_bags.get(bag_key)
            if not bag:
                bag = offsets.copy()
                rng.shuffle(bag)
                self._offset_bags[bag_key] = bag
            offset = bag.pop()
        else:
            offset = rng.choice(offsets)
        return self._record_at_offset(offset)

    def _record_at_offset(self, offset: JsonlRecordOffset) -> TrainingRecord:
        handle = self._offset_handles.get(offset.path_index)
        if handle is None:
            handle = self.paths[offset.path_index].open("rb")
            self._offset_handles[offset.path_index] = handle
        handle.seek(offset.byte_offset)
        line = handle.readline()
        if not line:
            raise ValueError(
                f"missing record at {self.paths[offset.path_index]}:{offset.line_number}"
            )
        return record_from_json(
            json.loads(line),
            self.paths[offset.path_index],
            offset.line_number,
        )

    def _ensure_split_available(self, split: str) -> None:
        while not any(entry.split == split for entry in self.entries):
            if self._reader_at_eof:
                self._reset_reader()
            entry = self._read_next_accepted()
            if entry is None:
                self._reset_reader()
                continue
            self._add_entry(entry)

    def _add_entry(
        self,
        entry: BufferedJsonlRecord,
        *,
        allow_eviction: bool = True,
        protected_split: str | None = None,
    ) -> None:
        if entry.byte_size > self.max_buffer_bytes:
            raise ValueError(
                "streaming buffer is too small to hold a single record; "
                "increase --stream-buffer-mb"
            )
        while self.entries and self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
            if not allow_eviction:
                raise ValueError("streaming buffer is too small for the requested prime size")
            candidates = [
                index
                for index, buffered in enumerate(self.entries)
                if protected_split is None or buffered.split != protected_split
            ]
            if not candidates:
                raise ValueError(
                    "streaming buffer is too small to preserve without-replacement "
                    f"sampling for split {protected_split}; increase --stream-buffer-mb"
                )
            index = self._evict_rng.choice(candidates)
            removed = self.entries.pop(index)
            self.buffer_bytes -= removed.byte_size
        self.entries.append(entry)
        self.buffer_bytes += entry.byte_size

    def _ruleset_counts(self, split: str, board_size: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.split == split and entry.board_size == board_size:
                counts[entry.ruleset_key] = counts.get(entry.ruleset_key, 0) + 1
        return counts

    def _add_entry_preserving_rulesets(
        self,
        entry: BufferedJsonlRecord,
        split: str,
        board_size: int,
        protected_rulesets: set[str],
    ) -> bool:
        if entry.byte_size > self.max_buffer_bytes:
            raise ValueError(
                "streaming buffer is too small to hold a single record; "
                "increase --stream-buffer-mb"
            )
        while self.entries and self.buffer_bytes + entry.byte_size > self.max_buffer_bytes:
            counts = self._ruleset_counts(split, board_size)
            candidates = []
            for index, buffered in enumerate(self.entries):
                is_protected = (
                    buffered.split == split
                    and buffered.board_size == board_size
                    and buffered.ruleset_key in protected_rulesets
                    and counts.get(buffered.ruleset_key, 0) <= 1
                )
                if not is_protected:
                    candidates.append(index)
            if not candidates:
                return False
            index = self._evict_rng.choice(candidates)
            removed = self.entries.pop(index)
            self.buffer_bytes -= removed.byte_size
        self.entries.append(entry)
        self.buffer_bytes += entry.byte_size
        return True

    def _ensure_rulesets_available(self, split: str, board_size: int) -> None:
        target_rulesets = set(self.metadata.rulesets_for_board(board_size))
        if len(target_rulesets) <= 1:
            return
        while target_rulesets - set(self._ruleset_counts(split, board_size)):
            if self._reader_at_eof:
                break
            entry = self._read_next_accepted(allowed_splits={split})
            if entry is None:
                break
            if not self._add_entry_preserving_rulesets(entry, split, board_size, target_rulesets):
                break

    def _sample_ruleset_batch_with_replacement(
        self,
        split: str,
        board_size: int,
        batch_size: int,
        rng: random.Random,
    ) -> list[TrainingRecord]:
        pools: dict[str, list[BufferedJsonlRecord]] = {}
        for entry in self.entries:
            if entry.split == split and entry.board_size == board_size:
                pools.setdefault(entry.ruleset_key, []).append(entry)
        if not pools:
            raise ValueError(f"streaming buffer has no {split} records for board size {board_size}")
        records: list[TrainingRecord] = []
        for ruleset_key in _balanced_key_order(sorted(pools), batch_size, rng):
            records.append(rng.choice(pools[ruleset_key]).record)
        return records

    def _pop_ruleset_batch_without_replacement(
        self,
        split: str,
        board_size: int,
        batch_size: int,
        rng: random.Random,
    ) -> list[TrainingRecord]:
        records: list[TrainingRecord] = []
        while len(records) < batch_size:
            self._ensure_rulesets_available(split, board_size)
            if not self._ruleset_counts(split, board_size):
                self._fill_until_pool(split, board_size, 1)
            rulesets = sorted(self._ruleset_counts(split, board_size))
            if not rulesets:
                if self._reader_at_eof:
                    self._reset_reader()
                    continue
                raise ValueError(f"streaming buffer has no {split} records for board size {board_size}")
            before = len(records)
            for ruleset_key in _balanced_key_order(rulesets, batch_size - len(records), rng):
                pool_indices = [
                    index
                    for index, entry in enumerate(self.entries)
                    if (
                        entry.split == split
                        and entry.board_size == board_size
                        and entry.ruleset_key == ruleset_key
                    )
                ]
                if not pool_indices:
                    continue
                index = rng.choice(pool_indices)
                removed = self.entries.pop(index)
                self.buffer_bytes -= removed.byte_size
                records.append(removed.record)
                if len(records) >= batch_size:
                    break
            if len(records) == before:
                if self._reader_at_eof:
                    self._reset_reader()
                    continue
                raise ValueError(f"streaming buffer could not sample {split} records for board size {board_size}")
        return records

    def _fill_until_pool(self, split: str, board_size: int, target_count: int) -> None:
        while (
            sum(1 for entry in self.entries if entry.split == split and entry.board_size == board_size)
            < target_count
            and not self._reader_at_eof
        ):
            entry = self._read_next_accepted(allowed_splits={split})
            if entry is None:
                break
            self._add_entry(entry, protected_split=split)

    def _refill_split(self, split: str, target_count: int) -> None:
        added = 0
        while added < target_count and not self._reader_at_eof:
            entry = self._read_next_accepted(allowed_splits={split})
            if entry is None:
                break
            self._add_entry(entry, protected_split=split)
            added += 1

    def _read_next_accepted(self, allowed_splits: set[str] | None = None) -> BufferedJsonlRecord | None:
        if self._pending_entry is not None:
            entry = self._pending_entry
            self._pending_entry = None
            if allowed_splits is None or entry.split in allowed_splits:
                return entry
        while True:
            path, line_number, line = self._read_next_line()
            if line is None:
                return None
            stripped = line.strip()
            if not stripped:
                continue
            record = record_from_json(json.loads(stripped), path, line_number)
            if record.board_size not in self.boards:
                continue
            split = split_for_position(
                record.board_size,
                record.position_key,
                self.val_fraction,
                self.seed,
                record.ruleset_key,
            )
            if allowed_splits is not None and split not in allowed_splits:
                continue
            return BufferedJsonlRecord(
                record=record,
                byte_size=record.array_nbytes() + 256,
                board_size=record.board_size,
                split=split,
                ruleset_key=record.ruleset_key,
            )

    def _reset_reader(self) -> None:
        self.close()
        self._path_index = 0
        self._line_number = 0
        self._pending_entry = None
        self._reader_at_eof = False

    def _read_next_line(self) -> tuple[Path, int, str | None]:
        if self._reader_at_eof:
            return self.paths[self._path_index], self._line_number, None
        while True:
            if self._handle is None:
                path = self.paths[self._path_index]
                self._handle = open_jsonl_text(path)
                self._line_number = 0
            line = self._handle.readline()
            if line:
                self._line_number += 1
                return self.paths[self._path_index], self._line_number, line
            self._handle.close()
            self._handle = None
            if self._path_index + 1 >= len(self.paths):
                self._reader_at_eof = True
                return self.paths[self._path_index], self._line_number, None
            self._path_index += 1


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
    size = _choose_board_size(sizes, rng, board_weights)
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

    for key, width in (("wdl", len(WDL_LABELS)), ("ownership", area), ("policy", action_count), ("budget", action_count)):
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


def batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, torch.Tensor]:
    if device.type == "cpu":
        return batch
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def collate(records: list[TrainingRecord], device: torch.device) -> dict[str, torch.Tensor]:
    """Synchronous collate for eval and one-off use: unpinned memory, blocking copies.

    Pinned + non_blocking staging is reserved for the training path, where
    PinnedBatchKeeper fences the staging buffers' lifetime with CUDA events.
    Freeing a pinned staging tensor while its async H2D copy is still queued
    lets the caching host allocator recycle the block before the copy finishes,
    which silently corrupts the copied batch.
    """
    return batch_to_device(collate_cpu(records, pin_memory=False), device, non_blocking=False)


class StreamingRulesetAwareBatchDataset(IterableDataset[list[TrainingRecord]]):
    """Infinite iterable dataset backed by StreamingJsonlBuffer."""

    def __init__(
        self,
        stream: StreamingJsonlBuffer,
        split: str,
        batch_size: int,
        rng: random.Random,
        board_weights: Mapping[int, float] | None = None,
        *,
        augment_d4: bool = False,
        advance: bool = True,
    ) -> None:
        self.stream = stream
        self.split = split
        self.batch_size = batch_size
        self.rng = rng
        self.board_weights = board_weights
        self.augment_d4 = augment_d4
        self.advance = advance

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError("StreamingRulesetAwareBatchDataset requires DataLoader(num_workers=0)")
        while True:
            records = self.stream.sample_ruleset_aware_batch(
                self.split,
                self.batch_size,
                self.rng,
                self.board_weights,
                advance=self.advance,
            )
            if self.augment_d4:
                records = [augment_record_d4(record, self.rng.randrange(8)) for record in records]
            yield records


def make_batch_dataloader(
    dataset: IterableDataset[list[TrainingRecord]],
    *,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=None,
        collate_fn=lambda records: collate_cpu(records, pin_memory=False),
        num_workers=0,
        pin_memory=pin_memory,
    )


class PinnedBatchKeeper:
    """Keeps pinned CPU batches alive until their async H2D copies have executed.

    Call fence(cpu_batch) right after batch_to_device: it records a CUDA event
    and holds the batch until the event reports completion.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._pending: list[tuple[torch.cuda.Event, dict[str, torch.Tensor]]] = []

    def fence(self, cpu_batch: dict[str, torch.Tensor]) -> None:
        if not self.enabled:
            return
        self._pending = [(event, batch) for event, batch in self._pending if not event.query()]
        event = torch.cuda.Event()
        event.record()
        self._pending.append((event, cpu_batch))

    def release_all(self) -> None:
        self._pending.clear()
