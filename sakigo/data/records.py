"""Record schema, JSONL IO, validation, split hashing, D4 augmentation, collate.

Ported behavior-identically from Training/common.py and Training/data.py —
these implement the versioned contracts in sakigo/CONTRACTS.md (record schema,
canonical split hashing, batch layout).
"""

from __future__ import annotations

import dataclasses
import glob
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import zstandard as zstd

from sakigo.constants import (
    BOARD_PLANE_COUNT,
    DISTILLATION_SCHEMA_VERSION,
    RULE_FEATURE_COUNT,
    SCHEMA_VERSION,
    WDL_LABELS,
)
from sakigo.rulesets import (
    ruleset_from_metadata,
    ruleset_key_from_raw,
    validate_rule_features,
)

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
ZSTD_JSONL_SUFFIXES = (".jsonl.zst", ".jsonl.zstd")
LEGACY_JSONL_SUFFIX = ".jsonl"


@dataclass(frozen=True, eq=False)
class TrainingRecord:
    """One training position. Array fields are numpy-backed for cheap collation."""

    schema_version: int
    board_size: int
    ply: int
    position_key: str
    board_planes: np.ndarray
    rule_features: np.ndarray
    ruleset_key: str = ""
    ruleset: dict[str, object] | None = None
    wdl: np.ndarray | None = None
    score: float | None = None
    ownership: np.ndarray | None = None
    policy: np.ndarray | None = None
    budget: np.ndarray | None = None
    legal_mask: np.ndarray | None = None


def is_zstd_jsonl_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ZSTD_JSONL_SUFFIXES)


def is_legacy_jsonl_path(path: Path) -> bool:
    return path.name.lower().endswith(LEGACY_JSONL_SUFFIX) and not is_zstd_jsonl_path(path)


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
        raise ValueError("no data files matched")
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

    def __enter__(self) -> "_ZstdTextReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def open_jsonl_text(path: Path) -> TextIO | _ZstdTextReader:
    if is_zstd_jsonl_path(path):
        return _ZstdTextReader(path)
    return path.open("r", encoding="utf-8")


class _ZstdTextWriter:
    def __init__(
        self,
        path: Path,
        compression_level: int = 3,
        threads: int = -1,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = path.open("wb")
        compressor = zstd.ZstdCompressor(level=compression_level, threads=threads)
        self._writer = compressor.stream_writer(self._raw, closefd=False)
        self._text = io.TextIOWrapper(self._writer, encoding="utf-8")

    def write(self, text: str) -> int:
        return self._text.write(text)

    def flush(self) -> None:
        self._text.flush()

    def close(self) -> None:
        self._text.close()
        self._raw.close()

    def __enter__(self) -> "_ZstdTextWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def open_jsonl_writer(
    path: Path,
    compression_level: int = 3,
    threads: int = -1,
) -> TextIO | _ZstdTextWriter:
    if is_zstd_jsonl_path(path):
        return _ZstdTextWriter(
            path,
            compression_level=compression_level,
            threads=threads,
        )
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
    if any(type(value) is not bool for value in raw):
        raise ValueError("legal_mask must contain only JSON booleans")
    if not raw[-1]:
        raise ValueError("legal_mask pass entry must be true")
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
        board_size = raw["board_size"]
        ply = raw["ply"]
        position_key = raw["position_key"]
        schema_version = raw["schema_version"]
    except KeyError as exc:
        raise ValueError(f"{label} is missing field {exc}") from exc
    for field, value in (
        ("board_size", board_size),
        ("ply", ply),
        ("schema_version", schema_version),
    ):
        if type(value) is not int:
            raise ValueError(f"{label} {field} must be an integer")
    if not isinstance(position_key, str):
        raise ValueError(f"{label} position_key must be a string")
    if schema_version not in {SCHEMA_VERSION, DISTILLATION_SCHEMA_VERSION}:
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
    if ruleset is None:
        raise ValueError(f"{label} must provide a schema-v1 ruleset object")
    parsed_ruleset = ruleset_from_metadata(ruleset)
    validate_rule_features(rule_features, parsed_ruleset)
    ownership = _optional_vector(targets, "ownership", area)
    if ownership is not None and ((ownership < -1.0) | (ownership > 1.0)).any():
        raise ValueError("ownership values must be in [-1, 1]")

    legal_source = targets if "legal_mask" in targets else raw
    policy = _optional_distribution(targets, "policy", action_count)
    budget = _optional_distribution(targets, "budget", action_count)
    legal_mask = _optional_legal_mask(legal_source, action_count)
    if schema_version == SCHEMA_VERSION and policy is not None:
        active = np.flatnonzero(policy > 1e-6)
        if len(active) != 1 or not np.isclose(float(policy[active[0]]), 1.0, atol=1e-6):
            raise ValueError("schema-v1 policy must be a one-hot top-1 distribution")
    if legal_mask is not None:
        illegal = ~legal_mask
        if policy is not None and float(policy[illegal].sum()) > 1e-6:
            raise ValueError("policy must assign zero mass to illegal actions")
        if budget is not None and float(budget[illegal].sum()) > 1e-6:
            raise ValueError("budget must assign zero mass to illegal actions")

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
        policy=policy,
        budget=budget,
        legal_mask=legal_mask,
    )
    if all(getattr(record, key) is None for key in ("wdl", "score", "ownership", "policy", "budget")):
        raise ValueError(f"{label} must provide at least one target")
    return record


def iter_records(paths: Iterable[Path]) -> Iterable[TrainingRecord]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing data file: {path}")
        with open_jsonl_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                yield record_from_json(json.loads(stripped), path, line_number)


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


def canonical_position_key(record: TrainingRecord) -> str:
    """Stable identity of the exact model-visible input, independent of move order."""
    digest = blake2b(digest_size=20)
    digest.update(np.asarray(record.board_planes, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(record.rule_features, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def split_for_record(record: TrainingRecord, val_fraction: float, seed: int) -> str:
    return split_for_position(
        record.board_size,
        canonical_position_key(record),
        val_fraction,
        seed,
        record.ruleset_key,
    )


def _d4_transform_planes(array: np.ndarray, transform: int) -> np.ndarray:
    """Apply one of the 8 dihedral transforms to the last two (spatial) axes."""
    out = np.rot90(array, k=transform % 4, axes=(-2, -1))
    if transform >= 4:
        out = out[..., ::-1]
    return np.ascontiguousarray(out)


def augment_record_d4(record: TrainingRecord, transform: int) -> TrainingRecord:
    """Apply a D4 board symmetry (0..7) to all spatial fields; the pass entry stays last."""
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


def batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, torch.Tensor]:
    if device.type == "cpu":
        return batch
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
