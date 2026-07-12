"""One-time conversion: schema-v1/v2 JSONL(.zst) -> mmap-able per-(split, board size)
tensor shards. Decode+validate once at prepare time; training then reads
numpy memmaps with zero JSON cost.

Layout under out_dir/:
  manifest.json
  generation_<id>/<split>_<N>/board_planes.npy, rule_features.npy, ply.npy, ruleset_code.npy,
              wdl.npy + wdl_mask.npy, score.npy + score_mask.npy,
              ownership.npy + ownership_mask.npy, policy.npy + policy_mask.npy,
              budget.npy + budget_mask.npy, legal_mask.npy + legal_available.npy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from sakigo.constants import SCHEMA_VERSION, WDL_LABELS
from sakigo.data.records import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    expand_data_paths,
    iter_records,
    split_for_record,
)

MANIFEST_NAME = "manifest.json"
PREPARE_FORMAT_VERSION = 2
PREPARED_ARRAY_NAMES = (
    "board_planes",
    "rule_features",
    "ply",
    "ruleset_code",
    "wdl",
    "wdl_mask",
    "score",
    "score_mask",
    "ownership",
    "ownership_mask",
    "policy",
    "policy_mask",
    "budget",
    "budget_mask",
    "legal_mask",
    "legal_available",
)


@dataclass(frozen=True)
class GroupInfo:
    split: str
    board_size: int
    count: int
    directory: str


def _source_fingerprint(paths: list[Path]) -> list[dict[str, object]]:
    return [
        {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


def _group_dir(
    out_dir: Path,
    split: str,
    board_size: int,
    generation: str | None = None,
) -> Path:
    root = out_dir / generation if generation is not None else out_dir
    return root / f"{split}_{board_size:02d}"


def load_manifest(out_dir: Path) -> dict[str, object]:
    return json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def manifest_is_current(
    out_dir: Path,
    data_paths: list[Path],
    seed: int,
    val_fraction: float,
) -> bool:
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False
    try:
        manifest = load_manifest(out_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    metadata_matches = (
        manifest.get("prepare_format_version") == PREPARE_FORMAT_VERSION
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("seed") == seed
        and manifest.get("val_fraction") == val_fraction
        and manifest.get("sources") == _source_fingerprint(data_paths)
    )
    if not metadata_matches:
        return False
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        return False
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("directory"), str):
            return False
        directory = out_dir / group["directory"]
        if not all((directory / f"{name}.npy").is_file() for name in PREPARED_ARRAY_NAMES):
            return False
    return True


def prepare_tensor_shards(
    data: list[Path | str],
    out_dir: Path,
    *,
    seed: int,
    val_fraction: float,
    force: bool = False,
) -> dict[str, object]:
    """Convert JSONL sources into tensor shards; skips work when up to date."""
    data_paths = expand_data_paths(Path(item) for item in data)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force and manifest_is_current(out_dir, data_paths, seed, val_fraction):
        return load_manifest(out_dir)
    generation = f"generation_{uuid4().hex}"

    # Pass 1: per-(split, size) counts and the ruleset key table.
    counts: dict[tuple[str, int], int] = {}
    ruleset_keys: list[str] = []
    ruleset_index: dict[str, int] = {}
    for record in iter_records(data_paths):
        split = split_for_record(record, val_fraction, seed)
        counts[(split, record.board_size)] = counts.get((split, record.board_size), 0) + 1
        if record.ruleset_key not in ruleset_index:
            ruleset_index[record.ruleset_key] = len(ruleset_keys)
            ruleset_keys.append(record.ruleset_key)
    if not counts:
        raise ValueError("no records found in the data sources")

    # Pass 2: fill memmaps.
    writers: dict[tuple[str, int], dict[str, np.memmap]] = {}
    cursors: dict[tuple[str, int], int] = {}
    for (split, board_size), count in counts.items():
        directory = _group_dir(out_dir, split, board_size, generation)
        directory.mkdir(parents=True, exist_ok=True)
        area = board_size * board_size
        action = area + 1

        def memmap(name: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
            return np.lib.format.open_memmap(
                directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )

        writers[(split, board_size)] = {
            "board_planes": memmap("board_planes", "float32", (count, 6, board_size, board_size)),
            "rule_features": memmap("rule_features", "float32", (count, 10)),
            "ply": memmap("ply", "int64", (count,)),
            "ruleset_code": memmap("ruleset_code", "int32", (count,)),
            "wdl": memmap("wdl", "float32", (count, len(WDL_LABELS))),
            "wdl_mask": memmap("wdl_mask", "bool", (count,)),
            "score": memmap("score", "float32", (count,)),
            "score_mask": memmap("score_mask", "bool", (count,)),
            "ownership": memmap("ownership", "float32", (count, area)),
            "ownership_mask": memmap("ownership_mask", "bool", (count,)),
            "policy": memmap("policy", "float32", (count, action)),
            "policy_mask": memmap("policy_mask", "bool", (count,)),
            "budget": memmap("budget", "float32", (count, action)),
            "budget_mask": memmap("budget_mask", "bool", (count,)),
            "legal_mask": memmap("legal_mask", "bool", (count, action)),
            "legal_available": memmap("legal_available", "bool", (count,)),
        }
        cursors[(split, board_size)] = 0

    for record in iter_records(data_paths):
        split = split_for_record(record, val_fraction, seed)
        key = (split, record.board_size)
        row = cursors[key]
        cursors[key] = row + 1
        arrays = writers[key]
        arrays["board_planes"][row] = record.board_planes
        arrays["rule_features"][row] = record.rule_features
        arrays["ply"][row] = record.ply
        arrays["ruleset_code"][row] = ruleset_index[record.ruleset_key]
        if record.wdl is not None:
            arrays["wdl"][row] = record.wdl
            arrays["wdl_mask"][row] = True
        if record.score is not None:
            arrays["score"][row] = record.score
            arrays["score_mask"][row] = True
        if record.ownership is not None:
            arrays["ownership"][row] = record.ownership
            arrays["ownership_mask"][row] = True
        if record.policy is not None:
            arrays["policy"][row] = record.policy
            arrays["policy_mask"][row] = True
        if record.budget is not None:
            arrays["budget"][row] = record.budget
            arrays["budget_mask"][row] = True
        if record.legal_mask is not None:
            arrays["legal_mask"][row] = record.legal_mask
            arrays["legal_available"][row] = True
        else:
            arrays["legal_mask"][row] = True

    for key, arrays in writers.items():
        assert cursors[key] == counts[key]
        for array in arrays.values():
            array.flush()
    del array, arrays
    writers.clear()

    manifest = {
        "prepare_format_version": PREPARE_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "val_fraction": val_fraction,
        "sources": _source_fingerprint(data_paths),
        "ruleset_keys": ruleset_keys,
        "generation": generation,
        "groups": [
            {
                "split": split,
                "board_size": board_size,
                "count": count,
                "directory": str(
                    _group_dir(out_dir, split, board_size, generation).relative_to(out_dir)
                ),
            }
            for (split, board_size), count in sorted(counts.items())
        ],
    }
    manifest_path = out_dir / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)
    return manifest


_SPLITS = (TRAIN_SPLIT, VAL_SPLIT)
