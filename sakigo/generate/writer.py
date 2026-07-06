"""Sharded zstd JSONL output writer (ported from Training/generate_katago_phase1.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sakigo.data.records import is_zstd_jsonl_path, open_jsonl_writer


def _strip_jsonl_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.zstd", ".jsonl.zst", ".jsonl"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem if path.suffix else "samples"


class GenerationOutputWriter:
    def __init__(self, output: Path, *, samples_per_file: int, zstd_level: int) -> None:
        if samples_per_file < 0:
            raise ValueError("--samples-per-file must be non-negative")
        if zstd_level < 1 or zstd_level > 22:
            raise ValueError("--zstd-level must be in [1, 22]")
        self.output = output
        self.samples_per_file = samples_per_file
        self.zstd_level = zstd_level
        self.paths: list[Path] = []
        self._handle: Any = None
        self._shard_index = 0
        self._samples_in_file = 0
        self._single_file = samples_per_file == 0

        if self._single_file:
            self.directory = output.parent
            self.prefix = output.stem
        elif output.suffix:
            self.directory = output.parent
            self.prefix = _strip_jsonl_suffix(output)
        else:
            self.directory = output
            self.prefix = "samples"
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def data_format(self) -> str:
        if self._single_file:
            return (
                "single_jsonl_zstd"
                if is_zstd_jsonl_path(self.output)
                else "legacy_single_jsonl_deprecated"
            )
        return "jsonl_zstd_shards"

    def __enter__(self) -> "GenerationOutputWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write_record(self, record: dict[str, Any]) -> None:
        if self._handle is None or (
            not self._single_file and self._samples_in_file >= self.samples_per_file
        ):
            self._rotate()
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._samples_in_file += 1

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _rotate(self) -> None:
        self.close()
        if self._single_file:
            path = self.output
        else:
            path = self.directory / f"{self.prefix}_{self._shard_index:06d}.jsonl.zst"
            self._shard_index += 1
        self.paths.append(path)
        self._samples_in_file = 0
        self._handle = open_jsonl_writer(path, compression_level=self.zstd_level)
