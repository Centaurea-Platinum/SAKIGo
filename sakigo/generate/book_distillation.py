"""Book-only 9x9 distillation dataset generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from sakigo.data.records import open_jsonl_writer
from sakigo.generate.book import (
    assign_validated_histories,
    freeze_uniform_sample,
    index_book_archive,
    iter_frozen_sample,
)
from sakigo.generate.record_builder import build_book_training_record
from sakigo.rulesets import ruleset_from_overrides

ROOT = Path(__file__).resolve().parents[2]
BOOK_URL = "https://katagobooks.org/downloads/book9x9tt-20260226.tar.gz"
DEFAULT_BOOK = ROOT / "Distillation/downloads/book9x9tt-20260226.tar.gz"
DEFAULT_TRAIN_SAMPLES = 1 << 20
DEFAULT_VALIDATION_SAMPLES = 1 << 12
DEFAULT_SAMPLES_PER_SHARD = 1 << 16


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a book-only 9x9 TT7 distillation dataset."
    )
    parser.add_argument("stage", choices=("artifacts", "index", "sample", "all"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--book-archive", type=Path, default=DEFAULT_BOOK)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument(
        "--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES
    )
    parser.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--zstd-level", type=int, default=3)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_samples <= 0 or args.validation_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if not 1 <= args.zstd_level <= 22:
        raise ValueError("--zstd-level must be in [1, 22]")


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(destination)


def stage_artifacts(args: argparse.Namespace) -> None:
    _download(BOOK_URL, args.book_archive)
    write_atomic_json(
        args.run_dir / "artifact_manifest.json",
        {
            "book": {
                "path": str(args.book_archive.resolve()),
                "url": BOOK_URL,
                "sha256": file_sha256(args.book_archive),
            },
            "teacher_inference": "not_used",
            "ownership": "not_generated",
        },
    )


def stage_index(args: argparse.Namespace) -> None:
    database = args.run_dir / "book_index.sqlite"
    parse_report = index_book_archive(args.book_archive, database)
    validation_report = assign_validated_histories(database)
    sample_report = freeze_uniform_sample(
        database,
        train_count=args.train_samples,
        validation_count=args.validation_samples,
        seed=args.selection_seed,
    )
    write_atomic_json(
        args.run_dir / "book_index_report.json",
        {
            **parse_report,
            **validation_report,
            "sample": sample_report,
            "database": str(database.resolve()),
        },
    )


def _shard_path(output: Path, split: str, shard_index: int) -> Path:
    return output / split / f"samples_{shard_index:06d}.jsonl.zst"


def _write_split(
    args: argparse.Namespace,
    *,
    split: str,
    count: int,
    progress: tqdm,
) -> list[Path]:
    database = args.run_dir / "book_index.sqlite"
    output = args.run_dir / "dataset"
    ruleset = ruleset_from_overrides(ruleset="tromp-taylor", komi=7.0)
    paths: list[Path] = []
    for shard_index, offset in enumerate(range(0, count, args.samples_per_shard)):
        shard_count = min(args.samples_per_shard, count - offset)
        path = _shard_path(output, split, shard_index)
        paths.append(path)
        if path.exists():
            progress.update(shard_count)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp.jsonl.zst")
        with open_jsonl_writer(temporary, compression_level=args.zstd_level) as handle:
            written = 0
            for task in iter_frozen_sample(
                database, split=split, offset=offset, limit=shard_count
            ):
                record = build_book_training_record(task, ruleset=ruleset)
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
                progress.update(1)
            if written != shard_count:
                raise RuntimeError(
                    f"{split} shard {shard_index} expected {shard_count} records, wrote {written}"
                )
        temporary.replace(path)
        write_atomic_json(
            path.with_suffix(path.suffix + ".status.json"),
            {"state": "complete", "split": split, "records": shard_count},
        )
    return paths


def stage_sample(args: argparse.Namespace) -> None:
    total = args.train_samples + args.validation_samples
    with tqdm(total=total, unit="record", dynamic_ncols=True) as progress:
        train_paths = _write_split(
            args, split="train", count=args.train_samples, progress=progress
        )
        validation_paths = _write_split(
            args,
            split="validation",
            count=args.validation_samples,
            progress=progress,
        )
    write_atomic_json(
        args.run_dir / "dataset_manifest.json",
        {
            "state": "complete",
            "train_records": args.train_samples,
            "validation_records": args.validation_samples,
            "ownership": "absent",
            "teacher_inference": "none",
            "selection": "uniform stable hash over validated canonical book nodes",
            "selection_seed": args.selection_seed,
            "train_shards": [str(path.resolve()) for path in train_paths],
            "validation_shards": [str(path.resolve()) for path in validation_paths],
        },
    )


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {
        "artifacts": stage_artifacts,
        "index": stage_index,
        "sample": stage_sample,
    }
    if args.stage == "all":
        for stage in (stage_artifacts, stage_index, stage_sample):
            stage(args)
    else:
        stages[args.stage](args)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
