"""Concurrent mixed-small-board KataGo book distillation pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from sakigo.data.prepare import prepare_tensor_shards
from sakigo.data.records import open_jsonl_writer
from sakigo.generate.book import (
    assign_validated_histories,
    count_book_target_eligible,
    freeze_uniform_sample,
    index_book_archive,
    iter_frozen_sample,
)
from sakigo.generate.book_catalog import (
    BookSpec,
    allocate_global_random_sample,
    resolve_books,
)
from sakigo.generate.record_builder import build_book_training_record

DEFAULT_TRAIN_SAMPLES = 1 << 20
DEFAULT_VALIDATION_SAMPLES = 1 << 12
DEFAULT_SAMPLES_PER_SHARD = 1 << 16
PIPELINE_VERSION = 1
SAMPLE_ALLOCATION_VERSION = 3
T = TypeVar("T")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _run_parallel(
    specs: Iterable[BookSpec],
    workers: int,
    operation: Callable[[BookSpec], T],
) -> dict[str, T]:
    selected = tuple(specs)
    results: dict[str, T] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
        futures = {executor.submit(operation, spec): spec for spec in selected}
        for future in as_completed(futures):
            spec = futures[future]
            results[spec.book_id] = future.result()
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a randomly mixed 7x7/8x8/9x9 KataGo book dataset with "
            "concurrent per-book processing."
        )
    )
    parser.add_argument(
        "stage", choices=("artifacts", "index", "sample", "prepare", "all")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--books",
        default="all",
        help="comma-separated book IDs, or all",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument(
        "--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES
    )
    parser.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[BookSpec, ...]:
    specs = resolve_books(args.books)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.train_samples <= 0 or args.validation_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if not 1 <= args.zstd_level <= 22:
        raise ValueError("--zstd-level must be in [1, 22]")
    return specs


def _download(spec: BookSpec) -> Path:
    destination = spec.archive
    if destination.is_file():
        size = destination.stat().st_size
        if size != spec.expected_bytes:
            raise ValueError(
                f"{destination} has {size:,} bytes, expected {spec.expected_bytes:,}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    if temporary.exists():
        temporary.unlink()
    try:
        with urllib.request.urlopen(spec.url) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        size = temporary.stat().st_size
        if size != spec.expected_bytes:
            raise ValueError(
                f"downloaded {size:,} bytes for {spec.book_id}, "
                f"expected {spec.expected_bytes:,}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def stage_artifacts(args: argparse.Namespace, specs: tuple[BookSpec, ...]) -> None:
    paths = _run_parallel(specs, args.workers, _download)
    # Hash sequentially to avoid turning one disk into a random-I/O benchmark.
    books = []
    for spec in specs:
        path = paths[spec.book_id]
        books.append({**spec.metadata(), "sha256": file_sha256(path)})
    write_atomic_json(
        args.run_dir / "artifact_manifest.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "books": books,
            "teacher_inference": "not_used",
            "ownership": "not_generated",
        },
    )


def _artifact_hashes(args: argparse.Namespace, specs: tuple[BookSpec, ...]) -> dict[str, str]:
    manifest_path = args.run_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("artifact manifest is missing; run the artifacts stage first")
    manifest = _read_json(manifest_path)
    raw_books = manifest.get("books")
    if not isinstance(raw_books, list):
        raise ValueError("artifact manifest has no book list")
    hashes = {
        str(item["book_id"]): str(item["sha256"])
        for item in raw_books
        if isinstance(item, dict) and "book_id" in item and "sha256" in item
    }
    missing = [spec.book_id for spec in specs if spec.book_id not in hashes]
    if missing:
        raise ValueError(f"artifact manifest is missing selected books: {missing}")
    for spec in specs:
        if not spec.archive.is_file() or spec.archive.stat().st_size != spec.expected_bytes:
            raise ValueError(f"book archive is missing or incomplete: {spec.archive}")
    return hashes


def _book_dir(args: argparse.Namespace, spec: BookSpec) -> Path:
    return args.run_dir / "books" / spec.book_id


def _index_and_validate_book(
    args: argparse.Namespace,
    spec: BookSpec,
    archive_sha256: str,
) -> dict[str, Any]:
    directory = _book_dir(args, spec)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "book_index.sqlite"
    index_marker = directory / "archive_index_complete.json"
    marker_matches = False
    if index_marker.is_file() and database.is_file():
        marker = _read_json(index_marker)
        marker_matches = (
            marker.get("pipeline_version") == PIPELINE_VERSION
            and marker.get("archive_sha256") == archive_sha256
        )
    if marker_matches:
        parse_report = _read_json(index_marker)
    else:
        parse_report = index_book_archive(spec.archive, database)
        parse_report = {
            **parse_report,
            "pipeline_version": PIPELINE_VERSION,
            "archive_sha256": archive_sha256,
        }
        write_atomic_json(index_marker, parse_report)

    validation_marker = directory / "validation_complete.json"
    validation_matches = False
    if validation_marker.is_file():
        validation = _read_json(validation_marker)
        validation_matches = (
            validation.get("pipeline_version") == PIPELINE_VERSION
            and validation.get("archive_sha256") == archive_sha256
            and validation.get("board_size") == spec.board_size
            and validation.get("ruleset") == spec.ruleset().metadata()
        )
    if not validation_matches:
        validation = assign_validated_histories(
            database,
            board_size=spec.board_size,
            ruleset=spec.ruleset(),
        )
        validation = {
            **validation,
            "pipeline_version": PIPELINE_VERSION,
            "archive_sha256": archive_sha256,
            "board_size": spec.board_size,
            "ruleset": spec.ruleset().metadata(),
        }
        write_atomic_json(validation_marker, validation)
    eligible = count_book_target_eligible(database, board_size=spec.board_size)
    return {
        **spec.metadata(),
        "database": str(database.resolve()),
        "parse": parse_report,
        "validation": validation,
        "eligible": eligible,
    }


def stage_index(args: argparse.Namespace, specs: tuple[BookSpec, ...]) -> None:
    hashes = _artifact_hashes(args, specs)
    reports = _run_parallel(
        specs,
        args.workers,
        lambda spec: _index_and_validate_book(args, spec, hashes[spec.book_id]),
    )
    capacities = {book_id: int(report["eligible"]) for book_id, report in reports.items()}
    allocation = allocate_global_random_sample(
        train_total=args.train_samples,
        validation_total=args.validation_samples,
        capacities=capacities,
        specs=specs,
        seed=args.selection_seed,
    )

    def freeze(spec: BookSpec) -> dict[str, int]:
        counts = allocation[spec.book_id]
        return freeze_uniform_sample(
            _book_dir(args, spec) / "book_index.sqlite",
            train_count=counts["train"],
            validation_count=counts["validation"],
            seed=args.selection_seed,
            board_size=spec.board_size,
            replace_existing=True,
        )

    frozen = _run_parallel(specs, args.workers, freeze)
    for spec in specs:
        reports[spec.book_id]["sample"] = frozen[spec.book_id]
    write_atomic_json(
        args.run_dir / "book_index_report.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "sample_allocation_version": SAMPLE_ALLOCATION_VERSION,
            "selection": (
                "balanced board-size/ruleset validation cohorts; global uniform "
                "training sample without replacement from remaining nodes"
            ),
            "selection_seed": args.selection_seed,
            "train_records": args.train_samples,
            "validation_records": args.validation_samples,
            "allocation": allocation,
            "books": [reports[spec.book_id] for spec in specs],
        },
    )


def _shard_path(
    args: argparse.Namespace, spec: BookSpec, split: str, shard_index: int
) -> Path:
    return (
        args.run_dir
        / "dataset"
        / split
        / f"{spec.book_id}_samples_{shard_index:06d}.jsonl.zst"
    )


def _write_book_shards(
    args: argparse.Namespace,
    spec: BookSpec,
    counts: dict[str, int],
    sample_identity: str,
) -> dict[str, list[dict[str, Any]]]:
    database = _book_dir(args, spec) / "book_index.sqlite"
    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        count = int(counts[split])
        for shard_index, offset in enumerate(range(0, count, args.samples_per_shard)):
            shard_count = min(args.samples_per_shard, count - offset)
            path = _shard_path(args, spec, split, shard_index)
            status_path = path.with_suffix(path.suffix + ".status.json")
            if path.is_file() and status_path.is_file():
                status = _read_json(status_path)
                expected_status = {
                    "state": "complete",
                    "pipeline_version": PIPELINE_VERSION,
                    "sample_identity": sample_identity,
                    "book_id": spec.book_id,
                    "board_size": spec.board_size,
                    "split": split,
                    "records": shard_count,
                    "offset": offset,
                }
                if any(
                    type(status.get(key)) is not type(value)
                    or status.get(key) != value
                    for key, value in expected_status.items()
                ):
                    raise ValueError(f"inconsistent completed shard status: {status_path}")
                actual_bytes = path.stat().st_size
                actual_sha256 = file_sha256(path)
                if (
                    status.get("bytes") != actual_bytes
                    or status.get("sha256") != actual_sha256
                ):
                    raise ValueError(f"completed shard content is inconsistent: {path}")
                shard_bytes = actual_bytes
                shard_sha256 = actual_sha256
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.stem}.tmp.jsonl.zst")
                temporary.unlink(missing_ok=True)
                written = 0
                with open_jsonl_writer(
                    temporary,
                    compression_level=args.zstd_level,
                    threads=1,
                ) as handle:
                    for task in iter_frozen_sample(
                        database, split=split, offset=offset, limit=shard_count
                    ):
                        task["task_id"] = f"{spec.book_id}-{task['task_id']}"
                        record = build_book_training_record(
                            task,
                            ruleset=spec.ruleset(),
                            board_size=spec.board_size,
                            book_id=spec.book_id,
                        )
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        written += 1
                if written != shard_count:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{spec.book_id} {split} shard {shard_index} expected "
                        f"{shard_count} records, wrote {written}"
                    )
                temporary.replace(path)
                shard_bytes = path.stat().st_size
                shard_sha256 = file_sha256(path)
                write_atomic_json(
                    status_path,
                    {
                        "state": "complete",
                        "pipeline_version": PIPELINE_VERSION,
                        "sample_identity": sample_identity,
                        "book_id": spec.book_id,
                        "board_size": spec.board_size,
                        "split": split,
                        "records": shard_count,
                        "offset": offset,
                        "bytes": shard_bytes,
                        "sha256": shard_sha256,
                    },
                )
            output[split].append(
                {
                    "path": str(path.resolve()),
                    "records": shard_count,
                    "bytes": shard_bytes,
                    "sha256": shard_sha256,
                }
            )
    return output


def stage_sample(args: argparse.Namespace, specs: tuple[BookSpec, ...]) -> None:
    report_path = args.run_dir / "book_index_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("book index report is missing; run the index stage first")
    report = _read_json(report_path)
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("book index report uses an incompatible pipeline version")
    if report.get("sample_allocation_version") != SAMPLE_ALLOCATION_VERSION:
        raise ValueError(
            "book index report predates board-size/ruleset validation cohorts; "
            "rerun the index stage after the active index process exits"
        )
    allocation = report.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("book index report has no sample allocation")
    raw_books = report.get("books")
    if not isinstance(raw_books, list):
        raise ValueError("book index report has no book metadata")
    report_books = [
        str(book.get("book_id")) for book in raw_books if isinstance(book, dict)
    ]
    selected_books = [spec.book_id for spec in specs]
    if report_books != selected_books:
        raise ValueError(
            f"selected books {selected_books} do not match indexed books {report_books}"
        )
    selection_seed = report.get("selection_seed")
    if type(selection_seed) is not int:
        raise ValueError("book index report selection_seed must be an integer")

    def sample_identity(spec: BookSpec) -> str:
        book_report = raw_books[selected_books.index(spec.book_id)]
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "sample_allocation_version": SAMPLE_ALLOCATION_VERSION,
            "selection_seed": selection_seed,
            "allocation": allocation[spec.book_id],
            "book_id": spec.book_id,
            "board_size": spec.board_size,
            "ruleset": spec.ruleset().metadata(),
            "archive_sha256": book_report.get("parse", {}).get("archive_sha256"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    results = _run_parallel(
        specs,
        args.workers,
        lambda spec: _write_book_shards(
            args,
            spec,
            dict(allocation[spec.book_id]),
            sample_identity(spec),
        ),
    )
    train_shards = [item for spec in specs for item in results[spec.book_id]["train"]]
    validation_shards = [
        item for spec in specs for item in results[spec.book_id]["validation"]
    ]
    write_atomic_json(
        args.run_dir / "dataset_manifest.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "sample_allocation_version": SAMPLE_ALLOCATION_VERSION,
            "state": "complete",
            "train_records": sum(int(item["records"]) for item in train_shards),
            "validation_records": sum(
                int(item["records"]) for item in validation_shards
            ),
            "selection": report.get("selection"),
            "selection_seed": selection_seed,
            "allocation": allocation,
            "preprocessing": {
                "deduplication": (
                    "canonical symmetry/transposition book nodes; unique node ID "
                    "across train and validation within each book"
                ),
                "score": "round Black score lead to nearest 0.5, then normalize by area",
                "policy": "uniform mass over concrete moves tied at rounded optimum",
                "budget": (
                    "normalize raw concrete visits after dropping other; apply each "
                    "representative count to every symmetry-equivalent action"
                ),
                "wdl": "draw at rounded score zero, otherwise book W/L",
                "other": "retain page, discard other row from concrete targets",
            },
            "ruleset_mixing": (
                "training is unstratified within each board size; validation is "
                "a stable cohort for every board-size/ruleset pair"
            ),
            "board_size_scheduling": (
                "randomly shuffled batch tickets proportional to records per size"
            ),
            "ownership": "absent",
            "teacher_inference": "none",
            "train_shards": train_shards,
            "validation_shards": validation_shards,
        },
    )


def stage_prepare(args: argparse.Namespace, specs: tuple[BookSpec, ...]) -> None:
    del specs
    manifest_path = args.run_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("dataset manifest is missing; run the sample stage first")
    dataset_manifest = _read_json(manifest_path)
    if (
        dataset_manifest.get("pipeline_version") != PIPELINE_VERSION
        or dataset_manifest.get("sample_allocation_version")
        != SAMPLE_ALLOCATION_VERSION
        or dataset_manifest.get("state") != "complete"
    ):
        raise ValueError("dataset manifest is incomplete or incompatible")
    for key in ("train_records", "validation_records", "selection_seed"):
        if type(dataset_manifest.get(key)) is not int:
            raise ValueError(f"dataset manifest {key} must be an integer")

    def shard_paths(key: str) -> list[Path]:
        raw_shards = dataset_manifest.get(key)
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError(f"dataset manifest has no {key}")
        paths: list[Path] = []
        for item in raw_shards:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError(f"dataset manifest has malformed {key}")
            if type(item.get("records")) is not int or item["records"] <= 0:
                raise ValueError(f"dataset manifest has invalid record count in {key}")
            if type(item.get("bytes")) is not int or item["bytes"] <= 0:
                raise ValueError(f"dataset manifest has invalid byte count in {key}")
            digest = item.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"dataset manifest has invalid SHA-256 in {key}")
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"dataset shard is missing: {path}")
            actual_bytes = path.stat().st_size
            actual_sha256 = file_sha256(path)
            if item.get("bytes") != actual_bytes or item.get("sha256") != actual_sha256:
                raise ValueError(f"dataset shard content does not match manifest: {path}")
            paths.append(path.resolve())
        if len(paths) != len(set(paths)):
            raise ValueError(f"dataset manifest repeats a shard in {key}")
        return paths

    train_shards = shard_paths("train_shards")
    validation_shards = shard_paths("validation_shards")
    if set(train_shards).intersection(validation_shards):
        raise ValueError("dataset manifest overlaps training and validation shards")
    manifest = prepare_tensor_shards(
        train_shards,
        args.run_dir / "prepared",
        validation_data=validation_shards,
        seed=int(dataset_manifest["selection_seed"]),
        val_fraction=0.0,
        force=args.force_prepare,
    )
    prepared_counts = {"train": 0, "validation": 0}
    for group in manifest["groups"]:
        split = "validation" if group["split"] == "val" else group["split"]
        if split not in prepared_counts:
            raise ValueError(f"prepared manifest has unexpected split {split!r}")
        prepared_counts[split] += int(group["count"])
    expected_counts = {
        "train": dataset_manifest.get("train_records"),
        "validation": dataset_manifest.get("validation_records"),
    }
    if prepared_counts != expected_counts:
        raise ValueError(
            f"prepared record counts {prepared_counts} do not match "
            f"dataset manifest {expected_counts}"
        )
    write_atomic_json(
        args.run_dir / "prepared_report.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "state": "complete",
            "manifest": str((args.run_dir / "prepared" / "manifest.json").resolve()),
            "groups": manifest["groups"],
            "ruleset_keys": manifest["ruleset_keys"],
        },
    )


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    specs = _validate_args(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stages: tuple[Callable[[argparse.Namespace, tuple[BookSpec, ...]], None], ...] = (
        stage_artifacts,
        stage_index,
        stage_sample,
        stage_prepare,
    )
    if args.stage == "all":
        for stage in stages:
            stage(args, specs)
        return
    by_name = {stage.__name__.removeprefix("stage_"): stage for stage in stages}
    by_name[args.stage](args, specs)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
