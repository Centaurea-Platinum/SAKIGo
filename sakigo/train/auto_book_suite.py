"""Wait for the book dataset, then run one epoch for all three models."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sakigo.generate.multi_book_distillation import SAMPLE_ALLOCATION_VERSION
from sakigo.train.suite import DEFAULT_SPECS, SuiteConfig, run_suite


def _write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**values, "updated_at": datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _book_ids_from_report(index_report: Path) -> list[str]:
    report = json.loads(index_report.read_text(encoding="utf-8"))
    books = report.get("books")
    if isinstance(books, list) and books:
        book_ids = [
            str(book["book_id"])
            for book in books
            if isinstance(book, dict) and "book_id" in book
        ]
        if len(book_ids) != len(books):
            raise ValueError("multi-book index report has malformed book metadata")
        return book_ids
    return []


def _sample_command(generation_run: Path, index_report: Path) -> list[str]:
    book_ids = _book_ids_from_report(index_report)
    if book_ids:
        return [
            sys.executable,
            "-m",
            "sakigo.generate.multi_book_distillation",
            "sample",
            "--run-dir",
            str(generation_run),
            "--books",
            ",".join(book_ids),
        ]
    return [
        sys.executable,
        "-m",
        "sakigo.generate.book_distillation",
        "sample",
        "--run-dir",
        str(generation_run),
    ]


def _index_command(
    generation_run: Path,
    index_report: Path,
    *,
    train_samples: int,
    validation_samples: int,
    workers: int,
) -> list[str]:
    book_ids = _book_ids_from_report(index_report)
    if not book_ids:
        raise ValueError("full-dataset allocation refresh requires a multi-book report")
    return [
        sys.executable,
        "-m",
        "sakigo.generate.multi_book_distillation",
        "index",
        "--run-dir",
        str(generation_run),
        "--books",
        ",".join(book_ids),
        "--train-samples",
        str(train_samples),
        "--validation-samples",
        str(validation_samples),
        "--workers",
        str(workers),
    ]


def _index_allocation_matches(
    report: dict[str, object], *, train_samples: int, validation_samples: int
) -> bool:
    return (
        report.get("sample_allocation_version") == SAMPLE_ALLOCATION_VERSION
        and report.get("train_records") == train_samples
        and report.get("validation_records") == validation_samples
    )


def _manifest_shards(
    manifest: dict[str, object], key: str, *, require_identity: bool
) -> tuple[Path, ...]:
    raw_shards = manifest.get(key)
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError(f"dataset manifest has no {key}")
    paths: list[Path] = []
    for item in raw_shards:
        raw_path = item.get("path") if isinstance(item, dict) else item
        if not isinstance(raw_path, str):
            raise ValueError(f"dataset manifest has malformed {key}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"dataset shard is missing: {path}")
        expected_bytes = item.get("bytes") if isinstance(item, dict) else None
        expected_sha256 = item.get("sha256") if isinstance(item, dict) else None
        expected_records = item.get("records") if isinstance(item, dict) else None
        if isinstance(item, dict) and (
            type(expected_records) is not int or expected_records <= 0
        ):
            raise ValueError(f"dataset manifest has invalid record count for {path}")
        if require_identity and (
            type(expected_bytes) is not int or not isinstance(expected_sha256, str)
        ):
            raise ValueError(f"dataset manifest has no content identity for {path}")
        if expected_sha256 is not None and (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"dataset manifest has invalid SHA-256 for {path}")
        if expected_bytes is not None or expected_sha256 is not None:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != expected_bytes or digest.hexdigest() != expected_sha256:
                raise ValueError(f"dataset shard content does not match manifest: {path}")
        paths.append(path.resolve())
    if len(paths) != len(set(paths)):
        raise ValueError(f"dataset manifest repeats a shard in {key}")
    return tuple(paths)


def run(
    generation_run: Path,
    suite_run: Path,
    poll_seconds: float = 30.0,
    prepared_dir: Path | None = None,
    model_compile: str = "reduce-overhead",
    train_samples: int = 1 << 20,
    validation_samples: int = 1 << 12,
    index_workers: int = 3,
    batch_size: int = 0,
    benchmark_budget_fraction: float = 0.85,
) -> None:
    if train_samples <= 0 or validation_samples <= 0:
        raise ValueError("training and validation sample counts must be positive")
    if index_workers <= 0:
        raise ValueError("index_workers must be positive")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if not 0.0 < benchmark_budget_fraction <= 1.0:
        raise ValueError("benchmark_budget_fraction must be in (0, 1]")
    launcher_status = suite_run / "launcher_status.json"
    dataset_manifest = generation_run / "dataset_manifest.json"
    index_report = generation_run / "book_index_report.json"
    _write_status(
        launcher_status,
        state="waiting",
        generation_run=str(generation_run.resolve()),
        dataset_manifest=str(dataset_manifest.resolve()),
    )
    while not dataset_manifest.exists():
        if index_report.exists():
            report = json.loads(index_report.read_text(encoding="utf-8"))
            if not _index_allocation_matches(
                report,
                train_samples=train_samples,
                validation_samples=validation_samples,
            ):
                _write_status(
                    launcher_status,
                    state="refreshing_dataset_allocation",
                    train_records=train_samples,
                    validation_records=validation_samples,
                )
                try:
                    subprocess.run(
                        _index_command(
                            generation_run,
                            index_report,
                            train_samples=train_samples,
                            validation_samples=validation_samples,
                            workers=index_workers,
                        ),
                        check=True,
                    )
                except Exception as error:
                    _write_status(launcher_status, state="failed", error=str(error))
                    raise
                continue
            _write_status(
                launcher_status,
                state="generating_dataset_shards",
                generation_run=str(generation_run.resolve()),
            )
            try:
                subprocess.run(
                    _sample_command(generation_run, index_report),
                    check=True,
                )
            except Exception as error:
                _write_status(launcher_status, state="failed", error=str(error))
                raise
            continue
        time.sleep(poll_seconds)
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise ValueError("dataset manifest is not complete")
    expected = (train_samples, validation_samples)
    raw_counts = (manifest.get("train_records"), manifest.get("validation_records"))
    if any(type(value) is not int for value in raw_counts):
        raise ValueError("dataset manifest record totals must be integers")
    actual = raw_counts
    if actual != expected:
        raise ValueError(f"expected book dataset counts {expected}, found {actual}")
    require_identity = isinstance(manifest.get("allocation"), dict)
    if (
        require_identity
        and manifest.get("sample_allocation_version") != SAMPLE_ALLOCATION_VERSION
    ):
        raise ValueError(
            "multi-book dataset predates board-size/ruleset validation cohorts; "
            "rerun index and sample before training"
        )
    allocation = manifest.get("allocation")
    if require_identity and isinstance(allocation, dict):
        validation_counts = []
        for book_id, counts in allocation.items():
            if not isinstance(book_id, str) or not isinstance(counts, dict):
                raise ValueError("dataset manifest has malformed cohort allocation")
            count = counts.get("validation")
            if type(count) is not int or count <= 0:
                raise ValueError(f"dataset manifest has no validation for cohort {book_id}")
            validation_counts.append(count)
        if not validation_counts or sum(validation_counts) != validation_samples:
            raise ValueError("dataset manifest validation allocation has the wrong total")
        if max(validation_counts) - min(validation_counts) > 1:
            raise ValueError("dataset manifest validation cohorts are not balanced")
    train_sources = _manifest_shards(
        manifest, "train_shards", require_identity=require_identity
    )
    validation_sources = _manifest_shards(
        manifest, "validation_shards", require_identity=require_identity
    )
    if set(train_sources).intersection(validation_sources):
        raise ValueError("dataset manifest overlaps training and validation shards")
    _write_status(
        launcher_status,
        state="starting",
        train_data=[str(path.resolve()) for path in train_sources],
        validation_data=[str(path.resolve()) for path in validation_sources],
        specs=list(DEFAULT_SPECS),
        epochs=1,
        score_weight=1.0,
        score_weighting="board_area",
        model_compile=model_compile,
        batch_size=batch_size,
        benchmark_budget_fraction=benchmark_budget_fraction,
        prepared_dir=str(prepared_dir.resolve()) if prepared_dir else None,
    )
    try:
        summary = run_suite(
            SuiteConfig(
                root=suite_run,
                data=train_sources,
                validation_data=validation_sources,
                prepared_dir=prepared_dir,
                specs=DEFAULT_SPECS,
                seed=20260713,
                batch_size=batch_size,
                benchmark_budget_fraction=benchmark_budget_fraction,
                steps=0,
                num_workers=0,
                checkpoint_interval=0,
                val_batches=0,
                val_fixed=True,
                model_compile=model_compile,
                amp="auto",
                device="cuda",
                augment_d4=False,
                score_weight=1.0,
                progress=False,
            )
        )
    except Exception as error:
        _write_status(launcher_status, state="failed", error=str(error))
        raise
    _write_status(
        launcher_status,
        state="complete",
        final_checkpoints=summary["final_checkpoints"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--suite-run", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument(
        "--compile",
        choices=("off", "default", "reduce-overhead"),
        default="reduce-overhead",
    )
    parser.add_argument("--train-samples", type=int, default=1 << 20)
    parser.add_argument("--validation-samples", type=int, default=1 << 12)
    parser.add_argument("--index-workers", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Fixed batch size to safety-preflight; 0 selects automatically.",
    )
    parser.add_argument(
        "--benchmark-budget-fraction",
        type=float,
        default=0.85,
        help="Maximum fraction of total GPU memory available to the training batch.",
    )
    args = parser.parse_args()
    run(
        generation_run=args.generation_run,
        suite_run=args.suite_run,
        poll_seconds=args.poll_seconds,
        prepared_dir=args.prepared_dir,
        model_compile=args.compile,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        index_workers=args.index_workers,
        batch_size=args.batch_size,
        benchmark_budget_fraction=args.benchmark_budget_fraction,
    )


if __name__ == "__main__":
    main()
