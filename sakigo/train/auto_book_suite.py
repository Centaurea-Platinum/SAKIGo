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

from sakigo.train.suite import DEFAULT_SPECS, SuiteConfig, run_suite


def _write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**values, "updated_at": datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _sample_command(generation_run: Path, index_report: Path) -> list[str]:
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
) -> None:
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
    expected = (1 << 20, 1 << 12)
    raw_counts = (manifest.get("train_records"), manifest.get("validation_records"))
    if any(type(value) is not int for value in raw_counts):
        raise ValueError("dataset manifest record totals must be integers")
    actual = raw_counts
    if actual != expected:
        raise ValueError(f"expected book dataset counts {expected}, found {actual}")
    require_identity = isinstance(manifest.get("allocation"), dict)
    if require_identity and manifest.get("sample_allocation_version") != 2:
        raise ValueError(
            "multi-book dataset predates board-size/ruleset validation cohorts; "
            "rerun index and sample before training"
        )
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
        score_weight=81.0,
        model_compile=model_compile,
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
                batch_size=0,
                steps=0,
                num_workers=0,
                checkpoint_interval=0,
                val_batches=0,
                val_fixed=True,
                model_compile=model_compile,
                amp="auto",
                device="cuda",
                augment_d4=False,
                score_weight=81.0,
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
    args = parser.parse_args()
    run(
        args.generation_run,
        args.suite_run,
        args.poll_seconds,
        args.prepared_dir,
        args.compile,
    )


if __name__ == "__main__":
    main()
