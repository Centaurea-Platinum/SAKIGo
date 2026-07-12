"""Wait for the book dataset, then run one compiled epoch for all three models."""

from __future__ import annotations

import argparse
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


def run(generation_run: Path, suite_run: Path, poll_seconds: float = 30.0) -> None:
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
                    [
                        sys.executable,
                        "-m",
                        "sakigo.generate.book_distillation",
                        "sample",
                        "--run-dir",
                        str(generation_run),
                    ],
                    check=True,
                )
            except Exception as error:
                _write_status(launcher_status, state="failed", error=str(error))
                raise
            continue
        time.sleep(poll_seconds)
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    expected = (1 << 20, 1 << 12)
    actual = (int(manifest["train_records"]), int(manifest["validation_records"]))
    if actual != expected:
        raise ValueError(f"expected book dataset counts {expected}, found {actual}")
    train_dir = generation_run / "dataset" / "train"
    validation_dir = generation_run / "dataset" / "validation"
    _write_status(
        launcher_status,
        state="starting",
        train_data=str(train_dir.resolve()),
        validation_data=str(validation_dir.resolve()),
        specs=list(DEFAULT_SPECS),
        epochs=1,
    )
    try:
        summary = run_suite(
            SuiteConfig(
                root=suite_run,
                data=(train_dir,),
                validation_data=(validation_dir,),
                specs=DEFAULT_SPECS,
                seed=20260713,
                batch_size=0,
                steps=0,
                num_workers=0,
                checkpoint_interval=0,
                val_batches=0,
                val_fixed=True,
                model_compile="reduce-overhead",
                amp="auto",
                device="cuda",
                augment_d4=False,
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
    args = parser.parse_args()
    run(args.generation_run, args.suite_run, args.poll_seconds)


if __name__ == "__main__":
    main()
