"""Source-owned multi-spec training suite orchestration.

The single-run Trainer stays small and standard. This module owns the larger
Phase-1 experiment layout:

  runs/<suite>/
    data/        # optional generator output location
    generation/  # generator logs/status, when used
    prepared/    # tensor shards
    train/<spec>/
    logs/
    sweeps/
    scripts/
    status.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sakigo.data import PreparedDataset, load_manifest, prepare_tensor_shards
from sakigo.train.benchmark import benchmark_batch_size
from sakigo.train.config import TrainConfig
from sakigo.train.trainer import Trainer, resolve_device

DEFAULT_SPECS = ("narrow-deep", "balanced", "wide-shallow")
DEFAULT_BENCHMARK_BATCH_SIZES = (4, 8, 12, 16, 24, 32, 48, 64)


@dataclass(frozen=True)
class SuitePaths:
    root: Path
    data: Path
    prepared: Path
    generation: Path
    train: Path
    logs: Path
    sweeps: Path
    scripts: Path
    status: Path

    def train_run_dir(self, spec: str) -> Path:
        return self.train / spec


@dataclass(frozen=True)
class SuiteConfig:
    root: Path
    data: tuple[Path, ...] = ()
    validation_data: tuple[Path, ...] = ()
    prepared_dir: Path | None = None
    specs: tuple[str, ...] = DEFAULT_SPECS
    seed: int = 0
    val_fraction: float = 0.05
    batch_size: int = 0
    steps: int = 0
    num_workers: int = 0
    benchmark_batch_sizes: tuple[int, ...] = DEFAULT_BENCHMARK_BATCH_SIZES
    benchmark_timed_steps: int = 8
    benchmark_budget_fraction: float = 0.85
    checkpoint_interval: int = 0  # 0 = eight checkpoints/evaluations per epoch
    val_batches: int = 0  # 0 = cover the complete explicit validation split
    val_fixed: bool = True
    model_compile: str = "reduce-overhead"
    amp: str = "auto"
    device: str = "auto"
    augment_d4: bool = False
    progress: bool = True
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("batch sizes must be positive")
    return values


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one value")
    return values


def build_suite_paths(config: SuiteConfig) -> SuitePaths:
    root = config.root
    return SuitePaths(
        root=root,
        data=root / "data",
        prepared=config.prepared_dir or (root / "prepared"),
        generation=root / "generation",
        train=root / "train",
        logs=root / "logs",
        sweeps=root / "sweeps",
        scripts=root / "scripts",
        status=root / "status.json",
    )


def ensure_suite_dirs(paths: SuitePaths) -> None:
    for path in (
        paths.root,
        paths.data,
        paths.prepared,
        paths.generation,
        paths.train,
        paths.logs,
        paths.sweeps,
        paths.scripts,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _default_data_sources(paths: SuitePaths) -> tuple[Path, ...]:
    if paths.data.exists() and any(paths.data.iterdir()):
        return (paths.data,)
    return ()


def _manifest_counts(manifest: dict[str, object]) -> tuple[int, int]:
    train_records = 0
    val_records = 0
    for group in manifest["groups"]:
        split = str(group["split"])
        count = int(group["count"])
        if split == "train":
            train_records += count
        elif split == "val":
            val_records += count
    return train_records, val_records


def _status_payload(
    config: SuiteConfig,
    paths: SuitePaths,
    *,
    state: str,
    stage: str,
    current_spec: str = "",
    data_sources: tuple[Path, ...] = (),
    train_records: int = 0,
    val_records: int = 0,
    batch_size: int = 0,
    steps: int = 0,
    benchmark_results: list[dict[str, Any]] | None = None,
    final_checkpoints: dict[str, str] | None = None,
    error: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": state,
        "stage": stage,
        "current_spec": current_spec,
        "root": str(paths.root),
        "data_dir": str(paths.data),
        "data_sources": [str(path) for path in data_sources],
        "validation_sources": [str(path) for path in config.validation_data],
        "prepared_dir": str(paths.prepared),
        "generation_dir": str(paths.generation),
        "train_dir": str(paths.train),
        "logs_dir": str(paths.logs),
        "sweeps_dir": str(paths.sweeps),
        "scripts_dir": str(paths.scripts),
        "train_specs": list(config.specs),
        "validation": {
            "fixed": config.val_fixed,
            "batches": config.val_batches or (
                math.ceil(val_records / batch_size) if val_records and batch_size else 0
            ),
        },
        "checkpoint_interval": config.checkpoint_interval or (
            max(1, math.ceil(steps / 8)) if steps else 0
        ),
        "metrics_policy": "step_000000 and checkpoint multiples; final step also logged",
        "log_interval": 0,
        "seed": config.seed,
        "val_fraction": config.val_fraction,
        "batch_size": batch_size,
        "train_steps": steps,
        "train_records": train_records,
        "val_records": val_records,
        "benchmark_specs": list(config.specs),
        "benchmark_batch_sizes": list(config.benchmark_batch_sizes),
        "benchmark_results": benchmark_results or [],
        "final_checkpoints": final_checkpoints or {},
        "updated_at": _now_iso(),
    }
    if error:
        payload["error"] = error
    return payload


def write_status(
    config: SuiteConfig,
    paths: SuitePaths,
    **kwargs: Any,
) -> None:
    _atomic_write_json(paths.status, _status_payload(config, paths, **kwargs))


def prepare_suite_data(
    config: SuiteConfig, paths: SuitePaths
) -> tuple[dict[str, object], tuple[Path, ...]]:
    data_sources = config.data or _default_data_sources(paths)
    if data_sources:
        manifest = prepare_tensor_shards(
            list(data_sources),
            paths.prepared,
            seed=config.seed,
            val_fraction=config.val_fraction,
            validation_data=list(config.validation_data),
        )
        return manifest, data_sources
    manifest_path = paths.prepared / "manifest.json"
    if manifest_path.exists():
        return load_manifest(paths.prepared), ()
    raise ValueError(
        "suite needs --data, non-empty run_dir/data, or an existing prepared manifest"
    )


def choose_batch_size(
    config: SuiteConfig,
    paths: SuitePaths,
) -> tuple[int, list[dict[str, Any]]]:
    if config.batch_size > 0:
        return config.batch_size, []
    if not config.specs:
        raise ValueError("suite needs at least one model spec")

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    budget = 0
    if device.type == "cuda":
        budget = int(
            torch.cuda.get_device_properties(device).total_memory
            * config.benchmark_budget_fraction
        )
    dataset = PreparedDataset(paths.prepared, "train")
    results: list[dict[str, Any]] = []
    best: tuple[int, float] | None = None
    for batch_size in config.benchmark_batch_sizes:
        if len(dataset) % batch_size != 0:
            results.append(
                {
                    "batch_size": batch_size,
                    "geometric_mean_samples_per_second": None,
                    "per_spec": [],
                    "reason": "not_an_exact_epoch_divisor",
                }
            )
            continue
        per_spec: list[dict[str, Any]] = []
        rates: list[float] = []
        stop_larger = False
        for spec in config.specs:
            rate, peak, reason = benchmark_batch_size(
                spec,
                dataset,
                batch_size,
                device,
                timed_steps=config.benchmark_timed_steps,
                memory_budget=budget,
                compile_mode=config.model_compile,
            )
            per_spec.append(
                {
                    "spec": spec,
                    "samples_per_second": rate,
                    "peak_bytes": peak,
                    "reason": reason,
                }
            )
            if rate is not None:
                rates.append(rate)
            if reason in {"over_budget", "oom"}:
                stop_larger = True
            if rate is None:
                break
        aggregate = (
            math.exp(sum(math.log(rate) for rate in rates) / len(rates))
            if len(rates) == len(config.specs)
            else None
        )
        result = {
            "batch_size": batch_size,
            "geometric_mean_samples_per_second": aggregate,
            "per_spec": per_spec,
        }
        results.append(result)
        if aggregate is not None and (best is None or aggregate > best[1]):
            best = (batch_size, aggregate)
        if stop_larger:
            break
    sweep_path = paths.sweeps / "batch_size_all_models.json"
    _atomic_write_json(
        sweep_path,
        {
            "specs": list(config.specs),
            "device": str(device),
            "compile": config.model_compile,
            "budget_fraction": config.benchmark_budget_fraction,
            "results": results,
            "best_batch_size": best[0] if best else None,
        },
    )
    if best is None:
        raise RuntimeError("batch-size benchmark found no candidate usable by every model")
    return best[0], results


def train_config_for_spec(
    config: SuiteConfig,
    paths: SuitePaths,
    spec: str,
    *,
    data_sources: tuple[Path, ...],
    batch_size: int,
    steps: int,
    checkpoint_interval: int | None = None,
    val_batches: int | None = None,
) -> TrainConfig:
    checkpoint_interval = (
        config.checkpoint_interval if checkpoint_interval is None else checkpoint_interval
    )
    val_batches = config.val_batches if val_batches is None else val_batches
    return TrainConfig(
        data=tuple(str(path) for path in data_sources),
        validation_data=tuple(str(path) for path in config.validation_data),
        prepared_dir=str(paths.prepared),
        seed=config.seed,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        augment_d4=config.augment_d4,
        model_spec=spec,
        steps=steps,
        batch_size=batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        amp=config.amp,
        compile=config.model_compile,
        run_dir=str(paths.train_run_dir(spec)),
        log_interval=0,
        checkpoint_interval=checkpoint_interval,
        val_batches=val_batches,
        val_fixed=config.val_fixed,
        progress=config.progress,
        device=config.device,
    )


def run_suite(config: SuiteConfig) -> dict[str, Any]:
    if not config.specs:
        raise ValueError("suite needs at least one model spec")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if config.val_batches < 0:
        raise ValueError("val_batches must be non-negative")
    if config.batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if config.steps < 0:
        raise ValueError("steps must be non-negative")
    paths = build_suite_paths(config)
    ensure_suite_dirs(paths)
    data_sources: tuple[Path, ...] = ()
    train_records = 0
    val_records = 0
    batch_size = config.batch_size
    steps = config.steps
    benchmark_results: list[dict[str, Any]] = []
    final_checkpoints: dict[str, str] = {}
    try:
        write_status(
            config,
            paths,
            state="running",
            stage="preparing_data",
            data_sources=data_sources,
        )
        manifest, data_sources = prepare_suite_data(config, paths)
        train_records, val_records = _manifest_counts(manifest)
        if train_records <= 0:
            raise ValueError("prepared data has no train records")

        write_status(
            config,
            paths,
            state="running",
            stage="benchmark_batch_size" if batch_size <= 0 else "planning",
            data_sources=data_sources,
            train_records=train_records,
            val_records=val_records,
            batch_size=batch_size,
            steps=steps,
        )
        batch_size, benchmark_results = choose_batch_size(config, paths)
        steps = steps or math.ceil(train_records / batch_size)
        checkpoint_interval = config.checkpoint_interval or max(1, math.ceil(steps / 8))
        val_batches = config.val_batches or (
            math.ceil(val_records / batch_size) if val_records else 0
        )

        for spec in config.specs:
            write_status(
                config,
                paths,
                state="running",
                stage="training",
                current_spec=spec,
                data_sources=data_sources,
                train_records=train_records,
                val_records=val_records,
                batch_size=batch_size,
                steps=steps,
                benchmark_results=benchmark_results,
                final_checkpoints=final_checkpoints,
            )
            trainer = Trainer(
                train_config_for_spec(
                    config,
                    paths,
                    spec,
                    data_sources=data_sources,
                    batch_size=batch_size,
                    steps=steps,
                    checkpoint_interval=checkpoint_interval,
                    val_batches=val_batches,
                )
            )
            final_checkpoints[spec] = str(trainer.train())

        summary = _status_payload(
            config,
            paths,
            state="complete",
            stage="done",
            data_sources=data_sources,
            train_records=train_records,
            val_records=val_records,
            batch_size=batch_size,
            steps=steps,
            benchmark_results=benchmark_results,
            final_checkpoints=final_checkpoints,
        )
        _atomic_write_json(paths.status, summary)
        return summary
    except Exception as error:
        write_status(
            config,
            paths,
            state="failed",
            stage="error",
            data_sources=data_sources,
            train_records=train_records,
            val_records=val_records,
            batch_size=batch_size,
            steps=steps,
            benchmark_results=benchmark_results,
            final_checkpoints=final_checkpoints,
            error=str(error),
        )
        raise


def parse_args(argv: list[str] | None = None) -> SuiteConfig:
    parser = argparse.ArgumentParser(prog="sakigo.train.suite")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--data", nargs="*", type=Path, default=())
    parser.add_argument("--validation-data", nargs="*", type=Path, default=())
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--specs", default=",".join(DEFAULT_SPECS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--benchmark-batch-sizes",
        default=",".join(str(size) for size in DEFAULT_BENCHMARK_BATCH_SIZES),
    )
    parser.add_argument("--benchmark-timed-steps", type=int, default=8)
    parser.add_argument("--benchmark-budget-fraction", type=float, default=0.85)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument(
        "--val-fixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay one fixed validation subset for each metrics row.",
    )
    parser.add_argument(
        "--compile",
        choices=("off", "default", "reduce-overhead"),
        default="reduce-overhead",
    )
    parser.add_argument("--amp", choices=("auto", "off"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--augment-d4", action="store_true")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args(argv)

    run_dir = args.run_dir or Path("runs") / f"phase1_suite_{datetime.now():%Y%m%d_%H%M%S}"
    return SuiteConfig(
        root=run_dir,
        data=tuple(args.data),
        validation_data=tuple(args.validation_data),
        prepared_dir=args.prepared_dir,
        specs=_parse_csv_strings(args.specs),
        seed=args.seed,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        steps=args.steps,
        num_workers=args.num_workers,
        benchmark_batch_sizes=_parse_csv_ints(args.benchmark_batch_sizes),
        benchmark_timed_steps=args.benchmark_timed_steps,
        benchmark_budget_fraction=args.benchmark_budget_fraction,
        checkpoint_interval=args.checkpoint_interval,
        val_batches=args.val_batches,
        val_fixed=args.val_fixed,
        model_compile=args.compile,
        amp=args.amp,
        device=args.device,
        augment_d4=args.augment_d4,
        progress=args.progress,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )


def main() -> None:
    summary = run_suite(parse_args())
    print(json.dumps({"status": summary["state"], "run_dir": summary["root"]}))


if __name__ == "__main__":
    main()
