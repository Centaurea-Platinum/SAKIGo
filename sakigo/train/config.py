"""Run configuration: a frozen dataclass, loadable from TOML with CLI overrides.

Replaces the legacy ~30-flag argparse surface + vars(args)-in-checkpoint.
"""

from __future__ import annotations

import argparse
import math
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    # data
    data: tuple[str, ...] = ()
    validation_data: tuple[str, ...] = ()
    prepared_dir: str = ""
    seed: int = 0
    val_fraction: float = 0.05
    num_workers: int = 2
    board_weights: dict[int, float] | None = None
    augment_d4: bool = False
    # model
    model_spec: str = "balanced"
    # optimization
    steps: int = 1000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lr_schedule: str = "warmup-cosine"  # or "constant"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.05
    amp: str = "auto"  # auto|off
    compile: str = "reduce-overhead"  # off|default|reduce-overhead
    # loss weights
    wdl_weight: float = 1.0
    score_weight: float = 1.0
    policy_weight: float = 1.0
    budget_weight: float = 1.0
    # logging / checkpoints
    run_dir: str = ""
    log_interval: int = 0  # 0 means follow checkpoint_interval
    checkpoint_interval: int = 1000
    val_batches: int = 16  # total cap; must cover every board-size/ruleset cohort
    val_fixed: bool = False  # replay cohort subsets instead of rotating within each cohort
    progress: bool = True
    device: str = "auto"
    resume: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_train_config(config: TrainConfig) -> TrainConfig:
    integer_bounds = (
        ("steps", config.steps, 0),
        ("batch_size", config.batch_size, 1),
        ("num_workers", config.num_workers, 0),
        ("warmup_steps", config.warmup_steps, 0),
        ("log_interval", config.log_interval, 0),
        ("checkpoint_interval", config.checkpoint_interval, 1),
        ("val_batches", config.val_batches, 0),
    )
    for label, value, minimum in integer_bounds:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.lr_schedule not in {"constant", "warmup-cosine"}:
        raise ValueError("lr_schedule must be constant or warmup-cosine")
    if config.amp not in {"auto", "off"}:
        raise ValueError("amp must be auto or off")
    if config.compile not in {"off", "default", "reduce-overhead"}:
        raise ValueError("compile must be off, default, or reduce-overhead")
    numeric_bounds = (
        ("lr", config.lr, 0.0, False),
        ("weight_decay", config.weight_decay, 0.0, True),
        ("grad_clip", config.grad_clip, 0.0, True),
        ("min_lr_ratio", config.min_lr_ratio, 0.0, True),
        ("wdl_weight", config.wdl_weight, 0.0, True),
        ("score_weight", config.score_weight, 0.0, True),
        ("policy_weight", config.policy_weight, 0.0, True),
        ("budget_weight", config.budget_weight, 0.0, True),
    )
    for label, value, minimum, inclusive in numeric_bounds:
        numeric = float(value)
        invalid_bound = numeric < minimum if inclusive else numeric <= minimum
        if not math.isfinite(numeric) or invalid_bound:
            relation = ">=" if inclusive else ">"
            raise ValueError(f"{label} must be finite and {relation} {minimum}")
    if config.min_lr_ratio > 1.0:
        raise ValueError("min_lr_ratio must be <= 1")
    if not any(
        float(weight) > 0
        for weight in (
            config.wdl_weight,
            config.score_weight,
            config.policy_weight,
            config.budget_weight,
        )
    ):
        raise ValueError("at least one loss weight must be positive")
    if config.board_weights is not None:
        for size, weight in config.board_weights.items():
            if type(size) is not int or size <= 0:
                raise ValueError("board weight sizes must be positive integers")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError("board weights must be finite and positive")
    return config


def config_from_dict(raw: dict[str, Any]) -> TrainConfig:
    known = {field.name for field in fields(TrainConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"unknown train config key {key!r}")
        if key in {"data", "validation_data"} and isinstance(value, list):
            value = tuple(value)
        if key == "board_weights" and isinstance(value, dict):
            value = {int(k): float(v) for k, v in value.items()}
        kwargs[key] = value
    return validate_train_config(TrainConfig(**kwargs))


def load_toml_config(path: Path) -> TrainConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(raw)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(prog="sakigo.train")
    parser.add_argument("--config", type=Path, default=None, help="TOML run config")
    parser.add_argument("--data", nargs="*", default=None, help="JSONL(.zst) sources")
    parser.add_argument(
        "--validation-data",
        nargs="*",
        default=None,
        help="Explicit validation JSONL(.zst) sources; disables hash re-splitting.",
    )
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--model-spec", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--augment-d4", action="store_true", default=None)
    parser.add_argument("--compile", choices=("off", "default", "reduce-overhead"), default=None)
    parser.add_argument("--amp", choices=("auto", "off"), default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        help="Total validation batch cap; must cover every board-size/ruleset cohort.",
    )
    parser.add_argument("--val-fixed", action="store_true", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-progress", action="store_true")
    namespace = parser.parse_args(argv)

    config = load_toml_config(namespace.config) if namespace.config else TrainConfig()
    overrides: dict[str, Any] = {}
    for key in (
        "data",
        "validation_data",
        "prepared_dir",
        "model_spec",
        "steps",
        "batch_size",
        "lr",
        "seed",
        "val_fraction",
        "num_workers",
        "augment_d4",
        "compile",
        "amp",
        "run_dir",
        "log_interval",
        "checkpoint_interval",
        "val_batches",
        "val_fixed",
        "device",
        "resume",
    ):
        value = getattr(namespace, key)
        if value is not None:
            overrides[key] = tuple(value) if key in {"data", "validation_data"} else value
    if namespace.no_progress:
        overrides["progress"] = False
    return replace(config, **overrides)
