"""Run configuration: a frozen dataclass, loadable from TOML with CLI overrides.

Replaces the legacy ~30-flag argparse surface + vars(args)-in-checkpoint.
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    # data
    data: tuple[str, ...] = ()
    prepared_dir: str = ""
    seed: int = 0
    val_fraction: float = 0.05
    num_workers: int = 2
    board_weights: dict[int, float] | None = None
    augment_d4: bool = False
    # model
    model_spec: str = "plain"
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
    compile: str = "default"  # off|default|reduce-overhead
    # loss weights
    wdl_weight: float = 1.0
    score_weight: float = 1.0
    ownership_weight: float = 1.0
    policy_weight: float = 1.0
    budget_weight: float = 1.0
    # logging / checkpoints
    run_dir: str = ""
    log_interval: int = 100
    checkpoint_interval: int = 1000
    val_batches: int = 16
    val_fixed: bool = False  # freeze one val subset instead of rotating through the val set
    progress: bool = True
    device: str = "auto"
    resume: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(raw: dict[str, Any]) -> TrainConfig:
    known = {field.name for field in fields(TrainConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"unknown train config key {key!r}")
        if key == "data" and isinstance(value, list):
            value = tuple(value)
        if key == "board_weights" and isinstance(value, dict):
            value = {int(k): float(v) for k, v in value.items()}
        kwargs[key] = value
    return TrainConfig(**kwargs)


def load_toml_config(path: Path) -> TrainConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(raw)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(prog="sakigo.train")
    parser.add_argument("--config", type=Path, default=None, help="TOML run config")
    parser.add_argument("--data", nargs="*", default=None, help="JSONL(.zst) sources")
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
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--val-fixed", action="store_true", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-progress", action="store_true")
    namespace = parser.parse_args(argv)

    config = load_toml_config(namespace.config) if namespace.config else TrainConfig()
    overrides: dict[str, Any] = {}
    for key in (
        "data",
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
            overrides[key] = tuple(value) if key == "data" else value
    if namespace.no_progress:
        overrides["progress"] = False
    return replace(config, **overrides)
