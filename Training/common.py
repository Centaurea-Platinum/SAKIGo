from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
BOARD_PLANE_COUNT = 6
RULE_FEATURE_COUNT = 10
WDL_LABELS = ("win", "draw", "loss")
HEADS = ("wdl", "score", "ownership", "policy", "budget")
ACTION_HEADS = ("policy", "budget")


@dataclass(frozen=True, eq=False)
class TrainingRecord:
    """One training position. Array fields are numpy-backed for cheap collation.

    board_planes: float32 [BOARD_PLANE_COUNT, n, n]; rule_features: float32 [RULE_FEATURE_COUNT];
    wdl: float32 [3]; ownership: float32 [n*n]; policy/budget: float32 [n*n+1] distributions;
    legal_mask: bool [n*n+1].
    """

    schema_version: int
    board_size: int
    ply: int
    position_key: str
    board_planes: np.ndarray
    rule_features: np.ndarray
    wdl: np.ndarray | None = None
    score: float | None = None
    ownership: np.ndarray | None = None
    policy: np.ndarray | None = None
    budget: np.ndarray | None = None
    legal_mask: np.ndarray | None = None

    def array_nbytes(self) -> int:
        total = 0
        for field in (self.board_planes, self.rule_features, self.wdl, self.ownership, self.policy, self.budget, self.legal_mask):
            if field is not None:
                total += field.nbytes
        return total


def resolve_root_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def board_sizes(raw: str, available: Iterable[int] | None = None) -> list[int]:
    if raw.strip():
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        sizes = sorted(set(available or ()))
    if not sizes:
        raise ValueError("at least one board size is required")
    if any(size <= 0 for size in sizes):
        raise ValueError("board sizes must be positive")
    return sizes


def board_sampling_weights(raw: str, sizes: list[int]) -> dict[int, float]:
    if not raw.strip():
        return {size: 1.0 for size in sizes}
    pieces = [part.strip() for part in raw.split(",") if part.strip()]
    if all(":" in piece for piece in pieces):
        weights = {size: 1.0 for size in sizes}
        valid_sizes = set(sizes)
        for piece in pieces:
            size_text, weight_text = piece.split(":", 1)
            size = int(size_text.strip())
            if size in valid_sizes:
                weights[size] = float(weight_text.strip())
    else:
        if len(pieces) != len(sizes):
            raise ValueError("--board-sampling-weights must match --boards or use n:weight pairs")
        weights = {size: float(weight) for size, weight in zip(sizes, pieces)}
    if any(weight <= 0.0 for weight in weights.values()):
        raise ValueError("board sampling weights must be positive")
    return weights


def step_set(raw: str) -> set[int]:
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def training_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def make_run_dir(raw: str, resume_path: str | Path | None = None) -> Path:
    if raw:
        path = resolve_root_path(raw)
    elif resume_path is not None:
        checkpoint_path = resolve_root_path(resume_path)
        path = checkpoint_path.parent.parent
    else:
        path = ROOT / "Training" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path

