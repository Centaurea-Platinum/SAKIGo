from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from Model.sakigo_model import (
    SakiGoModel,
    SakiGoModelConfig,
    ScalarSakiGoModel,
    config_from_checkpoint,
)


def model_from_config(config: SakiGoModelConfig) -> nn.Module:
    if config.architecture == "SakiGoModel":
        return SakiGoModel(config)
    if config.architecture == "ScalarSakiGoModel":
        return ScalarSakiGoModel(config)
    raise ValueError(f"unsupported architecture {config.architecture!r}")


def _cuda_rng_state() -> list[torch.Tensor] | None:
    return torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    args: argparse.Namespace,
    model_config: SakiGoModelConfig,
    train_rng: random.Random,
    val_rng: random.Random,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step:06}.pt"
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(args),
        "model_config": asdict(model_config),
        "python_rng_state": train_rng.getstate(),
        "val_python_rng_state": val_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": _cuda_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
    minimum_board_size: int,
) -> nn.Module:
    config = config_from_checkpoint(checkpoint, minimum_board_size=minimum_board_size)
    model = model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def restore_rng_state(checkpoint: dict[str, Any], train_rng: random.Random, val_rng: random.Random) -> None:
    if "python_rng_state" in checkpoint:
        train_rng.setstate(checkpoint["python_rng_state"])
    if "val_python_rng_state" in checkpoint:
        val_rng.setstate(checkpoint["val_python_rng_state"])
    if "torch_rng_state" in checkpoint:
        # torch.load(map_location=cuda) may have moved the saved ByteTensor to the GPU.
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_state = checkpoint.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
