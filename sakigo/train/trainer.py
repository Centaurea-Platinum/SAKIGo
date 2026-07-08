"""The Trainer: standard PyTorch loop with torch.compile, bf16 autocast,
fused AdamW, SequentialLR warmup-cosine, TensorBoard (+ CSV mirror for the
viewer), tqdm, atomic weights_only-loadable checkpoints, and full RNG capture.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sakigo.data import (
    FixedBatchSampler,
    PreparedDataset,
    RulesetBalancedBatchSampler,
    batch_to_device,
    make_dataloader,
    prepare_tensor_shards,
)
from sakigo.model import SakiGoNet, config_from_dict as model_config_from_dict, config_from_spec
from sakigo.train.config import TrainConfig, config_from_dict as train_config_from_dict
from sakigo.train.losses import LossWeights, compute_head_losses, weighted_total_loss
from sakigo.train.metrics import (
    MetricAccumulator,
    add_val_confusion,
    append_metrics,
    metric_fields,
    prefixed,
    write_metrics_header,
)

CHECKPOINT_SCHEMA_VERSION = 2


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        if (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in lower_name
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.lr_schedule == "constant":
        return None
    if config.lr_schedule != "warmup-cosine":
        raise ValueError(f"unknown lr schedule: {config.lr_schedule}")
    total = max(1, config.steps)
    warmup = min(max(0, config.warmup_steps), total)
    eta_min = config.lr * config.min_lr_ratio
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total - warmup), eta_min=eta_min
    )
    if warmup == 0:
        return cosine
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[linear, cosine], milestones=[warmup]
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        torch.set_float32_matmul_precision("high")
        # Seed before any parameter initialization so runs are reproducible
        # regardless of ambient RNG state (resume overwrites these below).
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.loss_weights = LossWeights(
            wdl=config.wdl_weight,
            score=config.score_weight,
            ownership=config.ownership_weight,
            policy=config.policy_weight,
            budget=config.budget_weight,
        )
        self.run_dir = self._make_run_dir()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.compile_status = "off"
        self.log_interval = (
            config.checkpoint_interval if config.log_interval <= 0 else config.log_interval
        )

        self.model_config = config_from_spec(config.model_spec)
        self.model = SakiGoNet(self.model_config).to(self.device)
        self.optimizer = self._make_optimizer()
        self.scheduler = make_scheduler(self.optimizer, config)
        self.start_step = 0
        if config.resume:
            self._resume(Path(config.resume))
        self.compiled_model = self._compile_model()
        self._prepare_data()

    # -- setup -------------------------------------------------------------

    def _make_run_dir(self) -> Path:
        if self.config.run_dir:
            path = Path(self.config.run_dir)
        elif self.config.resume:
            path = Path(self.config.resume).parent.parent
        else:
            path = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_optimizer(self) -> torch.optim.Optimizer:
        kwargs: dict[str, object] = {"lr": self.config.lr}
        if self.device.type == "cuda":
            kwargs["fused"] = True
        return torch.optim.AdamW(
            optimizer_param_groups(self.model, self.config.weight_decay), **kwargs
        )

    def _compile_model(self) -> nn.Module:
        if self.config.compile == "off":
            return self.model
        mode = None if self.config.compile == "default" else self.config.compile
        try:
            compiled = torch.compile(self.model, mode=mode)
            self.compile_status = f"enabled:{self.config.compile}"
            return compiled
        except Exception as error:  # noqa: BLE001 - fall back to eager, record why
            self.compile_status = f"failed:{error}"
            return self.model

    def _prepare_data(self) -> None:
        config = self.config
        if config.prepared_dir:
            prepared_dir = Path(config.prepared_dir)
            if config.data:
                prepare_tensor_shards(
                    list(config.data),
                    prepared_dir,
                    seed=config.seed,
                    val_fraction=config.val_fraction,
                )
        elif config.data:
            prepared_dir = self.run_dir / "prepared"
            prepare_tensor_shards(
                list(config.data),
                prepared_dir,
                seed=config.seed,
                val_fraction=config.val_fraction,
            )
        else:
            raise ValueError("train config needs data sources or a prepared_dir")
        self.prepared_dir = prepared_dir

        self.train_dataset = PreparedDataset(prepared_dir, "train", augment_d4=config.augment_d4)
        train_sampler = RulesetBalancedBatchSampler(
            self.train_dataset,
            config.batch_size,
            seed=config.seed,
            board_weights=config.board_weights,
        )
        pin = self.device.type == "cuda"
        self.train_loader = make_dataloader(
            self.train_dataset,
            train_sampler,
            num_workers=config.num_workers,
            pin_memory=pin,
            seed=config.seed,
            persistent_workers=True,
        )
        try:
            self.val_dataset: PreparedDataset | None = PreparedDataset(prepared_dir, "val")
        except ValueError:
            self.val_dataset = None
            self.val_loader = None
            return
        val_sampler = RulesetBalancedBatchSampler(
            self.val_dataset,
            config.batch_size,
            seed=config.seed + 1,
            board_weights=config.board_weights,
        )
        # Rotating (default): each eval takes the next without-replacement chunk,
        # covering the val set across evals. Fixed: replay one frozen subset for
        # minimal step-to-step metric noise.
        batch_sampler: FixedBatchSampler | RulesetBalancedBatchSampler = (
            FixedBatchSampler.freeze(val_sampler, config.val_batches)
            if config.val_fixed
            else val_sampler
        )
        self.val_loader = make_dataloader(
            self.val_dataset,
            batch_sampler,
            num_workers=0,
            pin_memory=pin,
            seed=config.seed + 1,
        )

    # -- checkpointing -----------------------------------------------------

    def _checkpoint_payload(self, step: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": asdict(self.model_config),
            "run_config": self.config.as_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()
        return payload

    def save_checkpoint(self, step: int) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"step_{step:06}.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self._checkpoint_payload(step), tmp)
        tmp.replace(path)
        return path

    def _resume(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        model_config = model_config_from_dict(payload["model_config"])
        if asdict(model_config) != asdict(self.model_config):
            self.model_config = model_config
            self.model = SakiGoNet(model_config).to(self.device)
            self.optimizer = self._make_optimizer()
            self.scheduler = make_scheduler(self.optimizer, self.config)
        self.model.load_state_dict(payload["model_state"])
        try:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except ValueError as error:
            print(f"warning: optimizer state not restored ({error})", flush=True)
        if self.scheduler is not None and "scheduler_state" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        rng = payload.get("rng", {})
        if "python" in rng:
            state = rng["python"]
            random.setstate((state[0], tuple(state[1]), state[2]))
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu().to(torch.uint8) for state in rng["cuda"]])
        self.start_step = int(payload["step"])

    # -- steps ---------------------------------------------------------------

    def _autocast(self):
        if self.config.amp != "off" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
    ]:
        self.compiled_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.compiled_model(batch["board"], batch["rules"])
            head_losses = compute_head_losses(output, batch)
            total = weighted_total_loss(head_losses, self.loss_weights)
        total.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return output, head_losses, total

    @torch.no_grad()
    def evaluate(self, batches: int) -> MetricAccumulator:
        accumulator = MetricAccumulator(self.loss_weights)
        if self.val_loader is None or batches <= 0:
            return accumulator
        self.compiled_model.eval()
        iterator = iter(self.val_loader)
        for _ in range(batches):
            try:
                batch = batch_to_device(next(iterator), self.device)
            except StopIteration:  # fixed val sampler is finite
                break
            with self._autocast():
                output = self.compiled_model(batch["board"], batch["rules"])
                head_losses = compute_head_losses(output, batch)
                total = weighted_total_loss(head_losses, self.loss_weights)
            accumulator.add_batch(output, batch, head_losses, total)
        return accumulator

    # -- loop ----------------------------------------------------------------

    def _log_row(
        self,
        step: int,
        train_accumulator: MetricAccumulator,
        writer: SummaryWriter,
        samples_per_second: float,
    ) -> dict[str, float | int]:
        row: dict[str, float | int] = {"step": step}
        row.update(prefixed("train", train_accumulator.averages()))
        val_accumulator = self.evaluate(self.config.val_batches)
        if val_accumulator.steps:
            row.update(prefixed("val", val_accumulator.averages()))
            add_val_confusion(row, val_accumulator)
        for field in metric_fields():
            value = row.get(field)
            if value is None or (isinstance(value, float) and value != value):
                continue
            if field != "step":
                writer.add_scalar(field.replace("_", "/", 1), float(value), step)
        writer.add_scalar("optim/lr", self.optimizer.param_groups[0]["lr"], step)
        if samples_per_second > 0:
            writer.add_scalar("perf/samples_per_second", samples_per_second, step)
        append_metrics(self.metrics_path, row)
        return row

    def _write_status(self, step: int, state: str) -> None:
        _atomic_write_json(
            self.run_dir / "status.json",
            {
                "state": state,
                "step": step,
                "total_steps": self.config.steps,
                "compile": self.compile_status,
                "device": str(self.device),
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def train(self) -> Path:
        config = self.config
        _atomic_write_json(
            self.run_dir / "config.json",
            {"run_config": config.as_dict(), "model_config": asdict(self.model_config)},
        )
        if self.start_step == 0 and not self.metrics_path.exists():
            write_metrics_header(self.metrics_path)
        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        accumulator = MetricAccumulator(self.loss_weights)
        iterator: Iterator[dict[str, torch.Tensor]] = iter(self.train_loader)

        if self.start_step == 0:
            self.save_checkpoint(0)
            self._log_row(0, accumulator, writer, 0.0)

        progress = tqdm(
            range(self.start_step + 1, config.steps + 1),
            initial=self.start_step,
            total=config.steps,
            disable=not config.progress,
            unit="step",
            dynamic_ncols=True,
        )
        window_start = time.monotonic()
        window_samples = 0
        last_step = self.start_step
        for step in progress:
            batch = batch_to_device(next(iterator), self.device)
            output, head_losses, total = self.train_step(batch)
            accumulator.add_batch(output, batch, head_losses, total)
            window_samples += batch["board"].shape[0]
            last_step = step

            if step % self.log_interval == 0 or step == config.steps:
                elapsed = max(time.monotonic() - window_start, 1e-9)
                row = self._log_row(step, accumulator, writer, window_samples / elapsed)
                progress.set_postfix(
                    loss=f"{float(row.get('train_loss', float('nan'))):.4f}",
                    val=f"{float(row.get('val_loss', float('nan'))):.4f}",
                )
                accumulator = MetricAccumulator(self.loss_weights)
                self._write_status(step, "running")
                window_start = time.monotonic()
                window_samples = 0
            if step % config.checkpoint_interval == 0 and step != config.steps:
                self.save_checkpoint(step)

        final_path = self.save_checkpoint(last_step)
        self._write_status(last_step, "finished")
        writer.close()
        return final_path


def train_from_config(config: TrainConfig) -> Path:
    return Trainer(config).train()
