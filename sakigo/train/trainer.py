"""The Trainer: standard PyTorch loop with torch.compile, bf16 autocast,
fused AdamW, SequentialLR warmup-cosine, TensorBoard (+ CSV mirror for the
viewer), tqdm, atomic weights_only-loadable checkpoints, and full RNG capture.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sakigo.data import (
    GroupedValidationBatchSampler,
    PreparedDataset,
    SizeGroupedBatchSampler,
    batch_to_device,
    load_manifest,
    make_dataloader,
    prepare_tensor_shards,
)
from sakigo.model import (
    CHECKPOINT_SCHEMA_VERSION,
    SakiGoNet,
    config_from_dict as model_config_from_dict,
    config_from_spec,
)
from sakigo.train.config import TrainConfig, validate_train_config
from sakigo.train.losses import LossWeights, compute_head_losses, weighted_total_loss
from sakigo.train.metrics import (
    MetricAccumulator,
    add_val_confusion,
    append_metrics,
    append_validation_groups,
    metric_fields,
    prefixed,
    write_metrics_header,
    write_validation_group_header,
)

def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def require_finite(label: str, value: torch.Tensor) -> None:
    """Stop before a non-finite result can corrupt optimizer state."""
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"non-finite {label}")


def require_model_finite(model: nn.Module) -> None:
    """Validate every trainable tensor after the optimizer's first update."""
    for name, parameter in model.named_parameters():
        require_finite(f"model parameter {name}", parameter)


def require_optimizer_finite(optimizer: torch.optim.Optimizer) -> None:
    """Validate tensor-valued optimizer state after it is first materialized."""
    for parameter_index, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value):
                require_finite(f"optimizer state {name} for parameter {id(parameter_index)}", value)


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


_EXACT_RESUME_FIELDS = (
    "seed",
    "val_fraction",
    "num_workers",
    "augment_d4",
    "model_spec",
    "steps",
    "batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "lr_schedule",
    "warmup_steps",
    "min_lr_ratio",
    "amp",
    "compile",
    "wdl_weight",
    "score_weight",
    "policy_weight",
    "budget_weight",
)


def _resume_properties(config: TrainConfig, device: torch.device) -> dict[str, object]:
    properties = {name: getattr(config, name) for name in _EXACT_RESUME_FIELDS}
    properties["board_weights"] = (
        sorted((int(size), float(weight)) for size, weight in config.board_weights.items())
        if config.board_weights is not None
        else None
    )
    properties["device"] = str(device)
    return properties


def _prepared_manifest_identity(manifest: Mapping[str, object]) -> str:
    raw_groups = manifest.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("prepared manifest has no groups")
    groups = []
    for group in raw_groups:
        if not isinstance(group, Mapping):
            raise ValueError("prepared manifest has a malformed group")
        groups.append(
            {
                "split": group.get("split"),
                "board_size": group.get("board_size"),
                "count": group.get("count"),
            }
        )
    identity_payload = {
        "prepare_format_version": manifest.get("prepare_format_version"),
        "schema_version": manifest.get("schema_version"),
        "seed": manifest.get("seed"),
        "val_fraction": manifest.get("val_fraction"),
        "split_mode": manifest.get("split_mode"),
        "sources": manifest.get("sources"),
        "ruleset_keys": manifest.get("ruleset_keys"),
        "groups": groups,
    }
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        config = validate_train_config(config)
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
            policy=config.policy_weight,
            budget=config.budget_weight,
        )
        self.run_dir = self._make_run_dir()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.validation_metrics_path = self.run_dir / "validation_metrics.csv"
        self.compile_status = "off"
        self.log_interval = (
            config.checkpoint_interval if config.log_interval <= 0 else config.log_interval
        )

        self.model_config = config_from_spec(config.model_spec)
        self.model = SakiGoNet(self.model_config).to(self.device)
        self.optimizer = self._make_optimizer()
        self.scheduler = make_scheduler(self.optimizer, config)
        self.start_step = 0
        self._optimizer_step_validated = False
        self._pending_sampler_state: dict[str, Any] | None = None
        self._pending_augmentation_state: dict[str, Any] | None = None
        self._pending_prepared_identity: str | None = None
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
        self.validation_batches = 0
        self.validation_cohorts = {}
        self.last_validation_groups = {}
        try:
            compiled = torch.compile(self.model, mode=mode)
            self.compile_status = f"pending_first_step:{self.config.compile}"
            return compiled
        except Exception as error:  # noqa: BLE001 - add context but never fall back
            self.compile_status = f"failed:{error}"
            raise RuntimeError(
                f"model compilation failed for mode {self.config.compile!r}; "
                "eager fallback is disabled"
            ) from error

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
                    validation_data=list(config.validation_data),
                )
        elif config.data:
            prepared_dir = self.run_dir / "prepared"
            prepare_tensor_shards(
                list(config.data),
                prepared_dir,
                seed=config.seed,
                val_fraction=config.val_fraction,
                validation_data=list(config.validation_data),
            )
        else:
            raise ValueError("train config needs data sources or a prepared_dir")
        self.prepared_dir = prepared_dir
        self.prepared_data_identity = _prepared_manifest_identity(
            load_manifest(prepared_dir)
        )
        if (
            self._pending_prepared_identity is not None
            and self.prepared_data_identity != self._pending_prepared_identity
        ):
            raise ValueError(
                "prepared data does not match the checkpoint; exact resume refused"
            )

        self.train_dataset = PreparedDataset(prepared_dir, "train", augment_d4=config.augment_d4)
        train_sampler = SizeGroupedBatchSampler(
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
        self.train_sampler = train_sampler
        if self._pending_sampler_state is not None:
            self.train_sampler.load_state_dict(self._pending_sampler_state)
        self.train_dataset.load_augmentation_state_dict(
            self._pending_augmentation_state
        )
        try:
            self.val_dataset: PreparedDataset | None = PreparedDataset(prepared_dir, "val")
        except ValueError:
            self.val_dataset = None
            self.val_loader = None
            return
        val_sampler = GroupedValidationBatchSampler(
            self.val_dataset,
            config.batch_size,
            seed=config.seed + 1,
            length=config.val_batches,
            fixed=config.val_fixed,
        )
        self.validation_cohorts = {
            (cohort.board_size, cohort.ruleset_code): cohort
            for cohort in val_sampler.cohorts
        }
        self.validation_batches = len(val_sampler)
        self.val_loader = make_dataloader(
            self.val_dataset,
            val_sampler,
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
            "resume_properties": _resume_properties(self.config, self.device),
            "prepared_data_identity": self.prepared_data_identity,
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()
        payload["sampler_state"] = self.train_sampler.state_dict()
        payload["augmentation_state"] = self.train_dataset.augmentation_state_dict()
        payload["sampler_state_exact"] = self.config.num_workers == 0
        return payload

    def save_checkpoint(self, step: int) -> Path:
        if type(step) is not int or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"step_{step:06}.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self._checkpoint_payload(step), tmp)
        tmp.replace(path)
        return path

    def _resume(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        version = payload.get("checkpoint_schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema {version!r} is incompatible; expected "
                f"{CHECKPOINT_SCHEMA_VERSION} for the book-only no-ownership model"
            )
        if payload.get("sampler_state_exact") is not True:
            raise ValueError(
                "checkpoint sampler state is not exact; resume requires num_workers=0"
            )
        if self.config.num_workers != 0:
            raise ValueError("exact resume requires current num_workers=0")
        saved_properties = payload.get("resume_properties")
        if not isinstance(saved_properties, Mapping):
            raise ValueError("checkpoint has no exact-resume properties")
        current_properties = _resume_properties(self.config, self.device)
        if dict(saved_properties) != current_properties:
            changed = sorted(
                key
                for key in set(saved_properties).union(current_properties)
                if saved_properties.get(key) != current_properties.get(key)
            )
            raise ValueError(
                "checkpoint resume properties changed: " + ", ".join(changed)
            )
        model_config = model_config_from_dict(payload["model_config"])
        if asdict(model_config) != asdict(self.model_config):
            self.model_config = model_config
            self.model = SakiGoNet(model_config).to(self.device)
            self.optimizer = self._make_optimizer()
            self.scheduler = make_scheduler(self.optimizer, self.config)
        model_state = payload.get("model_state")
        optimizer_state = payload.get("optimizer_state")
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise ValueError("checkpoint is missing model or optimizer state")
        self.model.load_state_dict(model_state)
        try:
            self.optimizer.load_state_dict(optimizer_state)
        except ValueError as error:
            raise RuntimeError(
                "optimizer state restore failed; exact resume cannot continue"
            ) from error
        if self.scheduler is not None:
            if "scheduler_state" not in payload:
                raise ValueError("checkpoint is missing scheduler state")
            self.scheduler.load_state_dict(payload["scheduler_state"])
        rng = payload.get("rng")
        if not isinstance(rng, Mapping) or "python" not in rng or "torch" not in rng:
            raise ValueError("checkpoint is missing RNG state")
        python_state = rng["python"]
        if not isinstance(python_state, (tuple, list)) or len(python_state) != 3:
            raise ValueError("checkpoint has invalid Python RNG state")
        random.setstate(
            (python_state[0], tuple(python_state[1]), python_state[2])
        )
        torch_state = rng["torch"]
        if not torch.is_tensor(torch_state):
            raise ValueError("checkpoint has invalid torch RNG state")
        torch.set_rng_state(torch_state.cpu().to(torch.uint8))
        cuda_state = rng.get("cuda")
        if self.device.type == "cuda":
            if not isinstance(cuda_state, (tuple, list)):
                raise ValueError("checkpoint is missing CUDA RNG state")
            torch.cuda.set_rng_state_all(
                [state.cpu().to(torch.uint8) for state in cuda_state]
            )
        raw_step = payload.get("step")
        if type(raw_step) is not int or raw_step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        self.start_step = raw_step
        if self.start_step > self.config.steps:
            raise ValueError(
                f"checkpoint step {self.start_step} exceeds configured steps "
                f"{self.config.steps}"
            )
        sampler_state = payload.get("sampler_state")
        if not isinstance(sampler_state, Mapping):
            raise ValueError("checkpoint is missing sampler state")
        self._pending_sampler_state = dict(sampler_state)
        augmentation_state = payload.get("augmentation_state")
        if self.config.augment_d4 and not isinstance(augmentation_state, Mapping):
            raise ValueError("checkpoint is missing augmentation RNG state")
        if not self.config.augment_d4 and augmentation_state is not None:
            raise ValueError("checkpoint has unexpected augmentation RNG state")
        self._pending_augmentation_state = (
            dict(augmentation_state)
            if isinstance(augmentation_state, Mapping)
            else None
        )
        prepared_identity = payload.get("prepared_data_identity")
        if not isinstance(prepared_identity, str) or not prepared_identity:
            raise ValueError("checkpoint is missing prepared-data identity")
        self._pending_prepared_identity = prepared_identity

    # -- steps ---------------------------------------------------------------

    def _autocast(self):
        if self.config.amp != "off" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
    ]:
        pending_compiled_step = self.compile_status.startswith("pending_first_step:")
        try:
            self.compiled_model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.compiled_model(batch["board"], batch["rules"])
                head_losses = compute_head_losses(output, batch)
                total = weighted_total_loss(head_losses, self.loss_weights)
            require_finite("training loss", total)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip if self.config.grad_clip > 0 else float("inf"),
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            if not self._optimizer_step_validated:
                require_model_finite(self.model)
                require_optimizer_finite(self.optimizer)
                self._optimizer_step_validated = True
            if pending_compiled_step:
                self.compile_status = f"enabled:{self.config.compile}"
            if self.scheduler is not None:
                self.scheduler.step()
            return output, head_losses, total
        except BaseException as error:
            if pending_compiled_step:
                self.compile_status = f"failed_first_step:{type(error).__name__}:{error}"
                raise RuntimeError(
                    "compiled first optimizer step failed; eager fallback is disabled"
                ) from error
            raise

    @torch.no_grad()
    def evaluate(self, batches: int) -> MetricAccumulator:
        accumulator = MetricAccumulator(self.loss_weights)
        group_accumulators: dict[tuple[int, int], MetricAccumulator] = {}
        self.last_validation_groups = group_accumulators
        if self.val_loader is None or batches <= 0:
            return accumulator
        self.compiled_model.eval()
        iterator = iter(self.val_loader)
        for _ in range(batches):
            try:
                batch = batch_to_device(next(iterator), self.device)
            except StopIteration:  # fixed val sampler is finite
                break
            try:
                with self._autocast():
                    output = self.compiled_model(batch["board"], batch["rules"])
                    head_losses = compute_head_losses(output, batch)
                    total = weighted_total_loss(head_losses, self.loss_weights)
            except BaseException as error:
                if self.compile_status.startswith("pending_first_step:"):
                    self.compile_status = (
                        f"failed_initial_forward:{type(error).__name__}:{error}"
                    )
                    raise RuntimeError(
                        "compiled initial validation forward failed; "
                        "eager fallback is disabled"
                    ) from error
                raise
            require_finite("validation loss", total)
            accumulator.add_batch(output, batch, head_losses, total)
            board_sizes = torch.unique(batch["board_size"])
            ruleset_codes = torch.unique(batch["ruleset_code"])
            if board_sizes.numel() != 1 or ruleset_codes.numel() != 1:
                raise RuntimeError(
                    "validation batch crossed a board-size/ruleset cohort boundary"
                )
            cohort_key = (int(board_sizes.item()), int(ruleset_codes.item()))
            group_accumulator = group_accumulators.setdefault(
                cohort_key, MetricAccumulator(self.loss_weights)
            )
            group_accumulator.add_batch(output, batch, head_losses, total)
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
        val_accumulator = self.evaluate(self.validation_batches)
        if val_accumulator.steps:
            row.update(prefixed("val", val_accumulator.averages()))
            add_val_confusion(row, val_accumulator)
            group_rows: list[dict[str, object]] = []
            for cohort_key, accumulator in sorted(self.last_validation_groups.items()):
                cohort = self.validation_cohorts[cohort_key]
                averages = accumulator.averages()
                averages.pop("records", None)
                group_rows.append(
                    {
                        "step": step,
                        "board_size": cohort.board_size,
                        "ruleset_name": cohort.ruleset_name,
                        "komi": cohort.komi,
                        "ruleset_key": cohort.ruleset_key,
                        "records": accumulator.records,
                        "batches": accumulator.steps,
                        **averages,
                    }
                )
                for metric, value in averages.items():
                    if metric == "records" or value != value:
                        continue
                    writer.add_scalar(
                        f"val_groups/{metric}/{cohort.metric_name}", value, step
                    )
            append_validation_groups(self.validation_metrics_path, group_rows)
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

    def _write_status(self, step: int, state: str, *, error: str | None = None) -> None:
        payload = {
            "state": state,
            "step": step,
            "total_steps": self.config.steps,
            "compile": self.compile_status,
            "device": str(self.device),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        if error is not None:
            payload["error"] = error
        _atomic_write_json(
            self.run_dir / "status.json",
            payload,
        )

    def train(self) -> Path:
        config = self.config
        _atomic_write_json(
            self.run_dir / "config.json",
            {"run_config": config.as_dict(), "model_config": asdict(self.model_config)},
        )
        if self.start_step == 0 and not self.metrics_path.exists():
            write_metrics_header(self.metrics_path)
        if not self.validation_metrics_path.exists():
            write_validation_group_header(self.validation_metrics_path)
        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        last_step = self.start_step
        progress = None
        try:
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
            for step in progress:
                batch = batch_to_device(next(iterator), self.device)
                output, head_losses, total = self.train_step(batch)
                accumulator.add_batch(output, batch, head_losses, total)
                window_samples += batch["board"].shape[0]
                last_step = step

                if step % self.log_interval == 0 or step == config.steps:
                    elapsed = max(time.monotonic() - window_start, 1e-9)
                    row = self._log_row(
                        step, accumulator, writer, window_samples / elapsed
                    )
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
            return final_path
        except BaseException as error:
            self._write_status(
                last_step,
                "failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            if progress is not None:
                progress.close()
            writer.close()


def train_from_config(config: TrainConfig) -> Path:
    return Trainer(config).train()
