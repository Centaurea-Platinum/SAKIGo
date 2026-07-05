from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model.sakigo_model import config_from_spec  # noqa: E402
from Training.checkpoints import (  # noqa: E402
    load_checkpoint,
    model_from_config,
    restore_model_from_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from Training.common import (  # noqa: E402
    board_sampling_weights,
    board_sizes,
    make_run_dir,
    resolve_root_path,
    step_set,
    training_device,
)
from Training.data import (  # noqa: E402
    TRAIN_SPLIT,
    VAL_SPLIT,
    PinnedBatchKeeper,
    StreamingRulesetAwareBatchDataset,
    StreamingJsonlBuffer,
    batch_to_device,
    data_format_label,
    expand_data_paths,
    is_legacy_jsonl_path,
    make_batch_dataloader,
    scan_jsonl_stream_metadata,
)
from Training.losses import LossWeights, compute_head_losses, weighted_total_loss  # noqa: E402
from Training.metrics import (  # noqa: E402
    MetricAccumulator,
    ProgressBar,
    add_val_confusion,
    append_metrics,
    prefixed,
    progress_line,
    write_metrics_header,
)


def _progress_enabled(args: argparse.Namespace) -> bool:
    if args.progress is not None:
        return bool(args.progress)
    return sys.stdout.isatty()


def _check_eval_batch(batch: dict[str, torch.Tensor]) -> None:
    """Canary: a real batch always has nonzero board planes (boundary/empty planes are set).
    An all-zero board means the input pipeline delivered corrupted data."""
    if float(batch["board"].abs().sum()) == 0.0:
        raise RuntimeError("eval batch canary tripped: all-zero board planes (input pipeline corruption)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAKIGo from precomputed JSONL records.")
    parser.add_argument("--data", nargs="+", required=True, help="Training JSONL files.")
    parser.add_argument("--boards", default="", help="Comma-separated board sizes. Defaults to data.")
    parser.add_argument("--board-sampling-weights", default="", help="n:weight pairs or positional weights.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        default="warmup-cosine",
        choices=("constant", "warmup-cosine"),
        help="Learning-rate schedule. warmup-cosine linearly warms up then cosine decays.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps. 0 derives this from --warmup-fraction.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.03,
        help="Warmup fraction of --steps when --warmup-steps is 0.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Final LR as a fraction of --lr for warmup-cosine.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--loss-eval-batches", type=int, default=4)
    parser.add_argument("--early-eval-steps", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--model-spec", default="model1")
    parser.add_argument("--model-board-size", type=int, default=0)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--resume", default="", help="Checkpoint to resume from.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--stream-buffer-mb",
        type=float,
        default=1024.0,
        help="Decoded-record streaming buffer in MiB. Defaults to 1024.",
    )
    parser.add_argument(
        "--amp",
        default="auto",
        choices=("auto", "bf16", "off"),
        help="Mixed precision: auto/bf16 use bfloat16 autocast on CUDA, off disables it.",
    )
    parser.add_argument(
        "--augment-d4",
        action="store_true",
        help="Apply a random D4 board symmetry to each training sample (for non-equivariant models).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render an in-place progress bar. Defaults to on when stdout is a terminal.",
    )
    parser.add_argument("--wdl-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--ownership-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--budget-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def _amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if args.amp == "off" or device.type != "cuda":
        return None
    return torch.bfloat16


def _autocast(amp_dtype: torch.dtype | None):
    return torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()


def _optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        if (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in lower_name
            or name == "register_seed"
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


def _make_optimizer(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, object] = {"lr": args.lr}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(_optimizer_param_groups(model, args.weight_decay), **kwargs)


def _effective_warmup_steps(args: argparse.Namespace) -> int:
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must be in [0, 1)")
    if args.warmup_steps > 0:
        return min(args.warmup_steps, max(1, args.steps))
    if args.warmup_fraction == 0.0:
        return 0
    return min(max(1, math.ceil(args.steps * args.warmup_fraction)), max(1, args.steps))


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.lr_schedule == "constant":
        return None
    if args.lr_schedule != "warmup-cosine":
        raise ValueError(f"unknown lr schedule: {args.lr_schedule}")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")

    total_steps = max(1, args.steps)
    warmup_steps = _effective_warmup_steps(args)
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_factor(epoch: int) -> float:
        step_number = max(1, epoch + 1)
        if warmup_steps > 0 and step_number <= warmup_steps:
            return step_number / warmup_steps
        progress = min(1.0, max(0.0, (step_number - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict,
) -> None:
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as exc:
        print(
            "warning: optimizer state was not restored "
            f"({exc}); continuing with a fresh optimizer",
            flush=True,
        )


def _restore_scheduler_state(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    checkpoint: dict,
) -> None:
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])


@torch.no_grad()
def balanced_eval_loader(
    model: torch.nn.Module,
    loader_iter: Iterator[dict[str, torch.Tensor]],
    device: torch.device,
    batches: int,
    loss_weights: LossWeights,
    amp_dtype: torch.dtype | None = None,
) -> MetricAccumulator:
    model.eval()
    accumulator = MetricAccumulator(loss_weights)
    for _ in range(max(1, batches)):
        cpu_batch = next(loader_iter)
        batch = batch_to_device(cpu_batch, device, non_blocking=False)
        _check_eval_batch(batch)
        with _autocast(amp_dtype):
            output = model(batch["board"], batch["rules"])
            head_losses = compute_head_losses(output, batch)
            total_loss = weighted_total_loss(head_losses, loss_weights)
        accumulator.add_batch(output, batch, head_losses, total_loss)
    return accumulator


def _train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    loss_weights: LossWeights,
    grad_clip: float,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    with _autocast(amp_dtype):
        output = model(batch["board"], batch["rules"])
        head_losses = compute_head_losses(output, batch)
        loss = weighted_total_loss(head_losses, loss_weights)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return output, head_losses, loss


def _loss_weights(args: argparse.Namespace) -> LossWeights:
    return LossWeights(
        wdl=args.wdl_weight,
        score=args.score_weight,
        ownership=args.ownership_weight,
        policy=args.policy_weight,
        budget=args.budget_weight,
    )


def _prepare_model(
    args: argparse.Namespace,
    boards: list[int],
    device: torch.device,
) -> tuple[torch.nn.Module, object, int, dict | None]:
    resume_checkpoint = None
    if args.resume:
        resume_path = resolve_root_path(args.resume)
        resume_checkpoint = load_checkpoint(resume_path, device)
        model = restore_model_from_checkpoint(
            resume_checkpoint,
            device,
            minimum_board_size=max(boards),
        )
        args.model_board_size = model.config.board_size
        return model, model.config, int(resume_checkpoint.get("step", 0)), resume_checkpoint

    base_config = config_from_spec(args.model_spec)
    model_board_size = args.model_board_size or max(base_config.board_size, max(boards))
    if max(boards) > model_board_size:
        raise ValueError("--model-board-size must cover every training board")
    args.model_board_size = model_board_size
    model_config = config_from_spec(args.model_spec, board_size=model_board_size)
    model = model_from_config(model_config).to(device)
    return model, model_config, 0, None


def _write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    data_paths: list[Path],
    record_count: int,
    train_count: int,
    val_count: int,
    board_weights: dict[int, float],
    device: torch.device,
    model_config: object,
    extra_config: dict[str, object] | None = None,
) -> None:
    config_path = run_dir / "config.json"
    if args.resume and config_path.exists():
        return
    config_path.write_text(
        json.dumps(
            {
                **vars(args),
                "data_paths": [str(path) for path in data_paths],
                "record_count": record_count,
                "train_records": train_count,
                "val_records": val_count,
                "device": str(device),
                "board_sampling_weights": board_weights,
                "model_config": asdict(model_config),
                **(extra_config or {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _stream_buffer_bytes(args: argparse.Namespace) -> int:
    if args.stream_buffer_mb <= 0.0:
        raise ValueError("--stream-buffer-mb must be positive (the eager loading path was removed)")
    byte_count = int(args.stream_buffer_mb * 1024 * 1024)
    if byte_count <= 0:
        raise ValueError("--stream-buffer-mb is too small")
    return byte_count


def _warn_deprecated_legacy_jsonl(data_paths: list[Path]) -> None:
    if any(is_legacy_jsonl_path(path) for path in data_paths):
        print(
            "warning: plain .jsonl training data is deprecated; prefer numbered .jsonl.zst shards",
            file=sys.stderr,
            flush=True,
        )


def _run_training_loop(
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    model_config,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_weights: LossWeights,
    amp_dtype: torch.dtype | None,
    train_loader,
    val_loader,
    run_dir: Path,
    metrics_path: Path,
    start_step: int,
    train_rng: random.Random,
    val_rng: random.Random,
) -> None:
    """The single training loop shared by the streaming and eager data paths."""
    early_eval_steps = step_set(args.early_eval_steps)
    train_metrics = MetricAccumulator(loss_weights)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    progress = ProgressBar(args.steps, start_step, args.batch_size, _progress_enabled(args))
    keeper = PinnedBatchKeeper(device.type == "cuda")
    logged_loss = float("nan")
    try:
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            cpu_batch = next(train_iter)
            batch = batch_to_device(cpu_batch, device)
            keeper.fence(cpu_batch)
            output, head_losses, loss = _train_batch(
                model,
                optimizer,
                batch,
                loss_weights,
                args.grad_clip,
                amp_dtype,
            )
            train_metrics.add_batch(output, batch, head_losses, loss)
            if scheduler is not None:
                scheduler.step()
            progress.render(step, logged_loss)

            should_checkpoint = (
                step in early_eval_steps
                or step % args.checkpoint_interval == 0
                or step == args.steps
            )
            should_log = (
                step == 1
                or step % args.log_interval == 0
                or should_checkpoint
                or step == args.steps
            )
            if should_checkpoint:
                save_checkpoint(
                    model,
                    optimizer,
                    run_dir,
                    step,
                    args,
                    model_config,
                    train_rng,
                    val_rng,
                    scheduler=scheduler,
                )
            if should_log:
                val_metrics = balanced_eval_loader(
                    model,
                    val_iter,
                    device,
                    args.loss_eval_batches,
                    loss_weights,
                    amp_dtype,
                )
                row: dict[str, float | int] = {
                    "step": step,
                    **prefixed("train", train_metrics.averages()),
                    **prefixed("val", val_metrics.averages()),
                }
                logged_loss = float(row["train_loss"])
                add_val_confusion(row, val_metrics)
                append_metrics(metrics_path, row)
                progress.clear()
                print(progress_line(row), flush=True)
                train_metrics.reset()
    finally:
        progress.clear()
        keeper.release_all()


def _run_streaming(args: argparse.Namespace, data_paths: list[Path]) -> None:
    requested_boards = board_sizes(args.boards) if args.boards.strip() else None
    metadata = scan_jsonl_stream_metadata(
        data_paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
        boards=requested_boards,
    )
    boards = board_sizes(args.boards, metadata.board_sizes)
    board_weights = board_sampling_weights(args.board_sampling_weights, boards)

    train_split = TRAIN_SPLIT if metadata.train_count else VAL_SPLIT
    val_split = VAL_SPLIT if metadata.val_count else train_split
    effective_train_count = metadata.train_count if train_split == TRAIN_SPLIT else metadata.val_count
    effective_val_count = metadata.val_count if val_split == VAL_SPLIT else effective_train_count

    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 20_000)
    torch.manual_seed(args.seed)

    device = training_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, model_config, start_step, resume_checkpoint = _prepare_model(args, boards, device)
    loss_weights = _loss_weights(args)
    amp_dtype = _amp_dtype(args, device)
    optimizer = _make_optimizer(model, args, device)
    scheduler = _make_scheduler(optimizer, args)
    if resume_checkpoint is not None:
        _restore_optimizer_state(optimizer, resume_checkpoint)
        _restore_scheduler_state(scheduler, resume_checkpoint)
        restore_rng_state(resume_checkpoint, train_rng, val_rng)

    run_dir = make_run_dir(args.run_dir, args.resume or None)
    metrics_path = run_dir / "metrics.csv"
    if not args.resume or not metrics_path.exists():
        write_metrics_header(metrics_path)

    stream_buffer_bytes = _stream_buffer_bytes(args)
    with StreamingJsonlBuffer(
        paths=data_paths,
        boards=boards,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_buffer_bytes=stream_buffer_bytes,
        metadata=metadata,
    ) as stream:
        stream.prime(args.batch_size)
        stream.build_ruleset_index()
        _write_run_config(
            run_dir,
            args,
            data_paths,
            metadata.record_count,
            effective_train_count,
            effective_val_count,
            board_weights,
            device,
            model_config,
            extra_config={
                "data_loading": "streaming",
                "data_pipeline": "torch_dataloader",
                "data_format": data_format_label(data_paths),
                "stream_train_split": train_split,
                "stream_val_split": val_split,
                "stream_metadata": {
                    "record_count": metadata.record_count,
                    "train_count": metadata.train_count,
                    "val_count": metadata.val_count,
                    "board_counts": {str(key): value for key, value in metadata.board_counts.items()},
                    "ruleset_counts": metadata.ruleset_counts,
                },
                "stream_buffer": stream.stats(),
                "stream_offset_index": stream.supports_offset_index,
            },
        )

        pin = device.type == "cuda"
        train_loader = make_batch_dataloader(
            StreamingRulesetAwareBatchDataset(
                stream,
                train_split,
                args.batch_size,
                train_rng,
                board_weights,
                augment_d4=args.augment_d4,
            ),
            pin_memory=pin,
        )
        val_loader = make_batch_dataloader(
            StreamingRulesetAwareBatchDataset(
                stream,
                val_split,
                args.batch_size,
                val_rng,
                board_weights,
                advance=False,
            ),
            pin_memory=False,
        )
        _run_training_loop(
            args,
            device,
            model,
            model_config,
            optimizer,
            scheduler,
            loss_weights,
            amp_dtype,
            train_loader,
            val_loader,
            run_dir,
            metrics_path,
            start_step,
            train_rng,
            val_rng,
        )

    print(f"run_dir={run_dir}")
    print(f"metrics={metrics_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_paths = expand_data_paths(resolve_root_path(path) for path in args.data)
    _warn_deprecated_legacy_jsonl(data_paths)
    _run_streaming(args, data_paths)


if __name__ == "__main__":
    main()
