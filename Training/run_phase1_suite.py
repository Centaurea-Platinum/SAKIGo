"""Sequential Phase 1 training suite: sweep batch size per spec, then train each spec.

For every model spec (default: model1 and model2) this script
1. benchmarks candidate batch sizes on real records (bf16, fused AdamW, grad clip),
2. picks the highest-throughput batch that fits, and
3. launches Training.train as a subprocess with an equal samples-seen budget
   (--epochs over the train split), progress bar on, and D4 augmentation enabled
   automatically for non-equivariant specs.

Example:
    uv run python -m Training.run_phase1_suite --data Training/data/katago_phase1_20260705_shards
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model.sakigo_model import config_from_spec  # noqa: E402
from Training.checkpoints import model_from_config  # noqa: E402
from Training.common import resolve_root_path, training_device  # noqa: E402
from Training.data import (  # noqa: E402
    build_groups,
    collate,
    expand_data_paths,
    open_jsonl_text,
    record_from_json,
    sample_batch,
    scan_jsonl_stream_metadata,
)
from Training.losses import LossWeights  # noqa: E402
from Training.train import _make_optimizer, _train_batch  # noqa: E402

DEFAULT_SPECS = "model1,model2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="Phase 1 JSONL/ZSTD shard file(s), glob, or directory.")
    parser.add_argument("--specs", default=DEFAULT_SPECS, help="Comma-separated model specs to train.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Samples-seen budget in train-split epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Fixed batch size for every spec (a controlled variable across the A/B).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Instead of the fixed batch size, pick the fastest per spec from --batch-candidates.",
    )
    parser.add_argument("--batch-candidates", default="128,256,512,1024")
    parser.add_argument("--sweep-records", type=int, default=512, help="Records loaded for the benchmark sweep.")
    parser.add_argument("--sweep-steps", type=int, default=20, help="Timed steps per candidate (after 3 warmup).")
    parser.add_argument("--sweep-max-seconds", type=float, default=10.0, help="Time cap on each candidate's timed loop.")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.8,
        help="Skip batch sizes whose peak allocation would exceed this fraction of dedicated VRAM "
        "(prevents WDDM shared-memory paging on Windows).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stream-buffer-mb", type=float, default=1024.0)
    parser.add_argument("--run-prefix", default="", help="Run dir prefix; default phase1_<timestamp>.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        default="warmup-cosine",
        choices=("constant", "warmup-cosine"),
        help="Learning-rate schedule forwarded to Training.train.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--loss-eval-batches", type=int, default=4)
    return parser.parse_args(argv)


def load_sweep_records(paths: list[Path], count: int) -> list:
    records = []
    for path in paths:
        with open_jsonl_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(records) >= count:
                    break
                stripped = line.strip()
                if stripped:
                    records.append(record_from_json(json.loads(stripped), path, line_number))
        if len(records) >= count:
            break
    if not records:
        raise ValueError("no records read from data paths")
    return records


def benchmark_spec_batch(
    spec: str,
    records: list,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    timed_steps: int,
    memory_budget: int,
    max_seconds: float,
    lr: float,
    weight_decay: float,
) -> tuple[float | None, int, str]:
    """Benchmark one (spec, batch size). Returns (samples/s or None, peak bytes, reason).

    The peak-allocation check after the first step catches oversized batches
    before Windows/WDDM silently spills them into shared system memory and
    runs them at paging speed (that spill never raises OutOfMemoryError).
    """
    board_size = max(record.board_size for record in records)
    groups = build_groups(records)
    rng = random.Random(0)
    model = None
    optimizer = None
    try:
        config = config_from_spec(spec, board_size=board_size)
        model = model_from_config(config).to(device)
        model.train()
        optimizer_args = argparse.Namespace(lr=lr, weight_decay=weight_decay, cuda_graphs=False)
        optimizer = _make_optimizer(model, optimizer_args, device)
        loss_weights = LossWeights()
        batch = collate(sample_batch(groups, batch_size, rng), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _train_batch(model, optimizer, batch, loss_weights, 1.0, amp_dtype)
        peak = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if memory_budget and peak > memory_budget:
                return None, peak, "over_budget"
        for _ in range(2):
            _train_batch(model, optimizer, batch, loss_weights, 1.0, amp_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        completed = 0
        for _ in range(timed_steps):
            _train_batch(model, optimizer, batch, loss_weights, 1.0, amp_dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            completed += 1
            if time.monotonic() - start > max_seconds:
                break
        elapsed = time.monotonic() - start
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated()
        return batch_size * completed / elapsed, peak, "ok"
    except torch.OutOfMemoryError:
        return None, 0, "oom"
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def final_val_loss(run_dir: Path) -> float | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    try:
        return float(rows[-1]["val_loss"])
    except (KeyError, ValueError):
        return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_paths = expand_data_paths(resolve_root_path(path) for path in args.data)
    specs = [part.strip() for part in args.specs.split(",") if part.strip()]
    candidates = [int(part) for part in args.batch_candidates.split(",") if part.strip()]
    device = training_device("auto")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    run_prefix = args.run_prefix or time.strftime("phase1_%Y%m%d_%H%M%S")

    data_label = str(data_paths[0]) if len(data_paths) == 1 else f"{len(data_paths)} files"
    print(f"device={device}  data={data_label}")
    print("Scanning data for split counts (one full pass)...", flush=True)
    metadata = scan_jsonl_stream_metadata(data_paths, val_fraction=args.val_fraction, seed=args.seed)
    train_count = metadata.train_count or metadata.record_count
    print(f"records={metadata.record_count}  train={metadata.train_count}  val={metadata.val_count}")

    sweep_records = load_sweep_records(data_paths, args.sweep_records) if args.sweep else []
    memory_budget = 0
    if args.sweep and device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_budget = int(total_memory * args.memory_fraction)
        print(f"memory budget: {memory_budget / 2**30:.1f} GiB of {total_memory / 2**30:.1f} GiB dedicated VRAM")
    plans: list[dict] = []
    for spec in specs:
        architecture = config_from_spec(spec).architecture
        augment = architecture != "SakiGoModel"
        best_batch, best_rate = args.batch_size, 0.0
        if args.sweep:
            print(f"\n=== {spec} ({architecture}) batch sweep ===")
            best_batch = None
            previous: tuple[int, int] | None = None  # (batch, peak bytes)
            for batch_size in candidates:
                if previous is not None and memory_budget:
                    projected = previous[1] * batch_size / previous[0]
                    if projected > memory_budget:
                        print(
                            f"  batch {batch_size:>5}: skipped "
                            f"(projected {projected / 2**30:.1f} GiB > budget {memory_budget / 2**30:.1f} GiB)"
                        )
                        break
                rate, peak, reason = benchmark_spec_batch(
                    spec,
                    sweep_records,
                    batch_size,
                    device,
                    amp_dtype,
                    args.sweep_steps,
                    memory_budget,
                    args.sweep_max_seconds,
                    args.lr,
                    args.weight_decay,
                )
                if reason == "oom":
                    print(f"  batch {batch_size:>5}: OOM, skipping larger sizes")
                    break
                if reason == "over_budget":
                    print(
                        f"  batch {batch_size:>5}: over memory budget "
                        f"({peak / 2**30:.1f} GiB > {memory_budget / 2**30:.1f} GiB), skipping larger sizes"
                    )
                    break
                previous = (batch_size, peak)
                peak_text = f", peak {peak / 2**30:.1f} GiB" if peak else ""
                if rate > best_rate:
                    best_batch, best_rate = batch_size, rate
                    print(f"  batch {batch_size:>5}: {rate:,.0f} samples/s{peak_text}  <- best")
                elif rate < 0.97 * best_rate:
                    print(f"  batch {batch_size:>5}: {rate:,.0f} samples/s{peak_text}  (declining, stopping sweep)")
                    break
                else:
                    print(f"  batch {batch_size:>5}: {rate:,.0f} samples/s{peak_text}")
            if best_batch is None:
                print(f"  no runnable batch size for {spec}; skipping")
                continue
        steps = max(1, math.ceil(train_count * args.epochs / best_batch))
        plans.append(
            {
                "spec": spec,
                "batch": best_batch,
                "rate": best_rate,
                "steps": steps,
                "augment": augment,
                "run_dir": ROOT / "Training" / "runs" / f"{run_prefix}_{spec}",
            }
        )

    print("\n=== Training plan ===")
    for plan in plans:
        eta_text = f"  est {plan['steps'] * plan['batch'] / plan['rate'] / 60.0:,.0f} min" if plan["rate"] else ""
        print(
            f"  {plan['spec']:<24} batch={plan['batch']:<5} steps={plan['steps']:<6} "
            f"augment_d4={plan['augment']} schedule={args.lr_schedule}{eta_text}"
        )

    results = []
    for plan in plans:
        command = [
            sys.executable,
            "-m",
            "Training.train",
            "--data",
            *(str(path) for path in data_paths),
            "--stream-buffer-mb",
            str(args.stream_buffer_mb),
            "--steps",
            str(plan["steps"]),
            "--batch-size",
            str(plan["batch"]),
            "--model-spec",
            plan["spec"],
            "--val-fraction",
            str(args.val_fraction),
            "--seed",
            str(args.seed),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--lr-schedule",
            args.lr_schedule,
            "--warmup-steps",
            str(args.warmup_steps),
            "--warmup-fraction",
            str(args.warmup_fraction),
            "--min-lr-ratio",
            str(args.min_lr_ratio),
            "--log-interval",
            str(args.log_interval),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
            "--loss-eval-batches",
            str(args.loss_eval_batches),
            "--run-dir",
            str(plan["run_dir"]),
            "--progress",
        ]
        if plan["augment"]:
            command.append("--augment-d4")
        print(f"\n=== Training {plan['spec']} ===")
        print("  " + " ".join(command[3:]), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        results.append((plan, completed.returncode))

    print("\n=== Suite summary ===")
    for plan, returncode in results:
        status = "ok" if returncode == 0 else f"FAILED (exit {returncode})"
        val_loss = final_val_loss(plan["run_dir"])
        val_text = f"val_loss={val_loss:.4f}" if val_loss is not None else "val_loss=?"
        print(f"  {plan['spec']:<24} {status:<18} {val_text}  {plan['run_dir']}")


if __name__ == "__main__":
    main()
