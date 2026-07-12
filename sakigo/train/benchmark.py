"""Batch-size benchmark: find the largest batch that fits the VRAM budget.

Ported from the legacy run_phase1_suite sweep. The peak-allocation check
after the first step catches oversized batches before Windows/WDDM silently
spills them into shared system memory and runs them at paging speed (that
spill never raises OutOfMemoryError).

Usage:
  uv run python -m sakigo.train.benchmark --prepared-dir <dir> --spec balanced
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from sakigo.data import PreparedDataset, RulesetBalancedBatchSampler, collate_prepared
from sakigo.data.records import batch_to_device
from sakigo.model import SakiGoNet, config_from_spec
from sakigo.train.losses import LossWeights, compute_head_losses, weighted_total_loss
from sakigo.train.trainer import optimizer_param_groups, resolve_device


def _train_step(model, optimizer, batch, loss_weights, amp_dtype) -> None:
    optimizer.zero_grad(set_to_none=True)
    if amp_dtype is not None:
        with torch.autocast("cuda", dtype=amp_dtype):
            output = model(batch["board"], batch["rules"])
            total = weighted_total_loss(compute_head_losses(output, batch), loss_weights)
    else:
        output = model(batch["board"], batch["rules"])
        total = weighted_total_loss(compute_head_losses(output, batch), loss_weights)
    total.backward()
    optimizer.step()


def benchmark_batch_size(
    spec: str,
    dataset: PreparedDataset,
    batch_size: int,
    device: torch.device,
    *,
    timed_steps: int = 8,
    memory_budget: int = 0,
    max_seconds: float = 20.0,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    compile_mode: str = "off",
) -> tuple[float | None, int, str]:
    """Benchmark one (spec, batch size). Returns (samples/s or None, peak bytes, reason)."""
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    board_size = max(dataset.board_sizes)
    model = None
    optimizer = None
    try:
        config = config_from_spec(spec, board_size=max(board_size, config_from_spec(spec).board_size))
        model = SakiGoNet(config).to(device)
        if compile_mode != "off":
            mode = None if compile_mode == "default" else compile_mode
            model = torch.compile(model, mode=mode)
        model.train()
        optimizer = torch.optim.AdamW(
            optimizer_param_groups(model, weight_decay),
            lr=lr,
            **({"fused": True} if device.type == "cuda" else {}),
        )
        loss_weights = LossWeights()
        sampler = RulesetBalancedBatchSampler(dataset, batch_size, seed=0, length=1)
        indices = next(iter(sampler))
        batch = batch_to_device(collate_prepared(dataset.fetch_batch(indices)), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _train_step(model, optimizer, batch, loss_weights, amp_dtype)
        peak = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if memory_budget and peak > memory_budget:
                return None, peak, "over_budget"
        for _ in range(2):
            _train_step(model, optimizer, batch, loss_weights, amp_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        completed = 0
        for _ in range(timed_steps):
            _train_step(model, optimizer, batch, loss_weights, amp_dtype)
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
    except Exception as error:  # noqa: BLE001 - benchmark records rejected candidates
        return None, 0, f"error:{type(error).__name__}:{error}"
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep batch sizes for a model spec.")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--spec", default="balanced")
    parser.add_argument("--batch-sizes", default="8,16,24,32,48,64,96,128")
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--budget-fraction", type=float, default=0.85)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--compile", choices=("off", "default", "reduce-overhead"), default="reduce-overhead"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    budget = 0
    if device.type == "cuda":
        budget = int(torch.cuda.get_device_properties(device).total_memory * args.budget_fraction)
    dataset = PreparedDataset(args.prepared_dir, "train")

    results = []
    best: tuple[int, float] | None = None
    for batch_size in (int(part) for part in args.batch_sizes.split(",") if part.strip()):
        rate, peak, reason = benchmark_batch_size(
            args.spec,
            dataset,
            batch_size,
            device,
            timed_steps=args.timed_steps,
            memory_budget=budget,
            compile_mode=args.compile,
        )
        results.append(
            {"batch_size": batch_size, "samples_per_second": rate, "peak_bytes": peak, "reason": reason}
        )
        rate_text = f"{rate:8.1f}/s" if rate else f"   {reason}"
        print(f"batch {batch_size:4d}: {rate_text}  peak {peak / 2**30:.2f} GiB")
        if rate is not None and (best is None or rate > best[1]):
            best = (batch_size, rate)
        if reason in ("over_budget", "oom"):
            break
    if best is not None:
        print(f"best: batch {best[0]} at {best[1]:.1f} samples/s")
    print(json.dumps({"spec": args.spec, "device": str(device), "results": results}))


if __name__ == "__main__":
    main()
