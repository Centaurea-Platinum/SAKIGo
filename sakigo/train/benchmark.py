"""Deterministic batch safety preflight and throughput benchmark.

Every compiled candidate is compared with an eager BF16 reference created from
the same model state and exact batch.  The comparison spans the complete first
optimizer step and the following forward pass, which prevents a candidate that
silently corrupts parameters or Adam state from being selected for training.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sakigo.data import PreparedDataset, SizeGroupedBatchSampler, collate_prepared
from sakigo.data.records import batch_to_device
from sakigo.model import SakiGoModelConfig, SakiGoNet, config_from_spec
from sakigo.model.config import config_from_dict
from sakigo.train.losses import LossWeights, compute_head_losses, weighted_total_loss
from sakigo.train.trainer import optimizer_param_groups, require_finite, resolve_device


_PARITY_TOLERANCES: dict[str, tuple[float, float]] = {
    "outputs": (0.02, 0.025),
    "head_losses": (0.01, 0.01),
    "total_loss": (0.01, 0.01),
    "gradients": (0.05, 0.006),
    "parameters": (0.005, 0.00005),
    "optimizer_state": (0.05, 0.005),
    "next_outputs": (0.02, 0.025),
}

# Large tensor collections should not fail because one otherwise-benign BF16
# reduction outlier barely exceeds the elementwise tolerance.  These sections
# instead require small aggregate error, a small violating tail, and a hard cap
# that still rejects catastrophic compiler corruption.  Scalar loss checks and
# optimizer state retain strict elementwise comparison.
_DISTRIBUTION_PARITY_LIMITS: dict[str, dict[str, float]] = {
    "outputs": {
        "max_normalized_rmse": 1.0,
        "max_fraction_over_tolerance": 0.025,
        "max_scaled_error": 5.0,
    },
    "gradients": {
        "max_normalized_rmse": 1.0,
        "max_fraction_over_tolerance": 1e-5,
        "max_scaled_error": 5.0,
    },
    "parameters": {
        "max_normalized_rmse": 1.0,
        "max_fraction_over_tolerance": 1e-5,
        "max_scaled_error": 5.0,
    },
    "next_outputs": {
        "max_normalized_rmse": 1.0,
        "max_fraction_over_tolerance": 0.025,
        "max_scaled_error": 5.0,
    },
}


def _amp_dtype(device: torch.device, amp: str) -> torch.dtype | None:
    return torch.bfloat16 if device.type == "cuda" and amp != "off" else None


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is not None:
        return torch.autocast(device.type, dtype=amp_dtype)
    return torch.autocast("cpu", enabled=False)


def _make_optimizer(
    model: torch.nn.Module,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": lr}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(optimizer_param_groups(model, weight_decay), **kwargs)


def _make_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LinearLR | None:
    if warmup_steps <= 0:
        return None
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _clone_mapping(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: _clone_tensor(value) for name, value in values.items()}


def _clone_state_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return deepcopy(value)


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: _clone_tensor(parameter)
        for name, parameter in model.named_parameters()
    }


def _gradient_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: _clone_tensor(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _optimizer_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                snapshot[f"{name}:{state_name}"] = _clone_tensor(value)
    return snapshot


def _require_mapping_finite(label: str, values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        require_finite(f"{label} {name}", value)


def _captured_train_step(
    forward_model: torch.nn.Module,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    loss_weights: LossWeights,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    grad_clip: float,
    *,
    require_next_finite: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    forward_model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, amp_dtype):
        output = forward_model(batch["board"], batch["rules"])
        head_losses = compute_head_losses(output, batch)
        total = weighted_total_loss(
            head_losses,
            loss_weights,
            board_area=int(batch["board"].shape[-2] * batch["board"].shape[-1]),
        )
    outputs = _clone_mapping(output)
    losses = _clone_mapping(head_losses)
    total_snapshot = {"total": _clone_tensor(total)}
    _require_mapping_finite("output", outputs)
    _require_mapping_finite("head loss", losses)
    _require_mapping_finite("total loss", total_snapshot)

    total.backward()
    gradients = _gradient_snapshot(base_model)
    _require_mapping_finite("gradient", gradients)
    torch.nn.utils.clip_grad_norm_(
        base_model.parameters(),
        grad_clip if grad_clip > 0 else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()
    parameters = _parameter_snapshot(base_model)
    optimizer_state = _optimizer_snapshot(base_model, optimizer)
    _require_mapping_finite("post-step parameter", parameters)
    _require_mapping_finite("post-step optimizer state", optimizer_state)

    with _autocast(device, amp_dtype):
        next_output = forward_model(batch["board"], batch["rules"])
    next_outputs = _clone_mapping(next_output)
    if require_next_finite:
        _require_mapping_finite("next output", next_outputs)
    return {
        "outputs": outputs,
        "head_losses": losses,
        "total_loss": total_snapshot,
        "gradients": gradients,
        "parameters": parameters,
        "optimizer_state": optimizer_state,
        "next_outputs": next_outputs,
    }


def _train_step(
    forward_model: torch.nn.Module,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    loss_weights: LossWeights,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    grad_clip: float,
) -> None:
    forward_model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, amp_dtype):
        output = forward_model(batch["board"], batch["rules"])
        total = weighted_total_loss(
            compute_head_losses(output, batch),
            loss_weights,
            board_area=int(batch["board"].shape[-2] * batch["board"].shape[-1]),
        )
    require_finite("benchmark loss", total)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        base_model.parameters(),
        grad_clip if grad_clip > 0 else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()


def _compare_mapping(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
    distribution_limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        return {
            "ok": False,
            "reason": "tensor_keys_differ",
            "missing": sorted(reference.keys() - candidate.keys()),
            "unexpected": sorted(candidate.keys() - reference.keys()),
        }
    worst_name = ""
    worst_scaled_error = 0.0
    worst_max_abs = 0.0
    element_count = 0
    elements_over_tolerance = 0
    normalized_error_squared = 0.0
    difference_squared = 0.0
    reference_squared = 0.0
    for name in reference:
        expected = reference[name]
        actual = candidate[name]
        if expected.shape != actual.shape:
            return {
                "ok": False,
                "reason": "tensor_shapes_differ",
                "tensor": name,
                "reference_shape": list(expected.shape),
                "candidate_shape": list(actual.shape),
            }
        if not bool(torch.isfinite(expected).all().item()) or not bool(
            torch.isfinite(actual).all().item()
        ):
            return {
                "ok": False,
                "reason": "non_finite_tensor",
                "tensor": name,
            }
        expected_float = expected.float()
        actual_float = actual.float()
        difference = (actual_float - expected_float).abs()
        tolerance = atol + rtol * expected_float.abs()
        normalized_error = difference / tolerance.clamp_min(1e-12)
        scaled_error = float(normalized_error.max().item())
        max_abs = float(difference.max().item())
        element_count += difference.numel()
        elements_over_tolerance += int((normalized_error > 1.0).sum().item())
        normalized_error_squared += float(
            normalized_error.double().square().sum().item()
        )
        difference_squared += float(difference.double().square().sum().item())
        reference_squared += float(expected_float.double().square().sum().item())
        if scaled_error > worst_scaled_error:
            worst_name = name
            worst_scaled_error = scaled_error
            worst_max_abs = max_abs
    fraction_over_tolerance = elements_over_tolerance / max(element_count, 1)
    normalized_rmse = (normalized_error_squared / max(element_count, 1)) ** 0.5
    relative_l2 = (difference_squared / max(reference_squared, 1e-30)) ** 0.5
    if distribution_limits is None:
        ok = worst_scaled_error <= 1.0
        decision = "strict_max"
    else:
        ok = (
            normalized_rmse <= distribution_limits["max_normalized_rmse"]
            and fraction_over_tolerance
            <= distribution_limits["max_fraction_over_tolerance"]
            and worst_scaled_error <= distribution_limits["max_scaled_error"]
        )
        decision = "distribution"
    return {
        "ok": ok,
        "decision": decision,
        "worst_tensor": worst_name,
        "max_scaled_error": worst_scaled_error,
        "max_absolute_error": worst_max_abs,
        "elements": element_count,
        "elements_over_tolerance": elements_over_tolerance,
        "fraction_over_tolerance": fraction_over_tolerance,
        "normalized_rmse": normalized_rmse,
        "relative_l2": relative_l2,
        "rtol": rtol,
        "atol": atol,
        "distribution_limits": distribution_limits,
    }


def compare_step_snapshots(
    reference: dict[str, dict[str, torch.Tensor]],
    candidate: dict[str, dict[str, torch.Tensor]],
    *,
    parameter_atol: float | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for section, (rtol, default_atol) in _PARITY_TOLERANCES.items():
        atol = (
            parameter_atol
            if section == "parameters" and parameter_atol is not None
            else default_atol
        )
        checks[section] = _compare_mapping(
            reference[section],
            candidate[section],
            rtol=rtol,
            atol=atol,
            distribution_limits=_DISTRIBUTION_PARITY_LIMITS.get(section),
        )
    failed = [name for name, result in checks.items() if not result["ok"]]
    return {"ok": not failed, "failed_checks": failed, "checks": checks}


def _combine_phase_reports(reports: dict[str, dict[str, Any]], amp_dtype: torch.dtype | None) -> dict[str, Any]:
    failed = [
        f"{phase}.{check}"
        for phase, report in reports.items()
        for check in report.get("failed_checks", ())
    ]
    return {
        "ok": not failed,
        "failed_checks": failed,
        "phases": reports,
        "mode": f"eager_vs_compiled_{'bf16' if amp_dtype is torch.bfloat16 else 'fp32'}",
    }


def _capture_peak_memory(result: dict[str, Any], device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()


def _model_config(
    spec: str,
    board_size: int,
    override: SakiGoModelConfig | None,
) -> SakiGoModelConfig:
    if override is None:
        return config_from_spec(
            spec,
            board_size=max(board_size, config_from_spec(spec).board_size),
        )
    if override.board_size < board_size:
        return replace(override, board_size=board_size)
    return override


def benchmark_batch_candidate(
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
    grad_clip: float = 1.0,
    warmup_steps: int = 100,
    compile_mode: str = "off",
    amp: str = "auto",
    loss_weights: LossWeights | None = None,
    seed: int = 0,
    model_config: SakiGoModelConfig | None = None,
) -> dict[str, Any]:
    """Run parity preflight and benchmark one model/batch candidate."""
    torch.set_float32_matmul_precision("high")
    weights = loss_weights or LossWeights()
    amp_dtype = _amp_dtype(device, amp)
    board_size = max(dataset.board_sizes)
    config = _model_config(spec, board_size, model_config)
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "samples_per_second": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "reason": "not_run",
        "safety": {"ok": False},
    }
    eager_model = None
    eager_optimizer = None
    prime_model = None
    prime_optimizer = None
    candidate_model = None
    candidate_forward = None
    candidate_optimizer = None
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        initial_model = SakiGoNet(config)
        initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in initial_model.state_dict().items()
        }
        del initial_model
        sampler = SizeGroupedBatchSampler(dataset, batch_size, seed=seed, length=1)
        indices = next(iter(sampler))
        host_batch = collate_prepared(dataset.fetch_batch(indices))
        batch = batch_to_device(host_batch, device)

        reference_snapshots: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        primed_model_state = None
        primed_optimizer_state = None
        if compile_mode != "off":
            if warmup_steps > 0:
                prime_model = SakiGoNet(config).to(device)
                prime_model.load_state_dict(initial_state)
                prime_optimizer = _make_optimizer(prime_model, device, lr, weight_decay)
                prime_scheduler = _make_warmup_scheduler(prime_optimizer, warmup_steps)
                prime_sampler = SizeGroupedBatchSampler(
                    dataset,
                    batch_size,
                    seed=seed + 1,
                    length=warmup_steps,
                )
                for prime_indices in prime_sampler:
                    prime_batch = batch_to_device(
                        collate_prepared(dataset.fetch_batch(prime_indices)),
                        device,
                    )
                    _train_step(
                        prime_model,
                        prime_model,
                        prime_optimizer,
                        prime_batch,
                        weights,
                        device,
                        amp_dtype,
                        grad_clip,
                    )
                    assert prime_scheduler is not None
                    prime_scheduler.step()
                primed_model_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in prime_model.state_dict().items()
                }
                primed_optimizer_state = _clone_state_to_cpu(prime_optimizer.state_dict())
                prime_model = None
                prime_optimizer = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            eager_model = SakiGoNet(config).to(device)
            eager_model.load_state_dict(initial_state)
            eager_optimizer = _make_optimizer(eager_model, device, lr, weight_decay)
            _make_warmup_scheduler(eager_optimizer, warmup_steps)
            reference_snapshots["initial"] = _captured_train_step(
                eager_model,
                eager_model,
                eager_optimizer,
                batch,
                weights,
                device,
                amp_dtype,
                grad_clip,
            )
            if primed_model_state is not None and primed_optimizer_state is not None:
                eager_model.load_state_dict(primed_model_state)
                eager_optimizer = _make_optimizer(eager_model, device, lr, weight_decay)
                eager_optimizer.load_state_dict(primed_optimizer_state)
                reference_snapshots["post_warmup"] = _captured_train_step(
                    eager_model,
                    eager_model,
                    eager_optimizer,
                    batch,
                    weights,
                    device,
                    amp_dtype,
                    grad_clip,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            eager_model = None
            eager_optimizer = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            torch._dynamo.reset()

        candidate_model = SakiGoNet(config).to(device)
        candidate_model.load_state_dict(initial_state)
        candidate_optimizer = _make_optimizer(candidate_model, device, lr, weight_decay)
        _make_warmup_scheduler(candidate_optimizer, warmup_steps)
        candidate_forward = candidate_model
        if compile_mode != "off":
            mode = None if compile_mode == "default" else compile_mode
            candidate_forward = torch.compile(candidate_model, mode=mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        candidate_snapshots = {
            "initial": _captured_train_step(
                candidate_forward,
                candidate_model,
                candidate_optimizer,
                batch,
                weights,
                device,
                amp_dtype,
                grad_clip,
                require_next_finite=False,
            )
        }
        previous_peak_allocated = result["peak_allocated_bytes"]
        previous_peak_reserved = result["peak_reserved_bytes"]
        _capture_peak_memory(result, device)
        result["peak_allocated_bytes"] = max(
            previous_peak_allocated, result["peak_allocated_bytes"]
        )
        result["peak_reserved_bytes"] = max(
            previous_peak_reserved, result["peak_reserved_bytes"]
        )
        initial_peak_allocated = result["peak_allocated_bytes"]
        initial_peak_reserved = result["peak_reserved_bytes"]

        if primed_model_state is not None and primed_optimizer_state is not None:
            # A reduce-overhead CUDA graph owns static parameter storage.  Build
            # a fresh graph from the warmed state instead of mutating a graph
            # that was captured from initialization.
            candidate_forward = None
            candidate_model = None
            candidate_optimizer = None
            gc.collect()
            if compile_mode != "off":
                torch._dynamo.reset()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            candidate_model = SakiGoNet(config).to(device)
            candidate_model.load_state_dict(primed_model_state)
            candidate_optimizer = _make_optimizer(candidate_model, device, lr, weight_decay)
            candidate_optimizer.load_state_dict(primed_optimizer_state)
            candidate_forward = candidate_model
            if compile_mode != "off":
                mode = None if compile_mode == "default" else compile_mode
                candidate_forward = torch.compile(candidate_model, mode=mode)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            candidate_snapshots["post_warmup"] = _captured_train_step(
                candidate_forward,
                candidate_model,
                candidate_optimizer,
                batch,
                weights,
                device,
                amp_dtype,
                grad_clip,
                require_next_finite=False,
            )
        _capture_peak_memory(result, device)
        result["peak_allocated_bytes"] = max(
            initial_peak_allocated, result["peak_allocated_bytes"]
        )
        result["peak_reserved_bytes"] = max(
            initial_peak_reserved, result["peak_reserved_bytes"]
        )

        if not reference_snapshots:
            safety = {"ok": True, "mode": "finite_eager_step"}
        else:
            phase_reports = {}
            for phase in reference_snapshots:
                effective_lr = lr
                if phase == "initial" and warmup_steps > 0:
                    effective_lr /= warmup_steps
                phase_reports[phase] = compare_step_snapshots(
                    reference_snapshots[phase],
                    candidate_snapshots[phase],
                    parameter_atol=max(
                        _PARITY_TOLERANCES["parameters"][1],
                        1.1 * effective_lr,
                    ),
                )
            safety = _combine_phase_reports(phase_reports, amp_dtype)
        result["safety"] = safety
        if not safety["ok"]:
            failed = ",".join(safety.get("failed_checks", ()))
            result["reason"] = f"parity_failed:{failed}"
            return result

        for _ in range(2):
            _train_step(
                candidate_forward,
                candidate_model,
                candidate_optimizer,
                batch,
                weights,
                device,
                amp_dtype,
                grad_clip,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        completed = 0
        for _ in range(max(1, timed_steps)):
            _train_step(
                candidate_forward,
                candidate_model,
                candidate_optimizer,
                batch,
                weights,
                device,
                amp_dtype,
                grad_clip,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            completed += 1
            if time.monotonic() - start > max_seconds:
                break
        elapsed = max(time.monotonic() - start, 1e-12)
        previous_peak_allocated = result["peak_allocated_bytes"]
        previous_peak_reserved = result["peak_reserved_bytes"]
        _capture_peak_memory(result, device)
        result["peak_allocated_bytes"] = max(
            previous_peak_allocated, result["peak_allocated_bytes"]
        )
        result["peak_reserved_bytes"] = max(
            previous_peak_reserved, result["peak_reserved_bytes"]
        )
        if memory_budget and result["peak_reserved_bytes"] > memory_budget:
            result["reason"] = "over_budget"
            return result
        result["samples_per_second"] = batch_size * completed / elapsed
        result["reason"] = "ok"
        return result
    except torch.OutOfMemoryError:
        result["reason"] = "oom"
        return result
    except Exception as error:  # noqa: BLE001 - candidate rejection is recorded
        result["reason"] = f"error:{type(error).__name__}:{error}"
        return result
    finally:
        eager_model = None
        eager_optimizer = None
        prime_model = None
        prime_optimizer = None
        candidate_forward = None
        candidate_model = None
        candidate_optimizer = None
        gc.collect()
        if compile_mode != "off":
            torch._dynamo.reset()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def benchmark_batch_size(
    spec: str,
    dataset: PreparedDataset,
    batch_size: int,
    device: torch.device,
    **kwargs: Any,
) -> tuple[float | None, int, str]:
    """Compatibility wrapper returning the legacy three-tuple."""
    result = benchmark_batch_candidate(spec, dataset, batch_size, device, **kwargs)
    return result["samples_per_second"], result["peak_allocated_bytes"], result["reason"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep safe batch sizes for a model spec.")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--spec", default="balanced")
    parser.add_argument("--batch-sizes", default="8,16,24,32,48,64,96,128")
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--budget-fraction", type=float, default=0.85)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--compile", choices=("off", "default", "reduce-overhead"), default="reduce-overhead"
    )
    parser.add_argument("--amp", choices=("auto", "off"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--wdl-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-weight",
        type=float,
        default=1.0,
        help="Base score multiplier; effective weight is this value times board area.",
    )
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--budget-weight", type=float, default=1.0)
    parser.add_argument("--model-config-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    budget = 0
    if device.type == "cuda":
        budget = int(torch.cuda.get_device_properties(device).total_memory * args.budget_fraction)
    dataset = PreparedDataset(args.prepared_dir, "train")
    weights = LossWeights(
        wdl=args.wdl_weight,
        score=args.score_weight,
        policy=args.policy_weight,
        budget=args.budget_weight,
    )
    model_config = None
    if args.model_config_checkpoint is not None:
        checkpoint = torch.load(
            args.model_config_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model_config = config_from_dict(checkpoint["model_config"])
    results = []
    best: tuple[int, float] | None = None
    for batch_size in (int(part) for part in args.batch_sizes.split(",") if part.strip()):
        result = benchmark_batch_candidate(
            args.spec,
            dataset,
            batch_size,
            device,
            timed_steps=args.timed_steps,
            memory_budget=budget,
            compile_mode=args.compile,
            amp=args.amp,
            loss_weights=weights,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
            model_config=model_config,
        )
        results.append(result)
        rate = result["samples_per_second"]
        rate_text = f"{rate:8.1f}/s" if rate else f"   {result['reason']}"
        print(
            f"batch {batch_size:4d}: {rate_text}  "
            f"allocated {result['peak_allocated_bytes'] / 2**30:.2f} GiB  "
            f"reserved {result['peak_reserved_bytes'] / 2**30:.2f} GiB"
        )
        if rate is not None and (best is None or rate > best[1]):
            best = (batch_size, rate)
        if result["reason"] in {"over_budget", "oom"}:
            break
    if best is not None:
        print(f"best: batch {best[0]} at {best[1]:.1f} samples/s")
    report = {
        "spec": args.spec,
        "device": str(device),
        "compile": args.compile,
        "amp": args.amp,
        "warmup_steps": args.warmup_steps,
        "loss_weights": weights.as_dict(),
        "model_config": model_config.as_dict() if model_config is not None else None,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
