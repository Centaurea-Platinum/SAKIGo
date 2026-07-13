from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import equivariant_attention.layers as attention_layers
from sakigo.data import PreparedDataset
from sakigo.train.benchmark import benchmark_batch_candidate, compare_step_snapshots
from sakigo.train.losses import LossWeights


def _snapshot(value: float = 0.0) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "outputs": {"x": torch.tensor([value])},
        "head_losses": {"x": torch.tensor([value])},
        "total_loss": {"total": torch.tensor([value])},
        "gradients": {"x": torch.tensor([value])},
        "parameters": {"x": torch.tensor([value])},
        "optimizer_state": {"x:exp_avg": torch.tensor([value])},
        "next_outputs": {"x": torch.tensor([value])},
    }


def test_step_parity_covers_post_step_state_and_next_forward() -> None:
    reference = _snapshot()
    candidate = _snapshot()
    candidate["optimizer_state"]["x:exp_avg"] = torch.tensor([float("nan")])
    candidate["next_outputs"]["x"] = torch.tensor([1.0])

    report = compare_step_snapshots(reference, candidate)

    assert not report["ok"]
    assert set(report["failed_checks"]) == {"optimizer_state", "next_outputs"}


def test_attention_primitive_is_an_explicit_compile_boundary() -> None:
    assert attention_layers._safe_scaled_dot_product_attention._torchdynamo_disable


def test_parameter_parity_can_use_one_optimizer_update_tolerance() -> None:
    reference = _snapshot()
    candidate = _snapshot()
    candidate["parameters"]["x"] = torch.tensor([2.5e-4])

    strict = compare_step_snapshots(reference, candidate)
    one_update = compare_step_snapshots(reference, candidate, parameter_atol=3.3e-4)

    assert strict["failed_checks"] == ["parameters"]
    assert one_update["ok"]


@pytest.mark.skipif(
    os.environ.get("SAKIGO_RUN_CUDA_COMPILE_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set SAKIGO_RUN_CUDA_COMPILE_TESTS=1 to run the compiled BF16 integration gate",
)
def test_current_models_complete_compiled_bf16_preflight() -> None:
    prepared = Path(
        os.environ.get(
            "SAKIGO_COMPILE_TEST_PREPARED_DIR",
            "runs/tt7-one-epoch/prepared",
        )
    )
    dataset = PreparedDataset(prepared, "train")
    for spec in ("narrow-deep", "balanced", "wide-shallow"):
        result = benchmark_batch_candidate(
            spec,
            dataset,
            batch_size=8,
            device=torch.device("cuda"),
            timed_steps=1,
            max_seconds=5.0,
            compile_mode="reduce-overhead",
            amp="auto",
            loss_weights=LossWeights(score=81.0),
            seed=20260713,
        )
        assert result["reason"] == "ok", (spec, result)
        assert result["safety"]["ok"], (spec, result["safety"])
