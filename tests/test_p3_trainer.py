"""P3 gate: trainer smoke run on synthetic prepared data (CPU, compile off),
loss decreases, run artifacts exist (TB events, CSV mirror, config, status),
checkpoints are weights_only-loadable, and resume is deterministic.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest
import torch

from sakigo.data import prepare_tensor_shards
from sakigo.train.config import TrainConfig, validate_train_config
from sakigo.train.trainer import Trainer, require_finite

from tests.test_p2_data import _make_raw_record

_SPEC_JSON = {
    "schema_version": 5,
    "default_model": "tiny",
    "includes": {"stem_shapes": "StemShapes.json", "head_shapes": "HeadShapes.json"},
    "models": {
        "tiny": {
            "name": "Tiny",
            "architecture": "d4_equivariant",
            "activation": "SiLU",
            "max_board_size": 9,
            "stem_shape": "scalar_lift_v1",
            "head_shape": "standard_v1",
            "trunk": {
                "block_count": 2,
                "register_count": 2,
                "expanded_channel": 16,
                "bottleneck_channel": 16,
                "q_heads": 2,
                "kv_heads": 1,
                "global_rope_frequencies": ["pi"],
                "local_rope_frequencies": ["pi/2"],
            },
            "norm_eps": 0.000001,
        }
    },
}

_STEM_JSON = {
    "schema_version": 1,
    "stem_shapes": {
        "scalar_lift_v1": {
            "stem_channels": [6, 8, "expanded_channel"],
            "rule_mlp_channels": [10, 16, "expanded_channel * register_count"],
        }
    },
}

_HEAD_JSON = {
    "schema_version": 1,
    "head_shapes": {
        "standard_v1": {
            "spatial_shape": ["expanded_channel", 8, "output"],
            "global_shape": ["expanded_channel * register_count", 8, "output"],
            "global_heads": {"wdl": 4, "score": 1, "pass_policy": 1, "pass_budget": 1},
            "spatial_heads": {"policy": 1, "budget": 1},
        }
    },
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("p3trainer")
    rng = random.Random(9)
    jsonl = root / "samples_000000.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for index in range(240):
            handle.write(
                json.dumps(
                    _make_raw_record(rng, 5, ("tromp-taylor", "chinese")[index % 2], index)
                )
                + "\n"
            )
    prepared = root / "prepared"
    prepare_tensor_shards([jsonl], prepared, seed=1, val_fraction=0.1)

    spec_dir = root / "specs"
    spec_dir.mkdir()
    (spec_dir / "ModelSpecs.json").write_text(json.dumps(_SPEC_JSON), encoding="utf-8")
    (spec_dir / "StemShapes.json").write_text(json.dumps(_STEM_JSON), encoding="utf-8")
    (spec_dir / "HeadShapes.json").write_text(json.dumps(_HEAD_JSON), encoding="utf-8")
    return {"root": root, "prepared": prepared, "specs": spec_dir}


def _config(workspace: dict[str, Path], run_name: str, **overrides) -> TrainConfig:
    base = dict(
        prepared_dir=str(workspace["prepared"]),
        model_spec="tiny",
        steps=48,
        batch_size=8,
        lr=3e-3,
        seed=5,
        num_workers=0,
        compile="off",
        amp="off",
        device="cpu",
        run_dir=str(workspace["root"] / run_name),
        log_interval=12,
        checkpoint_interval=12,
        val_batches=4,
        progress=False,
        warmup_steps=4,
        score_weight=1.0,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_reduce_overhead_compilation_is_default() -> None:
    from sakigo.train.suite import SuiteConfig

    assert TrainConfig().compile == "reduce-overhead"
    assert SuiteConfig(root=Path("suite")).model_compile == "reduce-overhead"


def test_non_finite_losses_fail_fast() -> None:
    require_finite("finite test value", torch.tensor(1.0))
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        require_finite("training loss", torch.tensor(float("nan")))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (TrainConfig(batch_size=0), "batch_size"),
        (TrainConfig(checkpoint_interval=0), "checkpoint_interval"),
        (TrainConfig(lr=float("nan")), "lr"),
        (
            TrainConfig(
                wdl_weight=0.0,
                score_weight=0.0,
                policy_weight=0.0,
                budget_weight=0.0,
            ),
            "at least one loss weight",
        ),
    ],
)
def test_train_config_rejects_invalid_values(
    config: TrainConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_train_config(config)


def test_compile_failure_never_falls_back_to_eager(
    workspace: dict[str, Path], spec_patch, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_compile(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="eager fallback is disabled"):
        Trainer(_config(workspace, "run_compile_failure", compile="default"))


def test_lazy_compile_failure_is_reported_on_first_optimizer_step(
    workspace: dict[str, Path], spec_patch
) -> None:
    class FailingCompiledModel(torch.nn.Module):
        def forward(self, board, rules):
            raise RuntimeError("synthetic lazy compiler failure")

    trainer = Trainer(_config(workspace, "run_lazy_compile_failure"))
    trainer.compiled_model = FailingCompiledModel()
    trainer.compile_status = "pending_first_step:reduce-overhead"
    batch = next(iter(trainer.train_loader))
    with pytest.raises(RuntimeError, match="compiled first optimizer step failed"):
        trainer.train_step(batch)
    assert trainer.compile_status.startswith("failed_first_step:RuntimeError")


def test_training_failure_writes_failed_status(
    workspace: dict[str, Path], spec_patch, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(_config(workspace, "run_failed_status", steps=1))

    def fail_step(batch):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(trainer, "train_step", fail_step)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        trainer.train()
    status = json.loads((trainer.run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "synthetic training failure" in status["error"]


@pytest.fixture(scope="module")
def spec_patch(workspace: dict[str, Path]):
    import sakigo.model.specs as specs_module

    original = specs_module.DEFAULT_SPEC_PATH
    specs_module.DEFAULT_SPEC_PATH = workspace["specs"] / "ModelSpecs.json"
    yield
    specs_module.DEFAULT_SPEC_PATH = original


def test_trainer_smoke_run(workspace: dict[str, Path], spec_patch) -> None:
    trainer = Trainer(_config(workspace, "run_smoke"))
    final = trainer.train()
    run_dir = Path(trainer.run_dir)

    assert final.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "validation_metrics.csv").exists()
    assert list((run_dir / "tb").glob("events.out.tfevents.*"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "finished"
    assert status["step"] == 48

    # The randomly mixed sampler makes individual aggregate windows noisy;
    # verify that both dense distribution heads memorize the synthetic set.
    lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    training_rows = [row for row in rows if row["train_loss"]]
    assert len(training_rows) >= 2
    assert float(training_rows[-1]["train_policy_loss"]) < float(
        training_rows[0]["train_policy_loss"]
    )
    assert float(training_rows[-1]["train_budget_loss"]) < float(
        training_rows[0]["train_budget_loss"]
    )

    validation_lines = (run_dir / "validation_metrics.csv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    validation_rows = list(csv.DictReader(validation_lines))
    cohorts = {(row["board_size"], row["ruleset_name"]) for row in validation_rows}
    assert cohorts == {("5", "chinese"), ("5", "tromp-taylor")}
    assert all(row["loss"] for row in validation_rows)

    # weights_only load must succeed (checkpoint contract)
    payload = torch.load(final, map_location="cpu", weights_only=True)
    assert payload["checkpoint_schema_version"] == 8
    assert payload["step"] == 48
    assert payload["run_config"]["model_spec"] == "tiny"
    assert payload["model_config"]["architecture"] == "SakiGoModel"


def test_resume_restores_exact_control_state(workspace: dict[str, Path], spec_patch) -> None:
    part = Trainer(_config(workspace, "run_part", steps=16, checkpoint_interval=8))
    final_part = part.train()
    continued_payload = torch.load(final_part, map_location="cpu", weights_only=True)
    midpoint = Path(part.run_dir) / "checkpoints" / "step_000008.pt"
    assert midpoint.exists()
    resumed = Trainer(
        _config(
            workspace,
            "run_part",
            steps=16,
            checkpoint_interval=8,
            resume=str(midpoint),
        )
    )
    final_resumed = resumed.train()

    resumed_payload = torch.load(final_resumed, map_location="cpu", weights_only=True)

    def assert_state(left, right, label: str) -> None:
        if torch.is_tensor(left):
            if left.is_floating_point():
                # Batch/RNG/control state is exact. Floating kernels may not be
                # bit-reproducible, even across two uninterrupted CPU runs.
                torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-4, msg=label)
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=0, msg=label)
        elif isinstance(left, dict):
            assert left.keys() == right.keys(), label
            for key in left:
                assert_state(left[key], right[key], f"{label}.{key}")
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right), label
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                assert_state(left_item, right_item, f"{label}[{index}]")
        else:
            assert left == right, label

    for section in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng",
        "sampler_state",
        "augmentation_state",
        "prepared_data_identity",
    ):
        assert_state(continued_payload[section], resumed_payload[section], section)
    status = json.loads((Path(resumed.run_dir) / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "finished" and status["step"] == 16


def test_resume_rejects_changed_trajectory_property(
    workspace: dict[str, Path], spec_patch
) -> None:
    source = Trainer(_config(workspace, "run_resume_properties", steps=1))
    checkpoint = source.save_checkpoint(0)
    with pytest.raises(ValueError, match="resume properties changed: batch_size"):
        Trainer(
            _config(
                workspace,
                "run_resume_properties",
                steps=1,
                batch_size=9,
                resume=str(checkpoint),
            )
        )


def test_resume_rejects_optimizer_state_failure(
    workspace: dict[str, Path],
    spec_patch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Trainer(_config(workspace, "run_resume_source", steps=1))
    checkpoint = source.save_checkpoint(0)

    def reject_state(self, state):
        raise ValueError("synthetic optimizer mismatch")

    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", reject_state)
    with pytest.raises(RuntimeError, match="exact resume cannot continue"):
        Trainer(
            _config(
                workspace,
                "run_resume_rejected",
                steps=1,
                resume=str(checkpoint),
            )
        )


def test_val_fixed_replays_identical_grouped_batches(
    workspace: dict[str, Path], spec_patch
) -> None:
    from sakigo.data import GroupedValidationBatchSampler

    fixed = Trainer(_config(workspace, "run_valfixed", val_fixed=True))
    sampler = fixed.val_loader.batch_sampler
    assert isinstance(sampler, GroupedValidationBatchSampler)
    assert len(sampler) == fixed.config.val_batches
    first = [list(batch) for batch in sampler]
    second = [list(batch) for batch in sampler]
    assert first == second
    # two evaluate() passes must not crash and must see the full fixed subset
    accumulator = fixed.evaluate(fixed.config.val_batches)
    assert accumulator.steps == fixed.config.val_batches

    # default (rotating): successive evaluation iterators advance independently
    # within every cohort while preserving cohort coverage.
    rotating = Trainer(_config(workspace, "run_valrot", val_batches=2))
    draw_a = list(rotating.val_loader.batch_sampler)
    draw_b = list(rotating.val_loader.batch_sampler)
    assert draw_a != draw_b


def test_default_log_interval_follows_checkpoint(
    workspace: dict[str, Path], spec_patch
) -> None:
    defaulted = Trainer(
        _config(
            workspace,
            "run_logdefault",
            log_interval=0,
            checkpoint_interval=7,
            steps=1,
        )
    )
    assert defaulted.log_interval == 7

    explicit = Trainer(
        _config(
            workspace,
            "run_logexplicit",
            log_interval=3,
            checkpoint_interval=7,
            steps=1,
        )
    )
    assert explicit.log_interval == 3


def test_default_metrics_rows_follow_checkpoint_cadence(
    workspace: dict[str, Path], spec_patch
) -> None:
    trainer = Trainer(
        _config(
            workspace,
            "run_logrows",
            log_interval=0,
            checkpoint_interval=5,
            steps=10,
            val_batches=2,
        )
    )
    trainer.train()

    metrics_path = Path(trainer.run_dir) / "metrics.csv"
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    assert [int(row["step"]) for row in rows] == [0, 5, 10]


def test_suite_layout_uses_structured_train_dirs(
    workspace: dict[str, Path], spec_patch
) -> None:
    from sakigo.train.suite import SuiteConfig, build_suite_paths, train_config_for_spec

    config = SuiteConfig(
        root=workspace["root"] / "suite_layout",
        prepared_dir=workspace["prepared"],
        specs=("tiny",),
        batch_size=8,
        steps=10,
        checkpoint_interval=5,
        val_batches=1,
        model_compile="off",
        amp="off",
        device="cpu",
        progress=False,
        score_weight=3.0,
    )
    paths = build_suite_paths(config)
    train_config = train_config_for_spec(
        config,
        paths,
        "tiny",
        data_sources=(),
        batch_size=8,
        steps=10,
    )

    assert paths.data == config.root / "data"
    assert paths.prepared == workspace["prepared"]
    assert paths.generation == config.root / "generation"
    assert paths.train == config.root / "train"
    assert paths.logs == config.root / "logs"
    assert paths.sweeps == config.root / "sweeps"
    assert paths.scripts == config.root / "scripts"
    assert train_config.run_dir == str(config.root / "train" / "tiny")
    assert train_config.log_interval == 0
    assert train_config.checkpoint_interval == 5
    assert train_config.score_weight == 3.0


def test_explicit_suite_batch_still_runs_safety_preflight(
    workspace: dict[str, Path], spec_patch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakigo.train.suite as suite_module
    from sakigo.train.suite import (
        SuiteConfig,
        build_suite_paths,
        choose_batch_size,
        ensure_suite_dirs,
    )

    seen: list[dict[str, object]] = []

    def safe_candidate(spec, dataset, batch_size, device, **kwargs):
        seen.append({"spec": spec, "batch_size": batch_size, **kwargs})
        return {
            "batch_size": batch_size,
            "samples_per_second": 12.0,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
            "reason": "ok",
            "safety": {"ok": True},
        }

    monkeypatch.setattr(suite_module, "benchmark_batch_candidate", safe_candidate)
    config = SuiteConfig(
        root=workspace["root"] / "suite_explicit_preflight",
        prepared_dir=workspace["prepared"],
        specs=("tiny",),
        batch_size=8,
        model_compile="reduce-overhead",
        amp="auto",
        device="cpu",
        score_weight=3.0,
        grad_clip=0.75,
    )
    paths = build_suite_paths(config)
    ensure_suite_dirs(paths)
    selected, results = choose_batch_size(config, paths)

    assert selected == 8 and results
    assert seen[0]["batch_size"] == 8
    assert seen[0]["loss_weights"].score == 3.0
    assert seen[0]["grad_clip"] == 0.75
    assert seen[0]["warmup_steps"] == config.warmup_steps


def test_explicit_suite_batch_rejects_failed_preflight(
    workspace: dict[str, Path], spec_patch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakigo.train.suite as suite_module
    from sakigo.train.suite import (
        SuiteConfig,
        build_suite_paths,
        choose_batch_size,
        ensure_suite_dirs,
    )

    def unsafe_candidate(spec, dataset, batch_size, device, **kwargs):
        return {
            "batch_size": batch_size,
            "samples_per_second": None,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
            "reason": "parity_failed:gradients",
            "safety": {"ok": False},
        }

    monkeypatch.setattr(suite_module, "benchmark_batch_candidate", unsafe_candidate)
    config = SuiteConfig(
        root=workspace["root"] / "suite_explicit_reject",
        prepared_dir=workspace["prepared"],
        specs=("tiny",),
        batch_size=8,
        model_compile="reduce-overhead",
        device="cpu",
    )
    paths = build_suite_paths(config)
    ensure_suite_dirs(paths)
    with pytest.raises(RuntimeError, match="mandatory safety preflight"):
        choose_batch_size(config, paths)
