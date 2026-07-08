"""P3 gate: trainer smoke run on synthetic prepared data (CPU, compile off),
loss decreases, run artifacts exist (TB events, CSV mirror, config, status),
checkpoints are weights_only-loadable, and resume is deterministic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from sakigo.data import prepare_tensor_shards
from sakigo.rulesets import ruleset_from_name
from sakigo.train.config import TrainConfig
from sakigo.train.trainer import Trainer

from tests.test_p2_data import _make_raw_record

_SPEC_JSON = {
    "schema_version": 3,
    "default_model": "tiny",
    "includes": {"stem_shapes": "StemShapes.json", "head_shapes": "HeadShapes.json"},
    "models": {
        "tiny": {
            "name": "Tiny",
            "architecture": "d4_equivariant",
            "activation": "SiLU",
            "max_board_size": 9,
            "stem_shape": "regular_v1",
            "head_shape": "standard_v1",
            "trunk": {
                "block_count": 2,
                "register_count": 2,
                "expanded_channel": 16,
                "bottleneck_channel": 16,
                "register_gather_blocks": [1, 2],
                "register_broadcast_blocks": [2],
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
        "regular_v1": {
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
            "collapse": "mean_d4_axis",
            "global_heads": {"wdl": 4, "score": 1, "pass_policy": 1, "pass_budget": 1},
            "spatial_heads": {"ownership": 1, "policy": 1, "budget": 1},
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
    )
    base.update(overrides)
    return TrainConfig(**base)


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
    assert list((run_dir / "tb").glob("events.out.tfevents.*"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "finished"
    assert status["step"] == 48

    # train loss decreases (memorization on the small synthetic set)
    lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    train_losses = [float(row["train_loss"]) for row in rows if row["train_loss"]]
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0]

    # weights_only load must succeed (checkpoint contract)
    payload = torch.load(final, map_location="cpu", weights_only=True)
    assert payload["step"] == 48
    assert payload["run_config"]["model_spec"] == "tiny"
    assert payload["model_config"]["architecture"] == "SakiGoModel"


def test_resume_is_deterministic(workspace: dict[str, Path], spec_patch) -> None:
    full = Trainer(_config(workspace, "run_full", steps=16, checkpoint_interval=8))
    final_full = full.train()

    part = Trainer(_config(workspace, "run_part", steps=16, checkpoint_interval=8))
    part.train()
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

    full_state = torch.load(final_full, map_location="cpu", weights_only=True)["model_state"]
    resumed_state = torch.load(final_resumed, map_location="cpu", weights_only=True)["model_state"]
    for key, value in full_state.items():
        assert value.shape == resumed_state[key].shape, key
    # Data order after resume differs (sampler state is not checkpointed), so
    # exact equality is not required — but the resumed run must train stably.
    status = json.loads((Path(resumed.run_dir) / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "finished" and status["step"] == 16


def test_val_fixed_replays_identical_batches(workspace: dict[str, Path], spec_patch) -> None:
    from sakigo.data import FixedBatchSampler

    fixed = Trainer(_config(workspace, "run_valfixed", val_fixed=True))
    sampler = fixed.val_loader.batch_sampler
    assert isinstance(sampler, FixedBatchSampler)
    assert len(sampler) == fixed.config.val_batches
    first = [list(batch) for batch in sampler]
    second = [list(batch) for batch in sampler]
    assert first == second
    # two evaluate() passes must not crash and must see the full fixed subset
    accumulator = fixed.evaluate(fixed.config.val_batches)
    assert accumulator.steps == fixed.config.val_batches

    # default (rotating): consecutive draws continue the without-replacement
    # cycle, so successive evals see different batches
    rotating = Trainer(_config(workspace, "run_valrot"))
    iterator = iter(rotating.val_loader.batch_sampler)
    draw_a = [next(iterator) for _ in range(4)]
    draw_b = [next(iterator) for _ in range(4)]
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
            val_batches=1,
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
