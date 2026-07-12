"""Spec gates for the fixed-schedule, D4-only model family."""

from __future__ import annotations

import pytest
from torch import nn

from sakigo.model.layers import BoardBlock, RegisterBroadcast, RegisterGather
from sakigo.model.model import SakiGoNet
from sakigo.model.specs import (
    config_from_spec,
    load_model_specs,
    model_from_spec,
    model_spec_names,
)


SPEC_TRUNK_PARAMETER_COUNTS = {
    "narrow-deep": 5_450_444,
    "balanced": 5_405_426,
    "wide-shallow": 5_398_737,
}
TRUNK_PARAMETER_TARGET = 5_418_202
TRUNK_PARAMETER_TOLERANCE = 0.006


def test_schema_and_spec_names() -> None:
    specs = load_model_specs()
    assert specs["schema_version"] == 5
    assert "trunk_layouts" not in specs
    assert specs["comparison"]["target_trunk_parameters"] == TRUNK_PARAMETER_TARGET
    assert specs["comparison"]["relative_tolerance"] == TRUNK_PARAMETER_TOLERANCE
    assert model_spec_names() == tuple(SPEC_TRUNK_PARAMETER_COUNTS)


@pytest.mark.parametrize("name", SPEC_TRUNK_PARAMETER_COUNTS)
def test_all_specs_build_with_fixed_d4_schedule(name: str) -> None:
    config = config_from_spec(name)
    model = model_from_spec(name)

    assert config.architecture == "SakiGoModel"
    assert config.group_size == 8
    assert config.board_size == 32
    assert config.trunk_channels == 128
    assert not hasattr(config, "trunk_mlp_variant")
    assert not hasattr(config, "trunk_delta_conditioning")
    assert not hasattr(config, "trunk_sequence")

    assert isinstance(model, SakiGoNet)
    assert isinstance(model.broadcast, RegisterBroadcast)
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == config.block_count
    assert all(isinstance(block, BoardBlock) for block in model.blocks)
    assert isinstance(model.gather, RegisterGather)

    count = model.trunk_parameter_count()
    assert count == SPEC_TRUNK_PARAMETER_COUNTS[name]
    assert (
        abs(count - TRUNK_PARAMETER_TARGET) / TRUNK_PARAMETER_TARGET
        <= TRUNK_PARAMETER_TOLERANCE
    )


def test_balanced_is_default_model() -> None:
    assert config_from_spec() == config_from_spec("balanced")


def test_balanced_uses_narrower_register_stream() -> None:
    config = config_from_spec("balanced")
    assert config.trunk_channels == 128
    assert config.register_channels == 64
    assert config.register_bottleneck_channels == 32
    assert config.register_head_dim == 16
    assert config.rule_mlp_channels[-1] == config.register_count * config.register_channels
    assert config.wdl_channels is not None
    assert config.wdl_channels[0] == config.register_count * config.register_channels


def test_board_size_override() -> None:
    assert config_from_spec("balanced", board_size=19).board_size == 19
