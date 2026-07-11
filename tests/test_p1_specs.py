"""Spec gates: the JSON loader builds all published packaged specs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakigo.model.model import SakiGoNet
from sakigo.model.specs import SPEC_DIR, config_from_spec, model_from_spec, model_spec_names


def test_spec_names() -> None:
    assert model_spec_names() == ("non-bottleneck", "plain", "swiglu", "scalar-control")


@pytest.mark.parametrize("name", ["non-bottleneck", "plain", "swiglu"])
def test_all_specs_build(name: str) -> None:
    config = config_from_spec(name)
    assert config.architecture == "SakiGoModel"
    assert config.board_size == 32


def test_scalar_control_is_available_to_training() -> None:
    config = config_from_spec("scalar-control")
    assert config.architecture == "ScalarSakiGoModel"
    assert config.group_size == 1


def test_plain_uses_narrower_register_stream() -> None:
    config = config_from_spec("plain")
    assert config.trunk_channels == 128
    assert config.register_channels == 64
    assert config.bottleneck_channels == 64
    assert config.register_bottleneck_channels == 32
    assert config.register_head_dim == 16
    assert config.rule_mlp_channels[-1] == config.register_count * config.register_channels
    assert config.wdl_channels is not None
    assert config.wdl_channels[0] == config.register_count * config.register_channels


def test_non_bottleneck_uses_equal_trunk_and_attention_width() -> None:
    config = config_from_spec("non-bottleneck")
    assert config.trunk_channels == config.bottleneck_channels == 64
    assert config.register_channels == 64


def test_default_model_is_plain() -> None:
    assert config_from_spec() == config_from_spec("plain")


def test_model_from_spec_builds() -> None:
    model = model_from_spec("plain")
    assert isinstance(model, SakiGoNet)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_board_size_override() -> None:
    assert config_from_spec("plain", board_size=19).board_size == 19


def test_spec_can_select_swiglu_trunk(tmp_path: Path) -> None:
    specs = json.loads((SPEC_DIR / "ModelSpecs.json").read_text(encoding="utf-8"))
    specs["includes"] = {}
    specs["stem_shapes"] = json.loads((SPEC_DIR / "StemShapes.json").read_text(encoding="utf-8"))[
        "stem_shapes"
    ]
    specs["head_shapes"] = json.loads((SPEC_DIR / "HeadShapes.json").read_text(encoding="utf-8"))[
        "head_shapes"
    ]
    specs["models"]["plain"]["trunk"]["mlp_variant"] = "swiglu"
    path = tmp_path / "swiglu_specs.json"
    path.write_text(json.dumps(specs), encoding="utf-8")

    config = config_from_spec("plain", path)

    assert config.trunk_mlp_variant == "swiglu"
