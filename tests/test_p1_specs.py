"""Spec gates: packaged spec copies stay in sync with Design/ModelSpecs, and
the JSON loader builds all published specs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakigo.model.model import SakiGoNet
from sakigo.model.specs import SPEC_DIR, config_from_spec, model_from_spec, model_spec_names

ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = ROOT / "Design" / "ModelSpecs"

_SYNC_PAIRS = (
    ("ModelSpecs.md", "ModelSpecs.json"),
    ("StemShapes.md", "StemShapes.json"),
    ("HeadShapes.md", "HeadShapes.json"),
)


@pytest.mark.parametrize("design_name,package_name", _SYNC_PAIRS)
def test_packaged_specs_in_sync_with_design(design_name: str, package_name: str) -> None:
    design = json.loads((DESIGN_DIR / design_name).read_text(encoding="utf-8"))
    packaged = json.loads((SPEC_DIR / package_name).read_text(encoding="utf-8"))
    assert packaged == design, (
        f"{package_name} is out of sync with Design/ModelSpecs/{design_name}; "
        "re-copy the design file into sakigo/model/specs/"
    )


def test_spec_names() -> None:
    assert model_spec_names() == ("model1", "model2", "model3")


@pytest.mark.parametrize("name", ["model1", "model2", "model3"])
def test_all_specs_build(name: str) -> None:
    config = config_from_spec(name)
    assert config.architecture == "SakiGoModel"
    assert config.board_size == 32


def test_model2_uses_narrower_register_stream() -> None:
    config = config_from_spec("model2")
    assert config.trunk_channels == 128
    assert config.register_channels == 64
    assert config.bottleneck_channels == 64
    assert config.register_bottleneck_channels == 32
    assert config.register_head_dim == 16
    assert config.rule_mlp_channels[-1] == config.register_count * config.register_channels
    assert config.wdl_channels is not None
    assert config.wdl_channels[0] == config.register_count * config.register_channels


def test_default_model_is_model1() -> None:
    assert config_from_spec() == config_from_spec("model1")


def test_model_from_spec_builds() -> None:
    model = model_from_spec("model1")
    assert isinstance(model, SakiGoNet)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_board_size_override() -> None:
    assert config_from_spec("model1", board_size=19).board_size == 19
