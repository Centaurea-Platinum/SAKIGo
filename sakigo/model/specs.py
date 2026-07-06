"""Slim spec loader: JSON model specs -> SakiGoModelConfig.

The spec *format* is frozen (schema_version 3, includes, named stem/head
shapes, `expanded_channel * register_count`-style derived dims, pass_-prefixed
global heads appended to spatial action heads). This replaces the legacy
590-line parser; markdown-fence stripping, YAML fallback, `base` shape
inheritance, and legacy key aliases are intentionally gone.

Package copies of the design specs live next to this module; a test keeps
them byte-equivalent with Design/ModelSpecs/*.md (design remains the source
of truth without runtime repo-layout coupling).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from math import pi
from pathlib import Path
from typing import Any, Mapping

from sakigo.model.config import SakiGoModelConfig

SPEC_DIR = Path(__file__).parent / "specs"
DEFAULT_SPEC_PATH = SPEC_DIR / "ModelSpecs.json"

ARCHITECTURE_TAGS = {
    "d4_equivariant": "SakiGoModel",
    "saki_go_model": "SakiGoModel",
    "SakiGoModel": "SakiGoModel",
    "scalar_control": "ScalarSakiGoModel",
    "ScalarSakiGoModel": "ScalarSakiGoModel",
}

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _evaluate(expression: str, context: Mapping[str, float], label: str) -> float:
    """Evaluate `name (*|/ name-or-number)*` left to right. No other operators."""
    tokens = [token.strip() for token in re.split(r"([*/])", expression)]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{label}: cannot parse expression {expression!r}")

    def atom(token: str) -> float:
        if _NAME_PATTERN.match(token):
            if token not in context:
                available = ", ".join(sorted(context))
                raise ValueError(f"{label}: unknown name {token!r}; available: {available}")
            return float(context[token])
        return float(token)

    value = atom(tokens[0])
    for operator, operand in zip(tokens[1::2], tokens[2::2]):
        if operator == "*":
            value *= atom(operand)
        else:
            value /= atom(operand)
    return value


def _resolve_int(raw: Any, context: Mapping[str, float], label: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be an integer or expression")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"{label}: {raw} is not an integer")
        return int(raw)
    if isinstance(raw, str):
        value = _evaluate(raw, context, label)
        if abs(value - round(value)) > 1e-9:
            raise ValueError(f"{label}: expression {raw!r} = {value} is not an integer")
        return int(round(value))
    raise ValueError(f"{label} must be an integer or expression, got {type(raw).__name__}")


def _resolve_float(raw: Any, label: str) -> float:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        return _evaluate(raw, {"pi": pi}, label)
    raise ValueError(f"{label} must be a number or expression, got {type(raw).__name__}")


def _int_tuple(raw: Any, label: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of integers")
    return tuple(int(item) for item in raw)


def _channels(raw: Any, context: Mapping[str, float], label: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"{label} must be a list of at least two channel entries")
    return tuple(_resolve_int(item, context, f"{label}[{index}]") for index, item in enumerate(raw))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _resolve_include(spec_path: Path, target: str) -> Path:
    candidate = spec_path.parent / target
    if candidate.exists():
        return candidate
    swapped = candidate.with_suffix(".json" if candidate.suffix == ".md" else ".md")
    if swapped.exists():
        return swapped
    raise FileNotFoundError(f"spec include {target!r} not found next to {spec_path}")


def load_model_specs(path: str | Path | None = None) -> dict[str, Any]:
    spec_path = Path(path) if path is not None else DEFAULT_SPEC_PATH
    if spec_path.is_dir():
        spec_path = spec_path / "ModelSpecs.json"
    specs = _load_json(spec_path)
    includes = specs.get("includes", {})
    if not isinstance(includes, Mapping):
        raise ValueError("includes must be a mapping")
    for key, field in (("stem_shapes", "stem_shapes"), ("head_shapes", "head_shapes")):
        target = includes.get(key)
        if target is not None:
            included = _load_json(_resolve_include(spec_path, str(target)))
            specs[field] = included.get(field, {})
    return specs


def model_spec_names(path: str | Path | None = None) -> tuple[str, ...]:
    return tuple(load_model_specs(path).get("models", {}))


def _shape(specs: Mapping[str, Any], spec: Mapping[str, Any], kind: str, name_field: str) -> dict[str, Any]:
    shape_name = spec.get(name_field)
    if shape_name is None:
        raise ValueError(f"spec must name a {name_field}")
    shapes = specs.get(kind, {})
    if shape_name not in shapes:
        available = ", ".join(str(item) for item in shapes)
        raise ValueError(f"unknown {name_field} {shape_name!r}; available: {available}")
    shape = shapes[shape_name]
    if not isinstance(shape, Mapping):
        raise ValueError(f"{name_field} {shape_name!r} must be a mapping")
    return dict(shape)


def _head_tuple(
    shape: tuple[int, ...],
    output: int,
    context: Mapping[str, float],
) -> tuple[int, ...]:
    return tuple(
        output if item == "output" else _resolve_int(item, context, "head channel")
        for item in shape
    )


def config_from_spec(
    model_name: str | None = None,
    path: str | Path | None = None,
    board_size: int | None = None,
) -> SakiGoModelConfig:
    specs = load_model_specs(path)
    name = model_name or str(specs.get("default_model", "model1"))
    models = specs.get("models", {})
    if name not in models:
        available = ", ".join(str(item) for item in models)
        raise ValueError(f"unknown model spec {name!r}; available specs: {available}")
    spec = models[name]
    if not isinstance(spec, Mapping):
        raise ValueError(f"model spec {name!r} must be a mapping")

    architecture_tag = str(spec.get("architecture", "d4_equivariant"))
    if architecture_tag not in ARCHITECTURE_TAGS:
        available = ", ".join(sorted(ARCHITECTURE_TAGS))
        raise ValueError(f"{name} uses unsupported architecture tag {architecture_tag!r}; available: {available}")
    architecture = ARCHITECTURE_TAGS[architecture_tag]

    trunk = spec.get("trunk")
    if not isinstance(trunk, Mapping):
        raise ValueError(f"{name} must contain a trunk mapping")

    register_count = int(trunk["register_count"])
    expanded_channel = int(trunk["expanded_channel"])
    bottleneck_channels = int(trunk["bottleneck_channel"])
    q_heads = int(trunk["q_heads"])
    kv_heads = int(trunk["kv_heads"])
    if bottleneck_channels % q_heads != 0:
        raise ValueError("bottleneck_channel must be divisible by q_heads")
    head_dim = bottleneck_channels // q_heads

    context: dict[str, float] = {
        "register_count": register_count,
        "expanded_channel": expanded_channel,
        "bottleneck_channel": bottleneck_channels,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }

    stem_shape = _shape(specs, spec, "stem_shapes", "stem_shape")
    stem_channels = _channels(stem_shape["stem_channels"], context, "stem_channels")
    rule_mlp_channels = _channels(stem_shape["rule_mlp_channels"], context, "rule_mlp_channels")
    input_planes = stem_channels[0]
    rule_dim = rule_mlp_channels[0]

    head_shape = _shape(specs, spec, "head_shapes", "head_shape")
    spatial_shape = tuple(head_shape["spatial_shape"])
    global_shape = tuple(head_shape["global_shape"])
    global_heads = dict(head_shape["global_heads"])
    spatial_heads = dict(head_shape["spatial_heads"])

    def global_channels(head: str) -> tuple[int, ...]:
        return _head_tuple(global_shape, int(global_heads[head]), context)

    def spatial_channels(head: str) -> tuple[int, ...]:
        return _head_tuple(spatial_shape, int(spatial_heads[head]), context)

    wdl_channels = global_channels("wdl")
    score_channels = global_channels("score")
    policy_pass_channels = global_channels("pass_policy")
    budget_pass_channels = global_channels("pass_budget")
    ownership_channels = spatial_channels("ownership")
    policy_channels = spatial_channels("policy")
    budget_channels = spatial_channels("budget")

    gather_raw = trunk.get("register_gather_blocks", "all")
    gather_blocks = None if gather_raw == "all" else _int_tuple(gather_raw, "register_gather_blocks")

    global_frequencies = tuple(
        _resolve_float(item, "global_rope_frequencies")
        for item in trunk.get("global_rope_frequencies", [])
    )
    local_frequencies = tuple(
        _resolve_float(item, "local_rope_frequencies")
        for item in trunk.get("local_rope_frequencies", [])
    )

    max_board_size = spec.get("max_board_size", spec.get("board_size"))
    if max_board_size is None:
        raise ValueError("spec must define max_board_size")

    config = SakiGoModelConfig(
        architecture=architecture,
        board_size=int(max_board_size),
        input_planes=input_planes,
        rule_dim=rule_dim,
        stem_channels=stem_channels,
        rule_mlp_channels=rule_mlp_channels,
        activation=str(spec.get("activation", "none")),
        block_count=int(trunk["block_count"]),
        register_count=register_count,
        trunk_channels=expanded_channel,
        expanded_channel=expanded_channel,
        bottleneck_channels=bottleneck_channels,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        global_rope_frequencies=global_frequencies,
        local_rope_frequencies=local_frequencies,
        gather_blocks=gather_blocks,
        broadcast_blocks=_int_tuple(
            trunk.get("register_broadcast_blocks", []), "register_broadcast_blocks"
        ),
        wdl_hidden=wdl_channels[-2],
        wdl_outputs=wdl_channels[-1],
        score_hidden=score_channels[-2],
        score_outputs=score_channels[-1],
        ownership_hidden=ownership_channels[-2],
        ownership_outputs=ownership_channels[-1],
        policy_hidden=policy_channels[-2],
        policy_outputs=policy_channels[-1],
        policy_pass_hidden=policy_pass_channels[-2],
        policy_pass_outputs=policy_pass_channels[-1],
        budget_hidden=budget_channels[-2],
        budget_outputs=budget_channels[-1],
        budget_pass_hidden=budget_pass_channels[-2],
        budget_pass_outputs=budget_pass_channels[-1],
        wdl_channels=wdl_channels,
        score_channels=score_channels,
        ownership_channels=ownership_channels,
        policy_channels=policy_channels,
        policy_pass_channels=policy_pass_channels,
        budget_channels=budget_channels,
        budget_pass_channels=budget_pass_channels,
        norm_eps=float(spec.get("norm_eps", 1e-6)),
    )
    if board_size is not None:
        config = replace(config, board_size=int(board_size))
    return config


def model_from_spec(
    model_name: str | None = None,
    path: str | Path | None = None,
    board_size: int | None = None,
):
    from sakigo.model.model import SakiGoNet

    return SakiGoNet(config_from_spec(model_name, path, board_size))
