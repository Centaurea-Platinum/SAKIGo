from __future__ import annotations

import json
from dataclasses import fields, replace
from json import JSONDecodeError
from math import pi
from pathlib import Path
from typing import Any

from .config import SakiGoModelConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = ROOT / "Design" / "ModelSpecs.md"
NESTED_SPEC_PATH = ROOT / "Design" / "ModelSpecs" / "ModelSpecs.md"
CONFIG_FIELDS = {field.name for field in fields(SakiGoModelConfig)}
CONFIG_TUPLE_FIELDS = {
    "stem_channels",
    "rule_mlp_channels",
    "global_rope_frequencies",
    "local_rope_frequencies",
    "gather_blocks",
    "broadcast_blocks",
    "wdl_channels",
    "score_channels",
    "ownership_channels",
    "policy_channels",
    "policy_pass_channels",
    "budget_channels",
    "budget_pass_channels",
}
ARCHITECTURE_TAGS = {
    "d4_equivariant": "SakiGoModel",
    "saki_go_model": "SakiGoModel",
    "SakiGoModel": "SakiGoModel",
    "scalar_control": "ScalarSakiGoModel",
    "ScalarSakiGoModel": "ScalarSakiGoModel",
}


def _default_spec_path() -> Path:
    return DEFAULT_SPEC_PATH if DEFAULT_SPEC_PATH.exists() else NESTED_SPEC_PATH


def _resolve_spec_path(path: str | Path | None = None) -> Path:
    spec_path = Path(path) if path is not None else _default_spec_path()
    if spec_path.is_dir():
        spec_path = spec_path / "ModelSpecs.md"
    return spec_path


def _extract_data(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()
    for start, line in enumerate(lines):
        if not line.strip().startswith("```"):
            continue
        for end in range(start + 1, len(lines)):
            if lines[end].strip().startswith("```"):
                return "\n".join(lines[start + 1 : end]).strip()
        break
    return stripped


def _load_data_file(spec_path: Path) -> dict[str, Any]:
    data = _extract_data(spec_path.read_text(encoding="utf-8"))
    try:
        parsed = json.loads(data)
    except JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as yaml_error:
            raise ValueError(
                f"{spec_path} must be JSON-compatible YAML unless PyYAML is installed"
            ) from yaml_error
        parsed = yaml.safe_load(data)
        if parsed is None:
            raise ValueError(f"{spec_path} is empty") from json_error
    if not isinstance(parsed, dict):
        raise ValueError(f"{spec_path} must contain a mapping")
    return parsed


def load_model_specs(path: str | Path | None = None) -> dict[str, Any]:
    spec_path = _resolve_spec_path(path)
    parsed = _load_data_file(spec_path)
    if "models" not in parsed or not isinstance(parsed["models"], dict):
        raise ValueError(f"{spec_path} must contain a models mapping")
    includes = parsed.get("includes", {})
    if includes:
        if not isinstance(includes, dict):
            raise ValueError(f"{spec_path} includes must be a mapping")
        for key in ("stem_shapes", "head_shapes"):
            if key in parsed or key not in includes:
                continue
            include_path = spec_path.parent / str(includes[key])
            included = _load_data_file(include_path)
            if key not in included or not isinstance(included[key], dict):
                raise ValueError(f"{include_path} must contain a {key} mapping")
            parsed[key] = included[key]
    return parsed


def model_spec_names(path: str | Path | None = None) -> tuple[str, ...]:
    specs = load_model_specs(path)
    return tuple(str(name) for name in specs["models"])


def get_model_spec(model_name: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    specs = load_model_specs(path)
    name = model_name or str(specs.get("default_model", "model1"))
    models = specs["models"]
    if name not in models:
        available = ", ".join(str(item) for item in models)
        raise ValueError(f"unknown model spec {name!r}; available specs: {available}")
    model = models[name]
    if not isinstance(model, dict):
        raise ValueError(f"model spec {name!r} must be a mapping")
    return model


def _int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return tuple(int(item) for item in value)


def _resolve_int(value: Any, context: dict[str, int], label: str) -> int:
    if isinstance(value, str):
        raw = value.strip()
        if raw in context:
            return int(context[raw])
        if raw.lstrip("-").isdigit():
            return int(raw)
        if "*" in raw:
            product = 1
            for part in raw.split("*"):
                product *= _resolve_int(part.strip(), context, label)
            return product
        if raw not in context:
            available = ", ".join(sorted(context))
            raise ValueError(f"{label} references unknown channel {value!r}; available: {available}")
    return int(value)


def _channel_tuple(value: Any, context: dict[str, int], label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return tuple(
        _resolve_int(item, context, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _parse_float(value: Any, label: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a number or pi expression")
    raw = value.strip().lower().replace(" ", "")
    if raw == "pi":
        return pi
    if raw.startswith("pi/"):
        return pi / float(raw.removeprefix("pi/"))
    if raw.endswith("*pi"):
        return float(raw.removesuffix("*pi")) * pi
    return float(raw)


def _head_channels(heads: dict[str, Any], name: str) -> tuple[int, ...]:
    raw = heads.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{name} head must be a mapping")
    channels = _int_tuple(raw.get("channels"), f"{name}.channels")
    if len(channels) < 2:
        raise ValueError(f"{name}.channels must have at least 2 entries")
    return channels


def _pass_head_channels(heads: dict[str, Any], name: str) -> tuple[int, ...]:
    raw = heads.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{name} head must be a mapping")
    channels = _int_tuple(raw.get("pass_channels"), f"{name}.pass_channels")
    if len(channels) < 2:
        raise ValueError(f"{name}.pass_channels must have at least 2 entries")
    return channels


def _shape_map(specs: dict[str, Any], key: str) -> dict[str, Any]:
    raw = specs.get(key, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a mapping")
    return raw


def _merged_shape(
    shapes: dict[str, Any],
    name: str,
    label: str,
    seen: tuple[str, ...] = (),
) -> dict[str, Any]:
    if name not in shapes:
        available = ", ".join(str(item) for item in shapes)
        raise ValueError(f"unknown {label} {name!r}; available: {available}")
    raw = shapes[name]
    if not isinstance(raw, dict):
        raise ValueError(f"{label} {name!r} must be a mapping")
    if "base" not in raw:
        return dict(raw)
    if name in seen:
        chain = " -> ".join((*seen, name))
        raise ValueError(f"{label} inheritance cycle: {chain}")
    base = _merged_shape(shapes, str(raw["base"]), label, (*seen, name))
    merged = dict(base)
    merged.update({key: value for key, value in raw.items() if key != "base"})
    return merged


def _stem_channels_from_shape(
    spec: dict[str, Any],
    specs: dict[str, Any],
    context: dict[str, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if "stem_channels" in spec or "rule_mlp_channels" in spec:
        return (
            _int_tuple(spec.get("stem_channels"), "stem_channels"),
            _int_tuple(spec.get("rule_mlp_channels"), "rule_mlp_channels"),
        )
    shape_name = str(spec.get("stem_shape", ""))
    if not shape_name:
        raise ValueError("spec must define stem_shape or explicit stem_channels")
    shape = _merged_shape(_shape_map(specs, "stem_shapes"), shape_name, "stem_shape")
    return (
        _channel_tuple(shape.get("stem_channels"), context, f"{shape_name}.stem_channels"),
        _channel_tuple(shape.get("rule_mlp_channels"), context, f"{shape_name}.rule_mlp_channels"),
    )


def _head_input(shape: dict[str, Any], key: str, context: dict[str, int], shape_name: str) -> int:
    return _resolve_int(shape.get(key), context, f"{shape_name}.{key}")


def _head_shape_channels(
    raw_shape: Any,
    context: dict[str, int],
    output_channels: Any,
    label: str,
) -> list[int]:
    local_context = dict(context)
    local_context["output"] = _resolve_int(output_channels, context, f"{label}.output")
    return list(_channel_tuple(raw_shape, local_context, label))


def _heads_from_shape(
    spec: dict[str, Any],
    specs: dict[str, Any],
    context: dict[str, int],
) -> dict[str, Any]:
    raw_heads = spec.get("heads")
    if raw_heads is not None:
        if not isinstance(raw_heads, dict):
            raise ValueError("heads must be a mapping")
        return raw_heads
    shape_name = str(spec.get("head_shape", ""))
    if not shape_name:
        raise ValueError("spec must define head_shape or explicit heads")
    shape = _merged_shape(_shape_map(specs, "head_shapes"), shape_name, "head_shape")
    collapse = str(shape.get("collapse", "none"))
    heads: dict[str, dict[str, Any]] = {}

    global_heads = shape.get("global_heads", {})
    spatial_heads = shape.get("spatial_heads", {})
    pass_heads = shape.get("pass_heads", {})
    if (
        not isinstance(global_heads, dict)
        or not isinstance(spatial_heads, dict)
        or not isinstance(pass_heads, dict)
    ):
        raise ValueError(f"{shape_name} head groups must be mappings")
    global_shape = shape.get("global_shape")
    spatial_shape = shape.get("spatial_shape")
    if global_shape is not None or spatial_shape is not None:
        if global_shape is None or spatial_shape is None:
            raise ValueError(f"{shape_name} must define both global_shape and spatial_shape")
        for head_name, outputs in spatial_heads.items():
            name = str(head_name)
            heads[name] = {
                "channels": _head_shape_channels(
                    spatial_shape,
                    context,
                    outputs,
                    f"{shape_name}.{name}.spatial_shape",
                ),
                "collapse": collapse,
            }
        for head_name, outputs in global_heads.items():
            name = str(head_name)
            channels = _head_shape_channels(
                global_shape,
                context,
                outputs,
                f"{shape_name}.{name}.global_shape",
            )
            if name.startswith("pass_"):
                target = name.removeprefix("pass_")
                if target not in heads:
                    raise ValueError(
                        f"{shape_name} global pass head {name!r} has no matching spatial head"
                    )
                heads[target]["pass_channels"] = channels
            else:
                heads[name] = {"channels": channels, "collapse": collapse}
        return heads

    hidden = _resolve_int(shape.get("hidden"), context, f"{shape_name}.hidden")
    global_input = _head_input(shape, "global_input", context, shape_name)
    spatial_input = _head_input(shape, "spatial_input", context, shape_name)
    for head_name, outputs in global_heads.items():
        heads[str(head_name)] = {
            "channels": [
                global_input,
                hidden,
                _resolve_int(outputs, context, f"{shape_name}.{head_name}"),
            ],
            "collapse": collapse,
        }
    for head_name, outputs in spatial_heads.items():
        heads[str(head_name)] = {
            "channels": [
                spatial_input,
                hidden,
                _resolve_int(outputs, context, f"{shape_name}.{head_name}"),
            ],
            "collapse": collapse,
        }
    for head_name, outputs in pass_heads.items():
        name = str(head_name)
        if name not in heads:
            raise ValueError(f"{shape_name} pass head {name!r} has no matching spatial head")
        heads[name]["pass_channels"] = [
            global_input,
            hidden,
            _resolve_int(outputs, context, f"{shape_name}.{name}.pass"),
        ]
    return heads


def _expanded_channel(trunk: dict[str, Any]) -> int:
    for key in ("expanded_channel", "trunk_channel", "trunk_channels"):
        if key in trunk:
            return int(trunk[key])
    raise ValueError("trunk must define expanded_channel")


def _bottleneck_channel(trunk: dict[str, Any]) -> int:
    for key in ("bottleneck_channel", "bottleneck_channels"):
        if key in trunk:
            return int(trunk[key])
    raise ValueError("trunk must define bottleneck_channel")


def _head_dim(trunk: dict[str, Any], bottleneck_channel: int, q_heads: int) -> int:
    if "head_dim" in trunk:
        return int(trunk["head_dim"])
    if bottleneck_channel % q_heads != 0:
        raise ValueError("bottleneck_channel must be divisible by q_heads to derive head_dim")
    return bottleneck_channel // q_heads


def config_from_spec(
    model_name: str | None = None,
    path: str | Path | None = None,
    board_size: int | None = None,
) -> SakiGoModelConfig:
    specs = load_model_specs(path)
    name = model_name or str(specs.get("default_model", "model1"))
    models = specs["models"]
    if name not in models:
        available = ", ".join(str(item) for item in models)
        raise ValueError(f"unknown model spec {name!r}; available specs: {available}")
    spec = models[name]
    if not isinstance(spec, dict):
        raise ValueError(f"model spec {name!r} must be a mapping")
    architecture_tag = str(spec.get("architecture", "d4_equivariant"))
    if architecture_tag not in ARCHITECTURE_TAGS:
        available = ", ".join(sorted(ARCHITECTURE_TAGS))
        raise ValueError(
            f"{name} uses unsupported architecture tag {architecture_tag!r}; "
            f"available: {available}"
        )
    architecture = ARCHITECTURE_TAGS[architecture_tag]

    trunk = spec.get("trunk", {})
    if not isinstance(trunk, dict):
        raise ValueError(f"{name} must contain a trunk mapping")

    gather_blocks_raw = trunk.get("register_gather_blocks", "all")
    if gather_blocks_raw == "all":
        gather_blocks: tuple[int, ...] | None = None
    else:
        gather_blocks = _int_tuple(gather_blocks_raw, "register_gather_blocks")

    register_count = int(trunk["register_count"])
    expanded_channel = _expanded_channel(trunk)
    trunk_channels = expanded_channel
    expected_register_input = register_count * expanded_channel
    bottleneck_channels = _bottleneck_channel(trunk)
    q_heads = int(trunk["q_heads"])
    kv_heads = int(trunk["kv_heads"])
    head_dim = _head_dim(trunk, bottleneck_channels, q_heads)

    context = {
        "register_count": register_count,
        "expanded_channel": expanded_channel,
        "bottleneck_channel": bottleneck_channels,
        "trunk_channel": expanded_channel,
        "trunk_channels": expanded_channel,
        "bottleneck_channels": bottleneck_channels,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }
    if "input_planes" in spec:
        context["input_planes"] = int(spec["input_planes"])
    if "rule_dim" in spec:
        context["rule_dim"] = int(spec["rule_dim"])

    stem_channels, rule_mlp_channels = _stem_channels_from_shape(spec, specs, context)
    if len(stem_channels) < 2:
        raise ValueError("stem_channels must contain at least input and output channels")
    if len(rule_mlp_channels) < 2:
        raise ValueError("rule_mlp_channels must contain at least input and output channels")
    input_planes = int(spec.get("input_planes", stem_channels[0]))
    rule_dim = int(spec.get("rule_dim", rule_mlp_channels[0]))
    if stem_channels[0] != input_planes:
        raise ValueError("input_planes must match the first stem channel")
    if rule_mlp_channels[0] != rule_dim:
        raise ValueError("rule_dim must match the first rule MLP channel")
    context["input_planes"] = input_planes
    context["rule_dim"] = rule_dim

    if stem_channels[-1] != trunk_channels:
        raise ValueError("stem output must match expanded_channel")
    if rule_mlp_channels[-1] != expected_register_input:
        raise ValueError("rule MLP output must equal register_count * expanded_channel")

    heads = _heads_from_shape(spec, specs, context)
    wdl_channels = _head_channels(heads, "wdl")
    score_channels = _head_channels(heads, "score")
    ownership_channels = _head_channels(heads, "ownership")
    policy_channels = _head_channels(heads, "policy")
    policy_pass_channels = _pass_head_channels(heads, "policy")
    budget_channels = _head_channels(heads, "budget")
    budget_pass_channels = _pass_head_channels(heads, "budget")
    for head_name, channels in (
        ("wdl", wdl_channels),
        ("score", score_channels),
        ("policy.pass", policy_pass_channels),
        ("budget.pass", budget_pass_channels),
    ):
        if channels[0] != expected_register_input:
            raise ValueError(
                f"{head_name} head input must equal register_count * expanded_channel"
            )
    for head_name, channels in (
        ("ownership", ownership_channels),
        ("policy", policy_channels),
        ("budget", budget_channels),
    ):
        if channels[0] != trunk_channels:
            raise ValueError(f"{head_name} head input must equal expanded_channel")

    global_frequencies = tuple(
        _parse_float(item, "global_rope_frequencies")
        for item in trunk.get("global_rope_frequencies", [])
    )
    local_frequencies = tuple(
        _parse_float(item, "local_rope_frequencies")
        for item in trunk.get("local_rope_frequencies", [])
    )
    if not global_frequencies and not local_frequencies:
        raise ValueError("at least one global or local rope frequency is required")
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
        trunk_channels=trunk_channels,
        expanded_channel=expanded_channel,
        bottleneck_channels=bottleneck_channels,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        global_rope_frequencies=global_frequencies,
        local_rope_frequencies=local_frequencies,
        gather_blocks=gather_blocks,
        broadcast_blocks=_int_tuple(
            trunk.get("register_broadcast_blocks", []),
            "register_broadcast_blocks",
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


def model_from_spec(model_name: str | None = None):
    config = config_from_spec(model_name)
    if config.architecture == "SakiGoModel":
        from .model import SakiGoModel

        return SakiGoModel(config)
    if config.architecture == "ScalarSakiGoModel":
        from .scalar_model import ScalarSakiGoModel

        return ScalarSakiGoModel(config)
    raise ValueError(f"unsupported architecture {config.architecture!r}")


def config_from_checkpoint(
    checkpoint: dict[str, Any],
    minimum_board_size: int = 5,
) -> SakiGoModelConfig:
    raw_model_config = checkpoint.get("model_config")
    if isinstance(raw_model_config, dict):
        kwargs = {key: value for key, value in raw_model_config.items() if key in CONFIG_FIELDS}
        for key in CONFIG_TUPLE_FIELDS:
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        config = SakiGoModelConfig(**kwargs)
    else:
        raw_config = checkpoint.get("config", {})
        if isinstance(raw_config, dict) and raw_config.get("model_spec"):
            config = config_from_spec(str(raw_config["model_spec"]))
        else:
            config = SakiGoModelConfig()

    raw_config = checkpoint.get("config", {})
    saved_board_size = 0
    if isinstance(raw_config, dict):
        saved_board_size = int(raw_config.get("model_board_size", 0) or 0)
    board_size = max(minimum_board_size, saved_board_size, config.board_size)
    if board_size != config.board_size:
        config = replace(config, board_size=board_size)
    return config
