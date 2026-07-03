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
CONFIG_FIELDS = {field.name for field in fields(SakiGoModelConfig)}
CONFIG_TUPLE_FIELDS = {
    "stem_channels",
    "rule_mlp_channels",
    "global_rope_frequencies",
    "local_rope_frequencies",
    "gather_blocks",
    "broadcast_blocks",
}


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


def load_model_specs(path: str | Path | None = None) -> dict[str, Any]:
    spec_path = Path(path) if path is not None else DEFAULT_SPEC_PATH
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
    if "models" not in parsed or not isinstance(parsed["models"], dict):
        raise ValueError(f"{spec_path} must contain a models mapping")
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


def _head_channels(heads: dict[str, Any], name: str) -> tuple[int, int, int]:
    raw = heads.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{name} head must be a mapping")
    channels = _int_tuple(raw.get("channels"), f"{name}.channels")
    if len(channels) != 3:
        raise ValueError(f"{name}.channels must have exactly 3 entries")
    return channels  # type: ignore[return-value]


def _pass_head_channels(heads: dict[str, Any], name: str) -> tuple[int, int, int]:
    raw = heads.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{name} head must be a mapping")
    channels = _int_tuple(raw.get("pass_channels"), f"{name}.pass_channels")
    if len(channels) != 3:
        raise ValueError(f"{name}.pass_channels must have exactly 3 entries")
    return channels  # type: ignore[return-value]


def config_from_spec(
    model_name: str | None = None,
    path: str | Path | None = None,
    board_size: int | None = None,
) -> SakiGoModelConfig:
    spec = get_model_spec(model_name, path)
    name = model_name or "default"
    architecture = str(spec.get("architecture", "SakiGoModel"))
    if architecture not in {"SakiGoModel", "ScalarSakiGoModel"}:
        raise ValueError(f"{name} uses unsupported architecture {spec.get('architecture')!r}")

    trunk = spec.get("trunk", {})
    heads = spec.get("heads", {})
    if not isinstance(trunk, dict) or not isinstance(heads, dict):
        raise ValueError(f"{name} must contain trunk and heads mappings")

    gather_blocks_raw = trunk.get("register_gather_blocks", "all")
    if gather_blocks_raw == "all":
        gather_blocks: tuple[int, ...] | None = None
    else:
        gather_blocks = _int_tuple(gather_blocks_raw, "register_gather_blocks")

    stem_channels = _int_tuple(spec.get("stem_channels"), "stem_channels")
    rule_mlp_channels = _int_tuple(spec.get("rule_mlp_channels"), "rule_mlp_channels")
    if len(stem_channels) < 2:
        raise ValueError("stem_channels must contain at least input and output channels")
    if len(rule_mlp_channels) < 2:
        raise ValueError("rule_mlp_channels must contain at least input and output channels")

    register_count = int(trunk["register_count"])
    trunk_channels = int(trunk["trunk_channels"])
    expected_register_input = register_count * trunk_channels
    if stem_channels[-1] != trunk_channels:
        raise ValueError("stem output must match trunk_channels")
    if rule_mlp_channels[-1] != expected_register_input:
        raise ValueError("rule MLP output must equal register_count * trunk_channels")

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
            raise ValueError(f"{head_name} head input must equal register_count * trunk_channels")
    for head_name, channels in (
        ("ownership", ownership_channels),
        ("policy", policy_channels),
        ("budget", budget_channels),
    ):
        if channels[0] != trunk_channels:
            raise ValueError(f"{head_name} head input must equal trunk_channels")

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
        input_planes=int(spec["input_planes"]),
        rule_dim=int(spec["rule_dim"]),
        stem_channels=stem_channels,
        rule_mlp_channels=rule_mlp_channels,
        activation=str(spec.get("activation", "none")),
        block_count=int(trunk["block_count"]),
        register_count=register_count,
        trunk_channels=trunk_channels,
        bottleneck_channels=int(trunk["bottleneck_channels"]),
        q_heads=int(trunk["q_heads"]),
        kv_heads=int(trunk["kv_heads"]),
        head_dim=int(trunk["head_dim"]),
        global_rope_frequencies=global_frequencies,
        local_rope_frequencies=local_frequencies,
        gather_blocks=gather_blocks,
        broadcast_blocks=_int_tuple(
            trunk.get("register_broadcast_blocks", []),
            "register_broadcast_blocks",
        ),
        wdl_hidden=wdl_channels[1],
        wdl_outputs=wdl_channels[2],
        score_hidden=score_channels[1],
        score_outputs=score_channels[2],
        ownership_hidden=ownership_channels[1],
        ownership_outputs=ownership_channels[2],
        policy_hidden=policy_channels[1],
        policy_outputs=policy_channels[2],
        policy_pass_hidden=policy_pass_channels[1],
        policy_pass_outputs=policy_pass_channels[2],
        budget_hidden=budget_channels[1],
        budget_outputs=budget_channels[2],
        budget_pass_hidden=budget_pass_channels[1],
        budget_pass_outputs=budget_pass_channels[2],
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
