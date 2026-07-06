"""Model configuration (ported from Model/sakigo_model/config.py, field-compatible).

`architecture` selects the group size of the unified stack:
"SakiGoModel" -> 8 (D4-equivariant), "ScalarSakiGoModel" -> 1 (scalar control).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi

ARCHITECTURE_GROUP_SIZES = {
    "SakiGoModel": 8,
    "ScalarSakiGoModel": 1,
}


@dataclass(frozen=True)
class SakiGoModelConfig:
    architecture: str = "SakiGoModel"
    board_size: int = 32
    input_planes: int = 6
    rule_dim: int = 10
    stem_channels: tuple[int, ...] = (6, 16, 32)
    rule_mlp_channels: tuple[int, ...] = (10, 32, 64)
    activation: str = "silu"
    block_count: int = 5
    register_count: int = 2
    trunk_channels: int = 32
    expanded_channel: int | None = None
    bottleneck_channels: int = 16
    q_heads: int = 2
    kv_heads: int = 1
    head_dim: int = 8
    global_rope_frequencies: tuple[float, ...] = (pi,)
    local_rope_frequencies: tuple[float, ...] = (pi / 2,)
    gather_blocks: tuple[int, ...] | None = None
    broadcast_blocks: tuple[int, ...] = (5,)
    wdl_hidden: int = 8
    wdl_outputs: int = 4
    wdl_channels: tuple[int, ...] | None = None
    score_hidden: int = 8
    score_outputs: int = 1
    score_channels: tuple[int, ...] | None = None
    ownership_hidden: int = 8
    ownership_outputs: int = 1
    ownership_channels: tuple[int, ...] | None = None
    policy_hidden: int = 8
    policy_outputs: int = 1
    policy_channels: tuple[int, ...] | None = None
    policy_pass_hidden: int = 8
    policy_pass_outputs: int = 1
    policy_pass_channels: tuple[int, ...] | None = None
    budget_hidden: int = 8
    budget_outputs: int = 1
    budget_channels: tuple[int, ...] | None = None
    budget_pass_hidden: int = 8
    budget_pass_outputs: int = 1
    budget_pass_channels: tuple[int, ...] | None = None
    norm_eps: float = 1e-6

    @property
    def group_size(self) -> int:
        if self.architecture not in ARCHITECTURE_GROUP_SIZES:
            available = ", ".join(sorted(ARCHITECTURE_GROUP_SIZES))
            raise ValueError(f"unknown architecture {self.architecture!r}; available: {available}")
        return ARCHITECTURE_GROUP_SIZES[self.architecture]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def config_from_dict(raw: dict[str, object]) -> SakiGoModelConfig:
    """Rebuild a config from a plain dict (checkpoint round-trip), tolerating lists."""
    field_names = SakiGoModelConfig.__dataclass_fields__
    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key not in field_names:
            continue
        if isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return SakiGoModelConfig(**kwargs)  # type: ignore[arg-type]
