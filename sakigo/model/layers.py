"""Unified group-equivariant building blocks (group_size ∈ {1, 8}).

State-dict compatible with the legacy Model/sakigo_model regular stack for
group_size=8 (same module/parameter names and shapes). For group_size=1 the
legacy scalar weights map via `unsqueeze(-1)` on GroupLinear1x1 weights.

torch.compile-ready: no module-global tensor caches, no mutable state in
forward; static tables live in non-persistent registered buffers, and RoPE
tables are computed functionally (constant-folded per board-size graph).
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from sakigo.model.group import (
    SUPPORTED_GROUP_SIZES,
    canonical_coordinates,
    relative_component_table,
)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
}


def build_activation(name: str) -> nn.Module:
    """Build the trunk activation by case-insensitive name."""
    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        available = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"unknown activation {name!r}; available: {available}")
    return _ACTIVATIONS[key]()


def _check_group_size(group_size: int) -> int:
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUP_SIZES}")
    return group_size


class GroupLift(nn.Module):
    """Lift scalar board planes [B,C,H,W] to group features [B,C,G,H,W]."""

    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = _check_group_size(group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("GroupLift expects [B,C,H,W]")
        return x.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)


class GroupLinear1x1(nn.Module):
    """Equivariant pointwise fiber mixing with a relative-group kernel.

    weight: [out_channels, in_channels, G]. For G=1 this is a plain linear map.
    """

    def __init__(self, in_channels: int, out_channels: int, group_size: int, bias: bool = True) -> None:
        super().__init__()
        self.group_size = _check_group_size(group_size)
        scale = 1.0 / sqrt(max(in_channels * group_size, 1))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, group_size))
        nn.init.uniform_(self.weight, -scale, scale)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer("relative", relative_component_table(group_size), persistent=False)

    def _flat_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        group = self.group_size
        kernel = self.weight[:, :, self.relative]
        out_channels, in_channels = kernel.shape[:2]
        weight = kernel.permute(0, 2, 1, 3).reshape(out_channels * group, in_channels * group)
        bias = None
        if self.bias is not None:
            bias = (
                self.bias.view(out_channels, 1)
                .expand(out_channels, group)
                .reshape(out_channels * group)
            )
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = self.group_size
        weight, bias = self._flat_weight_bias()
        out_channels = weight.shape[0] // group
        in_channels = weight.shape[1] // group
        if x.dim() == 5:
            if x.shape[2] != group:
                raise ValueError(f"group board features must have shape [B,C,{group},H,W]")
            batch, _, _, height, width = x.shape
            flat = x.permute(0, 3, 4, 1, 2).reshape(batch * height * width, in_channels * group)
            y = F.linear(flat, weight, bias)
            return y.reshape(batch, height, width, out_channels, group).permute(0, 3, 4, 1, 2)
        if x.dim() == 4:
            if x.shape[3] != group:
                raise ValueError(f"group register features must have shape [B,R,C,{group}]")
            batch, registers, _, _ = x.shape
            flat = x.reshape(batch * registers, in_channels * group)
            y = F.linear(flat, weight, bias)
            return y.reshape(batch, registers, out_channels, group)
        raise ValueError("GroupLinear1x1 expects [B,C,G,H,W] or [B,R,C,G]")


class GroupRMSNorm(nn.Module):
    """RMS norm over channels and group components."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            rms = torch.mean(x.square(), dim=(1, 2), keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, -1, 1, 1, 1)
        if x.dim() == 4:
            rms = torch.mean(x.square(), dim=(2, 3), keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, 1, -1, 1)
        raise ValueError("GroupRMSNorm expects [B,C,G,H,W] or [B,R,C,G]")


class GroupPointwiseMLP(nn.Module):
    def __init__(self, channels: tuple[int, ...], group_size: int, final_activation: bool = False) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("GroupPointwiseMLP needs at least two channel sizes")
        layers: list[nn.Module] = []
        last = len(channels) - 2
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            layers.append(GroupLinear1x1(in_channels, out_channels, group_size))
            if index < last or final_activation:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvariantHead(nn.Module):
    """Collapse the group axis (mean or sum); a no-op squeeze for group_size=1."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            dim = 2
        elif x.dim() == 4:
            dim = 3
        else:
            raise ValueError("InvariantHead expects group board or register features")
        if self.reduction == "mean":
            return x.mean(dim=dim)
        return x.sum(dim=dim)


def _rope_cos_sin(
    group_size: int,
    board_size: int,
    device: torch.device,
    dtype: torch.dtype,
    global_frequencies: tuple[float, ...],
    local_frequencies: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical-frame RoPE tables, computed functionally (traceable, no cache)."""
    rows, cols = canonical_coordinates(group_size, board_size, device, dtype)
    denom = max(board_size - 1, 1)
    channels: list[torch.Tensor] = []
    for frequency in global_frequencies:
        channels.append(frequency * rows / denom)
        channels.append(frequency * cols / denom)
    for frequency in local_frequencies:
        channels.append(frequency * rows)
        channels.append(frequency * cols)
    angles = torch.stack(channels, dim=-1)
    return angles.cos().unsqueeze(0).unsqueeze(2), angles.sin().unsqueeze(0).unsqueeze(2)


def _apply_rope(
    tokens: torch.Tensor,
    group_size: int,
    board_size: int,
    global_frequencies: tuple[float, ...],
    local_frequencies: tuple[float, ...],
) -> torch.Tensor:
    """Apply 2D RoPE to [B,G,H,L,D] board tokens; each frequency rotates 4 head dims."""
    frequency_count = len(global_frequencies) + len(local_frequencies)
    pair_count = 2 * frequency_count
    rope_dim = 2 * pair_count
    if tokens.shape[-1] < rope_dim:
        raise ValueError(
            f"{frequency_count}-frequency 2D RoPE needs at least {rope_dim} head dimensions"
        )
    cos, sin = _rope_cos_sin(
        group_size,
        board_size,
        tokens.device,
        tokens.dtype,
        global_frequencies,
        local_frequencies,
    )
    pairs = tokens[..., :rope_dim].reshape(*tokens.shape[:-1], pair_count, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
    rotated = rotated.reshape(*tokens.shape[:-1], rope_dim)
    if tokens.shape[-1] == rope_dim:
        return rotated
    return torch.cat((rotated, tokens[..., rope_dim:]), dim=-1)


def _board_tokens(x: torch.Tensor, heads: int, head_dim: int, group_size: int) -> torch.Tensor:
    batch, channels, group, height, width = x.shape
    if channels != heads * head_dim or group != group_size:
        raise ValueError("projected board channels do not match heads * head_dim")
    return x.reshape(batch, heads, head_dim, group, height * width).permute(0, 3, 1, 4, 2)


def _register_tokens(x: torch.Tensor, heads: int, head_dim: int, group_size: int) -> torch.Tensor:
    batch, registers, channels, group = x.shape
    if channels != heads * head_dim or group != group_size:
        raise ValueError("projected register channels do not match heads * head_dim")
    return x.reshape(batch, registers, heads, head_dim, group).permute(0, 4, 2, 1, 3)


def _tokens_to_board(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, group, heads, cells, head_dim = x.shape
    return x.permute(0, 2, 4, 1, 3).reshape(batch, heads * head_dim, group, height, width)


def _tokens_to_registers(x: torch.Tensor) -> torch.Tensor:
    batch, group, heads, registers, head_dim = x.shape
    return x.permute(0, 3, 2, 4, 1).reshape(batch, registers, heads * head_dim, group)


class GroupCrossAttention(nn.Module):
    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        output_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        query_is_board: bool,
        key_is_board: bool,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group_size: int,
    ) -> None:
        super().__init__()
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        self.group_size = _check_group_size(group_size)
        self.board_size = board_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.query_is_board = query_is_board
        self.key_is_board = key_is_board
        self.global_rope_frequencies = global_rope_frequencies
        self.local_rope_frequencies = local_rope_frequencies

        self.q_proj = GroupLinear1x1(query_channels, q_heads * head_dim, group_size)
        self.k_proj = GroupLinear1x1(key_channels, kv_heads * head_dim, group_size)
        self.v_proj = GroupLinear1x1(key_channels, kv_heads * head_dim, group_size)
        self.out_proj = GroupLinear1x1(q_heads * head_dim, output_channels, group_size)

    def _to_tokens(self, x: torch.Tensor, heads: int, is_board: bool) -> torch.Tensor:
        if is_board:
            return _board_tokens(x, heads, self.head_dim, self.group_size)
        return _register_tokens(x, heads, self.head_dim, self.group_size)

    def _active_board_size(self, query: torch.Tensor, key_value: torch.Tensor) -> int:
        if self.query_is_board:
            height, width = query.shape[-2:]
        elif self.key_is_board:
            height, width = key_value.shape[-2:]
        else:
            return self.board_size
        if height != width:
            raise ValueError("board features must be square")
        if height > self.board_size:
            raise ValueError(f"board size {height} exceeds configured maximum {self.board_size}")
        return height

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key_value)
        v_proj = self.v_proj(key_value)
        board_size = self._active_board_size(query, key_value)

        q = self._to_tokens(q_proj, self.q_heads, self.query_is_board)
        k = self._to_tokens(k_proj, self.kv_heads, self.key_is_board)
        v = self._to_tokens(v_proj, self.kv_heads, self.key_is_board)
        if self.query_is_board:
            q = _apply_rope(
                q,
                self.group_size,
                board_size,
                self.global_rope_frequencies,
                self.local_rope_frequencies,
            )
        if self.key_is_board:
            k = _apply_rope(
                k,
                self.group_size,
                board_size,
                self.global_rope_frequencies,
                self.local_rope_frequencies,
            )

        batch, group, _, query_tokens, head_dim = q.shape
        key_tokens = k.shape[-2]
        q_flat = q.reshape(batch * group, self.q_heads, query_tokens, head_dim)
        k_flat = k.reshape(batch * group, self.kv_heads, key_tokens, head_dim)
        v_flat = v.reshape(batch * group, self.kv_heads, key_tokens, head_dim)
        attended = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, enable_gqa=True)
        attended = attended.reshape(batch, group, self.q_heads, query_tokens, head_dim)
        if self.query_is_board:
            height = query.shape[-2]
            width = query.shape[-1]
            return self.out_proj(_tokens_to_board(attended, height, width))
        return self.out_proj(_tokens_to_registers(attended))


class GroupGQAAttention(GroupCrossAttention):
    def __init__(
        self,
        channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group_size: int,
    ) -> None:
        super().__init__(
            query_channels=channels,
            key_channels=channels,
            output_channels=channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_board=True,
            key_is_board=True,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group_size=group_size,
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor | None = None) -> torch.Tensor:
        if key_value is None:
            key_value = query
        return super().forward(query, key_value)


class RegisterToBoardAttention(GroupCrossAttention):
    def __init__(
        self,
        register_channels: int,
        board_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group_size: int,
    ) -> None:
        super().__init__(
            query_channels=register_channels,
            key_channels=board_channels,
            output_channels=register_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_board=False,
            key_is_board=True,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group_size=group_size,
        )


class BoardToRegisterAttention(GroupCrossAttention):
    def __init__(
        self,
        board_channels: int,
        register_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group_size: int,
    ) -> None:
        super().__init__(
            query_channels=board_channels,
            key_channels=register_channels,
            output_channels=board_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_board=True,
            key_is_board=False,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group_size=group_size,
        )


class TrunkBlock(nn.Module):
    def __init__(
        self,
        trunk_channels: int,
        bottleneck_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        block_count: int,
        eps: float,
        group_size: int,
        enable_gather: bool = True,
        enable_broadcast: bool = True,
        activation: str = "none",
    ) -> None:
        super().__init__()
        self.enable_gather = enable_gather
        self.enable_broadcast = enable_broadcast
        scale = 1.0 / sqrt(2.0 * block_count)
        self.alpha_1 = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.alpha_2 = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.gamma_1 = nn.Parameter(torch.tensor(scale, dtype=torch.float32)) if enable_gather else None
        self.gamma_2 = nn.Parameter(torch.tensor(scale, dtype=torch.float32)) if enable_broadcast else None

        self.norm_in = GroupRMSNorm(trunk_channels, eps)
        self.norm_reg = GroupRMSNorm(trunk_channels, eps) if enable_gather else None
        self.norm_bottleneck_1 = GroupRMSNorm(bottleneck_channels, eps)
        self.norm_bottleneck_2 = GroupRMSNorm(bottleneck_channels, eps)
        self.norm_bottleneck_3 = GroupRMSNorm(bottleneck_channels, eps)
        self.norm_out = GroupRMSNorm(trunk_channels, eps) if enable_broadcast else None
        self.norm_reg_out = GroupRMSNorm(trunk_channels, eps) if enable_broadcast else None

        self.f1 = GroupPointwiseMLP((trunk_channels, bottleneck_channels), group_size)
        self.f1_activation = build_activation(activation)
        self.attn1 = GroupGQAAttention(
            bottleneck_channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )
        self.attn2 = GroupGQAAttention(
            bottleneck_channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )
        self.f4 = GroupPointwiseMLP((bottleneck_channels, trunk_channels), group_size)
        self.register_gather = (
            RegisterToBoardAttention(
                trunk_channels,
                trunk_channels,
                board_size,
                q_heads,
                kv_heads,
                head_dim,
                global_rope_frequencies,
                local_rope_frequencies,
                group_size,
            )
            if enable_gather
            else None
        )
        self.register_broadcast = (
            BoardToRegisterAttention(
                trunk_channels,
                trunk_channels,
                board_size,
                q_heads,
                kv_heads,
                head_dim,
                global_rope_frequencies,
                local_rope_frequencies,
                group_size,
            )
            if enable_broadcast
            else None
        )

    def forward(self, x: torch.Tensor, registers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        normed_in = self.norm_in(residual)
        register_gather = self.register_gather
        if register_gather is not None:
            norm_reg = self.norm_reg
            gamma_1 = self.gamma_1
            if norm_reg is None or gamma_1 is None:
                raise RuntimeError("gather block is missing its paired modules")
            registers = registers + gamma_1 * register_gather(
                norm_reg(registers),
                normed_in,
            )
        x1 = self.f1_activation(self.f1(normed_in))
        x2 = x1 + self.alpha_1 * self.attn1(self.norm_bottleneck_1(x1))
        x3 = x2 + self.alpha_2 * self.attn2(self.norm_bottleneck_2(x2))
        out = residual + self.beta * self.f4(self.norm_bottleneck_3(x3))
        register_broadcast = self.register_broadcast
        if register_broadcast is not None:
            norm_out = self.norm_out
            norm_reg_out = self.norm_reg_out
            gamma_2 = self.gamma_2
            if norm_out is None or norm_reg_out is None or gamma_2 is None:
                raise RuntimeError("broadcast block is missing its paired modules")
            out = out + gamma_2 * register_broadcast(
                norm_out(out),
                norm_reg_out(registers),
            )
        return out, registers
