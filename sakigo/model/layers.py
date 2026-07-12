"""SAKIGo trunk layers built from the reusable equivariant_attention package."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn

from equivariant_attention import (
    InvariantPool,
    RegularCrossAttention,
    RegularLift,
    RegularLinear1x1,
    RegularPointwiseMLP,
    RegularRMSNorm,
    RegularSelfAttention,
    RegisterToSpatialAttention,
    SpatialToRegisterAttention,
)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
}


def build_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        available = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"unknown activation {name!r}; available: {available}")
    return _ACTIVATIONS[key]()


class GroupLift(RegularLift):
    def __init__(self, group_size: int) -> None:
        super().__init__(group_size)


class GroupLinear1x1(RegularLinear1x1):
    def __init__(self, in_channels: int, out_channels: int, group_size: int, bias: bool = True) -> None:
        super().__init__(in_channels, out_channels, group_size, bias=bias)


class GroupRMSNorm(RegularRMSNorm):
    pass


class GroupPointwiseMLP(RegularPointwiseMLP):
    def __init__(self, channels: tuple[int, ...], group_size: int, final_activation: bool = False) -> None:
        super().__init__(channels, group_size, final_activation=final_activation)


class InvariantHead(InvariantPool):
    pass


class GroupCrossAttention(RegularCrossAttention):
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
        super().__init__(
            query_channels=query_channels,
            key_channels=key_channels,
            output_channels=output_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_spatial=query_is_board,
            key_is_spatial=key_is_board,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group=group_size,
        )


class GroupGQAAttention(RegularSelfAttention):
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
            channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )


class RegisterGatherAttention(RegisterToSpatialAttention):
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
            register_channels,
            board_channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )


class RegisterBroadcastAttention(SpatialToRegisterAttention):
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
            board_channels,
            register_channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )


class BoardBlock(nn.Module):
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
        activation: str = "none",
    ) -> None:
        super().__init__()
        scale = 1.0 / sqrt(2.0 * block_count)
        self.alpha_1 = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.alpha_2 = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

        self.norm_in = GroupRMSNorm(trunk_channels, eps)
        self.norm_bottleneck_1 = GroupRMSNorm(bottleneck_channels, eps)
        self.norm_bottleneck_2 = GroupRMSNorm(bottleneck_channels, eps)
        self.norm_bottleneck_3 = GroupRMSNorm(bottleneck_channels, eps)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normed_in = self.norm_in(residual)
        x1 = self.f1_activation(self.f1(normed_in))
        x2 = x1 + self.alpha_1 * self.attn1(self.norm_bottleneck_1(x1))
        x3 = x2 + self.alpha_2 * self.attn2(self.norm_bottleneck_2(x2))
        delta = self.norm_bottleneck_3(x3)
        return residual + self.beta * self.f4(delta)


class RegisterGather(nn.Module):
    """Add a board-sourced cross-attention delta to the register residual."""

    def __init__(
        self,
        trunk_channels: int,
        register_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        register_head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        eps: float,
        group_size: int,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0 / sqrt(2.0), dtype=torch.float32))
        self.norm_board = GroupRMSNorm(trunk_channels, eps)
        self.norm_registers = GroupRMSNorm(register_channels, eps)
        self.attention = RegisterGatherAttention(
            register_channels,
            trunk_channels,
            board_size,
            q_heads,
            kv_heads,
            register_head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )

    def forward(self, x: torch.Tensor, registers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        registers = registers + self.gamma * self.attention(
            self.norm_registers(registers), self.norm_board(x)
        )
        return x, registers


class RegisterBroadcast(nn.Module):
    """Add a register-sourced cross-attention delta to the board residual."""

    def __init__(
        self,
        trunk_channels: int,
        register_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        register_head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        eps: float,
        group_size: int,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0 / sqrt(2.0), dtype=torch.float32))
        self.norm_board = GroupRMSNorm(trunk_channels, eps)
        self.norm_registers = GroupRMSNorm(register_channels, eps)
        self.attention = RegisterBroadcastAttention(
            trunk_channels,
            register_channels,
            board_size,
            q_heads,
            kv_heads,
            register_head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
            group_size,
        )

    def forward(self, x: torch.Tensor, registers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.gamma * self.attention(
            self.norm_board(x), self.norm_registers(registers)
        )
        return x, registers
