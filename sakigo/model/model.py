"""Unified SAKIGo network: one implementation for the D4-equivariant model
(group_size=8) and the scalar control (group_size=1), selected by
config.architecture. State-dict compatible with the legacy SakiGoModel for
group_size=8; legacy scalar checkpoints load via
sakigo.model.checkpoints.remap_legacy_scalar_state_dict.
"""

from __future__ import annotations

import torch
from torch import nn

from sakigo.model.config import SakiGoModelConfig
from sakigo.model.layers import (
    GroupLift,
    GroupPointwiseMLP,
    InvariantHead,
    TrunkBlock,
)


def scalar_mlp(channels: tuple[int, ...]) -> nn.Sequential:
    if len(channels) < 2:
        raise ValueError("scalar MLP needs at least two channel sizes")
    layers: list[nn.Module] = []
    last = len(channels) - 2
    for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
        layers.append(nn.Linear(in_channels, out_channels))
        if index < last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class SakiGoNet(nn.Module):
    """Spec-backed SAKIGo model with an explicit group axis (G ∈ {1, 8})."""

    def __init__(self, config: SakiGoModelConfig | str | None = None) -> None:
        super().__init__()
        if config is None or isinstance(config, str):
            from sakigo.model.specs import config_from_spec

            self.config = config_from_spec(config)
        else:
            self.config = config
        self.group_size = self.config.group_size
        self._validate_config()
        group_size = self.group_size

        self.lift = GroupLift(group_size)
        self.stem = GroupPointwiseMLP(self.config.stem_channels, group_size)
        register_channels = self._register_channels()
        self.register_seed = nn.Parameter(
            torch.zeros(self.config.register_count, register_channels)
        )
        nn.init.normal_(self.register_seed, mean=0.0, std=0.02)
        self.rule_mlp = scalar_mlp(self.config.rule_mlp_channels)

        gather_blocks = self._gather_blocks()
        broadcast_blocks = set(self.config.broadcast_blocks)
        self.blocks = nn.ModuleList(
            [
                TrunkBlock(
                    trunk_channels=self.config.trunk_channels,
                    register_channels=register_channels,
                    bottleneck_channels=self.config.bottleneck_channels,
                    register_bottleneck_channels=self._register_bottleneck_channels(),
                    board_size=self.config.board_size,
                    q_heads=self.config.q_heads,
                    kv_heads=self.config.kv_heads,
                    head_dim=self.config.head_dim,
                    register_head_dim=self._register_head_dim(),
                    global_rope_frequencies=self.config.global_rope_frequencies,
                    local_rope_frequencies=self.config.local_rope_frequencies,
                    block_count=self.config.block_count,
                    eps=self.config.norm_eps,
                    group_size=group_size,
                    enable_gather=index + 1 in gather_blocks,
                    enable_broadcast=index + 1 in broadcast_blocks,
                    activation=self.config.activation,
                )
                for index in range(self.config.block_count)
            ]
        )

        register_input = self._register_input_channels()
        self.invariant = InvariantHead("mean")

        def head(
            channels: tuple[int, ...] | None,
            input_channels: int,
            hidden: int,
            outputs: int,
        ) -> GroupPointwiseMLP:
            return GroupPointwiseMLP(
                channels or (input_channels, hidden, outputs),
                group_size,
            )

        config = self.config
        self.wdl_head = head(config.wdl_channels, register_input, config.wdl_hidden, config.wdl_outputs)
        self.score_head = head(config.score_channels, register_input, config.score_hidden, config.score_outputs)
        self.policy_pass_head = head(
            config.policy_pass_channels, register_input, config.policy_pass_hidden, config.policy_pass_outputs
        )
        self.budget_pass_head = head(
            config.budget_pass_channels, register_input, config.budget_pass_hidden, config.budget_pass_outputs
        )
        self.ownership_head = head(
            config.ownership_channels, config.trunk_channels, config.ownership_hidden, config.ownership_outputs
        )
        self.policy_head = head(
            config.policy_channels, config.trunk_channels, config.policy_hidden, config.policy_outputs
        )
        self.budget_head = head(
            config.budget_channels, config.trunk_channels, config.budget_hidden, config.budget_outputs
        )

    def _gather_blocks(self) -> set[int]:
        return (
            set(range(1, self.config.block_count + 1))
            if self.config.gather_blocks is None
            else set(self.config.gather_blocks)
        )

    def _register_channels(self) -> int:
        return self.config.register_channels or self.config.trunk_channels

    def _register_bottleneck_channels(self) -> int:
        return self.config.register_bottleneck_channels or self.config.bottleneck_channels

    def _register_head_dim(self) -> int:
        return self.config.register_head_dim or (
            self._register_bottleneck_channels() // self.config.q_heads
        )

    def _register_input_channels(self) -> int:
        register_input = self.config.register_count * self._register_channels()
        if (
            self.config.expanded_channel is not None
            and self.config.expanded_channel != self.config.trunk_channels
        ):
            raise ValueError("expanded_channel must equal trunk_channels")
        return register_input

    def _validate_config(self) -> None:
        config = self.config
        if config.input_planes != config.stem_channels[0]:
            raise ValueError("input_planes must match the first stem channel")
        if config.rule_dim != config.rule_mlp_channels[0]:
            raise ValueError("rule_dim must match the first rule MLP channel")
        register_input = self._register_input_channels()
        if config.rule_mlp_channels[-1] != register_input:
            raise ValueError("rule MLP output must equal register_count * register_channels")
        if config.stem_channels[-1] != config.trunk_channels:
            raise ValueError("stem output must match trunk_channels")
        if config.q_heads * config.head_dim != config.bottleneck_channels:
            raise ValueError("q_heads * head_dim must match bottleneck_channels")
        if config.q_heads * self._register_head_dim() != self._register_bottleneck_channels():
            raise ValueError("q_heads * register_head_dim must match register_bottleneck_channels")
        if config.q_heads % config.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        rope_frequency_count = len(config.global_rope_frequencies) + len(config.local_rope_frequencies)
        if rope_frequency_count < 1:
            raise ValueError("at least one global or local RoPE frequency is required")
        if config.head_dim < 4 * rope_frequency_count:
            raise ValueError("head_dim must be at least 4 * total RoPE frequency count")
        if self._register_head_dim() < 4 * rope_frequency_count:
            raise ValueError("register_head_dim must be at least 4 * total RoPE frequency count")
        for label, channels, expected in (
            ("wdl_channels", config.wdl_channels, register_input),
            ("score_channels", config.score_channels, register_input),
            ("policy_pass_channels", config.policy_pass_channels, register_input),
            ("budget_pass_channels", config.budget_pass_channels, register_input),
            ("ownership_channels", config.ownership_channels, config.trunk_channels),
            ("policy_channels", config.policy_channels, config.trunk_channels),
            ("budget_channels", config.budget_channels, config.trunk_channels),
        ):
            if channels is not None and channels[0] != expected:
                raise ValueError(f"{label} must start with {expected} input channels")
        for label, outputs in (
            ("score_outputs", config.score_outputs),
            ("ownership_outputs", config.ownership_outputs),
            ("policy_outputs", config.policy_outputs),
            ("policy_pass_outputs", config.policy_pass_outputs),
            ("budget_outputs", config.budget_outputs),
            ("budget_pass_outputs", config.budget_pass_outputs),
        ):
            if outputs != 1:
                raise ValueError(f"{label} must be 1")
        if config.wdl_outputs != 4:
            raise ValueError("wdl_outputs must be 4")
        broadcast_blocks = set(config.broadcast_blocks)
        gather_blocks = self._gather_blocks()
        if not gather_blocks:
            raise ValueError("at least one register gather block is required")
        for block in gather_blocks | broadcast_blocks:
            if block < 1 or block > config.block_count:
                raise ValueError("register blocks must be 1-based trunk block numbers")

    def initial_registers(self, rules: torch.Tensor) -> torch.Tensor:
        batch_size = rules.shape[0]
        seed = self.register_seed.to(device=rules.device, dtype=rules.dtype).unsqueeze(0)
        rule_delta = self.rule_mlp(rules).reshape(
            batch_size,
            self.config.register_count,
            self._register_channels(),
        )
        registers = seed + rule_delta
        return registers.unsqueeze(-1).expand(-1, -1, -1, self.group_size).contiguous()

    def _merged_registers(self, registers: torch.Tensor, batch_size: int) -> torch.Tensor:
        return registers.reshape(
            batch_size,
            1,
            self.config.register_count * self._register_channels(),
            self.group_size,
        )

    def _spatial_logits(self, features: torch.Tensor) -> torch.Tensor:
        board = self.invariant(features).squeeze(1)
        return board.reshape(features.shape[0], features.shape[-2] * features.shape[-1])

    def forward(self, board: torch.Tensor, rules: torch.Tensor) -> dict[str, torch.Tensor]:
        if board.dim() != 4:
            raise ValueError("SakiGoNet expects board shape [B,6,N,N]")
        if board.shape[1] != self.config.input_planes:
            raise ValueError(f"expected {self.config.input_planes} input planes")
        if board.shape[-1] != board.shape[-2]:
            raise ValueError("expected a square board")
        if board.shape[-1] > self.config.board_size:
            raise ValueError(f"expected board size <= {self.config.board_size}")
        if rules.dim() != 2:
            raise ValueError("rules must have shape [B,rule_dim]")
        if rules.shape[0] != board.shape[0]:
            raise ValueError("board and rules batch sizes must match")
        if rules.shape[1] != self.config.rule_dim:
            raise ValueError(f"expected {self.config.rule_dim} rule features")
        rules = rules.to(device=board.device, dtype=board.dtype)

        board_features = self.stem(self.lift(board))
        registers = self.initial_registers(rules)
        for block in self.blocks:
            board_features, registers = block(board_features, registers)

        merged_registers = self._merged_registers(registers, board.shape[0])
        wdl_logits = self.invariant(self.wdl_head(merged_registers)).squeeze(1)
        score = self.invariant(self.score_head(merged_registers)).squeeze(1)
        policy_pass = self.invariant(self.policy_pass_head(merged_registers)).squeeze(1)
        budget_pass = self.invariant(self.budget_pass_head(merged_registers)).squeeze(1)

        ownership_logits = self._spatial_logits(self.ownership_head(board_features))
        policy_board = self._spatial_logits(self.policy_head(board_features))
        budget_board = self._spatial_logits(self.budget_head(board_features))
        policy_logits = torch.cat((policy_board, policy_pass), dim=1)
        budget_logits = torch.cat((budget_board, budget_pass), dim=1)
        return {
            "wdl_logits": wdl_logits,
            "score": score,
            "ownership_logits": ownership_logits,
            "policy_logits": policy_logits,
            "budget_logits": budget_logits,
        }
