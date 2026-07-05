from __future__ import annotations

import torch
from torch import nn

from .config import SakiGoModelConfig
from .model import _scalar_mlp
from .scalar_layers import ScalarPointwiseMLP, ScalarTrunkBlock


class ScalarSakiGoModel(nn.Module):
    """Non-equivariant scalar control model with the same public API as SakiGoModel."""

    def __init__(self, config: SakiGoModelConfig | str | None = None) -> None:
        super().__init__()
        if config is None:
            self.config = SakiGoModelConfig(architecture="ScalarSakiGoModel")
        elif isinstance(config, str):
            from .specs import config_from_spec

            self.config = config_from_spec(config)
        else:
            self.config = config
        self._validate_config()

        self.stem = ScalarPointwiseMLP(self.config.stem_channels)
        self.register_seed = nn.Parameter(
            torch.zeros(self.config.register_count, self.config.trunk_channels)
        )
        nn.init.normal_(self.register_seed, mean=0.0, std=0.02)
        self.rule_mlp = _scalar_mlp(self.config.rule_mlp_channels)

        gather_blocks = self._gather_blocks()
        broadcast_blocks = set(self.config.broadcast_blocks)
        self.blocks = nn.ModuleList(
            [
                ScalarTrunkBlock(
                    trunk_channels=self.config.trunk_channels,
                    bottleneck_channels=self.config.bottleneck_channels,
                    board_size=self.config.board_size,
                    q_heads=self.config.q_heads,
                    kv_heads=self.config.kv_heads,
                    head_dim=self.config.head_dim,
                    global_rope_frequencies=self.config.global_rope_frequencies,
                    local_rope_frequencies=self.config.local_rope_frequencies,
                    block_count=self.config.block_count,
                    eps=self.config.norm_eps,
                    enable_gather=index + 1 in gather_blocks,
                    enable_broadcast=index + 1 in broadcast_blocks,
                    activation=self.config.activation,
                )
                for index in range(self.config.block_count)
            ]
        )

        register_input = self._register_input_channels()
        self.wdl_head = ScalarPointwiseMLP(
            self._head_channels(self.config.wdl_channels, register_input, self.config.wdl_hidden, self.config.wdl_outputs)
        )
        self.score_head = ScalarPointwiseMLP(
            self._head_channels(self.config.score_channels, register_input, self.config.score_hidden, self.config.score_outputs)
        )
        self.policy_pass_head = ScalarPointwiseMLP(
            self._head_channels(
                self.config.policy_pass_channels,
                register_input,
                self.config.policy_pass_hidden,
                self.config.policy_pass_outputs,
            )
        )
        self.budget_pass_head = ScalarPointwiseMLP(
            self._head_channels(
                self.config.budget_pass_channels,
                register_input,
                self.config.budget_pass_hidden,
                self.config.budget_pass_outputs,
            )
        )
        self.ownership_head = ScalarPointwiseMLP(
            self._head_channels(
                self.config.ownership_channels,
                self.config.trunk_channels,
                self.config.ownership_hidden,
                self.config.ownership_outputs,
            )
        )
        self.policy_head = ScalarPointwiseMLP(
            self._head_channels(
                self.config.policy_channels,
                self.config.trunk_channels,
                self.config.policy_hidden,
                self.config.policy_outputs,
            )
        )
        self.budget_head = ScalarPointwiseMLP(
            self._head_channels(
                self.config.budget_channels,
                self.config.trunk_channels,
                self.config.budget_hidden,
                self.config.budget_outputs,
            )
        )

    def _gather_blocks(self) -> set[int]:
        return (
            set(range(1, self.config.block_count + 1))
            if self.config.gather_blocks is None
            else set(self.config.gather_blocks)
        )

    def _head_channels(
        self,
        channels: tuple[int, ...] | None,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
    ) -> tuple[int, ...]:
        return channels or (input_channels, hidden_channels, output_channels)

    def _register_input_channels(self) -> int:
        register_input = self.config.register_count * self.config.trunk_channels
        if (
            self.config.expanded_channel is not None
            and self.config.expanded_channel != self.config.trunk_channels
        ):
            raise ValueError("expanded_channel must equal trunk_channels")
        return register_input

    def _validate_config(self) -> None:
        if self.config.architecture != "ScalarSakiGoModel":
            raise ValueError(
                f"ScalarSakiGoModel cannot build architecture {self.config.architecture!r}"
            )
        if self.config.input_planes != self.config.stem_channels[0]:
            raise ValueError("input_planes must match the first stem channel")
        if self.config.rule_dim != self.config.rule_mlp_channels[0]:
            raise ValueError("rule_dim must match the first rule MLP channel")
        register_input = self._register_input_channels()
        if self.config.rule_mlp_channels[-1] != register_input:
            raise ValueError("rule MLP output must equal register_count * trunk_channels")
        if self.config.stem_channels[-1] != self.config.trunk_channels:
            raise ValueError("stem output must match trunk_channels")
        if self.config.q_heads * self.config.head_dim != self.config.bottleneck_channels:
            raise ValueError("q_heads * head_dim must match bottleneck_channels")
        if self.config.q_heads % self.config.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        rope_frequency_count = len(self.config.global_rope_frequencies) + len(
            self.config.local_rope_frequencies
        )
        if rope_frequency_count < 1:
            raise ValueError("at least one global or local RoPE frequency is required")
        if self.config.head_dim < 4 * rope_frequency_count:
            raise ValueError("head_dim must be at least 4 * total RoPE frequency count")
        for label, outputs in (
            ("score_outputs", self.config.score_outputs),
            ("ownership_outputs", self.config.ownership_outputs),
            ("policy_outputs", self.config.policy_outputs),
            ("policy_pass_outputs", self.config.policy_pass_outputs),
            ("budget_outputs", self.config.budget_outputs),
            ("budget_pass_outputs", self.config.budget_pass_outputs),
        ):
            if outputs != 1:
                raise ValueError(f"{label} must be 1")
        if self.config.wdl_outputs != 4:
            raise ValueError("wdl_outputs must be 4")
        broadcast_blocks = set(self.config.broadcast_blocks)
        gather_blocks = self._gather_blocks()
        if not gather_blocks:
            raise ValueError("at least one register gather block is required")
        for block in gather_blocks | broadcast_blocks:
            if block < 1 or block > self.config.block_count:
                raise ValueError("register blocks must be 1-based trunk block numbers")

    def initial_registers(self, rules: torch.Tensor) -> torch.Tensor:
        batch_size = rules.shape[0]
        seed = self.register_seed.to(device=rules.device, dtype=rules.dtype).unsqueeze(0)
        rule_delta = self.rule_mlp(rules).reshape(
            batch_size,
            self.config.register_count,
            self.config.trunk_channels,
        )
        return seed + rule_delta

    def _merged_registers(self, registers: torch.Tensor, batch_size: int) -> torch.Tensor:
        return registers.reshape(
            batch_size,
            1,
            self.config.register_count * self.config.trunk_channels,
        )

    def _spatial_logits(self, scalar: torch.Tensor) -> torch.Tensor:
        board = scalar.squeeze(1)
        return board.reshape(scalar.shape[0], scalar.shape[-2] * scalar.shape[-1])

    def forward(self, board: torch.Tensor, rules: torch.Tensor) -> dict[str, torch.Tensor]:
        if board.dim() != 4:
            raise ValueError("ScalarSakiGoModel expects board shape [B,6,N,N]")
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

        board_features = self.stem(board)
        registers = self.initial_registers(rules)
        for block in self.blocks:
            board_features, registers = block(board_features, registers)

        merged_registers = self._merged_registers(registers, board.shape[0])
        wdl_logits = self.wdl_head(merged_registers).squeeze(1)
        score = self.score_head(merged_registers).squeeze(1)
        policy_pass = self.policy_pass_head(merged_registers).squeeze(1)
        budget_pass = self.budget_pass_head(merged_registers).squeeze(1)

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
