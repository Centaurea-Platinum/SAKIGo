"""Model gates: D4 equivariance, shape/gradient flow, and state-dict hygiene.
(Legacy-parity comparisons were removed at the P6 cutover; parity was
verified against real checkpoints before deletion.)
"""

from __future__ import annotations

import torch

from sakigo.model import d4 as new_d4
from sakigo.model.config import SakiGoModelConfig as NewConfig
from sakigo.model.layers import (
    BoardBlock,
    GroupPointwiseMLP,
    RegisterBroadcast,
    RegisterGather,
)
from sakigo.model.model import SakiGoNet

_SMALL = dict(
    board_size=9,
    stem_channels=(6, 8, 16),
    rule_mlp_channels=(10, 16, 32),
    block_count=2,
    register_count=2,
    trunk_channels=16,
    expanded_channel=16,
    register_channels=16,
    bottleneck_channels=16,
    register_bottleneck_channels=16,
    q_heads=2,
    kv_heads=1,
    head_dim=8,
    register_head_dim=8,
    activation="silu",
)


def _inputs(batch: int, size: int, seed: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    board = (torch.rand(batch, 6, size, size, generator=generator) > 0.6).float()
    rules = torch.zeros(batch, 10)
    rules[:, 0] = 1.0
    rules[:, 4] = 1.0
    rules[:, 6] = 1.0
    rules[:, 8] = 0.1
    rules[:, 9] = -0.05
    return board, rules


def test_unified_regular_model_is_equivariant() -> None:
    torch.manual_seed(2)
    model = SakiGoNet(NewConfig(architecture="SakiGoModel", **_SMALL)).eval()
    size = 9
    board, rules = _inputs(2, size)
    with torch.no_grad():
        base = model(board, rules)
        for transform in range(new_d4.GROUP_SIZE):
            transformed = model(new_d4.transform_board(board, transform), rules)
            torch.testing.assert_close(
                transformed["wdl_logits"], base["wdl_logits"], rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(transformed["score"], base["score"], rtol=1e-4, atol=1e-4)
            for head in ("policy_logits", "budget_logits"):
                torch.testing.assert_close(
                    transformed[head],
                    new_d4.transform_action_logits(base[head], transform, size),
                    rtol=1e-4,
                    atol=1e-4,
                )


def test_scalar_stem_then_lift_matches_collapsed_regular_stem() -> None:
    torch.manual_seed(5)
    model = SakiGoNet(NewConfig(architecture="SakiGoModel", **_SMALL)).double()
    regular_stem = GroupPointwiseMLP(_SMALL["stem_channels"], 8).double()
    scalar_layers = (model.stem[0], model.stem[2])
    regular_layers = (regular_stem.layers[0], regular_stem.layers[2])
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in scalar_layers)

    with torch.no_grad():
        for scalar_layer, regular_layer in zip(scalar_layers, regular_layers):
            scalar_layer.weight.copy_(regular_layer.weight.sum(dim=-1)[..., None, None])
            scalar_layer.bias.copy_(regular_layer.bias)

    scalar_input = torch.randn(2, 6, 7, 7, dtype=torch.float64, requires_grad=True)
    regular_input = scalar_input.detach().clone().requires_grad_(True)
    scalar_output = model.lift(model.stem(scalar_input))
    regular_output = regular_stem(model.lift(regular_input))
    cotangent = torch.randn_like(scalar_output)

    torch.testing.assert_close(scalar_output, regular_output, rtol=1e-10, atol=1e-10)
    (scalar_output * cotangent).sum().backward()
    (regular_output * cotangent).sum().backward()
    torch.testing.assert_close(scalar_input.grad, regular_input.grad, rtol=1e-10, atol=1e-10)

    expected_parameters = 6 * 8 + 8 + 8 * 16 + 16
    assert sum(parameter.numel() for parameter in model.stem.parameters()) == expected_parameters


def test_rectangular_register_model_is_equivariant() -> None:
    torch.manual_seed(6)
    config = NewConfig(
        architecture="SakiGoModel",
        **{
            **_SMALL,
            "register_channels": 8,
            "register_bottleneck_channels": 16,
            "register_head_dim": 8,
            "rule_mlp_channels": (10, 16, 16),
        },
    )
    model = SakiGoNet(config).eval()
    board, rules = _inputs(2, 9)
    with torch.no_grad():
        base = model(board, rules)
        transformed = model(new_d4.transform_board(board, 1), rules)
    torch.testing.assert_close(
        transformed["wdl_logits"],
        base["wdl_logits"],
        rtol=1e-4,
        atol=1e-4,
    )


def test_forward_backward_and_state_dict_hygiene() -> None:
    model = SakiGoNet(NewConfig(architecture="SakiGoModel", **_SMALL))
    board, rules = _inputs(2, 7)
    output = model(board, rules)
    assert output["wdl_logits"].shape == (2, 4)
    assert output["score"].shape == (2, 1)
    assert output["policy_logits"].shape == (2, 50)
    assert output["budget_logits"].shape == (2, 50)
    total = sum(value.sum() for value in output.values())
    total.backward()
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )
    # Non-persistent buffers must not leak into checkpoints.
    assert all("relative" not in key for key in model.state_dict())


def test_board_block_is_a_board_only_transition() -> None:
    model = SakiGoNet(NewConfig(architecture="SakiGoModel", **_SMALL)).eval()
    block = model.blocks[0]
    assert isinstance(block, BoardBlock)
    features = torch.randn(2, _SMALL["trunk_channels"], 8, 7, 7)

    with torch.no_grad():
        output = block(features)

    assert isinstance(output, torch.Tensor)
    assert output.shape == features.shape


class _RecordingDelta(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query: torch.Tensor | None = None
        self.key_value: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        self.query = query.detach().clone()
        self.key_value = key_value.detach().clone()
        return torch.ones_like(query)


def test_register_exchange_updates_only_the_query_residual() -> None:
    common = dict(
        trunk_channels=16,
        register_channels=8,
        board_size=9,
        q_heads=2,
        kv_heads=1,
        register_head_dim=8,
        global_rope_frequencies=(torch.pi,),
        local_rope_frequencies=(),
        eps=1e-6,
        group_size=8,
    )
    board = torch.randn(2, 16, 8, 7, 7)
    registers = torch.randn(2, 2, 8, 8)
    original_board = board.clone()
    original_registers = registers.clone()

    gather = RegisterGather(**common)
    gather_recorder = _RecordingDelta()
    gather.attention = gather_recorder
    with torch.no_grad():
        gather.gamma.fill_(0.25)
        expected_register_query = gather.norm_registers(registers)
        expected_board_source = gather.norm_board(board)
        gathered_board, gathered_registers = gather(board, registers)

    assert gathered_board is board
    torch.testing.assert_close(gathered_registers, registers + 0.25)
    torch.testing.assert_close(gather_recorder.query, expected_register_query)
    torch.testing.assert_close(gather_recorder.key_value, expected_board_source)
    torch.testing.assert_close(board, original_board)
    torch.testing.assert_close(registers, original_registers)

    broadcast = RegisterBroadcast(**common)
    broadcast_recorder = _RecordingDelta()
    broadcast.attention = broadcast_recorder
    with torch.no_grad():
        broadcast.gamma.fill_(0.25)
        expected_board_query = broadcast.norm_board(board)
        expected_register_source = broadcast.norm_registers(registers)
        broadcast_board, broadcast_registers = broadcast(board, registers)

    assert broadcast_registers is registers
    torch.testing.assert_close(broadcast_board, board + 0.25)
    torch.testing.assert_close(broadcast_recorder.query, expected_board_query)
    torch.testing.assert_close(broadcast_recorder.key_value, expected_register_source)
    torch.testing.assert_close(board, original_board)
    torch.testing.assert_close(registers, original_registers)
