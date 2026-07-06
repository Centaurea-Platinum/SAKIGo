"""Model gates: D4 equivariance of the unified SakiGoNet, scalar-control
non-equivariance, gradient flow, and state-dict hygiene.
(Legacy-parity comparisons were removed at the P6 cutover; parity was
verified against real checkpoints before deletion.)
"""

from __future__ import annotations

import torch

from sakigo.model import d4 as new_d4
from sakigo.model.config import SakiGoModelConfig as NewConfig
from sakigo.model.model import SakiGoNet

_SMALL = dict(
    board_size=9,
    stem_channels=(6, 8, 16),
    rule_mlp_channels=(10, 16, 32),
    block_count=2,
    register_count=2,
    trunk_channels=16,
    bottleneck_channels=16,
    q_heads=2,
    kv_heads=1,
    head_dim=8,
    broadcast_blocks=(2,),
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
            torch.testing.assert_close(
                transformed["ownership_logits"],
                new_d4.transform_policy_logits(base["ownership_logits"], transform, size),
                rtol=1e-4,
                atol=1e-4,
            )
            for head in ("policy_logits", "budget_logits"):
                torch.testing.assert_close(
                    transformed[head],
                    new_d4.transform_action_logits(base[head], transform, size),
                    rtol=1e-4,
                    atol=1e-4,
                )


def test_unified_scalar_model_is_not_equivariant() -> None:
    torch.manual_seed(4)
    model = SakiGoNet(NewConfig(architecture="ScalarSakiGoModel", **_SMALL)).eval()
    board, rules = _inputs(2, 9)
    with torch.no_grad():
        base = model(board, rules)
        transformed = model(new_d4.transform_board(board, 1), rules)
    assert not torch.allclose(
        transformed["ownership_logits"],
        new_d4.transform_policy_logits(base["ownership_logits"], 1, 9),
        rtol=1e-3,
        atol=1e-3,
    )


def test_forward_backward_and_state_dict_hygiene() -> None:
    model = SakiGoNet(NewConfig(architecture="SakiGoModel", **_SMALL))
    board, rules = _inputs(2, 7)
    output = model(board, rules)
    total = sum(value.sum() for value in output.values())
    total.backward()
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )
    # Non-persistent buffers must not leak into checkpoints.
    assert all("relative" not in key for key in model.state_dict())
