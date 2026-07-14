from __future__ import annotations

import pytest
import torch

from sakigo.train.losses import LossWeights, weighted_total_loss


def _unit_head_losses() -> dict[str, torch.Tensor]:
    return {
        "wdl": torch.tensor(1.0),
        "score": torch.tensor(1.0),
        "policy": torch.tensor(1.0),
        "budget": torch.tensor(1.0),
    }


@pytest.mark.parametrize(("board_area", "expected"), ((49, 52.0), (64, 67.0), (81, 84.0)))
def test_score_loss_weight_uses_actual_board_area(
    board_area: int, expected: float
) -> None:
    total = weighted_total_loss(
        _unit_head_losses(),
        LossWeights(),
        board_area=board_area,
    )

    assert total.item() == pytest.approx(expected)


@pytest.mark.parametrize("board_area", (0, -1, 1.0, True))
def test_score_loss_weight_rejects_invalid_board_area(board_area: object) -> None:
    with pytest.raises(ValueError, match="board_area"):
        weighted_total_loss(
            _unit_head_losses(),
            LossWeights(),
            board_area=board_area,  # type: ignore[arg-type]
        )
