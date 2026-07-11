from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sakigo.engine import ENGINE_AVAILABLE
from sakigo.eval.matrix import MatrixGame
from sakigo.eval.selfplay import MatchGame, load_policy_model, paired_mean_interval
from sakigo.rulesets import ruleset_from_name


def test_safe_checkpoint_loading_never_falls_back_implicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_load(*args, **kwargs):
        calls.append(bool(kwargs["weights_only"]))
        raise RuntimeError("not safe-loadable")

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(RuntimeError, match="allow-unsafe-legacy-checkpoint"):
        load_policy_model(Path("untrusted.pt"), torch.device("cpu"))
    assert calls == [True]


def test_paired_interval_uses_pair_scores() -> None:
    low, high = paired_mean_interval([1.0, 0.5, 0.0, 0.5])
    assert 0.0 <= low <= 0.5 <= high <= 1.0


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="sakigo_engine wheel not installed")
def test_draw_and_ply_cap_are_not_awarded_to_white() -> None:
    area = 1
    draw = MatchGame(0, 0, [], 10, 1, 0.0)
    draw.play(area)
    draw.play(area)
    assert draw.ended_by == "passes"
    assert draw.score == 0.0
    assert draw.winner() is None

    capped = MatchGame(1, 0, [area, area], 1, 1, 0.0)
    assert capped.ended_by == "max_plies"
    assert capped.actions == [area]
    assert capped.winner() is None


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="sakigo_engine wheel not installed")
def test_matrix_rejects_unsupported_ancient_chinese_scoring() -> None:
    game = MatrixGame(
        0,
        "AB",
        [],
        None,
        1,
        ruleset_from_name("ancient-chinese").with_komi(0.0),
    )
    game.play(1)
    with pytest.raises(ValueError, match="Tromp-Taylor"):
        game.play(1)
