from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


BLACK = 1
WHITE = -1


@dataclass(frozen=True)
class GameStateBatch:
    """Canonical tensor view of game state before model-specific projection.

    Stones use absolute colors: black is 1, white is -1, empty is 0. The
    projections decide whether a color is "mine" or "opponent" from `to_move`.
    `non_trivial_illegal` is expected to come from the rules engine, because it
    depends on ko, superko, and suicide state that the model should not rebuild.
    """

    stones: torch.Tensor
    to_move: torch.Tensor | int
    scoring_rule: torch.Tensor | int = 0
    ko_rule: torch.Tensor | int = 0
    suicide_rule: torch.Tensor | int = 1
    komi: torch.Tensor | float = 7.5
    captures: torch.Tensor | None = None
    non_trivial_illegal: torch.Tensor | None = None
    board_mask: torch.Tensor | None = None
    katago_spatial: torch.Tensor | None = None
    katago_global: torch.Tensor | None = None


@dataclass(frozen=True)
class SakiGoInputs:
    board: torch.Tensor
    rules: torch.Tensor

    def as_args(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.board, self.rules


@dataclass(frozen=True)
class KataGoInputs:
    spatial: torch.Tensor
    global_features: torch.Tensor

    def as_args(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spatial, self.global_features


@dataclass(frozen=True)
class DistillationInputs:
    student: SakiGoInputs
    teacher: KataGoInputs


def _require_square_stones(stones: torch.Tensor) -> None:
    if stones.dim() != 3:
        raise ValueError("stones must have shape [B,N,N]")
    if stones.shape[-1] != stones.shape[-2]:
        raise ValueError("stones must describe a square board")


def _batch_vector(
    value: torch.Tensor | int | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)
    if tensor.dim() == 0:
        return tensor.expand(batch_size)
    if tensor.shape == (batch_size, 1):
        return tensor.reshape(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape [B]")
    return tensor


def _long_batch_vector(
    value: torch.Tensor | int,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    return _batch_vector(
        value,
        batch_size=batch_size,
        device=device,
        dtype=torch.long,
        name=name,
    )


def _one_hot(indices: torch.Tensor, count: int, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    if torch.any(indices < 0).item() or torch.any(indices >= count).item():
        raise ValueError(f"{name} must be in 0..{count - 1}")
    return F.one_hot(indices, num_classes=count).to(dtype=dtype)


def _board_mask(state: GameStateBatch, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    batch_size, board_size, _ = state.stones.shape
    if state.board_mask is None:
        return torch.ones(batch_size, board_size, board_size, device=device, dtype=dtype)
    mask = state.board_mask.to(device=device, dtype=dtype)
    if mask.shape != state.stones.shape:
        raise ValueError("board_mask must have shape [B,N,N]")
    return mask


def _illegal_mask(state: GameStateBatch, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if state.non_trivial_illegal is None:
        return torch.zeros_like(state.stones, device=device, dtype=dtype)
    illegal = state.non_trivial_illegal.to(device=device, dtype=dtype)
    if illegal.shape != state.stones.shape:
        raise ValueError("non_trivial_illegal must have shape [B,N,N]")
    return illegal


def _captures(
    state: GameStateBatch,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    batch_size = state.stones.shape[0]
    if state.captures is None:
        return torch.zeros(batch_size, 2, device=device, dtype=dtype)
    captures = state.captures.to(device=device, dtype=dtype)
    if captures.shape != (batch_size, 2):
        raise ValueError("captures must have shape [B,2] as [black_captures, white_captures]")
    return captures


class SakiGoInputProjection(nn.Module):
    """Project canonical game state to SAKIGo's compact board/rule tensors."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, state: GameStateBatch) -> SakiGoInputs:
        _require_square_stones(state.stones)
        device = state.stones.device
        dtype = self.dtype
        stones = state.stones.to(device=device)
        batch_size, board_size, _ = stones.shape
        to_move = _long_batch_vector(
            state.to_move,
            batch_size=batch_size,
            device=device,
            name="to_move",
        )
        if torch.any((to_move != BLACK) & (to_move != WHITE)).item():
            raise ValueError("to_move must use BLACK=1 or WHITE=-1")

        mask = _board_mask(state, dtype=dtype, device=device)
        empty = ((stones == 0).to(dtype=dtype) * mask).contiguous()
        perspective = to_move.reshape(batch_size, 1, 1)
        my_stones = ((stones == perspective).to(dtype=dtype) * mask).contiguous()
        opponent_stones = ((stones == -perspective).to(dtype=dtype) * mask).contiguous()

        corner, edge = self._boundary_planes(
            batch_size=batch_size,
            board_size=board_size,
            dtype=dtype,
            device=device,
        )
        corner = corner * mask
        edge = edge * mask
        illegal = _illegal_mask(state, dtype=dtype, device=device) * empty

        board = torch.stack(
            (my_stones, opponent_stones, empty, corner, edge, illegal),
            dim=1,
        )
        rules = self._rule_features(state, to_move=to_move, mask=mask, dtype=dtype, device=device)
        return SakiGoInputs(board=board, rules=rules)

    def _boundary_planes(
        self,
        *,
        batch_size: int,
        board_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.arange(board_size, device=device)
        row = coords.reshape(board_size, 1)
        col = coords.reshape(1, board_size)
        last = board_size - 1
        corner = ((row == 0) | (row == last)) & ((col == 0) | (col == last))
        boundary = (row == 0) | (row == last) | (col == 0) | (col == last)
        edge = boundary & ~corner
        return (
            corner.to(dtype=dtype).expand(batch_size, -1, -1),
            edge.to(dtype=dtype).expand(batch_size, -1, -1),
        )

    def _rule_features(
        self,
        state: GameStateBatch,
        *,
        to_move: torch.Tensor,
        mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = state.stones.shape[0]
        scoring = _long_batch_vector(
            state.scoring_rule,
            batch_size=batch_size,
            device=device,
            name="scoring_rule",
        )
        ko = _long_batch_vector(
            state.ko_rule,
            batch_size=batch_size,
            device=device,
            name="ko_rule",
        )
        suicide = _long_batch_vector(
            state.suicide_rule,
            batch_size=batch_size,
            device=device,
            name="suicide_rule",
        )
        komi = _batch_vector(
            state.komi,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name="komi",
        )
        captures = _captures(state, dtype=dtype, device=device)
        area = mask.sum(dim=(1, 2)).clamp_min(1.0)

        signed_komi = torch.where(to_move == BLACK, -komi, komi)
        black_capture_diff = captures[:, 0] - captures[:, 1]
        capture_diff = torch.where(to_move == BLACK, black_capture_diff, -black_capture_diff)

        return torch.cat(
            (
                _one_hot(scoring, 4, dtype=dtype, name="scoring_rule"),
                _one_hot(ko, 2, dtype=dtype, name="ko_rule"),
                _one_hot(suicide, 2, dtype=dtype, name="suicide_rule"),
                (signed_komi / area).clamp(-1.0, 1.0).unsqueeze(1),
                (capture_diff / area).clamp(-1.0, 1.0).unsqueeze(1),
            ),
            dim=1,
        )


class KataGoInputProjection(nn.Module):
    """Project canonical game state to a native KataGo model input pair.

    KataGo's exact feature construction is versioned and hand-engineered. This
    layer therefore accepts either an exact encoder callable or precomputed
    native tensors on `GameStateBatch`. That keeps the distillation loop honest:
    SAKIGo and KataGo share one game-state object, but each model sees its own
    intended projection.
    """

    def __init__(
        self,
        encoder: Callable[[GameStateBatch], KataGoInputs | tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, state: GameStateBatch) -> KataGoInputs:
        if self.encoder is not None:
            encoded = self.encoder(state)
            if isinstance(encoded, KataGoInputs):
                return encoded
            if isinstance(encoded, tuple) and len(encoded) == 2:
                return KataGoInputs(spatial=encoded[0], global_features=encoded[1])
            raise TypeError("KataGo encoder must return KataGoInputs or (spatial, global_features)")
        if state.katago_spatial is None or state.katago_global is None:
            raise ValueError(
                "KataGoInputProjection needs an exact encoder or precomputed "
                "katago_spatial and katago_global tensors"
            )
        return KataGoInputs(spatial=state.katago_spatial, global_features=state.katago_global)


class DistillationInputProjection(nn.Module):
    """Create student and teacher inputs from the same canonical state."""

    def __init__(
        self,
        student_projection: SakiGoInputProjection | None = None,
        teacher_projection: KataGoInputProjection | None = None,
    ) -> None:
        super().__init__()
        self.student_projection = student_projection or SakiGoInputProjection()
        self.teacher_projection = teacher_projection or KataGoInputProjection()

    def forward(self, state: GameStateBatch) -> DistillationInputs:
        return DistillationInputs(
            student=self.student_projection(state),
            teacher=self.teacher_projection(state),
        )


class ProjectedModelAdapter(nn.Module):
    """Wrap a model so callers can pass `GameStateBatch` directly."""

    def __init__(self, model: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.projection = projection

    def forward(self, state: GameStateBatch) -> Any:
        projected = self.projection(state)
        if hasattr(projected, "as_args"):
            return self.model(*projected.as_args())
        if isinstance(projected, tuple):
            return self.model(*projected)
        if isinstance(projected, dict):
            return self.model(**projected)
        return self.model(projected)
