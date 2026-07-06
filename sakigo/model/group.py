"""Group-axis abstraction: D4 (group_size=8) or trivial (group_size=1).

The unified network keeps an explicit group axis everywhere; group_size=1
degenerates to the plain (non-equivariant) scalar control model with
identical semantics to the legacy ScalarSakiGoModel.

All forward-path helpers here are pure tensor ops (no module-global caches),
so torch.compile can trace them without graph breaks; per-board-size shape
specialization constant-folds the coordinate math.
"""

from __future__ import annotations

import torch

from sakigo.model.d4 import COMPOSE, GROUP_SIZE, INVERSE

SUPPORTED_GROUP_SIZES = (1, GROUP_SIZE)


def group_inverse(group_size: int) -> tuple[int, ...]:
    if group_size == 1:
        return (0,)
    if group_size == GROUP_SIZE:
        return INVERSE
    raise ValueError(f"unsupported group size {group_size}")


def group_compose(group_size: int) -> tuple[tuple[int, ...], ...]:
    if group_size == 1:
        return ((0,),)
    if group_size == GROUP_SIZE:
        return COMPOSE
    raise ValueError(f"unsupported group size {group_size}")


def relative_component_table(group_size: int) -> torch.Tensor:
    """rel[out_g, in_g] = out_g^-1 · in_g, as a CPU LongTensor for buffer registration."""
    compose = group_compose(group_size)
    inverse = group_inverse(group_size)
    values = [
        [compose[inverse[out_component]][in_component] for in_component in range(group_size)]
        for out_component in range(group_size)
    ]
    return torch.tensor(values, dtype=torch.long)


def _transformed_coordinates(
    transform: int,
    row: torch.Tensor,
    col: torch.Tensor,
    last: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if transform == 0:
        return row, col
    if transform == 1:
        return col, last - row
    if transform == 2:
        return last - row, last - col
    if transform == 3:
        return last - col, row
    if transform == 4:
        return row, last - col
    if transform == 5:
        return last - row, col
    if transform == 6:
        return col, row
    if transform == 7:
        return last - col, last - row
    raise ValueError(f"invalid D4 transform {transform}")


def canonical_coordinates(
    group_size: int,
    board_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinates of g^-1(p) for every group component: two [G, N*N] tensors.

    Matches d4.canonical_coordinate_tensors for group_size=8 and plain
    row-major coordinates for group_size=1.
    """
    cells = torch.arange(board_size * board_size, device=device)
    row = (cells // board_size).to(dtype)
    col = (cells % board_size).to(dtype)
    last = float(board_size - 1)
    inverse = group_inverse(group_size)
    rows = []
    cols = []
    for component in range(group_size):
        transformed_row, transformed_col = _transformed_coordinates(inverse[component], row, col, last)
        rows.append(transformed_row)
        cols.append(transformed_col)
    return torch.stack(rows, dim=0), torch.stack(cols, dim=0)
