"""D4 transforms used to align canonical book pages with replayed histories."""

from __future__ import annotations

from typing import Any, Sequence


def transform_point(row: int, col: int, size: int, transform: int) -> tuple[int, int]:
    transform %= 8
    for _ in range(transform % 4):
        row, col = size - 1 - col, row
    if transform >= 4:
        col = size - 1 - col
    return row, col


def transform_action(action: int, size: int, transform: int) -> int:
    if action == size * size:
        return action
    if not 0 <= action < size * size:
        raise ValueError(f"action {action} is outside a {size}x{size} board")
    row, col = divmod(action, size)
    row, col = transform_point(row, col, size, transform)
    return row * size + col


def transform_spatial(values: Sequence[Any], size: int, transform: int) -> list[Any]:
    if len(values) != size * size:
        raise ValueError(f"spatial vector must have length {size * size}")
    output: list[Any] = [None] * len(values)
    for old_index, value in enumerate(values):
        output[transform_action(old_index, size, transform)] = value
    return output
