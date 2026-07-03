from __future__ import annotations

from functools import lru_cache

import torch

GROUP_SIZE = 8


def transform_cell(size: int, cell: int, transform: int) -> int:
    """Transform a row-major cell index using the Rust solver's D4 order."""
    row = cell // size
    col = cell % size
    last = size - 1
    if transform == 0:
        rr, cc = row, col
    elif transform == 1:
        rr, cc = col, last - row
    elif transform == 2:
        rr, cc = last - row, last - col
    elif transform == 3:
        rr, cc = last - col, row
    elif transform == 4:
        rr, cc = row, last - col
    elif transform == 5:
        rr, cc = last - row, col
    elif transform == 6:
        rr, cc = col, row
    elif transform == 7:
        rr, cc = last - col, last - row
    else:
        raise ValueError(f"invalid D4 transform {transform}")
    return rr * size + cc


def _build_compose() -> tuple[tuple[int, ...], ...]:
    size = 3
    cells = range(size * size)
    table: list[list[int]] = [[0 for _ in range(GROUP_SIZE)] for _ in range(GROUP_SIZE)]
    for first in range(GROUP_SIZE):
        for second in range(GROUP_SIZE):
            for candidate in range(GROUP_SIZE):
                if all(
                    transform_cell(
                        size,
                        transform_cell(size, cell, second),
                        first,
                    )
                    == transform_cell(size, cell, candidate)
                    for cell in cells
                ):
                    table[first][second] = candidate
                    break
            else:
                raise RuntimeError("D4 composition table construction failed")
    return tuple(tuple(row) for row in table)


def _build_inverse(compose: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    inverse = [0 for _ in range(GROUP_SIZE)]
    for transform in range(GROUP_SIZE):
        for candidate in range(GROUP_SIZE):
            if compose[transform][candidate] == 0 and compose[candidate][transform] == 0:
                inverse[transform] = candidate
                break
        else:
            raise RuntimeError("D4 inverse table construction failed")
    return tuple(inverse)


COMPOSE = _build_compose()
INVERSE = _build_inverse(COMPOSE)
_LONG_TENSOR_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_COORDINATE_TENSOR_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def _device_key(device: torch.device) -> tuple[str, int | None]:
    normalized = torch.device(device)
    return normalized.type, normalized.index


def _cached_long_tensor(key: tuple[object, ...], values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    cache_key = (*key, *_device_key(device))
    cached = _LONG_TENSOR_CACHE.get(cache_key)
    if cached is None:
        cached = torch.tensor(values, dtype=torch.long, device=device)
        _LONG_TENSOR_CACHE[cache_key] = cached
    return cached


@lru_cache(maxsize=None)
def _source_indices(size: int, transform: int) -> tuple[int, ...]:
    source = [0 for _ in range(size * size)]
    for old_cell in range(size * size):
        new_cell = transform_cell(size, old_cell, transform)
        source[new_cell] = old_cell
    return tuple(source)


def _source_index_tensor(size: int, transform: int, device: torch.device) -> torch.Tensor:
    return _cached_long_tensor(("source", size, transform), _source_indices(size, transform), device)


def transform_board(x: torch.Tensor, transform: int) -> torch.Tensor:
    """Apply the scalar board action: y[p] = x[t^-1 p]."""
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("board tensors must be square in the last two dimensions")
    size = x.shape[-1]
    source = _source_index_tensor(size, transform, x.device)
    flat = x.reshape(*x.shape[:-2], size * size)
    transformed = flat.index_select(-1, source)
    return transformed.reshape_as(x)


def _regular_source_components(transform: int, device: torch.device) -> torch.Tensor:
    inv = INVERSE[transform]
    source = tuple(COMPOSE[inv][component] for component in range(GROUP_SIZE))
    return _cached_long_tensor(("regular-source", transform), source, device)


def transform_regular_board(x: torch.Tensor, transform: int) -> torch.Tensor:
    """Apply (T_k v)_g(p) = v_{k^-1 g}(k^-1 p) to [B,C,G,H,W]."""
    if x.dim() != 5 or x.shape[2] != GROUP_SIZE:
        raise ValueError("regular board features must have shape [B,C,8,H,W]")
    source = _regular_source_components(transform, x.device)
    selected = x.index_select(2, source)
    return transform_board(selected, transform)


def transform_regular_registers(x: torch.Tensor, transform: int) -> torch.Tensor:
    """Apply the regular group-axis action to [B,R,C,G] register features."""
    if x.dim() != 4 or x.shape[3] != GROUP_SIZE:
        raise ValueError("regular registers must have shape [B,R,C,8]")
    source = _regular_source_components(transform, x.device)
    return x.index_select(3, source)


def transform_policy_logits(x: torch.Tensor, transform: int, board_size: int) -> torch.Tensor:
    """Transform row-major policy logits with the scalar board action."""
    if x.dim() != 2 or x.shape[1] != board_size * board_size:
        raise ValueError("policy logits must have shape [B,N*N]")
    board = x.reshape(x.shape[0], 1, board_size, board_size)
    return transform_board(board, transform).reshape(x.shape[0], board_size * board_size)


def transform_action_logits(x: torch.Tensor, transform: int, board_size: int) -> torch.Tensor:
    """Transform row-major board logits plus an invariant final pass logit."""
    if x.dim() != 2 or x.shape[1] != board_size * board_size + 1:
        raise ValueError("action logits must have shape [B,N*N+1]")
    board = transform_policy_logits(x[:, :-1], transform, board_size)
    return torch.cat((board, x[:, -1:]), dim=1)


def relative_component_table(device: torch.device) -> torch.Tensor:
    """Return rel[out_g, in_g] = out_g^-1 in_g for regular linear layers."""
    values = []
    for out_component in range(GROUP_SIZE):
        inv = INVERSE[out_component]
        values.extend(COMPOSE[inv][in_component] for in_component in range(GROUP_SIZE))
    return _cached_long_tensor(("relative-components",), tuple(values), device).reshape(
        GROUP_SIZE,
        GROUP_SIZE,
    )


@lru_cache(maxsize=None)
def _canonical_coordinate_values(size: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rows: list[float] = []
    cols: list[float] = []
    for component in range(GROUP_SIZE):
        inv = INVERSE[component]
        for cell in range(size * size):
            canonical_cell = transform_cell(size, cell, inv)
            rows.append(float(canonical_cell // size))
            cols.append(float(canonical_cell % size))
    return tuple(rows), tuple(cols)


def canonical_coordinate_tensors(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinates of g^-1(p) for every group component and board cell."""
    cache_key = ("canonical-coordinates", size, *_device_key(device), dtype)
    cached = _COORDINATE_TENSOR_CACHE.get(cache_key)
    if cached is None:
        row_values, col_values = _canonical_coordinate_values(size)
        rows = torch.tensor(row_values, device=device, dtype=dtype).reshape(GROUP_SIZE, size * size)
        cols = torch.tensor(col_values, device=device, dtype=dtype).reshape(GROUP_SIZE, size * size)
        cached = (rows, cols)
        _COORDINATE_TENSOR_CACHE[cache_key] = cached
    return cached
