"""Finite-group specs and square-grid presets.

The core abstraction is deliberately plain: a multiplication table, an inverse
table, and optional coordinate actions for groups that act exactly on square
grids. The layer code only needs the algebra tables and canonical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch

CoordinateAction = Callable[
    [int, torch.Tensor, torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]
]
CellAction = Callable[[int, int, int], int]


def _tuple_table(table: tuple[tuple[int, ...], ...] | list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in table)


def _infer_inverse(compose: tuple[tuple[int, ...], ...], identity: int) -> tuple[int, ...]:
    order = len(compose)
    inverse = [0 for _ in range(order)]
    for element in range(order):
        for candidate in range(order):
            if compose[element][candidate] == identity and compose[candidate][element] == identity:
                inverse[element] = candidate
                break
        else:
            raise ValueError(f"group element {element} has no two-sided inverse")
    return tuple(inverse)


@dataclass(frozen=True)
class FiniteGroupSpec:
    """A finite group in a regular-representation-friendly basis.

    `compose[a][b]` means "apply b, then a". This matches ordinary function
    composition and the left-regular action used by the attention layers.
    """

    name: str
    compose: tuple[tuple[int, ...], ...] | list[list[int]]
    inverse: tuple[int, ...] | list[int] | None = None
    identity: int = 0
    coordinate_action: CoordinateAction | None = None
    cell_action: CellAction | None = None

    def __post_init__(self) -> None:
        compose = _tuple_table(self.compose)
        order = len(compose)
        if order < 1:
            raise ValueError("a finite group needs at least one element")
        if not (0 <= self.identity < order):
            raise ValueError("identity index is out of range")
        if any(len(row) != order for row in compose):
            raise ValueError("compose table must be square")
        if any(value < 0 or value >= order for row in compose for value in row):
            raise ValueError("compose table contains an out-of-range element")
        for element in range(order):
            if compose[self.identity][element] != element or compose[element][self.identity] != element:
                raise ValueError("identity row/column do not act as identity")
        for a in range(order):
            for b in range(order):
                for c in range(order):
                    if compose[compose[a][b]][c] != compose[a][compose[b][c]]:
                        raise ValueError("compose table is not associative")
        inverse = (
            tuple(int(value) for value in self.inverse)
            if self.inverse is not None
            else _infer_inverse(compose, self.identity)
        )
        if len(inverse) != order:
            raise ValueError("inverse table length must match group order")
        for element, inv in enumerate(inverse):
            if inv < 0 or inv >= order:
                raise ValueError("inverse table contains an out-of-range element")
            if compose[element][inv] != self.identity or compose[inv][element] != self.identity:
                raise ValueError("inverse table is inconsistent with compose")
        object.__setattr__(self, "compose", compose)
        object.__setattr__(self, "inverse", inverse)

    @property
    def order(self) -> int:
        return len(self.compose)

    def relative_components(self) -> torch.Tensor:
        """Return rel[out_g, in_g] = out_g^-1 * in_g as a CPU LongTensor."""
        values = [
            [self.compose[self.inverse[out_component]][in_component] for in_component in range(self.order)]
            for out_component in range(self.order)
        ]
        return torch.tensor(values, dtype=torch.long)

    def regular_source_components(self, element: int, device: torch.device | None = None) -> torch.Tensor:
        """Source components for the left-regular action by `element`."""
        self._check_element(element)
        source = [self.compose[self.inverse[element]][component] for component in range(self.order)]
        return torch.tensor(source, dtype=torch.long, device=device)

    def source_indices(self, size: int, element: int, device: torch.device | None = None) -> torch.Tensor:
        """Source cells for scalar grid action: y[p] = x[element^-1 p]."""
        if self.cell_action is None:
            raise ValueError(f"group {self.name!r} has no square-grid cell action")
        self._check_element(element)
        source = [0 for _ in range(size * size)]
        for old_cell in range(size * size):
            new_cell = self.cell_action(size, old_cell, element)
            source[new_cell] = old_cell
        return torch.tensor(source, dtype=torch.long, device=device)

    def transform_spatial(self, x: torch.Tensor, element: int) -> torch.Tensor:
        """Apply the scalar grid action to the last two dimensions."""
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("spatial tensors must be square in the last two dimensions")
        size = x.shape[-1]
        source = self.source_indices(size, element, x.device)
        flat = x.reshape(*x.shape[:-2], size * size)
        return flat.index_select(-1, source).reshape_as(x)

    def transform_regular_spatial(self, x: torch.Tensor, element: int) -> torch.Tensor:
        """Apply the regular action to [B,C,G,H,W] features."""
        if x.dim() != 5 or x.shape[2] != self.order:
            raise ValueError(f"regular spatial features must have shape [B,C,{self.order},H,W]")
        source = self.regular_source_components(element, x.device)
        return self.transform_spatial(x.index_select(2, source), element)

    def transform_regular_fibers(self, x: torch.Tensor, element: int) -> torch.Tensor:
        """Apply the regular action to [B,R,C,G] register/fiber features."""
        if x.dim() != 4 or x.shape[3] != self.order:
            raise ValueError(f"regular fiber features must have shape [B,R,C,{self.order}]")
        source = self.regular_source_components(element, x.device)
        return x.index_select(3, source)

    def canonical_coordinates(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coordinates of g^-1(p) for every group component."""
        if self.coordinate_action is None:
            raise ValueError(f"group {self.name!r} has no square-grid coordinate action")
        cells = torch.arange(size * size, device=device)
        row = (cells // size).to(dtype)
        col = (cells % size).to(dtype)
        last = float(size - 1)
        rows = []
        cols = []
        for component in range(self.order):
            transformed_row, transformed_col = self.coordinate_action(
                self.inverse[component],
                row,
                col,
                last,
            )
            rows.append(transformed_row)
            cols.append(transformed_col)
        return torch.stack(rows, dim=0), torch.stack(cols, dim=0)

    def _check_element(self, element: int) -> None:
        if element < 0 or element >= self.order:
            raise ValueError(f"group element {element} is out of range for {self.name}")


def _d4_cell_action(size: int, cell: int, element: int) -> int:
    row = cell // size
    col = cell % size
    last = size - 1
    if element == 0:
        rr, cc = row, col
    elif element == 1:
        rr, cc = col, last - row
    elif element == 2:
        rr, cc = last - row, last - col
    elif element == 3:
        rr, cc = last - col, row
    elif element == 4:
        rr, cc = row, last - col
    elif element == 5:
        rr, cc = last - row, col
    elif element == 6:
        rr, cc = col, row
    elif element == 7:
        rr, cc = last - col, last - row
    else:
        raise ValueError(f"invalid D4 element {element}")
    return rr * size + cc


def _d4_coordinate_action(
    element: int,
    row: torch.Tensor,
    col: torch.Tensor,
    last: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if element == 0:
        return row, col
    if element == 1:
        return col, last - row
    if element == 2:
        return last - row, last - col
    if element == 3:
        return last - col, row
    if element == 4:
        return row, last - col
    if element == 5:
        return last - row, col
    if element == 6:
        return col, row
    if element == 7:
        return last - col, last - row
    raise ValueError(f"invalid D4 element {element}")


def _build_d4_compose() -> tuple[tuple[int, ...], ...]:
    order = 8
    size = 3
    cells = range(size * size)
    table: list[list[int]] = [[0 for _ in range(order)] for _ in range(order)]
    for first in range(order):
        for second in range(order):
            for candidate in range(order):
                if all(
                    _d4_cell_action(size, _d4_cell_action(size, cell, second), first)
                    == _d4_cell_action(size, cell, candidate)
                    for cell in cells
                ):
                    table[first][second] = candidate
                    break
            else:
                raise RuntimeError("D4 composition table construction failed")
    return tuple(tuple(row) for row in table)


@lru_cache(maxsize=None)
def trivial_group() -> FiniteGroupSpec:
    return FiniteGroupSpec(
        name="trivial",
        compose=((0,),),
        inverse=(0,),
        coordinate_action=lambda element, row, col, last: (row, col),
        cell_action=lambda size, cell, element: cell,
    )


@lru_cache(maxsize=None)
def dihedral_square_group() -> FiniteGroupSpec:
    """D4 in SAKIGo/KataGo-friendly order: rotations first, then reflections."""
    return FiniteGroupSpec(
        name="D4",
        compose=_build_d4_compose(),
        coordinate_action=_d4_coordinate_action,
        cell_action=_d4_cell_action,
    )


@lru_cache(maxsize=None)
def cyclic_square_group(order: int = 4) -> FiniteGroupSpec:
    """Cyclic rotation group acting exactly on square grids.

    Supported orders are 1, 2, and 4. `C2` is generated by 180-degree rotation;
    `C4` is generated by 90-degree rotation.
    """
    if order == 1:
        return trivial_group()
    if order not in {2, 4}:
        raise ValueError("square-grid cyclic presets support only orders 1, 2, and 4")
    d4_elements = (0, 2) if order == 2 else (0, 1, 2, 3)

    def cell_action(size: int, cell: int, element: int) -> int:
        return _d4_cell_action(size, cell, d4_elements[element])

    def coordinate_action(
        element: int,
        row: torch.Tensor,
        col: torch.Tensor,
        last: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _d4_coordinate_action(d4_elements[element], row, col, last)

    compose = tuple(tuple((a + b) % order for b in range(order)) for a in range(order))
    inverse = tuple((-element) % order for element in range(order))
    return FiniteGroupSpec(
        name=f"C{order}",
        compose=compose,
        inverse=inverse,
        coordinate_action=coordinate_action,
        cell_action=cell_action,
    )


def resolve_group(group: FiniteGroupSpec | int | None) -> FiniteGroupSpec:
    if group is None:
        return trivial_group()
    if isinstance(group, FiniteGroupSpec):
        return group
    if group == 1:
        return trivial_group()
    if group == 8:
        return dihedral_square_group()
    raise ValueError("integer group shortcuts only support 1 (trivial) and 8 (D4)")
