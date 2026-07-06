"""SAKIGo compatibility wrappers around the reusable finite-group library."""

from __future__ import annotations

import torch

from equivariant_attention import dihedral_square_group, trivial_group

SUPPORTED_GROUP_SIZES = (1, 8)


def _group(group_size: int):
    if group_size == 1:
        return trivial_group()
    if group_size == 8:
        return dihedral_square_group()
    raise ValueError(f"unsupported group size {group_size}")


def group_inverse(group_size: int) -> tuple[int, ...]:
    return _group(group_size).inverse


def group_compose(group_size: int) -> tuple[tuple[int, ...], ...]:
    return _group(group_size).compose


def relative_component_table(group_size: int) -> torch.Tensor:
    """rel[out_g, in_g] = out_g^-1 * in_g, as a CPU LongTensor."""
    return _group(group_size).relative_components()


def canonical_coordinates(
    group_size: int,
    board_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinates of g^-1(p) for every group component."""
    return _group(group_size).canonical_coordinates(board_size, device, dtype)
