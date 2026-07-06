"""Reusable equivariant_attention package gates."""

from __future__ import annotations

from math import pi

import torch

from equivariant_attention import (
    RegularLinear1x1,
    RegularSelfAttention,
    SpatialToRegisterAttention,
    cyclic_square_group,
    dihedral_square_group,
    trivial_group,
)
from sakigo.model import d4


def test_group_presets_match_sakigo_d4_tables() -> None:
    group = dihedral_square_group()
    assert group.order == 8
    assert group.compose == d4.COMPOSE
    assert group.inverse == d4.INVERSE
    torch.testing.assert_close(group.relative_components(), d4.relative_component_table(torch.device("cpu")))

    trivial = trivial_group()
    assert trivial.order == 1
    assert trivial.compose == ((0,),)


def test_regular_linear_is_d4_equivariant() -> None:
    torch.manual_seed(1)
    group = dihedral_square_group()
    layer = RegularLinear1x1(3, 5, group).eval()
    x = torch.randn(2, 3, group.order, 5, 5)
    with torch.no_grad():
        base = layer(x)
        for element in range(group.order):
            transformed = layer(group.transform_regular_spatial(x, element))
            expected = group.transform_regular_spatial(base, element)
            torch.testing.assert_close(transformed, expected, rtol=1e-5, atol=1e-5)


def test_regular_attention_is_d4_equivariant() -> None:
    torch.manual_seed(2)
    group = dihedral_square_group()
    layer = RegularSelfAttention(
        channels=8,
        board_size=5,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).eval()
    x = torch.randn(2, 8, group.order, 5, 5)
    with torch.no_grad():
        base = layer(x)
        for element in range(group.order):
            transformed = layer(group.transform_regular_spatial(x, element))
            expected = group.transform_regular_spatial(base, element)
            torch.testing.assert_close(transformed, expected, rtol=1e-4, atol=1e-4)


def test_cyclic_square_group_reuses_same_attention_layers() -> None:
    torch.manual_seed(3)
    group = cyclic_square_group(4)
    layer = RegularSelfAttention(
        channels=8,
        board_size=7,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).eval()
    x = torch.randn(1, 8, group.order, 7, 7)
    with torch.no_grad():
        base = layer(x)
        for element in range(group.order):
            transformed = layer(group.transform_regular_spatial(x, element))
            expected = group.transform_regular_spatial(base, element)
            torch.testing.assert_close(transformed, expected, rtol=1e-4, atol=1e-4)


def test_spatial_to_register_attention_updates_spatial_features_equivariantly() -> None:
    torch.manual_seed(4)
    group = dihedral_square_group()
    layer = SpatialToRegisterAttention(
        spatial_channels=8,
        register_channels=8,
        board_size=5,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).eval()
    spatial = torch.randn(2, 8, group.order, 5, 5)
    registers = torch.randn(2, 3, 8, group.order)
    with torch.no_grad():
        base = layer(spatial, registers)
        transformed = layer(
            group.transform_regular_spatial(spatial, 1),
            group.transform_regular_fibers(registers, 1),
        )
        expected = group.transform_regular_spatial(base, 1)
        torch.testing.assert_close(transformed, expected, rtol=1e-4, atol=1e-4)
