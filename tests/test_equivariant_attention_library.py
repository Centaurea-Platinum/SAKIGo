"""Reusable equivariant_attention package gates."""

from __future__ import annotations

from math import pi

import torch

from equivariant_attention import (
    RegularCrossAttention,
    RegularLinear1x1,
    RegularSelfAttention,
    RegisterToSpatialAttention,
    SpatialToRegisterAttention,
    cyclic_square_group,
    dihedral_square_group,
    trivial_group,
)
from sakigo.model import d4


def _projection_slice(
    source: RegularLinear1x1,
    start: int,
    end: int,
) -> RegularLinear1x1:
    target = RegularLinear1x1(
        source.weight.shape[1],
        end - start,
        source.group,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight[start:end])
        assert source.bias is not None and target.bias is not None
        target.bias.copy_(source.bias[start:end])
    return target


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


def test_fused_self_attention_matches_separate_qkv_projections() -> None:
    torch.manual_seed(21)
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
    ).double()
    q_end = layer.q_channels
    k_end = q_end + layer.kv_channels
    references = (
        _projection_slice(layer.qkv_proj, 0, q_end),
        _projection_slice(layer.qkv_proj, q_end, k_end),
        _projection_slice(layer.qkv_proj, k_end, k_end + layer.kv_channels),
    )
    fused_input = torch.randn(2, 8, group.order, 3, 3, dtype=torch.float64, requires_grad=True)
    reference_input = fused_input.detach().clone().requires_grad_(True)
    fused_parts = torch.split(
        layer.qkv_proj(fused_input),
        (layer.q_channels, layer.kv_channels, layer.kv_channels),
        dim=1,
    )
    reference_parts = tuple(projection(reference_input) for projection in references)
    cotangents = tuple(torch.randn_like(part) for part in fused_parts)

    for fused, reference in zip(fused_parts, reference_parts):
        torch.testing.assert_close(fused, reference, rtol=1e-10, atol=1e-10)
    sum((part * grad).sum() for part, grad in zip(fused_parts, cotangents)).backward()
    sum((part * grad).sum() for part, grad in zip(reference_parts, cotangents)).backward()

    torch.testing.assert_close(fused_input.grad, reference_input.grad, rtol=1e-10, atol=1e-10)
    for reference, start, end in zip(
        references,
        (0, q_end, k_end),
        (q_end, k_end, k_end + layer.kv_channels),
    ):
        torch.testing.assert_close(
            layer.qkv_proj.weight.grad[start:end], reference.weight.grad, rtol=1e-10, atol=1e-10
        )
        assert layer.qkv_proj.bias is not None and reference.bias is not None
        torch.testing.assert_close(
            layer.qkv_proj.bias.grad[start:end], reference.bias.grad, rtol=1e-10, atol=1e-10
        )


def test_fused_cross_attention_kv_matches_separate_projections() -> None:
    torch.manual_seed(22)
    group = dihedral_square_group()
    gather = RegisterToSpatialAttention(
        register_channels=8,
        spatial_channels=8,
        board_size=5,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).double()
    broadcast = SpatialToRegisterAttention(
        spatial_channels=8,
        register_channels=8,
        board_size=5,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).double()
    cases = (
        (gather.kv_proj, torch.randn(2, 8, group.order, 3, 3, dtype=torch.float64), 1),
        (broadcast.kv_proj, torch.randn(2, 3, 8, group.order, dtype=torch.float64), 2),
    )

    for fused_projection, value, channel_dim in cases:
        k_reference = _projection_slice(fused_projection, 0, 4)
        v_reference = _projection_slice(fused_projection, 4, 8)
        fused_input = value.detach().clone().requires_grad_(True)
        reference_input = value.detach().clone().requires_grad_(True)
        fused_parts = torch.split(fused_projection(fused_input), (4, 4), dim=channel_dim)
        reference_parts = (k_reference(reference_input), v_reference(reference_input))
        cotangents = tuple(torch.randn_like(part) for part in fused_parts)

        for fused, reference in zip(fused_parts, reference_parts):
            torch.testing.assert_close(fused, reference, rtol=1e-10, atol=1e-10)
        sum((part * grad).sum() for part, grad in zip(fused_parts, cotangents)).backward()
        sum((part * grad).sum() for part, grad in zip(reference_parts, cotangents)).backward()

        torch.testing.assert_close(
            fused_input.grad, reference_input.grad, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            fused_projection.weight.grad[:4], k_reference.weight.grad, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            fused_projection.weight.grad[4:], v_reference.weight.grad, rtol=1e-10, atol=1e-10
        )

    board = torch.randn(2, 8, group.order, 3, 3, dtype=torch.float64)
    registers = torch.randn(2, 3, 8, group.order, dtype=torch.float64)
    for attention, query, key_value in (
        (gather, registers, board),
        (broadcast, board, registers),
    ):
        q_sources: list[torch.Tensor] = []
        kv_sources: list[torch.Tensor] = []
        q_hook = attention.q_proj.register_forward_pre_hook(
            lambda _module, inputs: q_sources.append(inputs[0])
        )
        kv_hook = attention.kv_proj.register_forward_pre_hook(
            lambda _module, inputs: kv_sources.append(inputs[0])
        )
        with torch.no_grad():
            attention(query, key_value)
        q_hook.remove()
        kv_hook.remove()
        assert len(q_sources) == 1 and q_sources[0] is query
        assert len(kv_sources) == 1 and kv_sources[0] is key_value
        assert not hasattr(attention, "k_proj") and not hasattr(attention, "v_proj")


def test_attention_projection_fusion_preserves_full_self_attention() -> None:
    torch.manual_seed(23)
    group = dihedral_square_group()
    fused = RegularSelfAttention(
        channels=8,
        board_size=5,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).eval()
    reference = RegularCrossAttention(
        query_channels=8,
        key_channels=8,
        output_channels=8,
        board_size=5,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        query_is_spatial=True,
        key_is_spatial=True,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(),
        group=group,
    ).eval()
    with torch.no_grad():
        reference.q_proj.weight.copy_(fused.qkv_proj.weight[: fused.q_channels])
        reference.q_proj.bias.copy_(fused.qkv_proj.bias[: fused.q_channels])
        reference.kv_proj.weight.copy_(fused.qkv_proj.weight[fused.q_channels :])
        reference.kv_proj.bias.copy_(fused.qkv_proj.bias[fused.q_channels :])
        reference.out_proj.load_state_dict(fused.out_proj.state_dict())

    calls: list[torch.Tensor] = []
    hook = fused.qkv_proj.register_forward_pre_hook(lambda _module, inputs: calls.append(inputs[0]))
    x = torch.randn(2, 8, group.order, 5, 5)
    with torch.no_grad():
        actual = fused(x)
        expected = reference(x, x)
    hook.remove()

    assert len(calls) == 1 and calls[0] is x
    assert not hasattr(fused, "q_proj") and not hasattr(fused, "kv_proj")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


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
