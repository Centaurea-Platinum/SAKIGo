"""Torch layers for regular-representation finite-group attention."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from equivariant_attention.groups import FiniteGroupSpec, resolve_group


@torch.compiler.disable
def _safe_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Keep SDPA on PyTorch's proven eager path inside a compiled model.

    Inductor's compiled BF16 GQA backward can leave this model numerically
    corrupted after an otherwise-finite optimizer step.  The projections and
    surrounding trunk remain compiled; only the fused attention primitive is
    an explicit graph boundary.
    """
    return F.scaled_dot_product_attention(query, key, value, enable_gqa=True)


class RegularLift(nn.Module):
    """Lift scalar spatial features [B,C,H,W] to regular features [B,C,G,H,W]."""

    def __init__(self, group: FiniteGroupSpec | int | None = None) -> None:
        super().__init__()
        self.group = resolve_group(group)
        self.group_size = self.group.order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("RegularLift expects [B,C,H,W]")
        return x.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)


class RegularLinear1x1(nn.Module):
    """Equivariant pointwise fiber mixing in the regular basis.

    `weight` has shape [out_channels, in_channels, G]. For the trivial group
    this degenerates to an ordinary linear map.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: FiniteGroupSpec | int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.group = resolve_group(group)
        self.group_size = self.group.order
        scale = 1.0 / sqrt(max(in_channels * self.group_size, 1))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.group_size))
        nn.init.uniform_(self.weight, -scale, scale)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer("relative", self.group.relative_components(), persistent=False)

    def _flat_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        group = self.group_size
        kernel = self.weight[:, :, self.relative]
        out_channels, in_channels = kernel.shape[:2]
        weight = kernel.permute(0, 2, 1, 3).reshape(out_channels * group, in_channels * group)
        bias = None
        if self.bias is not None:
            bias = (
                self.bias.view(out_channels, 1)
                .expand(out_channels, group)
                .reshape(out_channels * group)
            )
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = self.group_size
        weight, bias = self._flat_weight_bias()
        out_channels = weight.shape[0] // group
        in_channels = weight.shape[1] // group
        if x.dim() == 5:
            if x.shape[2] != group:
                raise ValueError(f"regular spatial features must have shape [B,C,{group},H,W]")
            batch, _, _, height, width = x.shape
            flat = x.permute(0, 3, 4, 1, 2).reshape(batch * height * width, in_channels * group)
            y = F.linear(flat, weight, bias)
            return y.reshape(batch, height, width, out_channels, group).permute(0, 3, 4, 1, 2)
        if x.dim() == 4:
            if x.shape[3] != group:
                raise ValueError(f"regular fiber features must have shape [B,R,C,{group}]")
            batch, registers, _, _ = x.shape
            flat = x.reshape(batch * registers, in_channels * group)
            y = F.linear(flat, weight, bias)
            return y.reshape(batch, registers, out_channels, group)
        raise ValueError("RegularLinear1x1 expects [B,C,G,H,W] or [B,R,C,G]")


class RegularRMSNorm(nn.Module):
    """RMS norm over channels and group components."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            rms = torch.mean(x.square(), dim=(1, 2), keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, -1, 1, 1, 1)
        if x.dim() == 4:
            rms = torch.mean(x.square(), dim=(2, 3), keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, 1, -1, 1)
        raise ValueError("RegularRMSNorm expects [B,C,G,H,W] or [B,R,C,G]")


class RegularPointwiseMLP(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        group: FiniteGroupSpec | int | None = None,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("RegularPointwiseMLP needs at least two channel sizes")
        layers: list[nn.Module] = []
        last = len(channels) - 2
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            layers.append(RegularLinear1x1(in_channels, out_channels, group))
            if index < last or final_activation:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvariantPool(nn.Module):
    """Collapse the group axis by mean or sum."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            dim = 2
        elif x.dim() == 4:
            dim = 3
        else:
            raise ValueError("InvariantPool expects regular spatial or fiber features")
        if self.reduction == "mean":
            return x.mean(dim=dim)
        return x.sum(dim=dim)


def _rope_cos_sin(
    group: FiniteGroupSpec,
    board_size: int,
    device: torch.device,
    dtype: torch.dtype,
    global_frequencies: tuple[float, ...],
    local_frequencies: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = group.canonical_coordinates(board_size, device, dtype)
    denom = max(board_size - 1, 1)
    channels: list[torch.Tensor] = []
    for frequency in global_frequencies:
        channels.append(frequency * rows / denom)
        channels.append(frequency * cols / denom)
    for frequency in local_frequencies:
        channels.append(frequency * rows)
        channels.append(frequency * cols)
    angles = torch.stack(channels, dim=-1)
    return angles.cos().unsqueeze(0).unsqueeze(2), angles.sin().unsqueeze(0).unsqueeze(2)


def _apply_rope(
    tokens: torch.Tensor,
    group: FiniteGroupSpec,
    board_size: int,
    global_frequencies: tuple[float, ...],
    local_frequencies: tuple[float, ...],
) -> torch.Tensor:
    frequency_count = len(global_frequencies) + len(local_frequencies)
    pair_count = 2 * frequency_count
    rope_dim = 2 * pair_count
    if tokens.shape[-1] < rope_dim:
        raise ValueError(
            f"{frequency_count}-frequency 2D RoPE needs at least {rope_dim} head dimensions"
        )
    cos, sin = _rope_cos_sin(
        group,
        board_size,
        tokens.device,
        tokens.dtype,
        global_frequencies,
        local_frequencies,
    )
    pairs = tokens[..., :rope_dim].reshape(*tokens.shape[:-1], pair_count, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
    rotated = rotated.reshape(*tokens.shape[:-1], rope_dim)
    if tokens.shape[-1] == rope_dim:
        return rotated
    return torch.cat((rotated, tokens[..., rope_dim:]), dim=-1)


def _spatial_tokens(x: torch.Tensor, heads: int, head_dim: int, group_size: int) -> torch.Tensor:
    batch, channels, group, height, width = x.shape
    if channels != heads * head_dim or group != group_size:
        raise ValueError("projected spatial channels do not match heads * head_dim")
    return x.reshape(batch, heads, head_dim, group, height * width).permute(0, 3, 1, 4, 2)


def _fiber_tokens(x: torch.Tensor, heads: int, head_dim: int, group_size: int) -> torch.Tensor:
    batch, registers, channels, group = x.shape
    if channels != heads * head_dim or group != group_size:
        raise ValueError("projected fiber channels do not match heads * head_dim")
    return x.reshape(batch, registers, heads, head_dim, group).permute(0, 4, 2, 1, 3)


def _tokens_to_spatial(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, group, heads, cells, head_dim = x.shape
    return x.permute(0, 2, 4, 1, 3).reshape(batch, heads * head_dim, group, height, width)


def _tokens_to_fibers(x: torch.Tensor) -> torch.Tensor:
    batch, group, heads, registers, head_dim = x.shape
    return x.permute(0, 3, 2, 4, 1).reshape(batch, registers, heads * head_dim, group)


def _split_regular_channels(
    x: torch.Tensor,
    sizes: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    if x.dim() == 5:
        channel_dim = 1
    elif x.dim() == 4:
        channel_dim = 2
    else:
        raise ValueError("regular features must be spatial [B,C,G,H,W] or fiber [B,R,C,G]")
    return torch.split(x, sizes, dim=channel_dim)


class RegularCrossAttention(nn.Module):
    """Regular-representation grouped-query attention.

    Cross-attention keeps its query projection separate and fuses the K/V
    projections because keys and values share one source tensor. Self-attention
    additionally fuses Q with K/V into one projection.
    """

    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        output_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        query_is_spatial: bool,
        key_is_spatial: bool,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group: FiniteGroupSpec | int | None = None,
        fuse_qkv: bool = False,
    ) -> None:
        super().__init__()
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        self.group = resolve_group(group)
        self.group_size = self.group.order
        self.board_size = board_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.query_is_spatial = query_is_spatial
        self.key_is_spatial = key_is_spatial
        self.global_rope_frequencies = global_rope_frequencies
        self.local_rope_frequencies = local_rope_frequencies
        self.q_channels = q_heads * head_dim
        self.kv_channels = kv_heads * head_dim
        self.fuse_qkv = fuse_qkv

        if fuse_qkv:
            if query_channels != key_channels or query_is_spatial != key_is_spatial:
                raise ValueError("fused QKV requires query and key/value to share shape and source")
            self.qkv_proj = RegularLinear1x1(
                query_channels,
                self.q_channels + 2 * self.kv_channels,
                self.group,
            )
        else:
            self.q_proj = RegularLinear1x1(query_channels, self.q_channels, self.group)
            self.kv_proj = RegularLinear1x1(key_channels, 2 * self.kv_channels, self.group)
        self.out_proj = RegularLinear1x1(self.q_channels, output_channels, self.group)

    def _to_tokens(self, x: torch.Tensor, heads: int, is_spatial: bool) -> torch.Tensor:
        if is_spatial:
            return _spatial_tokens(x, heads, self.head_dim, self.group_size)
        return _fiber_tokens(x, heads, self.head_dim, self.group_size)

    def _active_board_size(self, query: torch.Tensor, key_value: torch.Tensor) -> int:
        if self.query_is_spatial:
            height, width = query.shape[-2:]
        elif self.key_is_spatial:
            height, width = key_value.shape[-2:]
        else:
            return self.board_size
        if height != width:
            raise ValueError("spatial features must be square")
        if height > self.board_size:
            raise ValueError(f"board size {height} exceeds configured maximum {self.board_size}")
        return height

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if self.fuse_qkv:
            if query is not key_value:
                raise ValueError("fused self-attention requires one shared QKV input tensor")
            q_proj, k_proj, v_proj = _split_regular_channels(
                self.qkv_proj(query),
                (self.q_channels, self.kv_channels, self.kv_channels),
            )
        else:
            q_proj = self.q_proj(query)
            k_proj, v_proj = _split_regular_channels(
                self.kv_proj(key_value),
                (self.kv_channels, self.kv_channels),
            )
        board_size = self._active_board_size(query, key_value)

        q = self._to_tokens(q_proj, self.q_heads, self.query_is_spatial)
        k = self._to_tokens(k_proj, self.kv_heads, self.key_is_spatial)
        v = self._to_tokens(v_proj, self.kv_heads, self.key_is_spatial)
        if self.query_is_spatial:
            q = _apply_rope(
                q,
                self.group,
                board_size,
                self.global_rope_frequencies,
                self.local_rope_frequencies,
            )
        if self.key_is_spatial:
            k = _apply_rope(
                k,
                self.group,
                board_size,
                self.global_rope_frequencies,
                self.local_rope_frequencies,
            )

        batch, group, _, query_tokens, head_dim = q.shape
        key_tokens = k.shape[-2]
        q_flat = q.reshape(batch * group, self.q_heads, query_tokens, head_dim)
        k_flat = k.reshape(batch * group, self.kv_heads, key_tokens, head_dim)
        v_flat = v.reshape(batch * group, self.kv_heads, key_tokens, head_dim)
        attended = _safe_scaled_dot_product_attention(q_flat, k_flat, v_flat)
        attended = attended.reshape(batch, group, self.q_heads, query_tokens, head_dim)
        if self.query_is_spatial:
            height = query.shape[-2]
            width = query.shape[-1]
            return self.out_proj(_tokens_to_spatial(attended, height, width))
        return self.out_proj(_tokens_to_fibers(attended))


class RegularSelfAttention(RegularCrossAttention):
    def __init__(
        self,
        channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group: FiniteGroupSpec | int | None = None,
    ) -> None:
        super().__init__(
            query_channels=channels,
            key_channels=channels,
            output_channels=channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_spatial=True,
            key_is_spatial=True,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group=group,
            fuse_qkv=True,
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor | None = None) -> torch.Tensor:
        if key_value is None:
            key_value = query
        return super().forward(query, key_value)


class RegisterToSpatialAttention(RegularCrossAttention):
    """Update registers: Q comes from registers; fused K/V come from the board."""

    def __init__(
        self,
        register_channels: int,
        spatial_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group: FiniteGroupSpec | int | None = None,
    ) -> None:
        super().__init__(
            query_channels=register_channels,
            key_channels=spatial_channels,
            output_channels=register_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_spatial=False,
            key_is_spatial=True,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group=group,
        )


class SpatialToRegisterAttention(RegularCrossAttention):
    """Update the board: Q comes from the board; fused K/V come from registers."""

    def __init__(
        self,
        spatial_channels: int,
        register_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        group: FiniteGroupSpec | int | None = None,
    ) -> None:
        super().__init__(
            query_channels=spatial_channels,
            key_channels=register_channels,
            output_channels=spatial_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_spatial=True,
            key_is_spatial=False,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
            group=group,
        )
