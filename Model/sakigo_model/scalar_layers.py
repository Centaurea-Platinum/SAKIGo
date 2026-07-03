from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .d4 import _device_key
from .layers import build_activation

_SCALAR_ROPE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}


class ScalarLinear1x1(nn.Module):
    """Pointwise scalar channel mixing for board and register features."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        scale = 1.0 / sqrt(max(in_channels, 1))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.uniform_(self.weight, -scale, scale)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch, _, height, width = x.shape
            flat = x.permute(0, 2, 3, 1).reshape(batch * height * width, self.weight.shape[1])
            y = F.linear(flat, self.weight, self.bias)
            return y.reshape(batch, height, width, self.weight.shape[0]).permute(0, 3, 1, 2)
        if x.dim() == 3:
            batch, registers, _ = x.shape
            flat = x.reshape(batch * registers, self.weight.shape[1])
            y = F.linear(flat, self.weight, self.bias)
            return y.reshape(batch, registers, self.weight.shape[0])
        raise ValueError("ScalarLinear1x1 expects [B,C,H,W] or [B,R,C]")


class ScalarRMSNorm(nn.Module):
    """RMS norm over scalar channels."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            rms = torch.mean(x.square(), dim=1, keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, -1, 1, 1)
        if x.dim() == 3:
            rms = torch.mean(x.square(), dim=2, keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, 1, -1)
        raise ValueError("ScalarRMSNorm expects [B,C,H,W] or [B,R,C]")


class ScalarPointwiseMLP(nn.Module):
    def __init__(self, channels: tuple[int, ...], final_activation: bool = False) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("ScalarPointwiseMLP needs at least two channel sizes")
        layers: list[nn.Module] = []
        last = len(channels) - 2
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            layers.append(ScalarLinear1x1(in_channels, out_channels))
            if index < last or final_activation:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _scalar_rope_cos_sin(
    board_size: int,
    device: torch.device,
    dtype: torch.dtype,
    global_frequencies: tuple[float, ...],
    local_frequencies: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        "scalar-rope",
        board_size,
        *_device_key(device),
        dtype,
        tuple(float(frequency) for frequency in global_frequencies),
        tuple(float(frequency) for frequency in local_frequencies),
    )
    cached = _SCALAR_ROPE_CACHE.get(cache_key)
    if cached is None:
        cells = torch.arange(board_size * board_size, device=device)
        rows = (cells // board_size).to(dtype)
        cols = (cells % board_size).to(dtype)
        denom = max(board_size - 1, 1)
        channels: list[torch.Tensor] = []
        for frequency in global_frequencies:
            channels.append(frequency * rows / denom)
            channels.append(frequency * cols / denom)
        for frequency in local_frequencies:
            channels.append(frequency * rows)
            channels.append(frequency * cols)
        angles = torch.stack(channels, dim=-1)
        cached = (angles.cos().unsqueeze(0).unsqueeze(0), angles.sin().unsqueeze(0).unsqueeze(0))
        _SCALAR_ROPE_CACHE[cache_key] = cached
    return cached


def _apply_scalar_rope(
    tokens: torch.Tensor,
    board_size: int,
    global_frequencies: tuple[float, ...],
    local_frequencies: tuple[float, ...],
) -> torch.Tensor:
    """Apply 2D RoPE to [B,H,L,D] board tokens."""
    frequency_count = len(global_frequencies) + len(local_frequencies)
    pair_count = 2 * frequency_count
    rope_dim = 2 * pair_count
    if tokens.shape[-1] < rope_dim:
        raise ValueError(
            f"{frequency_count}-frequency 2D RoPE needs at least {rope_dim} head dimensions"
        )
    cos, sin = _scalar_rope_cos_sin(
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


def _repeat_kv(tokens: torch.Tensor, q_heads: int) -> torch.Tensor:
    kv_heads = tokens.shape[1]
    if kv_heads == q_heads:
        return tokens
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    return tokens.repeat_interleave(q_heads // kv_heads, dim=1)


def _board_tokens(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if channels != heads * head_dim:
        raise ValueError("projected board channels do not match heads * head_dim")
    return x.reshape(batch, heads, head_dim, height * width).permute(0, 1, 3, 2)


def _register_tokens(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    batch, registers, channels = x.shape
    if channels != heads * head_dim:
        raise ValueError("projected register channels do not match heads * head_dim")
    return x.reshape(batch, registers, heads, head_dim).permute(0, 2, 1, 3)


def _tokens_to_board(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, heads, cells, head_dim = x.shape
    return x.permute(0, 1, 3, 2).reshape(batch, heads * head_dim, height, width)


def _tokens_to_registers(x: torch.Tensor) -> torch.Tensor:
    batch, heads, registers, head_dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch, registers, heads * head_dim)


class ScalarCrossAttention(nn.Module):
    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        output_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        query_is_board: bool,
        key_is_board: bool,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.query_is_board = query_is_board
        self.key_is_board = key_is_board
        self.global_rope_frequencies = global_rope_frequencies
        self.local_rope_frequencies = local_rope_frequencies

        self.q_proj = ScalarLinear1x1(query_channels, q_heads * head_dim)
        self.k_proj = ScalarLinear1x1(key_channels, kv_heads * head_dim)
        self.v_proj = ScalarLinear1x1(key_channels, kv_heads * head_dim)
        self.out_proj = ScalarLinear1x1(q_heads * head_dim, output_channels)

    def _to_tokens(self, x: torch.Tensor, heads: int, is_board: bool) -> torch.Tensor:
        if is_board:
            return _board_tokens(x, heads, self.head_dim)
        return _register_tokens(x, heads, self.head_dim)

    def _active_board_size(self, query: torch.Tensor, key_value: torch.Tensor) -> int:
        if self.query_is_board:
            height, width = query.shape[-2:]
        elif self.key_is_board:
            height, width = key_value.shape[-2:]
        else:
            return self.board_size
        if height != width:
            raise ValueError("board features must be square")
        if height > self.board_size:
            raise ValueError(f"board size {height} exceeds configured maximum {self.board_size}")
        return height

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key_value)
        v_proj = self.v_proj(key_value)
        board_size = self._active_board_size(query, key_value)

        q = self._to_tokens(q_proj, self.q_heads, self.query_is_board)
        k = self._to_tokens(k_proj, self.kv_heads, self.key_is_board)
        v = self._to_tokens(v_proj, self.kv_heads, self.key_is_board)
        if self.query_is_board:
            q = _apply_scalar_rope(q, board_size, self.global_rope_frequencies, self.local_rope_frequencies)
        if self.key_is_board:
            k = _apply_scalar_rope(k, board_size, self.global_rope_frequencies, self.local_rope_frequencies)
        k = _repeat_kv(k, self.q_heads)
        v = _repeat_kv(v, self.q_heads)

        attended = F.scaled_dot_product_attention(q, k, v)
        if self.query_is_board:
            height = query.shape[-2]
            width = query.shape[-1]
            return self.out_proj(_tokens_to_board(attended, height, width))
        return self.out_proj(_tokens_to_registers(attended))


class ScalarGQAAttention(ScalarCrossAttention):
    def __init__(
        self,
        channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
    ) -> None:
        super().__init__(
            query_channels=channels,
            key_channels=channels,
            output_channels=channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_board=True,
            key_is_board=True,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor | None = None) -> torch.Tensor:
        if key_value is None:
            key_value = query
        return super().forward(query, key_value)


class ScalarRegisterToBoardAttention(ScalarCrossAttention):
    def __init__(
        self,
        register_channels: int,
        board_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
    ) -> None:
        super().__init__(
            query_channels=register_channels,
            key_channels=board_channels,
            output_channels=register_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_board=False,
            key_is_board=True,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
        )


class ScalarBoardToRegisterAttention(ScalarCrossAttention):
    def __init__(
        self,
        board_channels: int,
        register_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
    ) -> None:
        super().__init__(
            query_channels=board_channels,
            key_channels=register_channels,
            output_channels=board_channels,
            board_size=board_size,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_is_board=True,
            key_is_board=False,
            global_rope_frequencies=global_rope_frequencies,
            local_rope_frequencies=local_rope_frequencies,
        )


class ScalarTrunkBlock(nn.Module):
    def __init__(
        self,
        trunk_channels: int,
        bottleneck_channels: int,
        board_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        global_rope_frequencies: tuple[float, ...],
        local_rope_frequencies: tuple[float, ...],
        block_count: int,
        eps: float,
        enable_gather: bool = True,
        enable_broadcast: bool = True,
        activation: str = "none",
    ) -> None:
        super().__init__()
        self.enable_gather = enable_gather
        self.enable_broadcast = enable_broadcast
        scale = 1.0 / sqrt(2.0 * block_count)
        self.alpha_1 = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.alpha_2 = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.gamma_1 = nn.Parameter(torch.tensor(scale, dtype=torch.float32)) if enable_gather else None
        self.gamma_2 = nn.Parameter(torch.tensor(scale, dtype=torch.float32)) if enable_broadcast else None

        self.norm_in = ScalarRMSNorm(trunk_channels, eps)
        self.norm_reg = ScalarRMSNorm(trunk_channels, eps) if enable_gather else None
        self.norm_bottleneck_1 = ScalarRMSNorm(bottleneck_channels, eps)
        self.norm_bottleneck_2 = ScalarRMSNorm(bottleneck_channels, eps)
        self.norm_bottleneck_3 = ScalarRMSNorm(bottleneck_channels, eps)
        self.norm_out = ScalarRMSNorm(trunk_channels, eps) if enable_broadcast else None
        self.norm_reg_out = ScalarRMSNorm(trunk_channels, eps) if enable_broadcast else None

        self.f1 = ScalarPointwiseMLP((trunk_channels, bottleneck_channels))
        self.f1_activation = build_activation(activation)
        self.attn1 = ScalarGQAAttention(
            bottleneck_channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
        )
        self.attn2 = ScalarGQAAttention(
            bottleneck_channels,
            board_size,
            q_heads,
            kv_heads,
            head_dim,
            global_rope_frequencies,
            local_rope_frequencies,
        )
        self.f4 = ScalarPointwiseMLP((bottleneck_channels, trunk_channels))
        self.register_gather = (
            ScalarRegisterToBoardAttention(
                trunk_channels,
                trunk_channels,
                board_size,
                q_heads,
                kv_heads,
                head_dim,
                global_rope_frequencies,
                local_rope_frequencies,
            )
            if enable_gather
            else None
        )
        self.register_broadcast = (
            ScalarBoardToRegisterAttention(
                trunk_channels,
                trunk_channels,
                board_size,
                q_heads,
                kv_heads,
                head_dim,
                global_rope_frequencies,
                local_rope_frequencies,
            )
            if enable_broadcast
            else None
        )

    def forward(self, x: torch.Tensor, registers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        normed_in = self.norm_in(residual)
        register_gather = self.register_gather
        if register_gather is not None:
            norm_reg = self.norm_reg
            gamma_1 = self.gamma_1
            if norm_reg is None or gamma_1 is None:
                raise RuntimeError("gather block is missing its paired modules")
            registers = registers + gamma_1 * register_gather(
                norm_reg(registers),
                normed_in,
            )
        x1 = self.f1_activation(self.f1(normed_in))
        x2 = x1 + self.alpha_1 * self.attn1(self.norm_bottleneck_1(x1))
        x3 = x2 + self.alpha_2 * self.attn2(self.norm_bottleneck_2(x2))
        out = residual + self.beta * self.f4(self.norm_bottleneck_3(x3))
        register_broadcast = self.register_broadcast
        if register_broadcast is not None:
            norm_out = self.norm_out
            norm_reg_out = self.norm_reg_out
            gamma_2 = self.gamma_2
            if norm_out is None or norm_reg_out is None or gamma_2 is None:
                raise RuntimeError("broadcast block is missing its paired modules")
            out = out + gamma_2 * register_broadcast(
                norm_out(out),
                norm_reg_out(registers),
            )
        return out, registers
