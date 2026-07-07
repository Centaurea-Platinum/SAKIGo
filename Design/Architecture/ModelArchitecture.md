The model trunk uses D4-equivariant grouped-query attention. Reference [./EquivariantAttention.md].

G = D4 for the main equivariant model. The scalar control uses the same implementation with G = 1.

The stem first lifts scalar input planes into D4-regular features, then applies D4-regular equivariant 1x1 fiber mixing to reach trunk width. The lift repeats scalar planes across the group axis, and regular 1x1 mixing keeps group-constant fibers group-constant. The 8 D4 components first diverge at the first RoPE'd board attention layer.

Rule features initialize the register stream:

```python
x = stem(lift(board))                         # [B, trunk_channel, G, H, W]

registers = rule_mlp(rules)                   # [B, register_count * register_channel]
registers = reshape(registers)
registers = expand_group_axis(registers)      # [B, register_count, register_channel, G]
```

Each trunk block is a nested residual bottleneck around the board stream, with optional register gather/broadcast residuals on selected 1-based block indices:

```python
# x has m regular reps: [B, m, G, H, W]
# registers have r regular reps per register token: [B, register_count, r, G]

residual = x
x_norm = norm_in(residual)

if block_index in register_gather_blocks:
    registers = registers + gamma_1 * gather(
        norm_reg(registers),                  # Q from registers
        x_norm,                               # K/V from board
    )
    # gather uses register_bottleneck_channel as its internal attention width.
    # Its output projection returns register_channel, so the register residual shape is stable.

if trunk_mlp_variant == "plain":
    x_1 = activation(f1(x_norm))              # f1: m -> n regular reps
elif trunk_mlp_variant == "swiglu":
    value, gate = split(f1(x_norm), 2)        # f1: m -> 2n regular reps
    x_1 = value * SiLU(gate)                  # output: n regular reps

x_2 = x_1 + alpha_1 * attn_1(norm_1(x_1))     # board self-attention, n reps
x_3 = x_2 + alpha_2 * attn_2(norm_2(x_2))     # board self-attention, n reps

out = residual + beta * f4(norm_3(x_3))       # f4: n -> m regular reps, linear

if block_index in register_broadcast_blocks:
    out = out + gamma_2 * broadcast(
        norm_out(out),                        # Q from board
        norm_reg_out(registers),              # K/V from registers
    )
    # broadcast also uses register_bottleneck_channel internally.
    # Its output projection returns trunk_channel, so the board residual shape is stable.

return out, registers
```

`alpha_1`, `alpha_2`, `beta`, and enabled `gamma_1`/`gamma_2` are independent learned scalar residual multipliers in each block. They initialize to `1 / sqrt(2 * block_count)`.

The trunk bottleneck width `bottleneck_channel` controls the board stream between `f1` and `f4`. The register attention width `register_bottleneck_channel` controls only the hidden Q/K/V width of register gather/broadcast attention. It does not change the persistent register width; register tensors remain `register_channel` wide across residual adds.

Register gather and broadcast are cross-attention updates with explicit output projections back to the persistent stream width. Let:

```python
board_dim = trunk_channel
register_dim = register_channel
register_bottleneck_dim = register_bottleneck_channel
register_head_dim = register_bottleneck_dim // q_heads
raw_kv_dim = kv_heads * register_head_dim
```

With grouped-query attention, Q has `q_heads` heads while K/V have `kv_heads` heads. If `kv_heads < q_heads`, K/V are projected to `raw_kv_dim`, then repeated across query-head groups inside attention. The attended result is still `q_heads * register_head_dim == register_bottleneck_dim`.

Register gather updates the register stream from the board stream:

```python
# registers query the board
register input:    register_dim
board input:       board_dim

Q projection:      register_dim -> register_bottleneck_dim
K projection:      board_dim -> raw_kv_dim
V projection:      board_dim -> raw_kv_dim
attention result:  register_bottleneck_dim
output projection: register_bottleneck_dim -> register_dim
residual add:      register_dim + register_dim
```

Register broadcast updates the board stream from the register stream:

```python
# board queries the registers
board input:       board_dim
register input:    register_dim

Q projection:      board_dim -> register_bottleneck_dim
K projection:      register_dim -> raw_kv_dim
V projection:      register_dim -> raw_kv_dim
attention result:  register_bottleneck_dim
output projection: register_bottleneck_dim -> board_dim
residual add:      board_dim + board_dim
```

So `register_bottleneck_channel` is not a new register state size. It is the internal query/attention result width used before the output projection returns to either `register_dim` for gather or `board_dim` for broadcast.

Board attention uses spec-listed global and local 2D RoPE frequencies. Each frequency rotates 4 head dimensions: row sine/cosine and column sine/cosine. Any head dimensions beyond `4 * frequency_count` remain unrotated. RoPE applies only to board-side Q/K. Register tokens are unrotated.

Global heads merge register tokens into one D4-regular vector, apply D4-regular head MLPs, then average over the D4 axis to produce invariant logits:

```python
merged_registers = reshape(registers)         # [B, 1, register_count * register_channel, G]
wdl_logits       = invariant(wdl_head(merged_registers))
score            = invariant(score_head(merged_registers))
policy_pass      = invariant(policy_pass_head(merged_registers))
budget_pass      = invariant(budget_pass_head(merged_registers))
```

Spatial heads apply D4-regular equivariant head MLPs to the final board features, then average over the D4 axis to produce invariant board logits:

```python
ownership_logits = invariant(ownership_head(out))
policy_board     = invariant(policy_head(out))
budget_board     = invariant(budget_head(out))

policy_logits = concat(policy_board, policy_pass)
budget_logits = concat(budget_board, budget_pass)
```

Stem and head MLPs use fixed SiLU activations between projections and no activation after the final projection. The trunk's plain bottleneck mixer uses the spec-selected activation after `f1`; the SwiGLU variant replaces that activation site with `value * SiLU(gate)`. In both variants, `f4` remains linear so the residual update is sign-unconstrained.
