# Model Architecture

SAKIGo uses one deliberately narrow model family: a D4-equivariant board
stream, a small rule/register stream, and a fixed trunk program. The packaged
models vary only the board bottleneck width `n` and the number of board blocks
`L` while holding the persistent board width and `L * n = 1024` fixed.

The implementation lives in `sakigo/model/model.py` and
`sakigo/model/layers.py`. Reusable regular-representation operations live in
`equivariant_attention/layers.py`.

Use the [interactive architecture pipeline](../../Viewer/model_architecture.html)
to inspect the program and compare the depth/width sweep.

## Fixed Architecture

| Part | Value |
|---|---|
| Accepted board | square `N x N`, `N <= 32` |
| Inputs | 6 board planes, 10 rule features |
| Symmetry | D4 regular representation, `G = 8` |
| Board stem | scalar pointwise `6 -> 16 -> 128`, then one D4 lift |
| Rule MLP | `10 -> 32 -> 128`, reshaped to `2 x 64` registers |
| Persistent board width | `m = 128` |
| Register state | `R = 2`, `r = 64` per register |
| Register attention width | `a = 32` |
| Attention heads | `Hq = 2`, `Hkv = 1` |
| Board block | plain `m -> n`, two self-attentions, `n -> m` |
| Register exchange | one initial broadcast, one final gather |
| Attention-work proxy | `L * n = 1024` exactly |

There is no scalar-control model, SwiGLU variant, FiLM path, repeated register
cycle, or general trunk-layout experiment in the current architecture.

## Packaged Depth/Width Sweep

All three models keep `m = 128` and every non-board-block width fixed. The
power-of-two bottlenecks use reciprocal block counts so `L * n = 1024`
exactly. Parameter count is reported rather than controlled.

| Spec | Bottleneck `n` | Blocks `L` | `L*n` | Parameters in blocks | Broadcast | Gather | Trunk total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `narrow-deep` | 32 | 32 | 1,024 | 3,688,544 | 82,305 | 65,857 | 3,836,706 |
| `balanced` (default) | 64 | 16 | 1,024 | 5,257,264 | 82,305 | 65,857 | 5,405,426 |
| `wide-shallow` | 128 | 8 | 1,024 | 8,400,920 | 82,305 | 65,857 | 8,549,082 |

This is a controlled depth-versus-width comparison, not three unrelated model
designs.

## Tensor Conventions

```text
B   batch size
G   D4 group size = 8
N   active square-board size
S   board token count = N * N
R   register count = 2

m   persistent board width = 128
n   board-block bottleneck width = 32, 64, or 128 regular reps
r   persistent width of each register = 64
a   register cross-attention width = 32
```

The two persistent states are:

```text
board features   [B, m, G, N, N]
registers        [B, R, r, G]
```

A regular `1x1` layer mixes channels and the entire group fiber at each board
cell or register token using tied equivariant weights. `GroupRMSNorm`
normalizes the channel and group axes jointly at each cell or register, with
one learned scale per channel shared across `G`.

## Fixed Forward Program

```python
def forward(board, rules):
    # board: [B, 6, N, N], rules: [B, 10]
    x = lift(stem(board))                  # [B, 128, 8, N, N]
    registers = initial_registers(rules)  # [B, 2, 64, 8]

    x = broadcast(x, registers)
    for block in board_blocks:            # exactly L plain blocks
        x = block(x)
    registers = gather(registers, x)

    global_features = reshape(registers, [B, 1, R * r, G])

    wdl_logits = mean_group(wdl_head(global_features))
    score = mean_group(score_head(global_features))
    policy_pass = mean_group(policy_pass_head(global_features))
    budget_pass = mean_group(budget_pass_head(global_features))

    policy_board = flatten_board(mean_group(policy_head(x)))
    budget_board = flatten_board(mean_group(budget_head(x)))

    return {
        "wdl_logits": wdl_logits,                           # [B, 4]
        "score": score,                                     # [B, 1]
        "policy_logits": concat(policy_board, policy_pass), # [B, S + 1]
        "budget_logits": concat(budget_board, budget_pass), # [B, S + 1]
    }
```

The model returns raw values. It applies no legality mask, softmax, sigmoid,
dropout, causal mask, or final trunk normalization. Pass is the final action
logit.

Mean-pooling `G` makes WDL, score, and pass outputs invariant. Spatial heads
remain D4-equivariant because board positions still transform.

## Stem And Register Initialization

```python
x = scalar_conv1x1_6_16(board)            # [B, 16, N, N]
x = silu(x)
x = scalar_conv1x1_16_128(x)              # [B, 128, N, N]
x = lift(x)                               # [B, 128, G, N, N]

registers = scalar_mlp_10_32_128(rules)  # fixed SiLU between projections
registers = reshape(registers, [B, 2, 64, 1])
registers = expand(registers, group=G)
```

The pointwise scalar stem commutes with every board transformation, so its
output may be lifted only once at the end without weakening D4 equivariance.
Running the same maps after lifting produced eight identical output fibers and
also reread eight identical input fibers. Moving the lift reduces the stem from
17,296 to 2,288 parameters and its dense linear MAC count by 64x. The stem still
contains no spatial mixing; its `1x1` maps only mix channels at each cell.

The rule MLP directly creates the register state; its final bias already
supplies a rule-independent component, so there is no separate learned
register seed.

The initial broadcast is the only path from rules into the board trunk. Its
board-side queries use RoPE, so it conditions spatial board features before any
board self-attention. The final gather refreshes registers from the fully
processed board immediately before the global and pass heads.

## Plain Two-Attention Board Block

```python
def BoardBlock(x):
    residual = x                              # [B, m, G, N, N]
    work = silu(f1(rms_norm_in(x)))           # regular f1: m -> n

    work = work + alpha_1 * attn_1(rms_norm_1(work))
    work = work + alpha_2 * attn_2(rms_norm_2(work))

    delta = rms_norm_3(work)
    return residual + beta * f4(delta)        # regular f4: n -> m
```

The persistent `m`-wide residual bypasses the bottleneck. The two attention
updates share one `m -> n -> m` envelope but own independent Q/K/V parameter
slices and output projections. Within each attention update, Q/K/V are issued
by one fused regular projection. `alpha_1`, `alpha_2`, and `beta` are learned
scalars in every block, initialized to `1 / sqrt(2 * L)`.

No register operation or conditioning branch is embedded in a board block.

## Exact Trunk Parameter Formula

For D4 (`G = 8`), `Hq = 2`, `Hkv = 1`, a plain block has:

```text
f1 + f4                 16*m*n + n + m
two GQA modules          48*n^2 + 6*n
four RMSNorms + scalars       m + 3*n + 3
------------------------------------------------
P_block(m, n)           16*m*n + 48*n^2 + 10*n + 2*m + 3
```

At fixed `m = 128`:

```text
P_block(n) = 48*n^2 + 2058*n + 259
```

The fixed exchange modules contain:

```text
initial broadcast   82,305
final gather        65,857
exchange total     148,162
```

Therefore:

```text
P_trunk(n, L) = 148,162 + L * (48*n^2 + 2058*n + 259)
```

This trunk count is reported but not controlled by the comparison. It excludes
the fixed stem, rule MLP, and output heads; the controlled quantity is `L*n`.

## Grouped-Query Attention

All attention modules use equivariant regular projections followed by PyTorch
scaled-dot-product attention. Self-attention fuses Q/K/V because all three read
the same tensor. Register cross-attention keeps Q separate and fuses K/V because
K and V share the opposite stream:

```python
if self_attention:
    q, k, v = split_regular_channels(regular_qkv_projection(query))
else:
    q = regular_q_projection(query)
    k, v = split_regular_channels(regular_kv_projection(key_value))

# [B, G, heads, tokens, d]; attention is independent per group component
q, k, v = reshape_to_tokens(q, k, v)

if query_is_board:
    q = apply_2d_rope(q)
if key_is_board:
    k = apply_2d_rope(k)

attended = scaled_dot_product_attention(
    fold_batch_and_group(q),
    fold_batch_and_group(k),
    fold_batch_and_group(v),
    enable_gqa=True,
)
return regular_output_projection(unfold(attended))
```

Attention never treats group components as tokens. Regular projections mix the
group fiber before and after attention; SDPA folds `B * G` into the batch
dimension. Projection fusion only concatenates output-channel slices: it does
not share Q/K/V weights, change attention math, or change parameter counts.

| Module | Query tokens | K/V tokens | Per-head width | Q width | K/V width | Output width |
|---|---:|---:|---:|---:|---:|---:|
| board self-attention | `S` | `S` | `n / 2` | `n` | `n / 2` | `n` |
| final gather | `R` | `S` | 16 | 32 | 16 | 64 |
| initial broadcast | `S` | `R` | 16 | 32 | 16 | 128 |

## Positional Encoding

Only board-side queries and keys receive 2D RoPE. Register tokens have no
spatial position. For active board size `N`, each group component uses board
coordinates in its canonical `g^-1` frame:

```text
global angle = frequency * coordinate / max(N - 1, 1)
local angle  = frequency * coordinate
```

Each frequency consumes four head dimensions: a sine/cosine pair for rows and
one for columns. The fixed frequencies `pi` and `pi/2` consume eight dimensions;
remaining dimensions are unrotated.

## Register Exchange

Broadcast and gather are pre-normalized residual cross-attention modules. The
query stream is the destination of the update; the K/V stream is the sole
source of attended content:

```python
board_delta = broadcast_attention(
    query=rms_norm(board),
    key_value=rms_norm(registers),
)
board = board + gamma_broadcast * board_delta
# registers are unchanged

register_delta = gather_attention(
    query=rms_norm(registers),
    key_value=rms_norm(board),
)
registers = registers + gamma_gather * register_delta
# board is unchanged
```

Each owns one learned `gamma`, initialized to the depth-independent value
`1 / sqrt(2)`. Broadcast runs once before the board blocks; gather runs once
after them. Their widths, placement, and initialization are fixed rather than
sweep axes.

| Operation | Q projection reads | Fused K/V projection reads | Board-side RoPE | Residual updated | Other stream |
|---|---|---|---|---|---|
| initial broadcast | board | registers | Q | board | registers returned unchanged |
| final gather | registers | board | K | registers | board returned unchanged |

There is no query-side V in either operation. During gather, V is computed only
from the board. During broadcast, V is computed only from the registers. The
query stream contributes the attention weights and survives through its explicit
residual skip; V itself never receives RoPE.

## Heads

All seven subheads are built unconditionally.

| Source | Pointwise shape | Outputs |
|---|---|---|
| merged registers `[B, 1, R*r, G]` | `R*r -> 32 -> 8 -> output` | WDL, score, policy pass, budget pass |
| board `[B, m, G, N, N]` | `m -> 32 -> 8 -> 1` | policy board, budget board |

Hidden projections use SiLU and final projections are linear. The group axis is
always mean-pooled in the heads.

## What the Matched Proxy Controls

At fixed board size, board-attention mixing scales approximately as:

```text
O(L * G * S^2 * n)
```

The three models therefore match the leading board-token attention-mixing
proxy. They do not match parameters, activation depth, projection work, or
wall-clock latency:

| Spec | Attention calls | `L*n` | `L*n^2` width-work proxy | `L*m` retained-depth proxy |
|---|---:|---:|---:|---:|
| `narrow-deep` | 64 | 1,024 | 32,768 | 4,096 |
| `balanced` | 32 | 1,024 | 65,536 | 2,048 |
| `wide-shallow` | 16 | 1,024 | 131,072 | 1,024 |

Narrow/deep supplies more sequential transformations and activation depth.
Wide/shallow supplies richer per-step features and more width-dependent
projection work. The balanced model is the default until training and
paired-game evidence justify moving toward either extreme.
