# Trunk

The trunk has one fixed D4-equivariant program:

```text
rules -> registers
board, registers -> broadcast -> board
board -> L plain two-attention board blocks -> board
registers, board -> gather -> registers
```

The initial broadcast conditions the board on rules before spatial reasoning.
The final gather reads the completed board into the registers immediately
before the global and pass heads. There are no register exchanges between board
blocks.

Register attention follows ordinary cross-attention source semantics:

```text
broadcast: Q = board,     K/V = registers, update board only
gather:    Q = registers, K/V = board,     update registers only
```

Q is projected separately and K/V are produced by one fused regular projection.
The untouched stream is returned as-is. Thus gather has no register-derived V,
and broadcast has no board-derived V; the query stream is preserved by its
residual connection.

Each board block keeps a persistent `m = 128` residual stream, projects to a
working bottleneck `n`, applies two independent D4-equivariant self-attention
updates, and projects back to `m`. Blocks do not read registers and contain no
FiLM or other conditioning branch.

The packaged sweep holds every other width and the trunk parameter budget
fixed while trading bottleneck width against depth:

| Spec | Bottleneck `n` | Blocks `L` | Trunk parameters |
|---|---:|---:|---:|
| `narrow-deep` | 32 | 46 | 5,450,444 |
| `balanced` (default) | 64 | 16 | 5,405,426 |
| `wide-shallow` | 128 | 5 | 5,398,737 |

All board and register attention preserves whole-network D4 equivariance. The
exact block formula, exchange counts, and compute caveat are documented in
[ModelArchitecture.md](ModelArchitecture.md).

General operation layouts, repeated register cycles, FiLM blocks, SwiGLU
blocks, and scalar-control trunks are outside the current architecture.
