# Equivariant Fiber Features

## Variables
- `X`: the position space
- `G`: a finite group acting on the position space.
- `x`: a position. The dimension can be arbitrary, as long as every `g in G` has a defined action on that space.
- `v_g`: the feature vector in the fiber component indexed by `g`.
- `v`: the concatenated feature vector over all group components:
    v = [v_g]_{g in G}
    dim(v) = |G| * dim(v_g)

## Equivariance Convention

For `k in G`, features transform as:

    (T_k v)_g(x) = v_{k^{-1}g}(k^{-1}x)

All formulas below assume this left-regular action on the group index.

## Positional Embedding

Use one shared embedding function `f`. Each fiber component receives positional information in its own canonical frame:
    v_g(x) -> f(v_g(x), g^{-1}(x))

So the same shared function `f` can be applied to every component without breaking equivariance.

Tokens without a position (register tokens) receive no positional embedding and transform only on the group index. In cross-attention between board and register features, only the board side applies `f`.

## QKV Projections And Channel Mixing

QKV projections and channel mixing are linear maps applied to the concatenated vector `v`. To preserve equivariance, the weight matrix should not be arbitrary; it should reuse the same learned block weights across group components.

The implementation may evaluate several independent maps in one larger regular
linear operation. Board self-attention concatenates the Q, K, and V output
channels into one fused QKV projection, then splits them before attention.
Register cross-attention evaluates Q separately on the query stream and
concatenates K and V into one fused projection on the key/value stream. This is
only an execution/storage fusion: each slice retains independent weights, and
the equivariant map and parameter count are unchanged.

Equivalently, the full matrix is built from repeated block rows. If the first block row is:
    b = [B_e, B_{g_1}, B_{g_2}, ...]
then the row for output component `h` is the same collection of blocks, but permuted according to `h^{-1}`:
    row_h = [B_{h^{-1}g}]

So each learned block is reused across all output group components through a group-index permutation.

## Pointwise Nonlinearities

The action `T_k` only permutes group components and positions, so any elementwise function applied identically to every fiber component commutes with it. Shared activations are therefore equivariant, and so are elementwise products of two fiber features with matching component index (gating).
