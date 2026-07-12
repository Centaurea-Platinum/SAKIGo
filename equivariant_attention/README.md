# Equivariant Attention

Small torch-only layers for finite-group equivariant attention in the regular
representation.

The library assumes the pleasant case: finite, discrete groups with a clean
basis and, for spatial attention, an exact action on a square grid. It is meant
to be boring to reuse: define or pick a group spec, keep an explicit group axis,
and compose standard PyTorch modules.

## Tensor Shapes

```text
spatial features:  [B, C, G, H, W]
register/fibers:   [B, R, C, G]
```

`G` is the group order. The trivial group has `G = 1`, so the same modules also
serve as scalar controls.

## Quick Start

```python
from math import pi

from equivariant_attention import (
    RegularPointwiseMLP,
    RegularSelfAttention,
    dihedral_square_group,
)

group = dihedral_square_group()

# SAKIGo-style ModelSpecs fields:
trunk = {
    "expanded_channel": 32,
    "bottleneck_channel": 16,
    "q_heads": 2,
    "kv_heads": 1,
    "global_rope_frequencies": (pi,),
    "local_rope_frequencies": (pi / 2,),
}
head_dim = trunk["bottleneck_channel"] // trunk["q_heads"]

pre = RegularPointwiseMLP(
    (trunk["expanded_channel"], trunk["bottleneck_channel"]),
    group,
)
attn = RegularSelfAttention(
    channels=trunk["bottleneck_channel"],
    board_size=19,
    q_heads=trunk["q_heads"],
    kv_heads=trunk["kv_heads"],
    head_dim=head_dim,
    global_rope_frequencies=trunk["global_rope_frequencies"],
    local_rope_frequencies=trunk["local_rope_frequencies"],
    group=group,
)
post = RegularPointwiseMLP(
    (trunk["bottleneck_channel"], trunk["expanded_channel"]),
    group,
)

y = post(attn(pre(x)))  # x/y: [B, 32, 8, 19, 19]
```

Regular-domain head and post-processing shapes can use full pointwise MLP
tuples in the same style:

```python
spatial_head = (
    trunk["expanded_channel"],
    trunk["expanded_channel"],
    8,
    1,
)
```

## Design

- `FiniteGroupSpec` owns the multiplication table, inverse table, and optional
  square-grid coordinate action.
- `RegularLinear1x1` performs equivariant pointwise mixing using relative group
  components.
- `RegularSelfAttention` applies attention independently in canonical group
  frames, with RoPE coordinates computed as `g^-1(position)`.
- `RegisterToSpatialAttention` and `SpatialToRegisterAttention` provide the
  global-token exchange used by SAKIGo, without depending on Go-specific code.

Presets are intentionally modest: trivial, cyclic square rotations, and D4. Add
new groups by constructing `FiniteGroupSpec` rather than extending the layers.
