"""Reusable finite-group equivariant attention layers.

The package is intentionally small: it assumes a finite discrete symmetry group
represented in the regular basis and provides torch modules for pointwise
mixing, normalization, attention, and register/spatial cross-attention.
"""

from equivariant_attention.groups import (
    FiniteGroupSpec,
    cyclic_square_group,
    dihedral_square_group,
    resolve_group,
    trivial_group,
)
from equivariant_attention.layers import (
    InvariantPool,
    RegularCrossAttention,
    RegularLift,
    RegularLinear1x1,
    RegularPointwiseMLP,
    RegularRMSNorm,
    RegularSelfAttention,
    RegisterToSpatialAttention,
    SpatialToRegisterAttention,
)

__all__ = [
    "FiniteGroupSpec",
    "InvariantPool",
    "RegularCrossAttention",
    "RegularLift",
    "RegularLinear1x1",
    "RegularPointwiseMLP",
    "RegularRMSNorm",
    "RegularSelfAttention",
    "RegisterToSpatialAttention",
    "SpatialToRegisterAttention",
    "cyclic_square_group",
    "dihedral_square_group",
    "resolve_group",
    "trivial_group",
]
