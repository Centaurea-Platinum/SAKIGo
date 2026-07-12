# Stem

The board stem remains scalar until its final output:

```text
[B, 6, N, N]
  -> scalar 1x1, 6 -> 16
  -> SiLU
  -> scalar 1x1, 16 -> 128
  -> D4 lift
[B, 128, 8, N, N]
```

Every scalar `1x1` map is shared over board positions and therefore commutes
with rotations and reflections. Lifting after the scalar maps is exactly as
expressive as lifting first and applying regular maps to the resulting
group-constant fibers: each effective regular weight is the sum of its eight
group-component weights.

The late lift avoids computing the same stem feature in all eight D4 fibers.
For the fixed `6 -> 16 -> 128` shape, stem parameters fall from 17,296 to 2,288
and dense linear MACs fall by 64x. All spatial mixing and orientation-sensitive
processing remains in the regular-representation trunk after the lift.
