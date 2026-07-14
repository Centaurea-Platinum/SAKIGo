> **Current scope: distillation only.** Only the normalized scalar score target
> is active. Percentile and distributional score heads are not currently considered.

Include a simple scalar score head. Normalize each target by its board area.
The active mixed-small-board suite multiplies normalized smooth-L1 by each
record's actual board area, giving effective score weights of 49, 64, and 81
for 7×7, 8×8, and 9×9 respectively. The configurable score weight is a base
multiplier on top of that area scaling and defaults to 1. Report MAE in board
points by converting each record with its actual board area.
Include more complicated structures like percentile based score heads later.
    Percentile score heads are more generalizable across different board size compared to categorical heads, and more stable than MDNs. They can encode multimodal data and distribution well if finely grained enough.
