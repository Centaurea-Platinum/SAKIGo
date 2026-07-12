> **Current scope: distillation only.** Only the normalized scalar score target
> is active. Percentile and distributional score heads are not currently considered.

Include a simple scalar score head. Normalize by division of board area.
For the active 9x9 run, weight normalized smooth-L1 by the board area (81)
so normalization does not make the shared-trunk score gradient negligible.
Report MAE in board points alongside the normalized metric.
Include more complicated structures like percentile based score heads later.
    Percentile score heads are more generalizable across different board size compared to categorical heads, and more stable than MDNs. They can encode multimodal data and distribution well if finely grained enough.
