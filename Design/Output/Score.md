> **Current scope: distillation only.** Only the normalized scalar score target
> is active. Percentile and distributional score heads are not currently considered.

Include a simple scalar score head. Normalize by division of board area.
Include more complicated structures like percentile based score heads later.
    Percentile score heads are more generalizable across different board size compared to categorical heads, and more stable than MDNs. They can encode multimodal data and distribution well if finely grained enough.
