# Active distillation targets

The active 9×9 dataset is sampled directly from the 2026-02-26 Tromp–Taylor,
komi-7 KataGo book. Teacher-model inference, generated continuations, ownership,
and a later high-visit phase are not currently considered.

For each sampled book node:

- Policy is uniform across concrete moves whose mover-optimal book score ties
  after rounding to the nearest 0.5.
- Budget normalizes concrete `AVisits` after discarding the aggregate `other`
  row. Pages containing `other` remain eligible for sampling.
- Score is the rounded optimal book `ssM`, converted to mover perspective and
  divided by 81.
- WDL is one-hot draw when rounded score is zero. Otherwise book `wl` supplies
  soft W/L mass, with draw and no-result set to zero.

The frozen dataset contains `2^20` training positions and `2^12` validation
positions selected uniformly by stable hash from validated canonical nodes.

Implementation note: this one-off build sequentially replays the full book to
validate histories and symmetry mappings before sampling. Its runtime is
acceptable for the current run; if the dataset is ever regenerated, prefer
sample-first validation (and optionally parallel frontier replay) instead.
