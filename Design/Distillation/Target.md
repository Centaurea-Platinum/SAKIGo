# Active distillation targets

The active dataset is sampled directly from six KataGo books: Tromp–Taylor and
Japanese rules at each of 7×7, 8×8, and 9×9. Teacher-model inference, generated
continuations, ownership, and a later high-visit phase are not currently
considered.

For each sampled book node:

- Policy is uniform across concrete moves whose mover-optimal book score ties
  after rounding to the nearest 0.5.
- Budget normalizes raw concrete visits (`v`) after discarding the aggregate
  `other` row. If a row lists multiple symmetry-equivalent coordinates, every
  coordinate receives the row's full `v` count before normalization. Adjusted
  visits (`av`) are not used. Pages containing `other` remain eligible.
- Score is the rounded optimal book `ssM`, converted to mover perspective and
  divided by the active board area.
- WDL is one-hot draw when rounded score is zero. Otherwise book `wl` supplies
  soft W/L mass, with draw and no-result set to zero.

The current large run reserves 1,024 validation positions for each of the six
board-size/ruleset cohorts (6,144 total), then uses the remaining fully labeled
eligible capacity for 19,117,495 training positions. Training allocation follows
eligible population; there is no fixed board-size or ruleset quota.

Each book is streamed into its own SQLite index, then replay-validated under its
declared board size, rules, and komi. Separate books are indexed, validated, and
emitted concurrently. Frozen samples, emitted shard status, and preparation are
bound to manifest provenance so stale files cannot silently enter training.
