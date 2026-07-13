# Active distillation targets

The active dataset is sampled directly from six KataGo books: Tromp–Taylor and
Japanese rules at each of 7×7, 8×8, and 9×9. Teacher-model inference, generated
continuations, ownership, and a later high-visit phase are not currently
considered.

For each sampled book node:

- Policy is uniform across concrete moves whose mover-optimal book score ties
  after rounding to the nearest 0.5.
- Budget normalizes concrete `AVisits` after discarding the aggregate `other`
  row. Pages containing `other` remain eligible for sampling.
- Score is the rounded optimal book `ssM`, converted to mover perspective and
  divided by the active board area.
- WDL is one-hot draw when rounded score is zero. Otherwise book `wl` supplies
  soft W/L mass, with draw and no-result set to zero.

The frozen dataset contains `2^20` training positions and `2^12` validation
positions selected by one seeded global without-replacement draw from the union
of eligible validated canonical nodes. Book allocation follows eligible
population; there is no fixed board-size or ruleset quota.

Each book is streamed into its own SQLite index, then replay-validated under its
declared board size, rules, and komi. Separate books are indexed, validated, and
emitted concurrently. Frozen samples, emitted shard status, and preparation are
bound to manifest provenance so stale files cannot silently enter training.
