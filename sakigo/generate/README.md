# Mixed small-board book distillation

The active pipeline samples directly from six KataGo books:

- 9×9 Tromp–Taylor komi 7 and Japanese komi 6;
- 8×8 Tromp–Taylor komi 10 and Japanese komi 9;
- 7×7 Tromp–Taylor komi 9 and Japanese komi 8.

It performs no teacher inference, generates no continuations, and has no
ownership target. Default global sizes remain `2^20` training records and
`2^12` validation records; these totals are not multiplied by the number of
books. Validation is divided as evenly as capacity allows across all six
board-size/ruleset cohorts, while training is sampled globally from the
remaining eligible population.

```powershell
python -m sakigo.generate.multi_book_distillation all `
  --run-dir runs/smallboard-multibook --workers 3
```

The resumable stages are `artifacts`, `index`, `sample`, and `prepare`.
Indexing, validation, and JSONL emission process separate books concurrently.
Archives are streamed into one SQLite index per book rather than extracted as
millions of tiny files.

The automatic suite can wait behind an active index, refresh a stale allocation
to requested totals, emit and prepare the dataset, run mandatory batch-safety
preflight, and train all three packaged models sequentially. For example,
`--train-samples 23000000 --validation-samples 6144` reserves exactly `2^10`
validation records for each of the six active board-size/ruleset cohorts.

Completed shards carry byte counts and SHA-256 digests that are verified on
reuse and ingestion. Tensor preparation rejects duplicate or overlapping
train/validation sources and rechecks content identity around both decode
passes before atomically publishing a prepared generation.

Training has no fixed board-size, book, or ruleset quota. A seeded global
without-replacement draw selects from the remaining union of eligible nodes in
proportion to each book's population. Training then creates shape-compatible
batches from shuffled records of one board size. Rulesets are not stratified
inside a training batch, and shuffled board-size batch tickets occur in
proportion to the available records; consecutive batches may naturally have
the same size. Validation batches are homogeneous by both board size and
ruleset so their loss curves remain separately comparable.

Pages containing an `other` row remain eligible. Only the `other` row is
discarded when constructing concrete targets. Policy assigns equal mass to
concrete moves tied after score rounding, budget normalizes concrete `AVisits`,
score uses rounded `ssM`, and W/L uses `wl` with rounded zero score mapped to
draw. Canonical book nodes deduplicate symmetries and transposed move orders;
the frozen sample table additionally prevents a node from appearing twice or
crossing between training and validation. Positions with different board sizes
or rule features remain distinct because they are different model inputs.

The legacy single-book command remains available as
`sakigo.generate.book_distillation` for reproducing the original 9×9 TT7 run.
