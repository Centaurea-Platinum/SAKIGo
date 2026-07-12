# Book-only 9×9 distillation data

The active dataset is sampled directly from the 2026-02-26 Tromp–Taylor,
komi-7 KataGo book. It performs no teacher inference, generates no
continuations, and has no ownership target.

Default sizes are exactly:

- training: `2^20` = 1,048,576 records;
- validation: `2^12` = 4,096 records.

Nodes are selected uniformly by a stable hash over validated canonical book
positions. Pages containing an `other` row remain eligible; only the `other`
row is discarded when concrete move targets are constructed.

```powershell
python -m sakigo.generate.book_distillation all --run-dir runs/tt7-book-only
```

The stages may also be resumed separately:

```powershell
python -m sakigo.generate.book_distillation artifacts --run-dir runs/tt7-book-only
python -m sakigo.generate.book_distillation index --run-dir runs/tt7-book-only
python -m sakigo.generate.book_distillation sample --run-dir runs/tt7-book-only
```

The archive is streamed directly into SQLite rather than extracted as roughly
a million tiny HTML files. The index replays canonical parent histories with
the local rules engine and rejects board, player, orientation, or concrete-move
legality disagreements. The exact sampled node IDs are frozen in SQLite before
records are emitted. Completed zstd shards are atomic and skipped on resume.

Targets are book-derived: equal mass over concrete moves tied after score
rounding for policy, normalized concrete `AVisits` for budget, rounded `ssM`
for score, and `wl` for W/L (with rounded zero score mapped to draw).
