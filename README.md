# SAKIGo

SAKI currently stands for **SymmetryAwareKatago-DistillationImplementation**.
SAKIGo is the Go AI project built around that direction.

Current work is limited to direct distillation from KataGo's published
small-board books: deterministic dataset construction, model training, and
paired policy evaluation. Teacher-model inference, search, self-play training,
and auxiliary-head designs are not currently considered.

## Layout

- `sakigo/` — the Python stack: `model` (D4-equivariant fixed-schedule net),
  `data` (schema, immutable prepared mmap generations, loaders), `train` (Trainer: reduce-overhead torch.compile,
  TensorBoard, TOML config; `benchmark` for batch-size sweeps), `generate`
  (mixed 7x7/8x8/9x9 book data; legacy teacher generator retained), `eval`
  (paired checkpoint-policy matches), `engine`
  (Rust bindings). Frozen cross-module contracts: `sakigo/CONTRACTS.md`.
- `Engine/` — Rust rules/encoding crate; pyo3 wheel via
  `uvx maturin build --manifest-path Engine/Cargo.toml --release --out dist`.
- `Design/` — source-of-truth design notes; packaged model specs live in
  `sakigo/model/specs/`.
- `Training/data`, `Training/runs` — pre-rebuild datasets and checkpoints
  retained for reference; current architecture changes do not preserve old
  checkpoint compatibility.

- `Viewer/model_architecture.html` — matched-attention-work bottleneck/depth comparison.
- `Viewer/pipeline_viewer.html` — pipeline and aggregate-metrics viewer. Training
  runs also write `validation_metrics.csv` and TensorBoard `val_groups/` curves
  for each board-size/ruleset cohort.

Common commands:

```
uv run --frozen pytest --basetemp=.pytest-tmp -p no:cacheprovider
uv run --frozen python -m sakigo.train --data <shards> --model-spec balanced --steps 1000
uv run --frozen python -m sakigo.train.suite --data <shards> --run-dir runs/model-suite
uv run --frozen python -m sakigo.train.auto_book_suite --generation-run runs/smallboard-multibook --suite-run runs/smallboard-one-epoch
uv run --frozen python -m sakigo.generate.multi_book_distillation all --run-dir runs/smallboard-multibook --workers 3
uv run --frozen python -m sakigo.eval --player-a <ckpt> --player-b random --pairs 50
```

CI additionally builds and installs the PyO3 wheel before running the native
binding and generator gates, so a missing engine cannot turn those tests into
silent skips.
