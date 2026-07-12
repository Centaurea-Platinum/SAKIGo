# SAKIGo

SAKI currently stands for **SymmetryAwareKatago-DistillationImplementation**.
SAKIGo is the Go AI project built around that direction.

Current work is limited to KataGo-teacher distillation: teacher-data generation,
model training, and evaluation. Search, self-play training, and auxiliary-head
designs are not currently considered.

## Layout

- `sakigo/` — the Python stack: `model` (D4-equivariant fixed-schedule net),
  `data` (schema, tensor-shard prep, loaders), `train` (Trainer: reduce-overhead torch.compile,
  TensorBoard, TOML config; `benchmark` for batch-size sweeps), `generate`
  (KataGo phase-1 teacher data), `eval` (paired checkpoint-policy matches), `engine`
  (Rust bindings). Frozen cross-module contracts: `sakigo/CONTRACTS.md`.
- `Engine/` — Rust rules/encoding crate; pyo3 wheel via
  `uvx maturin build --manifest-path Engine/Cargo.toml --release --out dist`.
- `Design/` — source-of-truth design notes; packaged model specs live in
  `sakigo/model/specs/`.
- `Training/data`, `Training/runs` — pre-rebuild datasets and checkpoints
  retained for reference; current architecture changes do not preserve old
  checkpoint compatibility.

- `Viewer/model_architecture.html` — fixed-budget bottleneck/depth comparison.

Common commands:

```
uv run --frozen pytest --basetemp=.pytest-tmp -p no:cacheprovider
uv run --frozen python -m sakigo.train --data <shards> --model-spec balanced --steps 1000
uv run --frozen python -m sakigo.train.suite --data <shards> --run-dir runs/phase1_suite
uv run --frozen python -m sakigo.train.auto_book_suite --generation-run runs/tt7-book-only --suite-run runs/tt7-one-epoch
uv run --frozen python -m sakigo.generate --samples 4096 --output data/gen --run-dir runs/gen
uv run --frozen python -m sakigo.eval --player-a <ckpt> --player-b random --pairs 50
```

CI additionally builds and installs the PyO3 wheel before running the native
binding and generator gates, so a missing engine cannot turn those tests into
silent skips.
