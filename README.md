# SAKIGo

SAKI currently stands for **SymmetryAwareKatago-DistillationImplementation**.
SAKIGo is the Go AI project built around that direction.

## Layout

- `sakigo/` — the Python stack: `model` (D4-equivariant net, `group_size ∈ {1,8}`),
  `data` (schema, tensor-shard prep, loaders), `train` (Trainer: torch.compile,
  TensorBoard, TOML config; `benchmark` for batch-size sweeps), `generate`
  (KataGo phase-1 teacher data), `eval` (paired self-play matches), `engine`
  (Rust bindings). Frozen cross-module contracts: `sakigo/CONTRACTS.md`.
- `Engine/` — Rust rules/encoding crate; pyo3 wheel via
  `uvx maturin build --manifest-path Engine/Cargo.toml --release --out dist`.
- `Design/` — source-of-truth design notes; packaged model specs live in
  `sakigo/model/specs/`.
- `Training/data`, `Training/runs` — pre-rebuild datasets and checkpoints
  (still loadable; see `tests/test_legacy_checkpoints.py`).

Common commands:

```
uv run --frozen pytest --basetemp=.pytest-tmp -p no:cacheprovider
uv run --frozen python -m sakigo.train --data <shards> --model-spec plain --steps 1000
uv run --frozen python -m sakigo.train.suite --data <shards> --run-dir runs/phase1_suite
uv run --frozen python -m sakigo.generate --samples 4096 --output data/gen --run-dir runs/gen
uv run --frozen python -m sakigo.eval --player-a <ckpt> --player-b random --pairs 50
```

CI additionally builds and installs the PyO3 wheel before running the native
binding and generator gates, so a missing engine cannot turn those tests into
silent skips.
