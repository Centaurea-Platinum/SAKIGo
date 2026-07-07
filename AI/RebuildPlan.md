# Rebuild Plan — Python stack (2026-07-06)

**Status (2026-07-06, final): P0–P6 complete — cutover executed with owner approval.** Legacy `Training/*.py` and `Model/` deleted; `Training/data` + `Training/runs` (datasets, checkpoints) kept and still loadable (tests/test_legacy_checkpoints.py). Parity gates that required the legacy code were converted to self-contained invariant tests after passing (35/35 green post-cutover). The batch-size VRAM sweep survived as `sakigo/train/benchmark.py`. adapters.py was deleted with Model/ per the deferral decision — reimplement as a thin wrapper over the Rust encoder if/when the Phase-2 teacher-net path is reconciled with D10. Highlights: unified `SakiGoNet` reproduces both legacy models bit-for-bit from real checkpoints; trainer has torch.compile (verified on the 5070 Ti via triton-windows 3.7.1, ~140 samples/s steady on model1@19×19 vs ~70 legacy) + TensorBoard + tqdm + weights_only checkpoints; data = mmap tensor shards + standard DataLoader; pyo3 engine wheel built (`dist/`, needs repo-local `CARGO_HOME` — rsproxy mirror unreachable) and golden-tested against the legacy generator Game; generator decomposed onto the Rust engine and live-verified against real KataGo end-to-end (generate→validate→prepare→collate). Remaining: P6 (delete legacy after approval; port remaining legacy-only utilities: run_phase1_suite sweep, Viewer pointers, README/Design updates).

Goal: replace the grown Training/+Model/ stack (~8,500 lines) with a modular, standard, modern package. Drivers: too much hand-rolled machinery (streaming buffer, CUDA-graph wrapper, CSV metrics, spec parser), ~690 lines of scalar/equivariant duplication, three parallel encoder implementations, no torch.compile/TensorBoard.

## Audit summary (what exists)

- `Training/` ≈ 5,600 lines. Biggest artifact: `StreamingJsonlBuffer` (~500 lines of byte-budgeted buffer + offset-bag sampling + pinned-batch fencing) doing what a standard Dataset/DataLoader over pre-shuffled shards does. Metrics = hand-rolled 60-column CSV + two bespoke ASCII progress bars. Orchestration = CLI-string subprocess stitching. A full pure-Python Go engine lives *inside* the data-generator script; `selfplay_eval` imports it from there.
- `Model/` ≈ 2,850 lines. `scalar_layers.py`+`scalar_model.py` are ~85–90% verbatim copies of the equivariant stack. `specs.py` (~590 lines) hand-parses JSON-in-.md with a homemade expression evaluator. `inference.py` hand-rolls CUDA-graph capture. Forward-time module-global caches (`_ROPE_CACHE`, `_flat_weight_bias`) are the top torch.compile blockers.
- `Engine/` (Rust) is correct and tested but **unwired**: `python.rs` is parked (not compiled), and Python re-implements the rules+encoder independently in the generator.

## Target layout

One installable package (`pyproject.toml` packaging; kills all `sys.path.insert` hacks):

```
sakigo/
  engine/     # pyo3 bindings to Rust crate: Game + encoder. THE rules/encoding impl.
  model/      # unified stack, group_size ∈ {1, 8} (scalar = group_size 1)
  data/       # schema + record validation, rulesets, D4 augment, Dataset/collate
  train/      # Trainer, config, losses, metrics, checkpoints
  generate/   # KataGo phase-1 generator (decomposed)
  eval/       # selfplay_eval
tests/        # ported regression suites + golden-vector cross-tests
```

## Modernization decisions

| Hand-rolled today | Rebuild uses |
|---|---|
| `StreamingJsonlBuffer` + `PinnedBatchKeeper` + scan-cache (~700 ln) | Map-style `Dataset` over an index built from generation-time manifests; `BatchSampler` grouping by board size + round-robin rulesets; `DataLoader(num_workers>0, pin_memory=True)` |
| CSV metrics + 2 progress bars | TensorBoard `SummaryWriter` (+ small CSV mirror if wanted) + `tqdm` |
| `_make_scheduler` hand-derived warmup-cosine | `SequentialLR(LinearLR, CosineAnnealingLR)` |
| No compile; hand CUDA graphs in `inference.py` | `torch.compile` on trainer step (default on, `--no-compile` escape); `torch.compile(mode="reduce-overhead")` + `inference_mode` replaces `SakiGoInference` |
| `_repeat_kv` GQA copies | `F.scaled_dot_product_attention(enable_gqa=True)` (torch ≥2.5; we're on 2.11) |
| `RegularRMSNorm`/`ScalarRMSNorm` | `nn.RMSNorm` (regular case = RMSNorm over flattened C·G) |
| Module-global tensor caches, `_flat_weight_bias` eval cache | Non-persistent registered buffers; compute kernel every forward, let compile CSE it |
| `specs.py` expression evaluator + `.md` fallback chains | Same spec *format*, files renamed `.json`, explicit path, dataclass/pydantic schema validation, no string arithmetic (precompute derived dims in schema step). ~590→~150 ln |
| argparse-vars-in-checkpoint, `weights_only=False` | Config dataclass serialized as JSON alongside; `torch.save` payload loadable with `weights_only=True` |
| `run_phase1_suite` CLI stitching | TOML run-config consumed by one `sakigo.train` entry point; keep `benchmark_spec_batch` VRAM sweep as a utility |
| 3 encoder implementations (Rust, adapters.py, generator `Game`) | **One**: compile `python.rs` (pyo3), extend it to expose the encoder; generator + eval use it; golden-vector cross-test Rust↔recorded JSONL |
| Duplicated scalar stack (~690 ln) | Single parameterized implementation; `group_size=1` is the scalar control |

## Preserve exactly (correctness invariants)

1. Mover perspective everywhere: wdl/score/ownership targets, board planes (own/opp), signed komi + capture-diff features.
2. Pass logit = index N² (last) in policy/budget/legal_mask; D4 augment transforms `[:-1]` only.
3. Score ÷ area; komi/capture features ÷ area, clamped ±1.
4. Per-head masks with `mask.sum().clamp_min(1)` branchless averaging (losses.py ports as-is).
5. Policy = teacher top-1 one-hot; budget = full teacher policy renormalized over legal.
6. `split_for_position` blake2b split keyed (seed, board, ruleset, position_key) — byte-identical.
7. Ownership BCE (t+1)/2 mapping, ≥0 ⇔ own convention.
8. `rulesets.py` exactness enforcement — ports as-is.
9. One board size per batch; per-batch ruleset round-robin balance.
10. D4 augment only for non-equivariant specs; equivariant model trains unaugmented.
11. `d4.py` verbatim (brute-force-derived tables, canonical-frame RoPE, relative-component kernel); rule-conditioned register initialization; depth-scaled residual gains 1/√(2·blocks); head shapes incl. `pass_*` register heads.
12. Data schema v1 (JSONL/.zst fields) unchanged — existing generated data stays valid.
13. Optimizer decay/no-decay split for biases and normalization parameters.
14. Regression harnesses: 25 model tests (equivariance per layer + full model, stale-cache regression) and ~30 training tests, ported.

## Phasing (each gate = tests green before next phase)

**P0 — Scaffold + contracts**: package layout, pyproject packaging, pytest wiring (`--basetemp=.pytest-tmp -p no:cacheprovider`). Port unchanged: `d4.py`, `losses.py`, `rulesets.py`, schema constants. Freeze the stable contracts as short docs before any code depends on them: record schema (v1 JSONL fields), model forward/output schema, 10-dim rule-feature encoding, checkpoint payload schema, run-directory layout. Optional quick win, independent of the rebuild: bolt a TensorBoard `SummaryWriter` mirror onto the old trainer for immediate observability.

**P1 — Model unification**: single stack with group-size parameter; buffers replace global caches; `enable_gqa`; `nn.RMSNorm`; specs→JSON+schema. Gate: ported equivariance suite green **+ checkpoint parity** — load an existing `model1`/`model2` checkpoint (state-dict remap allowed, script it) and assert output equality vs old code on fixed inputs.

**P2 — Data**: record validation + hash split ported; manifest-based map-style Dataset + board-size BatchSampler + ruleset balancing; DataLoader workers. Also benchmark pre-tensorized shards (`.jsonl.zst` → tensor shards conversion) vs decode-on-load and keep whichever measures better. Gate: same train/val membership as old split on real shards; batch-composition stats match.

**P3 — Trainer**: config dataclass + TOML; Trainer loop with torch.compile, bf16 autocast, fused AdamW, SequentialLR, TensorBoard (+ small CSV mirror so the Viewer keeps working, + profiler hook), tqdm, clean checkpoints (atomic rename + RNG capture kept). Compile lands here *by design*: after P1 removed the forward-time caches (top graph-break sources) and separate from the P2 data rewrite — never benchmark compile against the old model or mix it with data changes. Flags: compile default-on with `--no-compile` escape; record compile status in run config. Gate: short A/B run old-vs-new on identical data/seed — loss curves statistically indistinguishable; compile speedup measured.

**P4 — Engine wiring**: compile pyo3 `python.rs`, add encoder to the binding; golden-vector tests Rust encoder ↔ existing JSONL records ↔ adapters.py before deleting Python copies. Gate: cross-tests green, `cargo test` green.

**P5 — Generator + eval**: decompose `generate_katago_phase1.run()` (KataGo client / scheduler / record builder / shard writer as modules) on the Rust engine; port `selfplay_eval` onto `sakigo.engine`. Gate: regenerate a small sample set and diff record semantics vs old generator (same positions → same targets).

**P6 — Cutover**: old `Training/`+`Model/` deleted only after P1–P5 gates; update READMEs/Design pointers (ask before touching non-AI files), AI/Context.md map.

**P7 — Search/self-play (outlook, not this rebuild)**: search + player interfaces, replay/target generation, subtree harvest (D11/D12/D16) — only on top of the rebuilt stack and the D10-vs-Target.md reconciliation.

### External review reconciliation (2026-07-06)

A second agent's plan was reviewed. Adopted: contracts-first (P0), compile/data-rewrite isolation (P3), tensor-shard benchmark (P2), CSV mirror + profiler hooks (P3), explicit search phase (P7), and its stale-notes catch (Context.md still advertised removed `--cuda-graphs`/eager path — fixed). Rejected: its ordering put torch.compile *before* model cleanup — the known top blockers (`_flat_weight_bias` mutable cache, global RoPE caches) live in the model, so compile benchmarks would graph-break against code slated for deletion; model unification (P1) stays ahead of compile (P3). Half-adopted: its "trainer shell first, keep current data path" slice — TB-on-old-trainer gives the observability win without shaping new Trainer hooks around the non-standard batch-yielding IterableDataset (`num_workers=0` contract).

## Resolved decisions (human, 2026-07-06)

- **One model fits all board sizes — confirmed, no design change.** Weights are size-agnostic (1×1 kernels, attention, head MLPs); only RoPE/coordinate buffers are per-size and derived. torch.compile "recompiles" = per-shape kernel specialization only (one cached graph per board size, same weights); checkpoints reusable across sizes. Fall back to `mark_dynamic` on N only if the size set grows large.
- **pyo3/maturin build is unblocked**: prior "build errors" were slow large-file downloads misread as timeouts. Network is fine — build with long timeouts and patience. P4 proceeds.
- **adapters.py is deferred (clarified after checking imports)**: Phase 1 needs *ruleset* projection (`rulesets.py`, preserved) and the encoder (Rust engine) — but nothing in Training/ imports adapters.py. Its `GameStateBatch`/`KataGoInputProjection` scaffolding serves the unreconciled Phase-2 teacher-net path; port it only when that phase is reconciled with D10.
- **TensorBoard only** — no wandb.
- **torch.compile on Windows is viable (verified 2026-07-06)**: the old note "Inductor needs Triton, unavailable on native Windows" is stale. Triton Windows support is now upstream (`triton-lang/triton-windows`), pip `triton-windows` wheels on PyPI, Blackwell sm120 supported. P3 checklist: install the triton-windows version matching torch 2.11 (pairing table publicly ends at 2.10↔3.6 — verify), enable Windows long-path support (known torchinductor temp-filename failure), vcredist present.
