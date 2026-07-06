# Session Log — SAKIGo

Dated, newest first. What changed, what is next. One entry per working session.

## 2026-07-07 - Register width decoupled from trunk width

- Created branch `codex-register-width-support` after the repo could not create a slash-nested `codex/...` branch under the local refs/sandbox layout.
- Added `register_channels`, `register_bottleneck_channels`, and `register_head_dim` to `SakiGoModelConfig`; old checkpoints/specs default to the prior full-width register behavior.
- Updated `SakiGoNet` and `TrunkBlock` so register seed/rule MLP/global heads use `register_channels`, while gather/broadcast cross-attention uses rectangular projections between board width and register width.
- Updated Design + packaged ModelSpecs/StemShapes/HeadShapes: `register_channel` and `register_bottleneck_channel` are first-class spec fields; `model2`/`model3` now use 128-wide trunks, 64-wide registers, and 32-wide register cross-attention. Parameter counts: `model1` 332,238 unchanged; `model2`/`model3` 8,426,602 -> 6,585,578.
- Tests added/updated for narrow-register specs and rectangular-register equivariance. Verification: targeted specs/model/legacy checkpoint tests passed (`17 passed`); full suite passed (`44 passed in 272.16s`).

## 2026-07-07 - Extracted reusable equivariant attention library

- Added `equivariant_attention/`: a torch-only finite-group regular-representation attention package with `FiniteGroupSpec`, trivial/Cn/D4 square-grid presets, regular linear/norm/MLP layers, invariant pooling, spatial self-attention, and spatial/register cross-attention. Included a short README with shapes and usage.
- Rewired `sakigo.model.group` and `sakigo.model.layers` to consume the library through compatibility wrappers, preserving SAKIGo's old class names, tensor contracts, and state-dict parameter layout.
- Added `tests/test_equivariant_attention_library.py`: D4 table compatibility, D4 regular-linear equivariance, D4 attention equivariance, C4 attention reuse, and spatial/register cross-attention equivariance.
- Verification: focused equivariance tests passed (`8 passed`); full suite passed with a fresh temp dir (`42 passed in 255.71s`). First full-suite attempt failed only because Windows denied cleanup of the pre-existing `.pytest-tmp` directory.

## 2026-07-06 - Post-cutover robustness review of sakigo/

- Full manual read of the training/data/model/generate/eval surface (no subagents); suites green before and after (36 pytest incl. new test, 17 cargo).
- Verdict: robust/standard/modern/modular — one real gap found and fixed: KataGo death mid-generation hung `run()` forever (`responses.get()` blocks; stdout reader exited silently on EOF). Reader now enqueues an `_engine_exited` sentinel with returncode + stderr tail; run() raises loudly. Regression test `test_katago_client_signals_engine_exit`.
- Noted, not fixed (by design or minor): val metrics rotate through the val set across evaluations (comparable only in expectation); `run_phase1_suite` was not ported in the cutover — multi-spec sweeps need per-spec `python -m sakigo.train` invocations (user hit this: `python -m Training.run_phase1_suite` now exits 1).
- Follow-up (same day, on request): added `--val-fixed` — freezes the first `val_batches` batches of the seeded val sampler (`FixedBatchSampler`) and replays them every eval for smooth step-to-step deltas; default stays rotating (unbiased, full coverage). Test: `test_val_fixed_replays_identical_batches`. 37 pytest green.

## 2026-07-06 - P6 cutover executed (legacy stack deleted)

- Owner approved ("lets cut!"). Deleted `Training/*.py` (12 modules) and `Model/` entirely; kept `Training/data` + `Training/runs`. Legacy checkpoints remain loadable — [test_legacy_checkpoints.py](../tests/test_legacy_checkpoints.py) pins the load+remap path.
- Parity tests that needed the legacy oracle were converted to self-contained invariant tests (equivariance, spec sync, hash-split/round-trip/collate contract, engine binding invariants, generator perspective-flip/schema/quota/shard tests). 35/35 pytest green post-cutover; pyproject testpaths now just `tests/`.
- Salvaged before deletion: the WDDM-aware batch-size sweep → [sakigo/train/benchmark.py](../sakigo/train/benchmark.py) (`python -m sakigo.train.benchmark`). adapters.py died with Model/ per the deferral decision (reimplement over the Rust encoder when Phase 2 is reconciled with D10).
- Updated: root README (layout + commands), AI/Context.md (phase, implementation map, boundary now names `sakigo/`), RebuildPlan status → complete.
- Rebuild summary across the day: ~8.5k-line legacy stack → `sakigo/` package (~4.3k lines incl. generator), one model implementation, one encoder implementation (Rust), torch.compile + TensorBoard, mmap shard data path, ≈ 2× training throughput.

## 2026-07-06 - Rebuild P0–P5 implemented (new `sakigo/` package, all gates green)

- Executed [RebuildPlan.md](RebuildPlan.md) through P5; 98/98 pytest (legacy suites untouched and green alongside the new gates in `tests/`).
- **P0**: `sakigo/` scaffold + [CONTRACTS.md](../sakigo/CONTRACTS.md) (frozen record/forward/rule-encoding/checkpoint/run-dir contracts); verbatim ports of d4/losses/rulesets/constants with byte-parity tests.
- **P1**: unified `SakiGoNet` (`group_size ∈ {1,8}`) replaces SakiGoModel+ScalarSakiGoModel (~690 dup lines): no forward-time caches (buffers + functional canonical-frame RoPE), SDPA `enable_gqa`. **Real trained model1+model2 checkpoints load (scalar via scripted remap) and reproduce outputs exactly**; equivariance suite ported. Specs: 590→~150-line JSON loader; packaged spec copies with a Design-sync test; configs identical to legacy parser.
- **P2**: JSONL→mmap tensor shards (`sakigo/data/prepare.py`, decode+validate once), map-style `PreparedDataset` + `RulesetBalancedBatchSampler` + standard DataLoader (workers, pin_memory). Gates: split membership == legacy blake2b split, round-trip exact, collate layout == legacy, D4-augment parity.
- **P3**: `Trainer` with torch.compile (default on, recorded fallback), bf16 autocast, fused AdamW, `SequentialLR`, TensorBoard + CSV mirror (Viewer-compatible), tqdm, atomic `weights_only=True` checkpoints + RNG capture, TOML/CLI config. GPU gate on real phase-1 data: compile works (triton-windows 3.7.1 + torch 2.11), ~140 samples/s steady vs ~70 legacy, loss trajectories match legacy A/B (both 13.0→10.3 in 200 steps). Fixed: seed at Trainer construction (init determinism).
- **P4**: pyo3 binding activated (feature-gated; compiled first try), encoder exposed; wheel built via maturin (repo-local `CARGO_HOME` bypasses the unreachable rsproxy mirror; `Engine/.cargo/config.toml` documents it) and installed. Golden gate: Rust engine == legacy generator `Game` on random playouts across all 4 preset rulesets (legality/board/captures/planes/rule features/simple-ko).
- **P5**: generator decomposed (`sakigo/generate/`: game/records/plan/writer/katago/run) onto the Rust engine; component parity gates (query/position-key/record construction/quotas/sharding) + **live KataGo run** (64 records @ ~112/s) fed back through validate→prepare→collate cleanly. `sakigo/eval/` = selfplay eval on the Rust engine (smoke-tested).
- **Next: P6 cutover needs human approval** — delete legacy `Training/`+`Model/` (parity tests go with them), port `run_phase1_suite` sweep if still wanted, update READMEs/Design pointers + Context.md map, decide fate of smoke artifacts (`runs/rebuild_gpu_smoke`, `runs/gen_smoke`, `.tmp-smoke/`).

## 2026-07-06 - Rebuild plan authored

- Human asked for a ground-up rebuild plan (modularize/standardize/modernize: torch.compile, TensorBoard). Two thorough audits (Training/ ~5,600 ln, Model/ ~2,850 ln) fed [RebuildPlan.md](RebuildPlan.md): one installable `sakigo/` package; delete `StreamingJsonlBuffer`/`PinnedBatchKeeper`/CSV-metrics/hand CUDA-graphs in favor of standard Dataset+DataLoader, TensorBoard+tqdm, `SequentialLR`, `torch.compile`; unify scalar+equivariant stacks via a group-size parameter (~690 dup lines); shrink specs.py to JSON+schema; wire the Rust engine via pyo3 as the single rules/encoder impl (3 copies today). 14 preserved invariants + 6 gated phases (P1 gate = checkpoint output parity). Open items for human: pyo3 buildability, adapters.py deferral, wandb, dynamic board sizes under compile. No code touched.

## 2026-07-06 - Step-0 baseline checkpoint/metrics

- Training now checkpoints and evaluates the *initialized* model at step 0 (fresh runs only, skipped on resume) and drops the step-1 special case: metric/checkpoint steps form an arithmetic sequence (0, interval, 2·interval, ...). Step-0 train columns are blank (no batches seen); val columns are real. Tests updated; 51 pytest green.

## 2026-07-06 - Simplification pass + HTML pipeline viewer

- Cut non-standard/stale machinery (-366 lines net): CUDA-graphs train-step capture (`graph_step.py`, `--cuda-graphs`, capturable-AdamW branches — measured ~3% at batch 128; inference has its own graph path in `SakiGoInference`), eager sampling remnants orphaned by streaming-only training (`RulesetAwareBatchDataset`, module-level `sample_ruleset_aware_batch`, `build_ruleset_groups`, `split_records`, `filter_records_by_boards`, `StreamingJsonlBuffer.sample_batch` + its pop path), and the speculative `prettyterm` adapter (plain ANSI now; also fixed progress-bar padding counting invisible escape codes). Kept: `load_records`/`build_groups`/`sample_batch` (suite sweep), both streaming sampling engines (offset index for plain jsonl, eviction buffer for zst — each is the only option for its format). The without-replacement test now runs on zst so it exercises the buffer path. 51 pytest green.
- Added [Viewer/pipeline_viewer.html](../Viewer/pipeline_viewer.html): single-file offline viewer (no server, no deps). Tabs: clickable pipeline map (generation/training/eval lanes with per-node invariants), record inspector (plain `.jsonl`: composite board, 6 planes, budget/policy/ownership/legal-mask heatmaps, WDL bar, decoded rule features), metrics.csv plotter (column toggles, log-y), status.json summary. Verified in-browser against real run artifacts; incidentally confirmed val now tracks train (13.2→6.4 over 18.6k steps) in the post-cache-fix suite run.

## 2026-07-06 - Rebuild improvements: suicide semantics, eager path removed, PyO3 parked

- Owner asked to implement the proposed improvements and was away; proceeded autonomously in stages.
- Engine now implements KataGo `multiStoneSuicideLegal` semantics: single-stone suicide is always illegal even under `SuicideRule::Allowed` — resolving the tracked engine/generator divergence with zero data change (generator already behaved this way). Tests updated + new `single_stone_suicide_is_always_illegal`; 17 cargo tests.
- Removed the deprecated eager training path: `train.py` is streaming-only (`--stream-buffer-mb` must be > 0), dead `balanced_eval`/`balanced_eval_streaming` deleted; the old eager smoke tests now exercise the streaming path. 52 pytest green. Eager record utilities in `data.py` remain (suite sweep + tests use them).
- PyO3 engine binding attempted and **parked**: crates mirror unreachable, and an optional `pyo3` dependency breaks offline `cargo test` (lock resolution). Binding code parked uncompiled at [python.rs](../Engine/src/python.rs); full activation checklist in Issues. No speculative Python glue landed — it cannot be exercised until the wheel builds.

## 2026-07-06 - Layered hashes; crash-safe infra; loop dedup

- Verified hash semantics against KataGo source: repetition hash is rule-defined (positional superko = board only; captures would break superko since every cycle increases them); metadata belongs in a separate situation hash. Engine now has incremental 128-bit Zobrist `PositionHash` + metadata-aware `StateHash` (to-move, simple-ko, captures, rules/komi) via `GameState::state_hash()`. 16 engine tests.
- Rebuild-lens robustness pass (manual, no subagents): atomic checkpoint saves, deduplicated the two identical training loops in train.py into `_run_training_loop`, fixed KataGo engine orphaning on generator exceptions, removed dead `--prefetch-batches`, atomic scan-cache writes. 52 pytest green. Details in [ScanResult.md](ScanResult.md) addendum.

## 2026-07-06 - Full verification scan; engine test gap fixed; verification skill

- Ran a full scan (suites + three parallel module audits + manual re-derivation of findings); results in [ScanResult.md](ScanResult.md). Verdict: no production bugs; 4 of 5 audit findings were false positives (notably "positional superko needs side-to-move" — wrong by definition).
- Fixed the one real gap: added 4 engine tests (positional superko repetition, initial-position superko on 1x1, pass semantics incl. ko clearing, multi-group capture). `cargo test` 12 passed; pytest 52 passed.
- Added `.github/skills/verification/SKILL.md`: evidence-driven verification procedure (discriminating tests over checklist ceremony) with SAKIGo invariants and past-incident lessons.

## 2026-07-05 - Audited implementation against checklist

- Scanned the current engine, model, training/data, KataGo generator, self-play eval, specs, tests, and project notes against [ImplementationChecklist.md](ImplementationChecklist.md).
- Verification: `uv run --frozen pytest` passed (`52 passed`), and `cargo test` in `Engine/` passed (`8 passed`).
- Corrected stale AI memory: WDL is now length 4 (win/draw/loss/no-result), matching design, model specs, generator, trainer, and tests.
- Added a watchlist risk: Phase 1 generator legality is duplicated in Python and currently differs from the Rust engine around allowed single-stone suicide.

## 2026-07-05 - Completed implementation checklist

- Expanded [ImplementationChecklist.md](ImplementationChecklist.md) from a rough seed into a practical implementation/review rubric.
- Covered correctness, source-of-truth boundaries, robustness, ML/training-specific traps, performance, tests, AI-note maintenance, and final-diff checks.
- Scope stayed inside `AI/`; no code or design sources changed.

## 2026-07-05 - Switched training data to DataLoader-backed zstd shards

- Added zstd JSONL support in [data.py](../Training/data.py): `.jsonl.zst` readers/writers, shard-directory/glob expansion, and format labels. Plain `.jsonl` remains readable but is treated as deprecated for training data.
- Reworked [train.py](../Training/train.py) so the default path is bounded streaming (`--stream-buffer-mb 1024`) through PyTorch `IterableDataset` + `DataLoader`; `--stream-buffer-mb 0` keeps the deprecated eager path. The training loop no longer uses the custom background prefetcher.
- Updated [generate_katago_phase1.py](../Training/generate_katago_phase1.py) to write numbered `.jsonl.zst` shards by default (`--samples-per-file 65536`, `--zstd-level 3`), with a legacy single-file escape hatch via `--samples-per-file 0`.
- Updated [run_phase1_suite.py](../Training/run_phase1_suite.py) to accept directories, globs, or multiple shard paths. Verification: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest Training/test_training_orchestrator.py` passed (`27 passed`).

## 2026-07-05 - Added Linux KataGo engine support

- Downloaded and checksum-verified the official KataGo v1.16.5 OpenCL Linux x64 archive into ignored local artifacts, then extracted it under `Distillation/engine/katago-v1.16.5-opencl-linux-x64/`.
- Updated [generate_katago_phase1.py](../Training/generate_katago_phase1.py) so the default engine lookup prefers `katago.exe` on Windows and `katago` on Linux/macOS; added regression coverage in [test_training_orchestrator.py](../Training/test_training_orchestrator.py).
- Added [Distillation/README.md](../Distillation/README.md) documenting the ignored artifact layout and Linux download command.

## 2026-07-05 - Denoted SAKI acronym

- Added the public README definition: **SAKI** currently stands for **SymmetryAwareKatago-DistillationImplementation**.
- Mirrored the same definition in [Context.md](Context.md) for future AI collaborators and corrected the model-spec context to include `model3`.

## 2026-07-03 - Fixed stale eval weight cache causing live val-loss mismatch

- Re-checked the active `phase1_20260703_215600_model1` run: its live CSV still reported uniform-ish validation at step 512 (`val_policy_loss` about 5.89) while train policy loss fell to about 3.95. Loading the same `step_000512.pt` checkpoint in a fresh CPU process scored real val slices much lower (`policy` about 1.89 near the file front and about 3.62 at val offset 5000), proving the checkpoint was good and the in-process eval metric was stale.
- Root cause: `RegularLinear1x1` cached flattened equivariant kernels during `torch.no_grad()` eval, keyed by parameter `_version`. Fused/low-level optimizer updates can leave that cache stale, so periodic validation reused the first eval's flattened weights while train forwards (grad enabled) recomputed from current weights. Changed the cache to activate only for frozen inference parameters (`requires_grad=False`), preserving `SakiGoInference` use while disabling it for trainable models being evaluated during training.
- Added a regression test that evals a trainable regular layer, mutates its parameter through a `_version`-bypassing low-level path, and confirms the second eval sees the new weights. Also fixed the earlier eager-mode metrics indentation bug and added a logging cadence regression test.
- Verification: full pytest passed (`37 passed`, using repo-local `--basetemp` and no cacheprovider because this Windows temp/cache setup denies access). The currently running model1 process imported the old code, so its live validation metrics remain invalid; its checkpoints/train updates are still usable and should be re-evaluated from a fresh process or rerun after the fix.

## 2026-07-03 - Flat val is real (no pipeline bug); sync-free metrics, scan cache, batch-128 suite

**CORRECTION (same day, after owner pushback on single-exposure sampling):** the "no pipeline bug" verdict below was itself wrong. Decisive contradiction: the live run's own step-512 checkpoint, probed offline, scored val-split policy CE 3.3-4.0 (real generalization, even on unreached file regions), while the live process's in-process val metric reported 5.89 (uniform) at the same step - and a resumed-checkpoint process agreed with the probe (~3.7-4.0), not the live metric. So the live run's val path WAS corrupted (timing-dependent, consistent with the async pinned-buffer reuse race); the "clean" A/B that drove the wrong verdict used a 38k-sample fresh model whose uniform val is genuinely correct - too short to discriminate. Also corrected: streaming exposure is ~once per record (draws = ingests = one file pass; Poisson mean 1), and train loss is computed pre-update, so the falling train loss was online generalization on the stream, NOT memorization. The hardening already landed (sync unpinned eval collate, event-fenced pinned train batches, zero-board canary) targets exactly the corrupting mechanism. Next full suite run on hardened code is the real test: expect val to track (~4 by mid-run) if the correction is right.

- Investigated the first full model1 run's frozen val_loss (13.45 while train fell to 9). Initial corruption hypothesis was WRONG: an in-process A/B showed the trainer's val eval and a known-good file-based probe agree exactly on the same live model. The earlier "checkpoint generalizes on val" readings were confounded: contiguous file slices oversample openings, and openings leak across the split because `position_key` hashes the move *sequence* - the same early position reached by different orders lands in both splits. Verdict: at ~131k samples seen, model1 genuinely generalizes only to transposed openings and nothing else; the val plateau is honest model behavior at this data scale, consistent with the 262k-is-small prediction.
- Real bug found along the way: resume crashed because `torch.load(map_location=cuda)` moves saved RNG ByteTensors to GPU; `restore_rng_state` now moves them back to CPU.
- GPU utilization: `MetricAccumulator` now accumulates on-device float64 tensors (weighted bincount for the confusion matrix, no boolean indexing) - the ~20 per-step host syncs are gone; host transfer happens only at log time. Progress bar shows last logged loss instead of forcing a per-step sync.
- Hardening (cheap, kept): eval collate is now synchronous/unpinned; train keeps pinned batches alive until a CUDA event confirms the H2D copy (`PinnedBatchKeeper`); eval raises loudly if a batch arrives with all-zero board planes (canary).
- Ergonomics: `scan_jsonl_stream_metadata` caches per-(file,size,mtime,split) results next to the data file - startup dropped ~60s -> ~7s; suite now trains all specs at a fixed `--batch-size 128` (controlled variable) with the sweep behind opt-in `--sweep`.
- Tests 8/8. Next: the A/B verdict needs game-based eval, and truly held-out validation would need a position-hash (not move-sequence) split or cross-game dedup.

## 2026-07-03 - Fixed slow batch sweep (WDDM paging, not OOM)

- Root cause: on Windows/WDDM, oversized batches do not raise `OutOfMemoryError` - the driver spills into shared system memory and runs at paging speed, so the sweep's OOM catch never fired and oversized candidates ground through all timed steps (model1 attention at batch 512 projects ~14 GiB on a 12 GiB card).
- Fix in [run_phase1_suite.py](../Training/run_phase1_suite.py): peak-allocation check after the first step against a `--memory-fraction` budget (default 0.8 x dedicated VRAM), predictive skip of larger candidates by linear peak extrapolation, `--sweep-max-seconds` time cap on the timed loop, and early stop when throughput declines below 0.97 x best. Sweep lines now print peak GiB.
- Verified on GPU across the full 128-1024 range: model1 picks 256 (7.1 GiB peak), 512 skipped by projection; control_params measures through 512 and skips 1024; whole sweep runs in seconds. Optional extra hardening: NVIDIA driver "Prefer No Sysmem Fallback" makes overflow fail fast globally.

## 2026-07-03 - Self-play eval harness (paired matches, Elo, SGF)

- Added [selfplay_eval.py](../Training/selfplay_eval.py): paired color-reversed matches between two players (checkpoint raw-policy agents with legal masking + optional temperature, or a uniform-random baseline that passes only when forced). Reuses the Phase 1 generator's `Game` (Tromp-Taylor rules and exact board/rule encoding), shares one seeded uniform-random opening per pair, adjudicates at two passes or a ply cap via a new Tromp-Taylor area scorer, and reports winrate with Wilson 95% CI, Elo diff, and per-pair 2-0/1-1/0-2 counts. Outputs games.jsonl, summary.json, and one SGF per game.
- Tests: scorer unit test (empty/one-color/contested boards) and a CPU random-vs-random match smoke (shared openings, outputs) - 8/8 passing. GPU smoke: 4-step suite_smoke model1 checkpoint vs random, 4 games completed (~31 plies/s), summary/SGF written.
- Known gaps recorded in Issues: Python legality is the throughput bottleneck (~30 plies/s -> long full matches), ply-cap adjudication scores unfinished games by raw area (vibego used a neutral strong judge), players are raw-policy only until search exists.

## 2026-07-03 - Three-model suite runner, D4 augmentation, progress bar

- Added [run_phase1_suite.py](../Training/run_phase1_suite.py): per-spec batch-size throughput sweep (OOM-aware, bf16 + fused AdamW, reuses `_train_batch`), then sequential subprocess training of `model1`, `model1_control_params`, `model1_control_compute` with an equal samples-seen budget (`--epochs` over the train split) and a final val-loss summary. Non-equivariant (ScalarSakiGoModel) specs automatically get `--augment-d4`.
- Added `augment_record_d4` to [data.py](../Training/data.py) (random D4 symmetry per training sample: board planes, ownership, policy/budget board parts, legal mask; pass entry and global fields untouched) plus a unit test; `--augment-d4` wired into both trainer loading paths (train batches only, val untouched).
- Added `--progress` to [train.py](../Training/train.py): in-place ASCII bar (step, samples/s, ETA, last loss), auto-on for TTY, clears around log lines.
- Verification: 6/6 pytest; full suite smoke on the 512-record subset ran all three specs (augment flag applied to the two controls) and printed plan + summary. Sweep datum at batch 128: model1 ~380 samples/s, control_params ~1,080, control_compute ~410 — consistent with the compute-matched design intent.

## 2026-07-03 - Training-throughput optimizations landed

- Data path: `TrainingRecord` is now numpy-backed; JSONL decoding moved to buffer-insert with vectorized validation (no per-batch `json.loads`), buffer budget counts decoded array bytes, and `collate` numpy-stacks with pinned memory + non-blocking H2D. Added `BatchPrefetcher` (default `--prefetch-batches 2`) assembling CPU batches on a background thread; the streaming buffer got an internal lock and eval sampling no longer advances the ingest stream.
- GPU path: bf16 autocast by default on CUDA (`--amp off` to disable), fused AdamW, branchless masked losses (removed 5 per-step `.item()` syncs), and opt-in `--cuda-graphs` (new [graph_step.py](../Training/graph_step.py)): 3 counted eager warmup steps on a side stream, then full-step capture (zero/forward/losses/backward/clip/step) with capturable AdamW. Skipped `torch.compile` — Inductor needs Triton, unavailable on native Windows.
- Verification: 5/5 pytest (needs `--basetemp` on this machine; system temp denies access). GPU smokes on real Phase 1 records: graphs trajectory matches eager to ~3 decimals at same seed. 300 steps @ batch 128: baseline 219s -> bf16+prefetch 126s -> +graphs 122s (~1.8x; batch-128 training is compute-bound, so graphs add little here — their payoff is batch-1 search latency later).
- Caveats: exact resume determinism is relaxed under prefetch (buffer contents timing-dependent; stream position was already not checkpointed). Startup full-file scan remains; binary shards still the right long-run fix.

## 2026-07-03 - Added streaming JSONL training buffer

- Added `--stream-buffer-mb` to [train.py](../Training/train.py). `0` keeps the old eager loader; positive values use a rolling raw-line JSONL buffer capped by the requested MiB budget, with deterministic train/validation splitting by `(seed, board_size, position_key)`.
- Added stream metadata scanning and `StreamingJsonlBuffer` in [data.py](../Training/data.py), plus a streaming train smoke in [test_training_orchestrator.py](../Training/test_training_orchestrator.py).
- Verification: `uv --cache-dir .uv-cache run --frozen pytest Training/test_training_orchestrator.py` passed (`5 passed`). A generated-slice streaming smoke passed. A full-file 1-step smoke on `Training/data/katago_phase1_20260703_172838/samples.jsonl` passed with `--stream-buffer-mb 64`; config recorded 262,144 rows, 236,411 train records, 25,733 validation records, and a 64 MiB buffer holding 2,886 raw records.

## 2026-07-03 - Checked Phase 1 sample training readiness

- Created a 512-record slice from `Training/data/katago_phase1_20260703_172838/samples.jsonl` and ran `Training.train` for 2 steps with `model1`, board size 19, batch size 2, and the current five-head losses. The smoke run completed and wrote metrics/checkpoints under `Training/runs/train_smoke_generated_readiness/`.
- Readiness verdict: the generated records have the labels needed by the current trainer, and the model/loss/data contract is working on a real generated subset. Full-scale training is not ready yet because `Training.train` eagerly loads the whole JSONL into Python records and groups, which is unsuitable for the 5.7 GB 2^18-sample file. Next implementation step is a streaming/sharded/binary data path.

## 2026-07-03 - Started 2^18 KataGo Phase 1 generation run

- Added [generate_katago_phase1.py](../Training/generate_katago_phase1.py), a schema-v1 JSONL generator that keeps concurrent KataGo analysis games/positions in flight, writes SAKIGo board/rule tensors, maps raw KataGo policy to budget, maps top-1 to policy, and converts WDL/score/ownership from BLACK to current-player perspective.
- Smoke-tested 64 samples and loaded them through `Training.data.load_records`: board planes/rules/ownership/legal-mask lengths and WDL/policy/budget normalization checked out.
- Launched run `katago_phase1_20260703_172838` for 262,144 samples using `nnMaxBatchSize=20`, `numAnalysisThreads=40`, and 40 concurrent games. Initial live checkpoint: 8,192 / 262,144 samples, about 230 samples/s, ETA about 18 minutes. Output: `Training/data/katago_phase1_20260703_172838/samples.jsonl`; status: `Training/data/katago_phase1_20260703_172838/status.json`; logs: `Training/runs/katago_phase1_20260703_172838/`.
- Completion check: status reached `complete`, process exited cleanly, output size is about 5.7 GB, and a streaming pass counted exactly 262,144 JSONL rows. Spot-validation through `record_from_json` at lines 1, 2, 3, 65,536, 131,072, 196,608, and 262,144 confirmed board/rule/target lengths and WDL/policy/budget sums.

## 2026-07-03 - Tuned KataGo batch-inference throughput

- Benchmarked the local TensorRT KataGo v1.16.5 teacher for Phase 1 sample generation using unique four-move 19x19 positions, `maxVisits: 1`, `includePolicy`, `includeOwnership`, and `includeNoResultValue`. Measured only steady-state after the engine reported ready.
- Best measured setting: `nnMaxBatchSize=20`, `numAnalysisThreads=40`, `numSearchThreadsPerAnalysisThread=1`, about 226-227 samples/s. Batch 16 / 32 threads was effectively tied but slightly lower on the confirmation run; larger batches filled correctly but were slower (`32` about 214, `64` about 204, `128` about 197 samples/s in the no-cache sweep).
- Recorded the empirical setting as D22 in [Decisions.md](Decisions.md) and added the missing reusable generator/config item to [Issues.md](Issues.md). Removed transient `analysis_logs/`; kept generated TensorRT timing caches under ignored `Distillation/engine/.../KataGoData/trtcache/`.

## 2026-07-03 - Checked Phase 1 KataGo data readiness

- Read the owner's updated [Target.md](../Design/Distillation/Target.md): Phase 1 maps KataGo raw policy to the budget head, derives the policy head from the top-1 move, and expects other current heads to map cleanly.
- Recorded D21 in [Decisions.md](Decisions.md) and updated [Issues.md](Issues.md): the labels are available from KataGo analysis output if queries include policy/ownership/no-result fields, but ingestion still needs a parser, saved position metadata, illegal-policy handling, perspective conversion, and a small emitted-schema pilot before large generation.
- Ran the local pilot through KataGo v1.16.5/TensorRT. First run built `KataGoData/trtcache/...` and showed this build rejects `analysisPVLen: 0`; reruns with `analysisPVLen: 1` succeeded. Verified empty-board output has policy length 362, ownership length 361, raw value fields, and policy sum ~= 1. Verified an occupied-point sample after `B D4` reports currentPlayer W and marks row-major index 288 as illegal (`-1`). Removed transient `analysis_logs/`; kept the useful TensorRT cache artifact.

## 2026-07-03 - Examined KataGo analysis output contract

- Read the bundled KataGo v1.16.5 README/config and the official Analysis Engine docs. The local smoke test reached TensorRT initialization on the RTX GPU but timed out while creating the timing cache before returning a JSON line; no KataGo process remained afterward, and the temporary `analysis_logs/` directory was removed.
- Recorded the missing KataGo analysis JSON importer in [Issues.md](Issues.md), including `includePolicy`, `includeOwnership`, row-major board order, pass-last policy, illegal `-1` policy entries, `rootInfo.raw*` value fields, and BLACK-to-current-player perspective conversion.
- Restored [Design/Distillation/Target.md](../Design/Distillation/Target.md) to the owner's short phase sketch after the human clarified that design docs should not be edited for findings unless asked.

## 2026-07-03 — Synced AI notes with code-bearing workspace

- Re-read root, model, and engine READMEs; current design docs; Rust engine code; PyTorch model/adapters/specs/tests; and the active diffs. Updated [Context.md](Context.md) from "design-only" to the current repo shape: `Design/`, `Engine/`, `Model/`, and local `Distillation/` assets.
- Added implemented decisions: the engine owns legality/history/encoding (D20); `Design/ModelSpecs.md` drives `model1` plus scalar controls (D19); and `SakiGoModel` implements register-seeded D4 attention with no FiLM branch (D18).
- Updated [Issues.md](Issues.md): architecture is now implemented but unvalidated, while search, scoring/adjudication, exact KataGo teacher projection, training/distillation reconciliation, and inference illegal-move masking remain open.
- Rewrote [Guide.md](Guide.md) to make AI-note maintenance an explicit duty after non-trivial work, not a separate task the human has to request.
- Worktree maintenance: added ignore coverage for local KataGo downloads/engines/model weights so artifact directories do not masquerade as source.

## 2026-07-03 — Reconciled model-design clarifications

- Updated AI notes to match the owner's clarifications: board input is six planes; trunk is D4-equivariant spatial attention plus register-token attention; global heads are MLPs on registers; rule conditioning seeds registers by default, with FiLM reserved as an add-on; captured-stone difference is `(opponent stones I captured - my stones opponent captured) / board_area`; pass is one extra logit in the same softmax as the board moves.
- Clarified in Issues that the SquareAccumulation register implementation uses equivariant regular-representation register features, not merely invariant registers; only collapsed global-head outputs are invariant.
- Left search and training-loop issues as deferred/open rather than trying to settle loss construction or search spec here; the current worktree also updates the matching `Design/` files.

## 2026-07-03 — Synced findings from the SquareAccumulation playground

- The owner's boardgame-AI repo (`D:\stuff\Documents\SquareAccumulationK-Isolation` — exact-solved n×n game, D4-equivariant attention model) doubles as a SAKIGo rehearsal ground; a 2026-07-02 session there produced SAKIGo-relevant results, now synced into these notes.
- **Equivariant read+write register attention is implemented and test-verified there** — regular-rep fibers, group-axis-batched invariant QKV, canonical-frame RoPE (board side only), no escnn. Re-scoped the Issues item from "design TBD" to "reference implementation exists; adapt to the conv trunk" and noted the update on D14. The playground's exact labels also make it the venue for the ×8-vs-augmentation A/B (D13).
- **Registers-vs-FiLM got a concrete candidate:** seed the registers at t=0 from an MLP over rule one-hots + komi/area + captures/area — one pathway, group-constant (equivariant) by construction, heads read it natively; FiLM held in reserve for multiplicative semantic gating. Decisive exact-label A/B available via the playground's center-ban rule variant. Expanded the Issues item.
- **Sizing datum:** full-width, every-block register gather was the largest parameter bucket at scale (~35%); a subset-of-blocks schedule cut params ~26%. New Issues watch-item against D14's every-block read+write. Also measured for the future self-play loop (same GPU): CUDA-graph replay ≈10× on batch-1 latency, bf16 ≈2–3× compute-bound (19×19 batch-64 on an 11.3M-param net ≈1.15 s → 0.36 s); and a trunk-activation A/B where SiLU beat a pure-linear attention trunk only modestly — attention softmax alone already supplies workable nonlinearity (attention-trunk evidence; conv trunks differ).
- Context.md gained the Sibling-testbed section and a refreshed current-phase line (the "specs mostly empty" wording was stale). Edits confined to `AI/`.

## 2026-06-19 — Track B chosen; self-play sandbox built & working

- Human picked Track B (custom PyTorch self-play sandbox). Built a minimal AlphaZero loop in `../VibeKatago/sandbox/`: Go rules (capture/suicide/positional-superko/Tromp-Taylor area score), SAKIGo 4-plane features, small policy+value CNN, PUCT MCTS, self-play, and a train/eval entry point. Launch: `uv run python -m sandbox.train [--smoke]`.
- Smoke run validated end-to-end on the GPU: loss 3.47→3.14, winrate-vs-random 0.60→0.70 in 2 quick iters (~20s each, 7×7, 16 sims). Engine has its own self-test (`uv run python -m sandbox.go`).
- This is the Milestone-0 harness. **Next:** a fuller run to confirm sustained learning, then the first SAKIGo A/B — equivariance ×8 (D13) or minimal-input (D1/D9). Sandbox layout + swap points in repo memory.

## 2026-06-19 — VibeKatago experiments sandbox set up

- Per human grant, established `../VibeKatago/` as an experiments sandbox (boundary recorded in [Context.md](Context.md)). KataGo cloned to `../VibeKatago/KataGo/` (human did the clone — my automated fetch kept hitting TLS/connection resets on a slow link). Added `VibeKatago/` to the SAKIGo repo's local `.git/info/exclude` so the design repo never tracks it.
- Environment probed: **RTX 5070 Ti Laptop, 12 GB, Blackwell (sm_120), driver 591.97**; Python 3.12.10; but PyTorch is `2.2.0+cpu` — **CPU-only and too old for Blackwell** (needs torch ≥2.7 / CUDA 12.8+). Blocking prerequisite before any GPU training.
- KataGo recon: the full self-play loop needs the **C++ engine** (`cpp/katago selfplay`+`gatekeeper`) built (CUDA 12.8 for Blackwell — heavy on Windows) plus bash loop scripts; the PyTorch model is editable at `python/katago/train/model_pytorch.py` / `modelconfigs.py`, but inputs are KataGo's hardcoded V7 features (changing the minimal 4-plane input needs C++ too).
- **Next (awaiting human):** pick a track — (A) build KataGo's native self-play loop vs (B) a small custom PyTorch self-play sandbox using KataGo as reference/opponent — then fix PyTorch and run a minimal first experiment. Recommended B for testing SAKIGo's architectural ideas. Setup details in repo memory.

## 2026-06-19 — Correction: self-play, not distillation

- Human clarified the training paradigm: SAKIGo trains by **self-play RL** (search-into-prior on its own games, z-grounded), **not** offline distillation from an external teacher — my vibego capture had conflated the two. Sharpened D10 (explicit self-play scope) and re-scoped the vibego external-evidence block in [Issues.md](Issues.md): its two "trap" findings (searched targets, Gumbel-loses-to-PUCT) are **distillation-specific and do not transfer**. Notably the Gumbel result is a *distilled-net* profile (strong prior / noisy value); self-play is the *balanced* regime where Gumbel's published low-visit gains hold, so that evidence now **supports** rather than tempers the D16 lean.
- Reaffirmed D9 (history stays out — the *wrong prior* / a last-move-response crutch + added complexity; and self-play keeps full history at generation time, so vibego's history-less-relabel failure cannot occur) and D11 (subtree harvest is well-suited to self-play, per author).
- Kept the paradigm-agnostic vibego carryovers (pattern-embed symmetry feature D13, games-not-val-loss measurement, capacity-vs-data sizing). Edits confined to `AI/`.

## 2026-06-19 — vibego distillation study (external evidence)

- Reviewed [sanderland/vibego](https://github.com/sanderland/vibego) (KaTrain author's agent-driven small-net Go lab; a June-2026 single-GPU distillation study with paired-game Elo). Captured the decision-bearing findings in [Issues.md](Issues.md) under a dated external-evidence block.
- Two land hardest: (1) **searched/amplified distillation targets lost to the raw 1-visit soft prior**, and history-less searched targets were *decisively* harmful — challenges the subtree-harvest thesis (D11) and intersects the dropped-history decision (D9); (2) **Gumbel root search lost to PUCT at low visits for distilled nets** (strong-prior / noisy-value profile) — tempers D16's Gumbel lean. Also logged a cheap D4 pattern-embed alternative to the ×8 G-CNN (D13) and vibego's games-not-val-loss measurement methodology (fills the eval-methodology gap).
- All small-scale / low-visit / single-teacher caveats noted. Edits confined to `AI/`.

## 2026-06-19 — Reconciled stale "board size → area" notes

- The `area` correction had already landed in the source ([NonBoardInput.md](../Design/Input/NonBoardInput.md) now divides komi/captures by board **area**) in commit `1fb5cf8`, but that same commit left the AI side still flagging it as to-be-fixed. Cleared the now-false references: removed the [Nit] in [Issues.md](Issues.md) (back to none open) and dropped D4's "source says board size, correct it" note in [Decisions.md](Decisions.md) — the D4 body already reads "area" (matches score head D7).
- Re-cross-checked every `Design/` note against the AI notes; no other discrepancies. Edits confined to `AI/`.

## 2026-06-18 — AI/ vs Design/ cross-check + equivariance correction

- Re-read all of `AI/` and cross-checked every `Design/` note. Found one real discrepancy: [Trunk.md](../Design/Architecture/Trunk.md) specifies **escnn-based equivariant** trunk convs, but the AI notes had recorded the trunk as *non*-equivariant (D14 omitted escnn; the "stem equivariance is partial" issue assumed a non-equivariant trunk).
- Human clarified: the **whole model is equivariant**, and the register tokens **read+write** spatial tokens but are to be **designed to preserve equivariance**. Rewrote D13 (whole-net equivariance, not a stem-only prior), D14 (escnn convs + equivariant read/write register attention), the Context doc map, and the Issues item (now "equivariant register-token attention" — plain QKV isn't equivariant). Kept the ×8-cost-vs-augmentation question.
- Normalization: human clarified NonBoardInput's "board size" meant board **area** (matches score head D7). Updated D4; logged a Nit to fix the Design wording when that file is next edited.
- Edits confined to `AI/`; no `Design/` files changed.

## 2026-06-17 — Policy-vs-budget clarification

- Human clarified the intended split: budget stays smooth to guide search allocation; policy can be comparatively sharp because it is liberated from being the search prior and acts as an auxiliary reward/ranking target. Recorded as D17 and narrowed the open issue to exact target construction/loss weighting.

## 2026-06-17 — Workspace evaluation + web context

- Reviewed current workspace from `AI/` outward. Confirmed project remains design/spec only: concise `Design/` notes, no implementation code, and no SAKIGo-specific search spec yet.
- Web-grounded comparison against AlphaGo Zero, KataGo, Gumbel MuZero, regularized MCTS, FiLM, and G-CNN sources. Main evaluation: direction is coherent and research-aligned; critical blockers remain search/target definitions, policy-vs-budget semantics, pass normalization, and training anchors for subtree harvest.
- Corrected stale AI context: `Design/Search/` is not empty anymore; it contains the Gumbel policy-improvement PDF reference, but no local spec.

## 2026-06-17 — Bayesian VOI search direction

- Author leaning toward Bayesian value-of-information leaf selection (expand the leaf most likely to change the root policy). Recorded as D16 (exploring) and detailed the cruxes in [Issues.md](Issues.md): decision-focus vs training-target conflict, argmax-vs-distribution ambiguity, max-node posterior cost.
- Flagged the synthesis: budget head as *learned* VOI + Gumbel-style root selection — makes the intractable analytic VOI a regression target and sharpens the budget head's definition (D12/D5).
- No `Design/` files changed.

## 2026-06-17 — Search landscape research

- At author's request, surveyed alternatives to vanilla PUCT (web-grounded: Wikipedia MCTS, Grill et al. arXiv:2007.12509, KataGo arXiv:1902.10565). Recorded a candidate shortlist under the Search gap in [Issues.md](Issues.md) for the pending PUCT redesign — Gumbel MuZero (low-visit policy-improvement), regularized-policy-opt (Grill), MENTS family, KataGo tricks (FPU-reduction, forced playouts, LCB), plus the Predictor-vs-Polynomial PUCT naming collision.
- Also explained "vanilla" = stock/unmodified reference version (default-ice-cream-flavor metaphor). No `Design/` files changed.

## 2026-06-17 — Reviewed Architecture

- Three `Architecture` notes filled. Captured D13 (G-CNN stem), D14 (nested-resblock trunk + register-token attention), D15 (1×1 spatial / attention-pool global heads); updated doc map. Two open items in [Issues.md](Issues.md): partial stem equivariance (end-to-end-or-nothing; ×8 cost vs augmentation) and register/FiLM redundancy.
- No `Design/` files changed.

## 2026-06-17 — BestMoveVisit + conciseness note

- New `Train` note BestMoveVisit → recorded as D12 (cutoff on best-move visits, refining D11's flat cutoff). Flagged the crux in [Issues.md](Issues.md): the "more compute to uncertain positions" benefit needs dynamic search-termination; as a fixed-budget filter it inverts. Updated doc map.
- Recorded the human's standing **be-concise** preference in [Context.md](Context.md).
- No `Design/` files changed.

## 2026-06-17 — Train review: author resolutions

- Author addressed the subtree-harvest risks. Recorded in D11 / [Issues.md](Issues.md): a min-visit **cutoff** (≈ KataGo playout-cap randomization) closes the visit-decay risk; **policy/budget-entropy gating** closes both selection-bias and intra-tree-correlation (excludes peaked/forced near-duplicate runs).
- Pushed back on one: harvest is structurally search-bootstrapped — it can't supply the game-outcome z anchor. Reframed as still-open (value target ← z vs f^m; keep a grounded:bootstrapped ratio) rather than closed, and flagged both gates depend on the pending PUCT redesign.
- No `Design/` files changed.

## 2026-06-17 — Reviewed Train design

- Evaluated the two new `Train` notes. Captured the framing as D10 (search-based student–teacher distillation) and D11 (subtree harvest) in [Decisions.md](Decisions.md); updated the doc map in [Context.md](Context.md) and the training gap in [Issues.md](Issues.md).
- Logged 6 review items under "Training design review" in [Issues.md](Issues.md) — mostly bounding subtree harvest (min-visit threshold, selection bias, correlation, outcome-target masking, per-ply sign flip) plus the soft-vs-hard student target.
- Author fixed the `Auxiliary.md` spelling; updated my references and dropped the resolved nit. No `Design/` files changed.

## 2026-06-17 — Corrected Input review after author feedback

- Author clarifications resolved 4 of the review items: (1) Komi+CapturedStones are two *retained* scalars — I misread "I can leave it out" (the rare overflow) as dropping the prisoner count; Territory scoring therefore has what it needs. (2) A `Boundary` plane is now in BoardInput (4 planes), also enabling non-rectangular boards — closes the mask gap. (3) The history / last-move prior is **intentionally** dropped ("Markovian enough" with the legality plane) → recorded as D9.
- Updated D1 (4 planes), D4 (captures retained), added D9; trimmed [Issues.md](Issues.md) and [Context.md](Context.md) to match. No `Design/` files changed.

## 2026-06-17 — Reviewed Input/Output design

- Evaluated the `Input` / `Output` notes against the intended KataGo-style design. Logged 8 open questions / risks in [Issues.md](Issues.md) under "Input/output design review": territory ⟂ captured-stones, captured-stones normalization, missing on-board mask, no history plane, policy-vs-budget targets, pass/policy normalization, quantile crossing, auxiliary target non-stationarity.
- Read-only review — no `Design/` files changed.

## 2026-06-17 — Rename charter

- Renamed `GuideTemplate.md` → [Guide.md](Guide.md) and trimmed the stale "drop this in the project root" advice (it stays in `AI/`). Updated the reference in [Context.md](Context.md).

## 2026-06-17 — Infrastructure bootstrap

- Built the AI-collaboration infrastructure inside `AI/`, per the charter ([Guide.md](Guide.md)): created [Context.md](Context.md), [Decisions.md](Decisions.md), [Issues.md](Issues.md), and this log.
- Seeded them from the existing design notes under `../Design/Input/` and `../Design/Output/`: captured 8 design decisions (D1–D8) with rationale, mapped every design doc, and logged 3 spec gaps.
- Touched nothing outside `AI/`.
- **Next:** flesh out the empty specs — Architecture (Stem / Trunk / Heads) first, then Search and Train. See [Issues.md](Issues.md).
