# Durable Context - SAKIGo

Long-term memory for AI collaborators. Read this before acting, then update it when the repo's ground truth changes.

## What SAKIGo is

SAKI currently stands for **SymmetryAwareKatago-DistillationImplementation**. The current project scope is direct distillation from published small-board KataGo books: derive book targets, train the student network, and evaluate distilled checkpoints.

## Current phase

The repo is no longer design-only. It now has:

- `Design/`: concise source-of-truth design notes.
- `Engine/`: a Rust rules and encoding crate with a pyo3 Python binding (`sakigo_engine` wheel).
- `equivariant_attention/`: reusable torch-only finite-group regular-representation attention library extracted from the SAKIGo model code; SAKIGo consumes it through compatibility wrappers.
- `sakigo/`: the rebuilt Python stack (2026-07-06 rebuild, see [RebuildPlan.md](RebuildPlan.md)) — model, data, train, generate, eval, engine subpackages plus [CONTRACTS.md](../sakigo/CONTRACTS.md).
- `Distillation/`: local KataGo teacher assets; downloaded engines/models are artifacts, not source.
- `Training/data`, `Training/runs`: pre-rebuild artifacts retained for reference; current model checkpoints are not required to remain compatible during the move-quick phase.

Search, self-play training, feature/time auxiliary heads, and high-visit Phase 2 are explicitly not considered in the current scope. Core inputs, model architecture, output heads, and engine support remain active because they serve distillation.

## Implementation map

- [Engine/README.md](../Engine/README.md) - Rust rules/encoding engine. Owns board topology, captures, suicide, ko/superko, history-aware cache hashes, feature encoding, combined model inputs/legal mask, and area scoring. Python binding: build with `uvx maturin build --manifest-path Engine/Cargo.toml --release --features python --out dist`, then install the wheel.
- [equivariant_attention/](../equivariant_attention) - reusable finite-group equivariant attention package. Provides `FiniteGroupSpec`, trivial/Cn/D4 square-grid presets, regular-representation linear/norm/MLP layers, invariant pooling, spatial self-attention, and spatial/register cross-attention. Tensor shapes: spatial `[B,C,G,H,W]`, registers `[B,R,C,G]`.
- [sakigo/CONTRACTS.md](../sakigo/CONTRACTS.md) - versioned contracts: strict record schemas v1/v2, prepared format v3, checkpoint schema 7, model/loss semantics, and run-dir layout.
- `sakigo/model/` - D4-only `SakiGoNet` with no forward-time caches (torch.compile-clean). A scalar pointwise `6 -> 16 -> 128` board stem runs before one D4 lift. The fixed trunk is one register-to-board broadcast, `L` plain two-attention board blocks, and one board-to-register gather. The packaged sweep fixes `m = 128`, register widths, heads, and `L*n = 1024` while comparing `n/L = 32/32`, `64/16`, and `128/8`; parameter count is reported rather than controlled.
- `sakigo/data/` - strict record validation, canonical model-input split, content-digested JSONL(.zst) to immutable-generation mmap shards with atomic manifest switch, a stateful size-grouped training sampler with natural ruleset mixing and duplicate-free full batches, board-size/ruleset validation cohorts, and checkpointable D4 augmentation.
- `sakigo/train/` - validated config, fail-closed `reduce-overhead` torch.compile by default, bf16/fused AdamW, mandatory eager/compiled optimizer-step parity preflight in suites, aggregate plus per-cohort validation curves, and atomic safe checkpoints with data/trajectory binding plus batch/control-state-exact `num_workers=0` resume. Backend kernels are not claimed to produce bit-identical weights.
- `sakigo/generate/` - active no-inference book distillation from six 7x7/8x8/9x9 KataGo books, plus the legacy Phase 1 teacher generator. Multi-book indexing and emission run per book concurrently; validation is balanced across the six board-size/ruleset cohorts, training remains globally sampled, and frozen samples/shard manifests fail closed on provenance mismatches.
- `sakigo/eval/` - safe checkpoint loading, paired color-reversed policy matches, canonical book openings for active small-board sizes, engine-owned area scoring, honest draw/void outcomes, paired uncertainty, and JSONL/SGF dumps.
- [pyproject.toml](../pyproject.toml) - Python 3.12, CUDA PyTorch `2.11.0+cu128`, tensorboard, tqdm, triton-windows (torch.compile works on this machine).

- [sakigo/model/specs/ModelSpecs.json](../sakigo/model/specs/ModelSpecs.json) - packaged `narrow-deep`, default `balanced`, and `wide-shallow` D4 depth/width sweep.

## Boundary

AI collaborators may write freely inside `AI/`. Do not modify `Design/`, `sakigo/`, `Engine/`, `README.md`, or other non-AI files unless the human explicitly asks. The human did explicitly ask on 2026-07-03 to maintain the worktree and reconcile AI notes with current docs/code, and on 2026-07-06 to implement the rebuild plan (including the P6 cutover).

Keep the AI notes current without waiting to be prompted: when code, design docs, READMEs, or project direction change, update `AI/Context.md`, `AI/Decisions.md`, `AI/Issues.md`, and `AI/Log.md` in the same working session.

**Authorized experiments area (2026-06-19):** the human granted permission to create and work inside `../VibeKatago/`, an experiments sandbox for training small-net ideas. It is outside this repo's git tracking.

**Sibling testbed (2026-07-03):** `D:\stuff\Documents\SquareAccumulationK-Isolation` is the owner's exact-solved boardgame-AI playground. It supplied the register/attention reference that SAKIGo's current model package now adapts.

## Collaboration preferences

- Be concise. Lead with the answer, avoid long recaps unless asked.
- Prefer updating existing notes over creating new process docs.
- If a design note and implementation disagree, name the discrepancy in `AI/Issues.md` instead of silently smoothing it over.

## Design-doc map

**Input - how a position is encoded**
- [BoardInput.md](../Design/Input/BoardInput.md) - six board planes: MyStones, OpponentStones, EmptyPositions, BoundaryCorner, BoundaryEdge, NonTrivialIllegal.
- [NonBoardInput.md](../Design/Input/NonBoardInput.md) - rule one-hots plus normalized komi and capture-difference scalars. The rule MLP directly initializes two registers, which broadcast once before the board blocks.
- [Markov.md](../Design/Input/Markov.md) - the neural input is lossy, but the engine keeps history/hash state and exposes legality through encoding.

**Architecture - the network**
- [Stem.md](../Design/Architecture/Stem.md) - scalar pointwise board stem followed by one late D4 lift.
- [EquivariantAttention.md](../Design/Architecture/EquivariantAttention.md) - left-regular feature convention, canonical-frame positional embedding, equivariant QKV/channel mixing, and shared pointwise nonlinearities.
- [Trunk.md](../Design/Architecture/Trunk.md) - fixed D4 program: one initial broadcast, `L` plain two-attention board blocks, and one final gather.
- [Heads.md](../Design/Architecture/Heads.md) - spatial heads use 1x1 convs over board features; global heads use MLPs over register tokens.

**Output - the heads**
- [SpatialGlobalDistinction.md](../Design/Output/SpatialGlobalDistinction.md) - spatial heads are board-shaped; global heads are spatial-independent register outputs.
- [Winrate.md](../Design/Output/Winrate.md) - length-4 win/draw/loss/no-result output.
- [Score.md](../Design/Output/Score.md) - scalar score divided by board area first; percentile heads later.
- [Ownership.md](../Design/Output/Ownership.md) - not currently considered; the active dataset and model have no ownership target/head.
- [Policy+Budget.md](../Design/Output/Policy+Budget.md) - current book mapping: budget is the normalized concrete `AVisits` distribution and policy is uniform over tied rounded-optimum concrete moves; pass is the final logit.
- [FeatureAuxiliary.md](../Design/Output/FeatureAuxiliary.md) and [TimeAuxiliary.md](../Design/Output/TimeAuxiliary.md) - **not currently considered**.

**Pipeline**
- [Design/Engine/Scope.md](../Design/Engine/Scope.md) - engine performs the lossy projection and handles history through hashing.
- `Design/Search/` and `Design/Train/` - **not currently considered**; retained only as future reference.
- [Design/Distillation/Target.md](../Design/Distillation/Target.md) - active direct-book contract: mixed 7x7/8x8/9x9 targets derived from concrete book rows with no teacher inference or continuations.

## Glossary

- **Budget head** - current distillation target for the normalized concrete-move `AVisits` distribution in a book row; no search-time role is assumed in the current scope.
- **Bottleneck width (`n`)** - working channel count inside each board block; the packaged sweep trades it against block count at fixed `m = 128` and fixed `L*n = 1024`, while parameter counts vary.
- **Regular representation** - D4 feature layout with one component for each of the 8 board symmetries.
- **Register tokens** - global tokens shaped as regular features in the main model; they are equivariant, not merely invariant.
