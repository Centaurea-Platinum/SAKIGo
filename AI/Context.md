# Durable Context - SAKIGo

Long-term memory for AI collaborators. Read this before acting, then update it when the repo's ground truth changes.

## What SAKIGo is

SAKI currently stands for **SymmetryAwareKatago-DistillationImplementation**. SAKIGo is the Go AI project built around that direction: KataGo-style neural network plus search, with selectable scoring / ko / suicide rules at inference instead of a single baked-in ruleset.

## Current phase

The repo is no longer design-only. It now has:

- `Design/`: concise source-of-truth design notes and model specs.
- `Engine/`: a Rust rules and encoding crate with a pyo3 Python binding (`sakigo_engine` wheel).
- `sakigo/`: the rebuilt Python stack (2026-07-06 rebuild, see [RebuildPlan.md](RebuildPlan.md)) — model, data, train, generate, eval, engine subpackages plus [CONTRACTS.md](../sakigo/CONTRACTS.md).
- `Distillation/`: local KataGo teacher assets; downloaded engines/models are artifacts, not source.
- `Training/data`, `Training/runs`: pre-rebuild datasets and checkpoints (legacy code deleted at the P6 cutover; checkpoints still load via `sakigo.model.remap_legacy_scalar_state_dict`).

Search, final scoring/adjudication beyond Tromp-Taylor eval, self-play training, and exact KataGo teacher projection are still not implemented.

## Implementation map

- [Engine/README.md](../Engine/README.md) - Rust rules/encoding engine. Owns board topology, captures, suicide, simple ko, positional superko, pass moves, history hashes, capture accounting, and feature encoding. Python binding: build with a repo-local `CARGO_HOME` (see [Engine/.cargo/config.toml](../Engine/.cargo/config.toml)), `uvx maturin build --manifest-path Engine/Cargo.toml --release --out dist`, then `uv pip install dist/*.whl`.
- [sakigo/CONTRACTS.md](../sakigo/CONTRACTS.md) - frozen cross-module contracts: record schema v1, board planes, rule features, model forward contract, loss semantics, hash split, checkpoint payload, run-dir layout.
- `sakigo/model/` - unified `SakiGoNet` with `group_size ∈ {1, 8}` (scalar control = 1); no forward-time caches (torch.compile-clean); spec JSONs packaged in `sakigo/model/specs/` with a Design-sync test.
- `sakigo/data/` - record validation, blake2b position split, JSONL(.zst) → mmap tensor shards (`prepare.py`), map-style `PreparedDataset` + `RulesetBalancedBatchSampler` + standard DataLoader, D4 augmentation.
- `sakigo/train/` - `python -m sakigo.train`: torch.compile (default on), bf16 autocast, fused AdamW, SequentialLR warmup-cosine, TensorBoard + metrics.csv mirror (Viewer-compatible), tqdm, atomic `weights_only=True` checkpoints with RNG capture, TOML/CLI config. `python -m sakigo.train.benchmark` = WDDM-aware batch-size sweep.
- `sakigo/generate/` - `python -m sakigo.generate`: Phase 1 KataGo teacher generation on the Rust engine (client/plan/records/writer/run modules), zstd shards + status.json.
- `sakigo/eval/` - `python -m sakigo.eval`: paired color-reversed policy matches, Tromp-Taylor adjudication, Elo + Wilson CI, JSONL/SGF dumps.
- [pyproject.toml](../pyproject.toml) - Python 3.12, CUDA PyTorch `2.11.0+cu128`, tensorboard, tqdm, triton-windows (torch.compile works on this machine).

- [Design/ModelSpecs/ModelSpecs.md](../Design/ModelSpecs/ModelSpecs.md) - JSON-compatible model specs; `sakigo/model/specs.py` consumes packaged copies kept in sync by test. Defines `model1`, `model2`, and `model3` with reusable stem/head shape files.

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
- [NonBoardInput.md](../Design/Input/NonBoardInput.md) - rule one-hots plus normalized komi and capture-difference scalars. Default path seeds register tokens; FiLM is optional future plumbing.
- [Markov.md](../Design/Input/Markov.md) - the neural input is lossy, but the engine keeps history/hash state and exposes legality through encoding.

**Architecture - the network**
- [Stem.md](../Design/Architecture/Stem.md) - small D4 group-equivariant stem using regular representations.
- [EquivariantAttention.md](../Design/Architecture/EquivariantAttention.md) - left-regular feature convention, canonical-frame positional embedding, equivariant QKV/channel mixing, and shared pointwise nonlinearities.
- [Trunk.md](../Design/Architecture/Trunk.md) - D4-equivariant spatial attention in nested residual blocks plus register-token attention.
- [Heads.md](../Design/Architecture/Heads.md) - spatial heads use 1x1 convs over board features; global heads use MLPs over register tokens.

**Output - the heads**
- [SpatialGlobalDistinction.md](../Design/Output/SpatialGlobalDistinction.md) - spatial heads are board-shaped; global heads are spatial-independent register outputs.
- [Winrate.md](../Design/Output/Winrate.md) - length-4 win/draw/loss/no-result output.
- [Score.md](../Design/Output/Score.md) - scalar score divided by board area first; percentile heads later.
- [Ownership.md](../Design/Output/Ownership.md) - end-of-game ownership.
- [Policy+Budget.md](../Design/Output/Policy+Budget.md) - policy and budget are separate targets; pass is the final logit in the shared `N*N + 1` action vector. Current design says train without masking illegal moves and mask them only as an inference precaution.
- [Auxiliary.md](../Design/Output/Auxiliary.md) - auxiliary heads predict future main-head values.

**Pipeline**
- [Design/Engine/Scope.md](../Design/Engine/Scope.md) - engine performs the lossy projection and handles history through hashing.
- `Design/Search/` - currently only contains the Gumbel MuZero / policy-improvement-by-planning PDF reference; no SAKIGo search spec yet.
- [SearchBasedStudentTeacher.md](../Design/Train/SearchBasedStudentTeacher.md) - self-play framing: student distills net+search results/statistics.
- [SubTreeHarvest.md](../Design/Train/SubTreeHarvest.md) - train interior search-tree nodes in addition to the root.
- [BestMoveVisit.md](../Design/Train/BestMoveVisit.md) - harvest cutoff keyed on best-move visits.
- [BranchedGames.md](../Design/Train/BranchedGames.md) - possible branching in high-policy-entropy positions.
- [Design/Distillation/Target.md](../Design/Distillation/Target.md) - phase sketch: phase 1 distills a 1-visit teacher net, mapping KataGo raw policy to budget and top-1 to policy; phase 2 fine-tunes on high-visit data. Reconcile this external-teacher bootstrap with the older self-play-only D10 framing before treating training as settled.

## Glossary

- **FiLM** - Feature-wise Linear Modulation, an optional future per-channel bias/scale add-on for rule conditioning.
- **Budget head** - predicts per-move search allocation / search prior, distinct from the policy head.
- **Regular representation** - D4 feature layout with one component for each of the 8 board symmetries.
- **Register tokens** - global tokens shaped as regular features in the main model; they are equivariant, not merely invariant.
