# Durable Context — SAKIGo

Long-term memory for AI collaborators: facts that stay true across sessions. Read this before acting; update it when the ground truth changes.

## What SAKIGo is

A new Go AI ("New GoAI" — see [../README.md](../README.md)). KataGo-style direction: a neural net (stem → trunk → heads) paired with search, and *multi-rule aware* — scoring / ko / suicide variants are selectable at inference rather than baked in.

## Current phase

**Design / specification only — no SAKIGo code yet** (running code lives outside the repo: the `../VibeKatago/` sandbox and the sibling testbed below). All substance lives as notes under `../Design/`. Input, Output, and Architecture are sketched; the search spec and most of the training loop are still open (see [Issues.md](Issues.md)).

## Sibling testbed (2026-07-03)

`D:\stuff\Documents\SquareAccumulationK-Isolation` — the owner's boardgame-AI playground (n×n fill game with exact-solved 3×3–5×5 game trees; a D4-equivariant attention model in PyTorch trained on exact labels; same laptop GPU). It doubles as SAKIGo's rehearsal ground, and its AI notes cross-reference SAKIGo. Already delivered: the equivariant read+write register attention (SAKIGo's hardest open architecture item — see [Issues.md](Issues.md)) and inference-lever measurements for a future self-play loop (CUDA-graph replay ≈10× on batch-1 latency; bf16 ≈2–3× compute-bound and halves memory). Natural venue for SAKIGo A/Bs that need exact ground truth: ×8 regular-rep vs augmentation, register-seeded rule conditioning (center-ban variant).

## Boundary

AI collaborators write freely inside `AI/`. Do **not** modify anything outside `AI/` (notably `../Design/`, `../README.md`) without an explicit request. Full charter: [Guide.md](Guide.md).

**Authorized experiments area (2026-06-19):** the human granted permission to create and work inside `../VibeKatago/` — an experiments sandbox holding a clone of KataGo, for training small-net experiments that test SAKIGo's ideas (self-play training, minimal input, subtree harvest, etc.). Write and run freely there; it is kept out of the SAKIGo design repo's git tracking. `Design/` and `README.md` stay read-only-without-asking.

## Collaboration preferences

- **Be concise.** The human optimizes for token cost and reading time — default to short answers, lead with the answer, cut preamble and recap. (Reinforces the charter's "Be concise.")

## Design-doc map

**Input — how a position is encoded**
- [BoardInput.md](../Design/Input/BoardInput.md) — 4 board planes: Boundary (also enables non-rectangular boards), MyStones, OpponentStones, NonTrivialIllegal (suicide / ko / superko). Deliberately minimal; no history plane (see D9 — prior intentionally dropped).
- [NonBoardInput.md](../Design/Input/NonBoardInput.md) — rule settings as one-hots → MLPs → FiLM (bias+scale) injected into the trunk; komi & captured-stones as normalized scalars.

**Architecture — the network**
- [Stem.md](../Design/Architecture/Stem.md) — small group-equivariant CNN, regular rep (D4 symmetry). (D13)
- [Trunk.md](../Design/Architecture/Trunk.md) — KataGo nested residual blocks on escnn **equivariant** convs + register-token QKV attention (no spatial self-attention yet); registers read+write spatial tokens and are designed equivariance-preserving (whole-net equivariant); FiLM sites here. (D14)
- [Heads.md](../Design/Architecture/Heads.md) — spatial = 1×1 conv, global = attention pooling → MLP. (D15)

**Output — the heads**
- [SpatialGlobalDistinction.md](../Design/Output/SpatialGlobalDistinction.md) — spatial heads = conv (board-shaped); global heads = pooled (scalars).
- [Winrate.md](../Design/Output/Winrate.md) — length-3 win / loss / draw.
- [Score.md](../Design/Output/Score.md) — scalar score ÷ board area now; percentile heads later.
- [Ownership.md](../Design/Output/Ownership.md) — end-of-game ownership (spatial).
- [Policy+Budget.md](../Design/Output/Policy+Budget.md) — budget = search prior; policy = reward signal (PolicyWinrate / PolicyScore); pass via separate global PassProb.
- [Auxiliary.md](../Design/Output/Auxiliary.md) — heads predicting a main head's future value, g(x_t) = f(x_{t+n}); e.g. ownership, score.

**Pipeline**
- `../Design/Search/` — contains the Gumbel MuZero / policy-improvement-by-planning PDF as a reference; SAKIGo's own search spec is still TBD.
- **Train** — [SearchBasedStudentTeacher.md](../Design/Train/SearchBasedStudentTeacher.md): **self-play RL** — net (student) distills its *own* net+search over self-play games (teacher) — result + statistics, outcome-grounded by z (D10; *not* offline external-teacher distillation). [SubTreeHarvest.md](../Design/Train/SubTreeHarvest.md): also train interior search-tree nodes f(x_t^p) → f^m(x_t^p), not just the root, to reuse subtrees. [BestMoveVisit.md](../Design/Train/BestMoveVisit.md): harvest cutoff keyed on best-move visits (vs flat playout cap), routing more compute to uncertain positions. (See D10, D11, D12.)

## Glossary (the non-obvious terms)

- **FiLM** — Feature-wise Linear Modulation: a per-channel bias+scale that conditions the trunk on the active rule settings.
- **Budget head** — predicts per-move search allocation; supplies the search prior (distinct from policy).
- **Percentile score head** — score expressed as predicted percentiles: generalizes across board sizes, encodes multimodal outcomes, and is steadier to train than an MDN.
