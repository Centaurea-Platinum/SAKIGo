# Decisions & Rationale — SAKIGo

The *why* behind design choices, so later sessions don't relitigate settled ground or lose the reasoning. Newest first. Entries are living until built, then they freeze.

## D23 - Large JSONL training uses a bounded streaming buffer (2026-07-03, implemented)

`Training.train` keeps eager loading as the default for tiny runs, but large Phase 1 JSONL training should pass `--stream-buffer-mb N`. Streaming mode scans metadata once, deterministically splits positions into train/validation by hash, keeps a rolling buffer of raw JSONL lines capped by the requested MiB budget, and decodes only sampled records into tensors. **Why:** the 2^18 Phase 1 file is about 5.7 GB on disk and would expand substantially if loaded as Python objects. A full-file smoke run passed with `--stream-buffer-mb 64`, yielding 236,411 train and 25,733 validation records from 262,144 rows. **Caveat:** the buffer budget is raw-line bytes, not exact Python heap usage; binary/sharded data may still be better for long production runs. Source: [train.py](../Training/train.py), [data.py](../Training/data.py), local smoke 2026-07-03.

## D22 - Local KataGo Phase 1 batch-inference setting (2026-07-03, empirical)

For local Phase 1 KataGo teacher generation on the RTX 5070 Ti Laptop GPU with the bundled TensorRT KataGo v1.16.5 and `kata1-zhizi-b40c768nbt-s11272M-d5935M`, use `maxVisits: 1`, `analysisPVLen: 1`, `includePolicy: true`, `includeOwnership: true`, `includeNoResultValue: true`, `numSearchThreadsPerAnalysisThread=1`, `nnMaxBatchSize=20`, and `numAnalysisThreads=40`. At the generator level, think "GPU batches of 20 NN evaluations" but keep about 40 positions/games in flight so KataGo actually fills those batches. **Why:** a steady-state benchmark on 4096-8192 unique four-move 19x19 positions found batch 20 / 40 analysis threads at about 226-227 samples/s, narrowly ahead of batch 16 / 32 threads and clearly ahead of 8, 12, 24, 32, 64, and 128. Bigger TensorRT batches filled well but were slower for this model/hardware. Source: local benchmark 2026-07-03.

## D21 - Phase 1 KataGo labels map budget to raw policy and policy to top-1 (2026-07-03)

For the external KataGo-teacher bootstrap path, [Target.md](../Design/Distillation/Target.md) now says Phase 1 uses the teacher net without search/one visit, trains the budget head on KataGo's raw policy output, and trains the policy head on the top-1 move derived from that output. Other current heads map directly: WDL from raw win/no-result values, score from raw lead, and ownership from the root ownership map, with perspective conversion as needed. **Why:** this preserves D17's split: the budget head stays smooth as the search prior, while the policy head becomes a sharp best-move reward/ranking signal. **Caveat:** this is an external-teacher bootstrap path and still needs reconciliation with D10's older self-play-only framing. Source: [Target.md](../Design/Distillation/Target.md); KataGo analysis output inspection 2026-07-03.

## D20 - Engine owns legality, history, and neural encoding (2026-07-03, implemented)

The Rust engine is the deterministic source of truth for board topology, captures, suicide, simple ko, positional superko, pass moves, capture accounting, history hashes, and model input encoding. The neural net still receives a lossy no-history projection, but `Engine/src/game.rs` keeps the state needed to compute legality, and `Engine/src/encoder.rs` emits the six board planes plus ten rule features from the side-to-move perspective. **Why:** search, training data generation, and inference masking should share one rule implementation instead of duplicating ko/capture/history logic in Python or in the model. Source: [Engine/README.md](../Engine/README.md), [Scope.md](../Design/Engine/Scope.md), [encoder.rs](../Engine/src/encoder.rs).

## D19 - Model specs are data-backed and include scalar controls (2026-07-03, implemented)

`Design/ModelSpecs.md` is a JSON-compatible spec consumed by `Model/sakigo_model/specs.py`. It defines the main D4-equivariant `model1` plus two non-equivariant scalar controls: `model1_control_params` for approximate trainable-parameter matching and `model1_control_compute` for active scalar feature-width matching. **Why:** the project needs to test whether the D4 regular model wins because of symmetry structure, parameter count, or raw dense width. Source: [ModelSpecs.md](../Design/ModelSpecs.md), [Model/README.md](../Model/README.md), [scalar_model.py](../Model/sakigo_model/scalar_model.py).

## D18 - Model v1 uses register-seeded D4 attention with no FiLM branch (2026-07-03, implemented)

The implemented `SakiGoModel` lifts `[B,6,N,N]` board tensors to regular features, initializes equivariant register tokens from a learned seed plus `rule_mlp(rules)`, runs regular spatial GQA attention with canonical-frame RoPE, gathers board state into registers, broadcasts registers back to board features on configured blocks, and collapses the D4 axis only in heads. There is no dormant FiLM module in code; FiLM remains a future add-on if register seeding underdelivers. **Why:** this turns the SquareAccumulation reference and current design docs into a concrete, tested baseline while keeping rule conditioning simple. Source: [model.py](../Model/sakigo_model/model.py), [layers.py](../Model/sakigo_model/layers.py), [EquivariantAttention.md](../Design/Architecture/EquivariantAttention.md), [Model/README.md](../Model/README.md).

## D17 — Policy can be sharp; budget should stay smooth (2026-06-17, clarified)

The policy head is liberated from being the search prior, so it may train toward a comparatively sharp "best move" / reward-preference target. The budget head remains the smooth action distribution that guides search allocation. **Why:** search needs graded probabilities so it does not prematurely starve plausible moves; the policy head is an auxiliary reward/ranking signal, so sharpness is less dangerous there. **Caveat:** sharp policy targets still affect the shared trunk and can amplify teacher noise, so keep loss weight / temperature / top-k shaping explicit. Source: human clarification, this session.

## D16 — Search direction: Bayesian value-of-information leaf selection (2026-06-17, exploring)

Leaning toward a Bayesian searcher: expand the leaf with the highest probability of changing the root policy, rather than a PUCT bonus. **Why:** it's the rational-metareasoning ideal (value of information / value of computation; Russell–Wefald, Tolpin–Shimony VOI-MCTS, Tesauro et al. Bayesian MCTS) that UCT/PUCT only approximate — mathematically the cleanest statement of "spend the next sim where it matters." **Tensions to resolve before committing** (see [Issues.md](Issues.md)): (1) VOI is decision-focused → concentrates sims on the top-2 moves → degenerate *training* target (the conflict KataGo patches with forced playouts) — clashes with D5/D10. (2) "Change the policy" is ambiguous: root *argmax* (→ binary VOI, top-2) vs root *visit distribution* (→ a KL magnitude, not a probability). (3) Bayesian cleanness needs posteriors propagated through max nodes (not closed-form) + myopia — approximations re-enter. **Pragmatic realization:** Gumbel/Sequential-Halving for the decision-focused behavior without posteriors, and reframe the **budget head as a learned VOI** (regress realized policy-change-per-visit) — turns the intractable analytic VOI into a training target and gives the budget head (D12) a crisp definition. Source: author, this session.

## D13 — Stem: small group-equivariant CNN, regular representation (2026-06-17; equivariance scope corrected 2026-06-18)

A small G-CNN (regular rep) as the stem. **Why:** bakes in Go's D4 board symmetry structurally instead of relying only on data augmentation. **Whole-net equivariance is the goal** (corrected 2026-06-18, clarified 2026-07-03): stem, D4-equivariant spatial attention trunk (D14), and register-token attention are all meant to be equivariant — not a stem-only front-end prior. **Open:** regular rep multiplies channels by |G| (×8 for D4) — costly, hence "small"; confirm that beats plain symmetry augmentation. The hard part is keeping spatial attention and register-token QKV read/write equivariance-preserving, since arbitrary QKV/channel mixing is not equivariant (see [Issues.md](Issues.md)). Color-swap / komi / Boundary on non-square boards aren't spatial symmetries and don't apply. Source: [Stem.md](../Design/Architecture/Stem.md); human clarification 2026-06-18 and 2026-07-03.

## D14 — Trunk: equivariant spatial attention + register-token attention (2026-06-17; clarified 2026-07-03)

The trunk uses D4-equivariant spatial attention nested residual blocks plus register-token QKV attention. Registers **read and write** spatial tokens (a bidirectional global-context exchange, not a read-only summary), and both spatial attention and register attention are designed to preserve whole-net equivariance (D13). **Why:** spatial attention provides board reasoning directly, while registers provide a compact global pathway and become the native input for global heads (D15). Rule conditioning seeds register tokens by default; FiLM can be added later as an extra multiplicative conditioning path if needed. **Update (2026-07-03):** a working reference implementation now exists in the owner's playground repo (`D:\stuff\Documents\SquareAccumulationK-Isolation`) — regular-rep fibers, group-axis-batched QKV, canonical-frame RoPE, equivariance-tested, no escnn; see [Issues.md](Issues.md). Source: [Trunk.md](../Design/Architecture/Trunk.md); human clarification 2026-06-18 and 2026-07-03.

## D15 — Heads: 1×1 conv (spatial) + register MLP (global) (2026-06-17; clarified 2026-07-03)

Spatial heads = 1×1 convs; global heads = MLPs applied to register tokens. **Why:** matches the spatial/global split ([SpatialGlobalDistinction.md](../Design/Output/SpatialGlobalDistinction.md)); the register stream is already the global state, so a separate attention-pooling mechanism is unnecessary unless later evidence asks for it. Source: [Heads.md](../Design/Architecture/Heads.md).

## D12 — Harvest cutoff keyed on best-move visits, not total visits (2026-06-17, proposed)

Gate harvest on the most-visited child's count (best-move visits ≥ K), not node-total N or a flat playout cap. **Why:** wide/uncertain positions split visits, so they must accumulate more total search before the top move reaches K — compute flows to noisy positions, and a node is harvested only once its principal variation is trustworthy (a flat-N gate can pass a wide node whose best move is still under-visited). **Open:** is this a *dynamic search-termination* rule (search until best move hits K → genuinely allocates more compute) or a *post-hoc filter* on fixed-budget trees (then it *rejects* high-entropy nodes — the opposite effect)? Near-ties need a total-visit ceiling. Source: [BestMoveVisit.md](../Design/Train/BestMoveVisit.md).

## D10 — Training is search-based self-play, distilling search into the prior (2026-06-17; scope clarified 2026-06-19)

The net (student) learns to approximate the result and statistics of net+search (teacher). **Why:** search is the policy-improvement operator; distilling it back into the prior is the AlphaZero/KataGo self-improvement loop. "Statistics" = visit distribution → budget/policy prior; "result" = root value → winrate/score. **Clarification (2026-06-19, human):** this is the *self-play RL loop* — the "teacher" is the net's **own search over its self-play games**, with targets outcome-grounded by the game result z. It is **not** offline distillation from a fixed external teacher (the vibego setting); "distillation" here means distilling *search into the prior*, not learning from a separate stronger net. This re-scopes the vibego evidence in [Issues.md](Issues.md) — most of its distillation-specific cautions do not transfer. **Current tension (2026-07-03):** [Target.md](../Design/Distillation/Target.md), `Distillation/` assets, and `Model/sakigo_model/adapters.py` now sketch a separate KataGo-teacher distillation/bootstrap path. Treat that as unresolved against this older self-play-only framing until the training plan is reconciled. Source: [SearchBasedStudentTeacher.md](../Design/Train/SearchBasedStudentTeacher.md), [Target.md](../Design/Distillation/Target.md).

## D11 — Subtree harvest: train interior tree nodes, not just the root (2026-06-17)

Root search with n visits yields f^n(x_t) as the root's target; each interior node x_t^p already accumulated m visits, so train f(x_t^p) → f^m(x_t^p) too instead of discarding the subtree. **Why:** the subtree NN evals are already paid for, so this adds training targets at zero extra search cost. **Two gates make it sound:** (1) a **min-visit cutoff** — only harvest nodes searched enough that f^m is a real refinement; directly analogous to KataGo's playout-cap randomization, where only full-playout moves supply policy targets. (2) **policy/budget-entropy gating** — harvest only high-entropy nodes, which excludes peaked/forced near-duplicate runs (kills intra-tree correlation) and de-skews away from confidently-greedy lines (kills most selection bias). **Still open:** harvest is search-bootstrapped, never outcome-grounded (no z at interior nodes), and both gates key off the *current* PUCT, which is itself under redesign — see [Issues.md](Issues.md). **Reaffirmed for self-play (2026-06-19, human):** author considers subtree harvest well-suited to self-play training — the interior searched values *are* the policy-improvement signal, and the self-play outcome z anchoring the played line partially mitigates the "no z at interior nodes" concern (a watch-item, not a blocker). Source: [SubTreeHarvest.md](../Design/Train/SubTreeHarvest.md).

## D9 — History / last-move prior intentionally dropped (2026-06-17)

No history planes. Go is only fully Markovian given complete history, but the `NonTrivialIllegal` plane already carries the ko / superko *legality* that history would otherwise be needed for — "Markovian enough" to work with. History's remaining value, a last-move local-response prior, is **intentionally** surrendered as a sample-efficiency trade, not an oversight. **Reaffirmed (2026-06-19, human):** history is left out because it is the **wrong prior** — a last-move-response crutch that biases the net instead of helping it read the position on its merits — *and* it adds input/engineering complexity; a deliberate inductive-bias choice, not merely a sample-efficiency trade. This is independent of vibego's "history-less relabel" failure (see [Issues.md](Issues.md)): that was an *offline* artifact of relabeling archive positions with no move record; under self-play (D10) the full history is always known at generation time, so legality (`NonTrivialIllegal`) and targets are computed correctly even though history is never a net input. Source: human, this session.

> **Seed (2026-06-17):** D1–D8 below were **captured from the existing `../Design/` notes**, not decided by the AI. They record the human's choices and stated rationale so the reasoning survives across sessions. (They share one date, so they read D1→D8; future entries go on top, newest first.)

## D1 — Minimal board input: 6 planes

MyStones, OpponentStones, EmptyPositions, BoundaryCorner, BoundaryEdge, NonTrivialIllegal. **Why:** "leave it to the model to figure it out" — avoid hand-engineered features while making occupancy, board geometry, and non-trivial legality explicit. Boundary planes also enable non-rectangular boards. Source: [BoardInput.md](../Design/Input/BoardInput.md); human clarification 2026-07-03.

## D2 — Rules condition registers by default; FiLM is an add-on

Rule settings are one-hot encoded and concatenated with scalar komi/capture-difference inputs. The default conditioning path feeds this non-board vector through an MLP to initialize the register tokens; FiLM bias+scale injection remains an optional add-on if register seeding is not enough. **Why:** feeding every rule as a board plane is overhead and rarely sampled; one-hot keeps correlated rules from being double-counted, and registers are the native global state read by global heads. Source: [NonBoardInput.md](../Design/Input/NonBoardInput.md); human clarification 2026-07-03.

## D3 — Only a subset of rules, one-hot per correlated group

Scoring {Area, Area+AncientChinese, Territory, TerritoryWithSekiScore}, Ko {SimpleKo, PositionalSuperKo}, Suicide {Yes, No}. **Why:** Go's rules don't map cleanly onto a hypercube — some rules exclude others, so a full cross-product would waste capacity and rarely be sampled. Source: [NonBoardInput.md](../Design/Input/NonBoardInput.md).

## D4 — Komi & captured stones as normalized scalars

Two *retained* scalars in [−1, 1], divided by board **area** (matching the score head, D7; handicap folded into komi implicitly). CapturedStones means `#opponent stones I captured - #my stones opponent captured`, also normalized by board area. **Why:** compact and board-size-agnostic; the prisoner count is needed for Territory scoring, so it stays. Source: [NonBoardInput.md](../Design/Input/NonBoardInput.md); human clarification 2026-06-18 and 2026-07-03.

## D5 — Policy and budget are separate targets

Budget head = search prior; policy head = an extra reward signal predicting the best move (variants PolicyWinrate, PolicyScore). Pass is represented as one additional logit beside the n^2 board-move logits; board moves plus pass enter the same softmax. **Why:** decouples "where to search" from "what is best," while keeping pass normalized with the legal action distribution instead of as an isolated scalar. Loss construction is deferred to the training phase. Source: [Policy+Budget.md](../Design/Output/Policy+Budget.md); human clarification 2026-07-03.

## D6 — Win / Loss / Draw as a length-3 vector

**Why:** draws are reachable (SimpleKo long-repeat = draw, per [NonBoardInput.md](../Design/Input/NonBoardInput.md)), so winrate cannot collapse to a single scalar. Source: [Winrate.md](../Design/Output/Winrate.md).

## D7 — Score: scalar first, percentile heads later

Scalar score normalized by board area to start; percentile-based score heads later. **Why:** percentiles generalize across board sizes better than categorical heads, are more stable than MDNs, and capture multimodal score distributions when finely grained. Source: [Score.md](../Design/Output/Score.md).

## D8 — Auxiliary heads as temporal predictions

An auxiliary head predicts a main head's value n plies ahead: g(x_t) = f(x_{t+n}); candidates are ownership and score. **Why:** cheap extra supervision / a lookahead signal. Source: [Auxiliary.md](../Design/Output/Auxiliary.md).
