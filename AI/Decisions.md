# Decisions & Rationale — SAKIGo

The *why* behind design choices, so later sessions don't relitigate settled ground or lose the reasoning. Newest first. Entries are living until built, then they freeze.

## D17 — Policy can be sharp; budget should stay smooth (2026-06-17, clarified)

The policy head is liberated from being the search prior, so it may train toward a comparatively sharp "best move" / reward-preference target. The budget head remains the smooth action distribution that guides search allocation. **Why:** search needs graded probabilities so it does not prematurely starve plausible moves; the policy head is an auxiliary reward/ranking signal, so sharpness is less dangerous there. **Caveat:** sharp policy targets still affect the shared trunk and can amplify teacher noise, so keep loss weight / temperature / top-k shaping explicit. Source: human clarification, this session.

## D16 — Search direction: Bayesian value-of-information leaf selection (2026-06-17, exploring)

Leaning toward a Bayesian searcher: expand the leaf with the highest probability of changing the root policy, rather than a PUCT bonus. **Why:** it's the rational-metareasoning ideal (value of information / value of computation; Russell–Wefald, Tolpin–Shimony VOI-MCTS, Tesauro et al. Bayesian MCTS) that UCT/PUCT only approximate — mathematically the cleanest statement of "spend the next sim where it matters." **Tensions to resolve before committing** (see [Issues.md](Issues.md)): (1) VOI is decision-focused → concentrates sims on the top-2 moves → degenerate *training* target (the conflict KataGo patches with forced playouts) — clashes with D5/D10. (2) "Change the policy" is ambiguous: root *argmax* (→ binary VOI, top-2) vs root *visit distribution* (→ a KL magnitude, not a probability). (3) Bayesian cleanness needs posteriors propagated through max nodes (not closed-form) + myopia — approximations re-enter. **Pragmatic realization:** Gumbel/Sequential-Halving for the decision-focused behavior without posteriors, and reframe the **budget head as a learned VOI** (regress realized policy-change-per-visit) — turns the intractable analytic VOI into a training target and gives the budget head (D12) a crisp definition. Source: author, this session.

## D13 — Stem: small group-equivariant CNN, regular representation (2026-06-17; equivariance scope corrected 2026-06-18)

A small G-CNN (regular rep) as the stem. **Why:** bakes in Go's D4 board symmetry structurally instead of relying only on data augmentation. **Whole-net equivariance is the goal** (corrected 2026-06-18): stem, escnn trunk convs (D14), *and* the register-token attention are all meant to be equivariant — not a stem-only front-end prior. **Open:** regular rep multiplies channels by |G| (×8 for D4) — costly, hence "small"; confirm that beats plain symmetry augmentation. The hard part shifts to the trunk — keeping the register-token QKV attention (which reads *and writes* spatial tokens, D14) equivariance-preserving, since plain QKV is not equivariant (see [Issues.md](Issues.md)). Color-swap / komi / Boundary on non-square boards aren't spatial symmetries and don't apply. Source: [Stem.md](../Design/Architecture/Stem.md); human clarification 2026-06-18.

## D14 — Trunk: KataGo nested residual blocks (escnn-equivariant) + register-token attention (2026-06-17; equivariance noted 2026-06-18)

Conv nested-bottleneck residual blocks (KataGo-style) built on **escnn-based equivariant convolutions**; no spatial self-attention yet; **register tokens** with QKV attention between registers and each spatial token. Registers **read and write** the spatial tokens (a bidirectional global-context exchange, not a read-only summary), and the attention is to be **designed equivariance-preserving** so the net stays equivariant end-to-end (D13). **Why:** registers are a cheap global-information pathway — O(N·R) vs O(N²) for full spatial attention — a principled alternative to KataGo's global pooling, and they double as the global-head summary (→ D15). FiLM injection sites live here. **Open:** an equivariant register attention with read+write injection is non-trivial (see [Issues.md](Issues.md)). Source: [Trunk.md](../Design/Architecture/Trunk.md); human clarification 2026-06-18.

## D15 — Heads: 1×1 conv (spatial) + attention pooling→MLP (global) (2026-06-17)

Spatial heads = 1×1 convs; global heads = attention pooling then MLP. **Why:** matches the spatial/global split ([SpatialGlobalDistinction.md](../Design/Output/SpatialGlobalDistinction.md)); attention pooling pairs naturally with the trunk's register tokens (they can *be* the pooled global state). Source: [Heads.md](../Design/Architecture/Heads.md).

## D12 — Harvest cutoff keyed on best-move visits, not total visits (2026-06-17, proposed)

Gate harvest on the most-visited child's count (best-move visits ≥ K), not node-total N or a flat playout cap. **Why:** wide/uncertain positions split visits, so they must accumulate more total search before the top move reaches K — compute flows to noisy positions, and a node is harvested only once its principal variation is trustworthy (a flat-N gate can pass a wide node whose best move is still under-visited). **Open:** is this a *dynamic search-termination* rule (search until best move hits K → genuinely allocates more compute) or a *post-hoc filter* on fixed-budget trees (then it *rejects* high-entropy nodes — the opposite effect)? Near-ties need a total-visit ceiling. Source: [BestMoveVisit.md](../Design/Train/BestMoveVisit.md).

## D10 — Training is search-based student–teacher distillation (2026-06-17)

The net (student) learns to approximate the result and statistics of net+search (teacher). **Why:** search is the policy-improvement operator; distilling it back into the prior is the AlphaZero/KataGo self-improvement loop. "Statistics" = visit distribution → budget/policy prior; "result" = root value → winrate/score. Source: [SearchBasedStudentTeacher.md](../Design/Train/SearchBasedStudentTeacher.md).

## D11 — Subtree harvest: train interior tree nodes, not just the root (2026-06-17)

Root search with n visits yields f^n(x_t) as the root's target; each interior node x_t^p already accumulated m visits, so train f(x_t^p) → f^m(x_t^p) too instead of discarding the subtree. **Why:** the subtree NN evals are already paid for, so this adds training targets at zero extra search cost. **Two gates make it sound:** (1) a **min-visit cutoff** — only harvest nodes searched enough that f^m is a real refinement; directly analogous to KataGo's playout-cap randomization, where only full-playout moves supply policy targets. (2) **policy/budget-entropy gating** — harvest only high-entropy nodes, which excludes peaked/forced near-duplicate runs (kills intra-tree correlation) and de-skews away from confidently-greedy lines (kills most selection bias). **Still open:** harvest is search-bootstrapped, never outcome-grounded (no z at interior nodes), and both gates key off the *current* PUCT, which is itself under redesign — see [Issues.md](Issues.md). Source: [SubTreeHarvest.md](../Design/Train/SubTreeHarvest.md).

## D9 — History / last-move prior intentionally dropped (2026-06-17)

No history planes. Go is only fully Markovian given complete history, but the `NonTrivialIllegal` plane already carries the ko / superko *legality* that history would otherwise be needed for — "Markovian enough" to work with. History's remaining value, a last-move local-response prior, is **intentionally** surrendered as a sample-efficiency trade, not an oversight. Source: human, this session.

> **Seed (2026-06-17):** D1–D8 below were **captured from the existing `../Design/` notes**, not decided by the AI. They record the human's choices and stated rationale so the reasoning survives across sessions. (They share one date, so they read D1→D8; future entries go on top, newest first.)

## D1 — Minimal board input: 4 planes

Boundary, MyStones, OpponentStones, NonTrivialIllegal. **Why:** "leave it to the model to figure it out" — avoid hand-engineered features. The Boundary plane also enables non-rectangular boards. Source: [BoardInput.md](../Design/Input/BoardInput.md).

## D2 — Rules conditioned via FiLM, not input planes

Rule settings are one-hot encoded, concatenated, and passed through two MLPs per injection site to produce FiLM bias+scale in the trunk. **Why:** feeding every rule as a board plane is overhead and rarely sampled; one-hot keeps correlated rules from being double-counted. Source: [NonBoardInput.md](../Design/Input/NonBoardInput.md).

## D3 — Only a subset of rules, one-hot per correlated group

Scoring {Area, Area+AncientChinese, Territory, TerritoryWithSekiScore}, Ko {SimpleKo, PositionalSuperKo}, Suicide {Yes, No}. **Why:** Go's rules don't map cleanly onto a hypercube — some rules exclude others, so a full cross-product would waste capacity and rarely be sampled. Source: [NonBoardInput.md](../Design/Input/NonBoardInput.md).

## D4 — Komi & captured stones as normalized scalars

Two *retained* scalars in [−1, 1], divided by board **area** (handicap folded into komi implicitly). **Why:** compact and board-size-agnostic; the prisoner count is needed for Territory scoring, so it stays. CapturedStones can exceed board area, but that overflow is deliberately left unhandled (rare, and harmless as a NN input). **Note (2026-06-18):** the source note says "board size," but the human clarified the intended denominator is board **area** — matching the score head (D7); the Design wording should be corrected to "area" (see [Issues.md](Issues.md)). Source: [NonBoardInput.md](../Design/Input/NonBoardInput.md); human clarification 2026-06-18.

## D5 — Policy and budget are separate targets

Budget head = search prior; policy head = an extra reward signal predicting the best move (variants PolicyWinrate, PolicyScore). Passing is a separate global PassProb. **Why:** decouples "where to search" from "what is best," so each can be supervised independently. Source: [Policy+Budget.md](../Design/Output/Policy+Budget.md).

## D6 — Win / Loss / Draw as a length-3 vector

**Why:** draws are reachable (SimpleKo long-repeat = draw, per [NonBoardInput.md](../Design/Input/NonBoardInput.md)), so winrate cannot collapse to a single scalar. Source: [Winrate.md](../Design/Output/Winrate.md).

## D7 — Score: scalar first, percentile heads later

Scalar score normalized by board area to start; percentile-based score heads later. **Why:** percentiles generalize across board sizes better than categorical heads, are more stable than MDNs, and capture multimodal score distributions when finely grained. Source: [Score.md](../Design/Output/Score.md).

## D8 — Auxiliary heads as temporal predictions

An auxiliary head predicts a main head's value n plies ahead: g(x_t) = f(x_{t+n}); candidates are ownership and score. **Why:** cheap extra supervision / a lookahead signal. Source: [Auxiliary.md](../Design/Output/Auxiliary.md).
