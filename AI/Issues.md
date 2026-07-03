# Open Issues & Risks — SAKIGo

Classified so later sessions know what is unsettled. Tags: **[Gap]** missing spec · **[Open]** undecided question · **[Risk]** could bite later · **[Nit]** cosmetic. Resolve an item by promoting it to [Decisions.md](Decisions.md) or deleting it.

## Current implementation watchlist

- **[Open] Training paradigm needs reconciliation.** D10 says SAKIGo's main paradigm is self-play search-into-prior, not offline external-teacher distillation. The current workspace also has [Design/Distillation/Target.md](../Design/Distillation/Target.md), KataGo assets under `Distillation/`, and teacher/student projection adapters in [adapters.py](../Model/sakigo_model/adapters.py). Decide whether KataGo distillation is only a bootstrap phase, a separate experiment, or a replacement for the earlier self-play plan.
- **[Gap] Exact KataGo teacher projection is missing.** `KataGoInputProjection` deliberately requires an exact encoder callable or precomputed native tensors; no exact KataGo feature encoder is implemented in this repo yet. This blocks honest teacher/student distillation from canonical `GameStateBatch` alone.
- **[Gap] KataGo analysis JSON target importer is missing.** [Design/Distillation/Target.md](../Design/Distillation/Target.md) now records the Phase 1 teacher-output contract: request `includePolicy` and `includeOwnership`, consume `policy`, `ownership`, and `rootInfo.raw*` fields, handle row-major top-left board order plus pass-last policy, ignore illegal `-1` entries, and convert the local `reportAnalysisWinratesAs = BLACK` perspective into SAKIGo's current-player perspective.
- **[Gap] Engine does not yet score games or adjudicate endings.** The Rust engine implements legality, captures, ko/superko, pass, history, and encoding, but not final scoring, end-of-game detection, search, neural inference, or target generation. See [Engine/README.md](../Engine/README.md).
- **[Open] Illegal-move policy masking location.** [Policy+Budget.md](../Design/Output/Policy+Budget.md) now says train without masking illegal moves and mask only at inference as a precaution. The model emits raw logits and the engine can compute legality, but the inference/search layer that applies this mask is not implemented yet.

## Gaps — spec not yet written

- **[Open] Model sizing is now concrete but not validated.** `model1` and two scalar controls are specified in [ModelSpecs.md](../Design/ModelSpecs.md) and implemented in `Model/`. The open question is no longer "what are the first sizes?" but whether these sizes train well and whether the D4 regular model beats parameter-matched or compute-width scalar controls.
- **[Open] Equivariant attention is implemented, but the A/B is still open.** `SakiGoModel` now adapts the SquareAccumulation reference: regular-rep board/register fibers, equivariant QKV/channel mixing, canonical-frame RoPE, gather, broadcast, and group-axis-collapsed heads. Tests cover equivariance and shape behavior. Remaining: confirm the regular-rep cost beats augmentation/scalar controls in actual Go or teacher-distillation experiments. (D13, D14, D18, D19)
- **[Open] FiLM remains reserve/add-on only.** Register-seeded rule conditioning is implemented; no FiLM module exists in the current model code. Keep the open question limited to whether future data shows a need for multiplicative rule gating, and if so where it enters.
- **[Open] Register read/write schedule and parameter cost.** `model1` gathers every block and broadcasts only in block 5. This encodes the parameter-cost lesson from the playground, but the schedule is still empirical and should be revisited after real training measurements.
- **[Gap] Search undefined.** `../Design/Search/` currently only contains a Gumbel MuZero / policy-improvement-by-planning PDF reference — SAKIGo's own algorithm (MCTS?) and how the budget head's prior feeds it are unspecified. Author has flagged a **PUCT redesign**; candidate directions below.

  *Search landscape (reference, 2026-06-17), most relevant first:*
  - **Gumbel MuZero/AlphaZero** (Danihelka et al., ICLR 2022) — Gumbel-top-k + Sequential Halving at root, completed-Q interior selection; provable policy improvement at *low* visit counts. Hits exactly where vanilla PUCT is weak — bears on subtree harvest's low-m nodes (D11) and the best-move-visit cutoff (D12).
  - **MCTS as Regularized Policy Optimization** (Grill et al., ICML 2020, arXiv:2007.12509) — PUCT/UCT ≈ approx. solution to a regularized policy-opt; act on the *exact* solution, not visit counts. Relevant to the policy-vs-budget target split (D5).
  - **MENTS / TENTS / RENTS** (Xiao et al., NeurIPS 2019) — entropy-regularized softmax-backup selection; faster convergence than UCT.
  - **KataGo practical tricks** (Wu, arXiv:1902.10565): FPU-reduction, forced-playouts + policy-target pruning, LCB final-move selection. Cheap, battle-tested, stackable.
  - **Backup/selection knobs:** Sequential Halving/SHOT (root, simple-regret), Power-UCT (mean↔max backup).
  - **Naming caveat:** "PUCT" is overloaded — Predictor-UCT (AlphaGo, Rosin 2011) vs Polynomial-UCT (Auger et al. 2013). Specify which in the redesign.
  - Note: the **budget head** is already non-vanilla — learned per-position visit allocation, adjacent to what Gumbel/Sequential-Halving do by hand at the root.

  *Bayesian VOI direction (D16, author-favored, 2026-06-17) — cruxes to resolve:*
  - **[Open] Decision-focus vs training target.** VOI concentrates sims on the top-2 moves → degenerate soft policy target (KataGo counters with forced playouts). May need two visit accountings: one to decide, one to train on. (D16, D5, D10)
  - **[Open] "Change the policy" — argmax or distribution?** Root argmax → binary VOI (top-2 collapse); root visit distribution → a KL magnitude (nonzero almost everywhere), not a probability. The KL-shift version likely fits the training goal better but isn't "probability of flipping." Pin which. (D16)
  - **[Open] Posterior propagation cost.** Bayesian VOI needs per-node posteriors pushed through negamax max-nodes (max of RVs isn't closed-form) + myopic single-step assumption → approximations re-enter, denting the "clean" appeal. (D16)
  - **[Open] Budget head as learned VOI.** Most promising synthesis: regress the budget head to realized policy-change-per-visit (amortized value-of-computation) instead of computing VOI analytically; pair with Gumbel-style root selection. Ties D16 to D12/D5. (D16)
- **[Gap] Training loop only sketched.** `../Design/Train/` now frames the principle (student–teacher distillation, D10) and one optimization (subtree harvest, D11), but per-head loss weighting, the self-play position distribution, target/replay-buffer mechanics, and the auxiliary-head targets are still unspecified.

## Open questions

- **[Open] Auxiliary horizon.** The n in g(x_t) = f(x_{t+n}) is unspecified — one horizon or several, and which heads get auxiliaries? (D8)
- **[Open] Percentile-score granularity.** Number and spacing of percentile bins is TBD. (D7)
- **[Open] Optional FiLM add-on sites.** If register seeding is not enough, decide how many FiLM sites to add and where their bias+scale enters the trunk. (D2)

## Nits

- *(none open)*

## Training design review (2026-06-17)

From evaluating the `Train` notes. (D10 = student–teacher, D11 = subtree harvest.)

**Resolved by design (2026-06-17):**
- *Target decays with visit count* → **best-move-visit cutoff** (D12), a refinement of the flat min-visit / playout-cap idea: gate on the top move's own visits so the harvested principal variation is always well-searched.
- *Selection bias* + *intra-tree correlation* → **policy/budget-entropy gating**: harvest only high-entropy nodes, excluding peaked/forced near-duplicate runs and de-skewing from greedy lines. (D11)

**Still open:**
- **[Open] Best-move-visit cutoff: search-control or filter?** The "more compute to high-entropy positions" benefit only holds if search *continues until* best-move-visits ≥ K (dynamic termination). As a post-hoc filter on fixed-budget trees it *rejects* high-entropy nodes (they split visits, never reach K) — contradicting the entropy gate. Pin which it is; if dynamic, add a total-visit ceiling so near-ties don't absorb unbounded search. (D12)
- **[Open] Harvest is search-bootstrapped, not outcome-grounded.** Interior nodes have no game outcome z, so they only ever supply match-the-search targets (value f^m, visit stats), never z-anchored ones. The author's "not an issue" holds *if* training is pure student–teacher distillation (D10) — but that's the premise to confirm: an outcome anchor is the one signal harvest structurally cannot give, and dropping it is the classic value-drift risk KataGo avoids by keeping z. (MuZero shows bootstrapping works, but with safeguards — target net, n-step returns, a capped harvest:root ratio.) Also: end-of-game Ownership/Score are defined only at terminals, so on harvested nodes either mask them or substitute the search's predicted values (same caveat). Pin the choice: value ← z, ← f^m, or a mix. (D11, D10, D6, D8)
- **[Open] Entropy gating skews the position mix.** Preferring high-entropy nodes over-represents chaotic midgame fights and won't surface *confidently-wrong* blind spots (those are low-entropy) — though catching those is root self-play noise's job, not harvest's. Minor; watch value calibration on quiet positions. (D11)
- **[Open] Both harvest gates assume the current PUCT.** The visit distribution and entropy profile the cutoff/gate key off are PUCT-determined; re-validate when the pending PUCT redesign lands. (D11)
- **[Open] Subtree harvest: per-ply perspective flip.** x_t^p alternates side-to-move; flip value/score sign, swap My/Opponent planes, and recompute the legality plane at each ply when materializing a node. Classic sign-bug source. (D11, D1)
- **[Open] Student–teacher: soft vs. hard statistics.** The student should match the teacher's *visit distribution* (soft), not its argmax — otherwise the policy target collapses to a near-one-hot. Same axis as the policy-vs-budget question below. (D10, D5)

## Input/output design review (2026-06-17)

Surfaced while evaluating the `Input` / `Output` notes; each is a question for the human, not a change. Four original items were **resolved on review** (2026-06-17) and removed: territory⟂captures and captures-range (CapturedStones is a retained scalar, not droppable), on-board mask (the new `Boundary` plane covers it), and the history plane (prior intentionally dropped → D9 in [Decisions.md](Decisions.md)).

- **[Deferred] Policy-vs-Budget loss construction.** Split is clarified: budget should be smooth because it guides search; policy may be sharp because it is an auxiliary reward/ranking target, not the search prior. Exact policy target construction, loss weighting, and inference use are training-phase decisions. (D5, D17)
- **[Risk] Percentile head: quantile crossing.** Pinball loss needs a monotonicity constraint or sorted outputs so q10 ≤ … ≤ q90. (D7)
- **[Risk] Auxiliary head non-stationary target.** Regress g to the *realized* future outcome (or a target net), not the live f(x_{t+n}), to avoid chasing a moving target. (D8)

## External evidence — vibego distillation study (2026-06-19; re-scoped for self-play 2026-06-19)

From [sanderland/vibego](https://github.com/sanderland/vibego) (Sander Land, KaTrain author): an agent-driven single-GPU study distilling tiny (0.8–4.2M-param) KataGo-style nets from the public `kata1-b18` teacher, judged by paired color-reversed games. Findings are small-scale / low-visit / single-teacher.

**Scope correction (2026-06-19, human):** vibego is an **offline distillation** study (relabel a fixed *external* teacher onto archive positions). SAKIGo trains by **self-play** (D10) — search-into-prior on its *own* games, outcome-grounded by z. So vibego's two headline "trap" findings are **distillation-specific and largely do not transfer**; the paradigm-agnostic findings (symmetry feature, measurement) do. Re-scoped:

- **"Searched targets are a trap" is an offline-relabel artifact — does *not* apply to self-play.** vibego: 32-visit relabel lost to 1-visit soft labels, and *history-less* relabel was decisively harmful (−38.7 sL) because search compounds the missing ko/capture context. **Both mechanisms are offline-relabel-specific.** Under self-play SAKIGo (a) generates its own games, so full history is present at target-generation time and the history-less failure cannot occur (see D9), and (b) the searched visit distribution *is* the standard, z-anchored policy-improvement target (AlphaZero/KataGo). So this does **not** challenge subtree harvest (D11). The one transferable grain: searched/greedy targets are *sharper* and can collapse toward argmax — already tracked as the "soft vs hard statistics" item (keep budget targets soft; D17 sharpens only the auxiliary policy head). (D11, D10, D9, D17)
- **Gumbel-loses-to-PUCT was a *distilled-net* result — self-play is the regime where Gumbel works.** vibego's stated mechanism: distillation gives a strong policy prior beside a noisy value head, and Gumbel's value-trusting completed-Q is wrong for that profile — "published Gumbel low-visit wins come from RL-loop nets where policy and value are balanced." SAKIGo *is* a self-play RL loop → the balanced regime, so this **supports rather than tempers** the Gumbel-style lean in D16. Residual caveat: D17's deliberately sharper policy + harvested (non-z) interior values could partially re-create the imbalance, so still A/B it at the target visit count. (D16, D10, D17)
- **[Open] Cheap local-symmetry alternative to the ×8 G-CNN.** ~+200 Elo for ~0 FLOPs at the small tier from a 3×3 **D4-canonical pattern-embed table** (each cell's 3×3 {empty/own/opp} neighborhood mapped through its dihedral-canonical form into a learned embedding, added to trunk input) — local symmetry as a cheap input feature instead of a full regular-rep trunk. A concrete data point for the "×8 regular-rep cost vs plain augmentation" question. Paradigm-agnostic — applies to self-play. (D13)
- **[Gap] The measurement methodology SAKIGo still lacks.** Per-game score sd ≈ 30–40 pts, and **val loss anti-correlated with strength** repeatedly (capacity and data-diversity gains were invisible-or-inverted in held-out loss) — judging by val loss "concludes the opposite of the truth." Load-bearing method: paired color-reversed openings + neutral-judge scoreLead + Elo±CI, sequential early stopping, and harness validation (deterministic batch-1 search trace, Q-equivalence CI assertion). Their day-eating **perspective bug** (Black-perspective judge values sign-flipping half the search tree) is precisely SAKIGo's flagged per-ply sign-flip risk. Paradigm-agnostic — applies to self-play. (training-loop gap, D11)
- **vibego chose distillation; SAKIGo deliberately chose self-play.** So vibego's distillation-recipe specifics (raw soft prior > searched labels, ensemble teachers > deeper search) are out of scope. Net-sizing carries over though: when a small net plateaus, add parameters before data (capacity-bound ≤~1.5M vs data-bound ~2.6M params). (D10, D13)
