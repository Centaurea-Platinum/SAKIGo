# Open Issues & Risks — SAKIGo

Classified so later sessions know what is unsettled. Tags: **[Gap]** missing spec · **[Open]** undecided question · **[Risk]** could bite later · **[Nit]** cosmetic. Resolve an item by promoting it to [Decisions.md](Decisions.md) or deleting it.

## Gaps — spec not yet written

- **[Gap] Network architecture: shape sketched, sizes open.** Stem (G-CNN, D13), trunk (nested resblocks + register attention, D14), heads (D15) are now framed, but block count, channel width, register-token count, and FiLM site count/placement are unspecified.
- **[Open] Equivariant register-token attention.** Direction settled (human, 2026-06-18): the whole net is to be equivariant — stem G-CNN, escnn trunk convs, *and* the register-token attention (D13/D14), not a stem-only prior. The hard part is the register attention: registers **read and write** spatial tokens, and plain QKV is **not** equivariant, so it must be built as equivariant/steerable attention over the regular-rep fibers (e.g. invariant query–key scores + equivariant value mixing) — design TBD. Also still confirm the ×8 regular-rep channel cost beats plain symmetry augmentation. (D13, D14)
- **[Open] Registers vs FiLM redundancy.** Both inject global context into spatial tokens (registers = learned global scratchpad; FiLM = rule conditioning). Confirm they're complementary, or let registers carry rule info and drop one path. (D14, D2)
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
- **[Open] FiLM injection sites.** How many sites, and where in the trunk the MLPs inject bias+scale. (D2)

## Nits

- **[Nit] NonBoardInput "board size" → "area."** [NonBoardInput.md](../Design/Input/NonBoardInput.md) says komi/captures are "normalized via division board size," but the human clarified the intended denominator is board **area** (D4). Correct the Design wording when that file is next edited. (D4)

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

- **[Open] Policy-vs-Budget exact targets still need pinning.** Split is now clarified: budget should be smooth because it guides search; policy may be sharp because it is an auxiliary reward/ranking target, not the search prior. Still specify policy target construction (argmax, top-k, low-temperature softmax over teacher values, PolicyWinrate/PolicyScore), loss weight, and whether the policy head is ever consulted at inference. (D5, D17)
- **[Open] Pass + board policy must form one distribution.** PassProb as a separate global scalar still has to combine with the board softmax into a single normalized prior for search (e.g. joint softmax over board ∪ pass logits). (D5)
- **[Risk] Percentile head: quantile crossing.** Pinball loss needs a monotonicity constraint or sorted outputs so q10 ≤ … ≤ q90. (D7)
- **[Risk] Auxiliary head non-stationary target.** Regress g to the *realized* future outcome (or a target net), not the live f(x_{t+n}), to avoid chasing a moving target. (D8)
