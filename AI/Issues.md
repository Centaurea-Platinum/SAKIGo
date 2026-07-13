# Open Issues & Risks — SAKIGo

Classified so later sessions know what is unsettled. Tags: **[Gap]** missing spec · **[Open]** undecided question · **[Risk]** could bite later · **[Nit]** cosmetic. Resolve an item by promoting it to [Decisions.md](Decisions.md) or deleting it.

> **Current scope:** direct published-book distillation only. Search, self-play
> training, feature/time auxiliaries, percentile score heads, and high-visit
> Phase 2 are deferred. Related material below is historical backlog, not an
> open requirement for current work.

## Current implementation watchlist

- **[Intentional for now] Pre-current-schema checkpoints do not resume after the architecture optimizations.** Fused attention projections, the scalar-before-lift stem, and removal of ownership changed model keys and AdamW slots. Earlier checkpoints are rejected explicitly; inference-only migration is possible, but training-resume migration is intentionally out of scope unless the owner revisits it. (D32-D35)
- **[Risk] Data prep and orchestration need scale hardening.** The 2026-07-07 486-combination run exposed three future-run hazards: JSONL(.zst) -> tensor-shard prep is single-process/two-pass and slow at ~500k records; Windows `DataLoader(num_workers>0)` can fail under the current launcher, forcing `num_workers=0`; ad-hoc PowerShell orchestration folders got messy when scripts/logs/sweeps/train runs shared one root. Immediate code mitigation: batched `PreparedDataset.fetch_batch` / `__getitems__` reduces row-by-row memmap overhead, the batch-size benchmark now uses the same fast path, default metrics logging follows checkpoint cadence, and `python -m sakigo.train.suite` owns the future multi-spec layout (`data/`, `prepared/`, `generation/`, `train/<spec>/`, `logs/`, `sweeps/`, `scripts/`). Remaining scale work: parallelize JSONL(.zst) prep by shard or replace it with a one-pass indexed format.
- **[Resolved by scope] Training paradigm.** Current work is direct distillation from published small-board books. The older live-teacher and self-play/search framings are deferred and need no reconciliation unless the project scope changes.
- **[Deferred legacy gap] The one-visit KataGo teacher generator lacks true resume.** It is not used by the active direct-book dataset. Failure status, stalled-response timeouts, atomic partial shards, explicit overwrite, and stale-shard protection are implemented; resume of an interrupted quota schedule remains unfinished. Sources: [generate/](../sakigo/generate), [writer.py](../sakigo/generate/writer.py).
- **[Gap] Engine scoring/adjudication remains partial.** The Rust engine now owns Tromp-Taylor/Chinese area scoring in addition to legality, history, and encoding. Ancient Chinese group tax, territory scoring/dead-stone adjudication, end-of-game detection, search, neural inference, and target generation remain open. See [Engine/README.md](../Engine/README.md).
- **[Resolved for current scope] Illegal-move policy masking.** Training uses raw logits; paired checkpoint-policy evaluation applies the engine's legal mask in `sakigo/eval/selfplay.py`.

## Architecture validation

- **[Open] The attention-work-matched depth/width sweep needs training evidence.** D36 fixes D4, `m = 128`, register widths, the two-attention plain block, one initial broadcast, one final gather, and `L*n=1024`. The packaged comparison is `narrow-deep` (`n=32, L=32`), `balanced` (`n=64, L=16`), and `wide-shallow` (`n=128, L=8`). Run all three with equal samples seen and judge them by paired game strength and throughput/memory measurements, not validation loss alone. (D36)
- **[Risk] Matching `L*n` does not match all compute or memory.** Width-dependent projections scale differently, while sequential depth plus saved activations grow with `L`; the three models also have substantially different parameter counts. Record step time, peak memory, achieved batch size, and parameters beside quality metrics. (D36)
- **[Open] D4 equivariance is tested structurally but not yet justified empirically at scale.** `SakiGoNet` uses regular-representation board/register fibers, equivariant QKV/channel mixing, canonical-frame RoPE, and group-axis-collapsed heads. The active architecture deliberately has no scalar-control package, so the current question is whether the fixed D4 family trains well—not a parallel symmetry ablation. (D13, D14, D18, D31)

## Deferred backlog - not currently considered

- **[Deferred] Search design.** `../Design/Search/` contains reference material only. No search specification is required in the current distillation scope.

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
- **[Deferred] Self-play training loop.** The `Design/Train/` material is retained only as future reference.

### Other deferred questions

- **[Deferred] Auxiliary horizon.** Feature/time auxiliary heads are not currently considered. (D8)
- **[Deferred] Percentile-score granularity.** Percentile score heads are not currently considered. (D7)

## Nits

- *(none open)*

## Deferred training/search review (2026-06-17)

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

- **[Open] Soft budget versus hard policy duplication.** Budget learns the raw teacher policy and policy learns its derived top-1 move. Decide whether the hard head adds useful ranking pressure or merely double-weights the same action target; keep separate loss weights explicit. (D5, D17, D21)
- **[Deferred] Percentile head: quantile crossing.** Retained only as a future caution. (D7)
- **[Deferred] Auxiliary head non-stationary target.** Retained only as a future caution. (D8)

## External evidence — vibego distillation study (2026-06-19; historical framing)

> **Current-scope note (2026-07-12):** direct external-distillation findings in
> this section may be relevant again. Gumbel, PUCT, self-play, and other search
> implications are deferred. Older statements that assume self-play is the
> active paradigm are superseded by D30.

From [sanderland/vibego](https://github.com/sanderland/vibego) (Sander Land, KaTrain author): an agent-driven single-GPU study distilling tiny (0.8–4.2M-param) KataGo-style nets from the public `kata1-b18` teacher, judged by paired color-reversed games. Findings are small-scale / low-visit / single-teacher.

**Current interpretation:** vibego is indirect evidence only: SAKIGo now learns from published book statistics rather than live one-visit teacher queries. Its history-less relabel failure still reinforces the engine's history-aware legality projection, and its measurement warning remains useful: validation loss alone may not predict playing strength, so compare checkpoints with paired color-reversed policy matches. Search/Gumbel conclusions are outside the current scope.
