# Session Log — SAKIGo

Dated, newest first. What changed, what is next. One entry per working session.

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
