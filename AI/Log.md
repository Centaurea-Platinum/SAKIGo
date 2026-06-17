# Session Log — SAKIGo

Dated, newest first. What changed, what is next. One entry per working session.

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
