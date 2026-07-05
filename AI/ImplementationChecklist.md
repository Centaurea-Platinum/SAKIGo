# Implementation Checklist

Use this before calling an implementation done, and also when reviewing a
finished change. Keep it lightweight: the point is to catch real mismatches,
not to perform ceremony.

## 1. Correctness

- [ ] Does the change match the user's actual request and implied intent?
- [ ] Does it align with the relevant design docs, README contracts, and AI
      decisions?
- [ ] If design, notes, and code disagree, is the discrepancy recorded instead
      of silently papered over?
- [ ] Does the implementation make basic sense when traced end to end with a
      tiny example?
- [ ] Are all important shapes, units, index orders, and normalizations correct?
      Check row-major board points, pass as the final action logit, rule feature
      length, komi/captures divided by board area, and score perspective.
- [ ] Are side-to-move conversions correct? Check My/Opponent planes, WDL,
      score sign, ownership, captured-stone difference, and per-ply perspective
      flips.
- [ ] Are edge cases handled deliberately: empty board, full board, pass moves,
      ko/superko, suicide rules, non-square or small board assumptions, zero or
      tiny datasets, missing optional fields, and terminal positions?
- [ ] Are nuanced details doing what they claim? For example, if a dataloader is
      meant to randomize samples, verify it is not feeding long correlated
      sequential runs.

## 2. Source Of Truth

- [ ] Does rules/legality/history logic use the engine or an existing canonical
      generator instead of reimplementing Go rules ad hoc?
- [ ] Does model construction use the spec loader and existing model APIs rather
      than duplicating dimensions in a second place?
- [ ] Does training data projection use the shared adapters/schema conventions?
- [ ] Are generated artifacts, downloaded engines, model weights, caches, and
      large data files kept out of source unless explicitly intended?
- [ ] Are module boundaries still clean: engine for rules/encoding, model for
      forward/inference, training for data/loss/loops, AI notes for memory?

## 3. Robustness

- [ ] Is the structure sturdy enough for the job without being overbuilt?
- [ ] Is it modular along existing ownership boundaries, and simple where a
      local helper is enough?
- [ ] Does it prefer standard library or established project infrastructure over
      fragile custom machinery?
- [ ] Do errors fail loudly with actionable messages rather than producing
      silent wrong labels, wrong masks, or stale metrics?
- [ ] Are CPU/CUDA, dtype, device movement, and optional acceleration paths
      handled explicitly?
- [ ] Is randomness seeded or documented where reproducibility matters?
- [ ] Are resume/checkpoint semantics honest? If resume is valid but not
      bit-exact, say so.
- [ ] Are memory and disk use bounded for large Phase 1 data, sharded data, and
      streaming buffers?

## 4. ML And Training Specifics

- [ ] Are train/validation splits honest for the claim being made? Watch for
      transposed openings or duplicated positions leaking across splits.
- [ ] Are targets normalized and masked consistently: WDL distribution, score,
      ownership, policy, budget, legal mask, and pass entry?
- [ ] Is the policy-vs-budget split preserved? Budget should remain the smooth
      search-prior target; policy may be sharper only when that is the intended
      auxiliary signal.
- [ ] Is illegal-move masking applied only at the intended layer? Current design
      says raw model logits during training, inference/search mask later.
- [ ] Are D4 transforms applied to every board-shaped field and not to global
      fields or pass logits?
- [ ] Does validation measure the live model, not stale cached weights or a
      mutated batch buffer?
- [ ] Are claims about strength backed by paired-game evals when feasible, not
      only by validation loss?

## 5. Performance

- [ ] Is the obvious hot path acceptable for the intended scale?
- [ ] Are large files streamed/sharded rather than eagerly expanded into Python
      objects?
- [ ] Are GPU paths free of avoidable host synchronizations in inner loops?
- [ ] Are batch-size choices checked against real peak memory, not only whether
      an out-of-memory exception appears?
- [ ] Does any caching strategy have a correct invalidation rule?
- [ ] Are slow fallbacks acceptable, documented, or guarded behind explicit
      options?

## 6. Tests

- [ ] Is there focused coverage for the changed behavior?
- [ ] Are existing tests updated when public behavior changes?
- [ ] For engine/rules changes: include direct legality, capture, ko/superko,
      pass, and encoding cases.
- [ ] For model changes: include shape, spec loading, D4 equivariance/invariance,
      device/dtype, and inference-cache behavior.
- [ ] For training/data changes: include tiny-dataset smokes, schema parsing,
      sharding/glob paths, split behavior, resume expectations, and target sums.
- [ ] For regressions: add a test that would have failed before the fix.
- [ ] If a full test is too expensive, run the smallest useful smoke and record
      what remains unverified.

## 7. Documentation And Memory

- [ ] Update README/design docs only when the human asked for that scope or the
      file is the native source of truth for the changed behavior.
- [ ] Update AI notes when ground truth changes: `Context.md` for stable facts,
      `Decisions.md` for settled rationale, `Issues.md` for remaining risk, and
      `Log.md` for the session record.
- [ ] Keep open risks open until code or a design decision actually resolves
      them.
- [ ] Do not use `Design/` as a scratchpad for inferred findings; put those in
      `AI/`.

## 8. Final Pass

- [ ] Re-read the files touched.
- [ ] Inspect the diff for accidental churn, unrelated edits, debug prints, dead
      code, stale TODOs, and formatting drift.
- [ ] Run the relevant lightweight checks and capture the result.
- [ ] Note any skipped checks and why.
- [ ] Make sure the final answer tells the user what changed, what was verified,
      and what remains, without burying them in process.
