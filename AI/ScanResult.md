# Codebase Scan Result — 2026-07-06

Full verification scan against (and beyond) [ImplementationChecklist.md](ImplementationChecklist.md).
Method: test suites first, then three parallel module audits (Engine, Model,
Training), then manual re-verification of every reported finding.

## Verdict

**Code is clean.** No production bugs found. One real gap fixed (engine test
coverage). Four of five audit-reported "bugs" were false positives, re-derived
and dismissed against the source.

## Evidence

- `uv run --frozen pytest`: **52 passed**.
- `cargo test` (Engine): **8 passed** before scan → **12 passed** after adding coverage.
- `get_errors`: no compile/lint diagnostics.
- Working tree at `abddff3` was clean except AI notes; the failed
  `run_phase1_suite` terminal exit (code 1) was a stale data path
  (`katago_phase1_20260703_225033` no longer exists), not a code bug. The
  `phase1_rules_fast_20260705_194303` orchestrator "aborted" state has empty
  error logs — manual abort, not a crash.

## Fixed

- **Engine test-coverage gap** (the one real finding): superko, pass, and
  multi-group capture were implemented but untested. Added 4 tests in
  [game.rs](../Engine/src/game.rs): positional-superko blocks board repetition
  (ko recapture), superko counts the *initial* position (1×1 allowed suicide),
  pass clears simple ko / toggles side / increments move number / allows
  recapture after passes, and one move capturing two groups. All pass.

## False positives dismissed (recorded so they aren't re-reported)

1. *"PositionalSuperKo hash lacks side-to-move"* — wrong by definition:
   **positional** superko keys on board position only; side-to-move belongs to
   *situational* superko, which the engine deliberately doesn't implement.
   Initial position is correctly seeded into `seen_positions`
   ([game.rs](../Engine/src/game.rs) `from_board`), now locked by a test.
2. *"`make_batch_dataloader` ignores `pin_memory`"* — the collate runs unpinned
   by design; `DataLoader(pin_memory=True)` pins the collated batch itself.
   Passing `pin_memory=True` into `collate_cpu` too would double-pin.
3. *"Offset bag exhaustion raises IndexError"* — `_read_indexed_record` refills
   the bag on `if not bag:` before every `pop()`; exhaustion just starts a new
   shuffled cycle.
4. *"RegularLinear1x1 cache needs requires_grad_(False) documentation"* —
   working as designed post-fix (2026-07-03): cache activates only for frozen
   params, `SakiGoInference` sets that up, trainable eval correctly recomputes.
   Regression test exists.

## Verified clean (spot-checked personally, beyond the subagent reports)

- **Perspective/sign conversions** in the Phase 1 generator (BLACK-perspective
  KataGo → current player: WDL-4, score, ownership) and
  [rulesets.py](../Training/rulesets.py) `rule_features` (komi negative for
  Black to move, capture diff = mine − opponent, both /area, clamped) — match
  the Rust encoder conventions.
- **[rulesets.py](../Training/rulesets.py)** (newest module): KataGo↔SAKIGo
  rule projection refuses inexact mappings (scoring/tax, ko, suicide) loudly;
  preset rulesets consistent; ruleset keys stable.
- **Ruleset-aware streaming sampler** ([data.py](../Training/data.py)):
  balanced key order, without-replacement bags per (split, board, ruleset),
  offset-index path for uncompressed JSONL, eval sampling (`advance=False`)
  doesn't consume training bags or advance the ingest stream.
- **Masked losses** ([losses.py](../Training/losses.py)): branchless,
  `clamp_min(1.0)` normalization, no illegal-move masking at train time (per
  design).
- **Streaming train loop** ([train.py](../Training/train.py)): eval collate
  unpinned+sync, `PinnedBatchKeeper` event fencing intact, val loader separate
  RNG, checkpoint/log cadence, RNG restore to CPU.
- **Suite runner** ([run_phase1_suite.py](../Training/run_phase1_suite.py)):
  WDDM peak-memory budget check, predictive skip, equal samples-seen budgets.
- **Model** (subagent, thorough): D4 group tables, regular-rep transform
  convention, RoPE canonical frames, head shapes (N²+1 pass-last, WDL-4),
  spatial-equivariant/global-invariant heads, spec loading for
  model1/2/3+controls, scalar control contract, inference freezing — all
  verified against Design/ModelSpecs.

## Still open (unchanged, tracked in [Issues.md](Issues.md))

Known watchlist items were confirmed still accurate, none newly resolved:
generator/engine legality divergence (single-stone suicide), move-sequence
split leak, non-exact streaming resume, engine lacks scoring/adjudication,
inference-time mask layer unimplemented, exact KataGo encoder missing.

## New infrastructure

- Wrote `.github/skills/verification/SKILL.md`: an evidence-driven verification
  procedure (run suites → parallel audits → re-derive every finding → fix +
  regression test → record), with SAKIGo invariants and past-incident lessons
  baked in. Intended to replace rote use of the checklist; the checklist
  remains as a reference appendix.
