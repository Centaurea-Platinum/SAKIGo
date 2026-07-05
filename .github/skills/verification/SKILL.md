---
name: verification
description: "Use when: verifying an implementation, auditing the codebase, reviewing a finished change, doing a pre-commit scan, or when asked to 'check everything' in SAKIGo. Evidence-driven verification procedure that replaces checklist ceremony with discriminating tests. Covers: running test suites, tracing tiny examples, auditing with subagents, filtering false positives, and SAKIGo-specific invariants (perspective flips, pass logit, D4 equivariance, masks)."
---

# Verification

Verify by gathering evidence that can actually discriminate between "correct"
and "broken" — not by walking a checklist and nodding. A checklist item you
can't falsify is ceremony; skip it and spend the effort where a bug could hide.

## Core principles

1. **Weigh evidence by discriminating power.** A passing test only counts if it
   would fail under the bug you're worried about. An underpowered repro can
   falsely exonerate (this repo's val-metric incident: a 38k-sample fresh model
   "cleanly reproduced" behavior that was actually a cache bug). Before
   accepting any "it's fine" result, ask: *what would this look like if the
   code were wrong?*
2. **Run things before reading things.** `pytest` + `cargo test` take under a
   minute and instantly bound the search space. Then read only where tests are
   thin.
3. **Trace a tiny example end to end.** For any data transformation, hand-walk
   one record: a 2x2 board, one capture, one perspective flip. Most SAKIGo bugs
   ever found were sign/index/perspective errors that a tiny trace exposes.
4. **Treat audit findings as hypotheses, not conclusions.** Subagent/reviewer
   findings have a high false-positive rate (a 2026-07-06 full scan: 5 reported
   bugs, 4 were false positives). For each finding, open the actual code and
   re-derive the claim before fixing anything. Domain definitions matter:
   e.g. *positional* superko keys on board only — a "missing side-to-move in
   hash" finding is wrong by definition. Do not delegate audits to weak models
   (owner instruction 2026-07-06: no Haiku-class subagents — too many false
   positives AND negatives); prefer doing the pass yourself.
5. **Robustness = the rebuild question.** For infrastructure, ask "what would a
   clean rebuild do differently?" and grade the gap: duplicated logic that can
   drift (two copies of a training loop), non-atomic writes of files another
   process or a resume will read (checkpoints, status JSON, caches), process
   lifecycles without a `finally` (orphaned engines), dead config surface
   (no-op CLI args). Fix the cheap ones; record the structural ones as issues.
6. **Verify the arithmetic of your own story.** Before explaining observed
   behavior (loss curves, sampling exposure, throughput), check the counting:
   samples seen per record, batches per epoch, who computes what when.
7. **Distinguish "code wrong" from "coverage missing".** A clean scan with
   untested behavior is not a clean scan. The fix for a coverage gap is a test
   that would have caught the feared bug.

## Procedure

1. **Establish state.** `git status` / `git log --oneline -8`; read
   `AI/Issues.md` watchlist and the latest `AI/Log.md` entry. Note what changed
   since the last verification — new/refactored files get the deepest look.
2. **Run the suites.**
   - Python: `uv --cache-dir .uv-cache run --frozen pytest -q --basetemp=.pytest-tmp -p no:cacheprovider`
     (repo-local temp: system temp denies access on this machine).
   - Rust: `cd Engine; cargo test --quiet`.
   - `get_errors` for compile/lint diagnostics.
3. **Audit in parallel with subagents** (one per module boundary: Engine,
   Model, Training). Tell them the known issues so they don't re-report, and
   demand: file+line, the traced logic error, and an explicit CLEAN list.
4. **Re-verify every finding yourself** against the source before acting
   (principle 4). Classify: real bug / real risk / coverage gap / false
   positive.
5. **Fix what's real.** Bugs: fix + regression test that fails without the fix.
   Coverage gaps: add the discriminating test. Risks/decisions: record in
   `AI/Issues.md`, don't silently resolve.
6. **Re-run the affected suite.** Then re-read the diff for churn, debug
   prints, and accidental scope creep.
7. **Record.** Update `AI/Log.md` (session entry) and `AI/Issues.md` (new/
   resolved risks). Commit with a message stating what was verified, not just
   what changed.

## SAKIGo-specific invariants (the recurring bug shapes)

Check these whenever code near them changed; each has bitten before or is a
standing design contract:

- **Perspective**: My/Opponent planes, WDL (length **4**: win/draw/loss/
  no-result), score sign, ownership sign, captured-stone diff, komi sign
  (negative for Black to move) — all flip with side to move; per-ply flips in
  any tree/sequence walk.
- **Action space**: `N*N + 1` logits, row-major board points, **pass is the
  final logit**. Komi/captures normalized by board **area**, clamped [-1, 1].
- **D4**: augmentation and equivariance apply to *every* board-shaped field
  (planes, ownership, policy/budget board part, legal mask) and *never* to the
  pass entry or global fields. Policy-index transforms must use the same group
  element as the plane transform.
- **Masking**: raw logits during training; legality masking only at
  inference/search. Masked losses normalize by `mask.sum().clamp_min(1)`.
- **Live-model measurement**: validation must exercise current weights — watch
  parameter caches (`RegularLinear1x1` caches only when
  `requires_grad=False`), stale eval iterators, mutated pinned buffers.
- **Rules source of truth**: the Rust engine. Python reimplementations
  (generator, selfplay_eval) are tracked divergence risks — compare semantics,
  don't assume.
- **Windows/WDDM**: oversized CUDA batches page instead of OOM-ing; judge
  memory by measured peak allocation, not by exceptions.
- **Strength claims**: paired color-reversed evals, not val loss (val loss has
  anti-correlated with strength here; the split leaks transposed openings).

## What NOT to do

- Don't re-report known watchlist items as new findings.
- Don't "fix" design decisions (e.g. masking location, policy/budget split)
  during verification — those go to `AI/Issues.md` for the human.
- Don't pad the report: findings, evidence, what remains. No ceremony.
