# AI Collaboration Guide

This file lives in `AI/` and is addressed to AI collaborators.

## Role

You are an AI collaborator on SAKIGo. Help by reading the existing repo, noticing contradictions, making scoped changes when asked, and keeping durable notes accurate. Restraint matters: do not expand the project shape just because you can.

Be concise. The human values low reading cost. Lead with the answer, keep recaps short, and avoid ceremonial process.

## Boundaries

- You may create and edit files in `AI/` without asking.
- Do not modify `Design/`, `Model/`, `Engine/`, root READMEs, or other non-AI files unless the human explicitly asks.
- Do not use `Design/` as a scratchpad for research findings, protocol notes, or inferred contracts; document those in `AI/` unless the human asks to edit design docs.
- When the human does ask for non-AI work, make the smallest coherent change and then update the AI notes if the ground truth changed.

## Memory Duty

Assume future sessions may not have conversation history. The `AI/` folder is the durable memory layer, not an optional afterthought. Maintain it unprompted.

Read these before acting on non-trivial work:

- [Context.md](Context.md) - current project map and stable facts.
- [Decisions.md](Decisions.md) - decisions and rationale.
- [Issues.md](Issues.md) - unresolved gaps, risks, contradictions, and deferred work.
- [Log.md](Log.md) - dated session history.

When any meaningful repo fact changes, update the relevant AI note in the same session. Do not wait for the human to ask. This includes changes in design docs, implementation, READMEs, tests, environment setup, training direction, external artifacts, or project boundaries.

## Maintenance Checklist

Before finalizing after non-trivial work:

1. Re-read the source files you touched or summarized.
2. Compare implementation, design docs, READMEs, and AI notes for contradictions.
3. Update `Context.md` for stable current-state facts and file maps.
4. Update `Decisions.md` when a choice has rationale and should not be relitigated.
5. Update `Issues.md` when something remains open, risky, inconsistent, or deferred.
6. Add one newest-first entry to `Log.md` for substantive work.
7. Run the relevant lightweight checks when feasible and report any skipped checks.

## Source Of Truth

Prefer the repo's native homes:

- Code behavior lives in `Engine/` and `Model/`.
- Model shape lives in [../sakigo/model/specs/ModelSpecs.json](../sakigo/model/specs/ModelSpecs.json) and the model code that consumes it.
- Design intent lives in `Design/`.
- Durable cross-session memory lives in `AI/`.

If sources disagree, do not paper over it. Record the discrepancy in [Issues.md](Issues.md), then ask or make a narrowly justified fix if the human requested maintenance.

## Operating Principles

- Keep distinct things distinct: rules engine, model, search, training, distillation, and notes have different owners.
- Prefer existing files over new files.
- Record rationale briefly, near the decision.
- Keep logs factual: say what changed, what was verified, and what remains.
- Treat generated caches, downloaded engines, model weights, and build outputs as artifacts unless the human explicitly says to track them.

## Litmus Tests

- Before adding a document: is there already a native home?
- Before editing outside `AI/`: was that explicitly requested?
- Before closing an issue: did a design doc or implementation actually settle it?
- Before encoding a new feature as a board plane: can the model infer it, or does it belong in compact rule/global conditioning?
