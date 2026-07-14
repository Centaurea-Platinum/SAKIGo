# SAKIGo contracts (frozen 2026-07-06, P0)

Stable interfaces every module must honor. Changing anything here requires a
deliberate schema-version bump and a migration note in AI/Decisions.md.

## 1. Training record schema (v1 and v2, JSONL / .jsonl.zst)

One JSON object per line. Fields (targets may nest under `targets`):

| Field | Type / shape | Notes |
|---|---|---|
| `schema_version` | int = 1, 2, or 3 | v2 permits equal-optimal soft policy targets; v3 identifies raw-visit budget semantics |
| `board_size` | JSON int N | square boards only; booleans/floats/strings are rejected |
| `ply` | JSON int | booleans/floats/strings are rejected |
| `position_key` | str | non-empty provenance key; current book records use a BLAKE2b digest of the model-visible input |
| `ruleset` | object | `RulesetSpec.metadata()`: name, katago_rules, katago_ko, katago_suicide, saki_scoring, saki_ko, saki_suicide, komi |
| `board_planes` | float[6·N²] flat | plane order below, **mover perspective** |
| `rule_features` | float[10] | encoding below |
| `wdl` | float[4] | win/draw/loss/no_result, **mover perspective** |
| `score` | float | mover-perspective lead **÷ board area** |
| `ownership` | float[N²] ∈ [−1,1] | optional legacy target, mover perspective (+1 = mine); absent from current book records |
| `policy` | float[N²+1] | action distribution; current books use uniform tied rounded-optimum concrete moves; **pass = last index** |
| `budget` | float[N²+1] | action distribution; current books normalize raw concrete visits (`v`) after discarding `other`; each symmetry-equivalent coordinate receives its representative row's full count |
| `legal_mask` | bool[N²+1] | JSON booleans only; pass last and always true |
| `source` | object | provenance, not consumed by training |

Every target is optional per record; absent targets get mask=False in batches.
Schema v1 retains the original strictly one-hot policy invariant. Schema v2
permits policy mass to be uniform across rounded-equal optimal book moves.
Schema v3 retains that policy contract and identifies corrected book budgets
that use raw visits (`v`) expanded across symmetry-equivalent actions. Both action targets must
assign zero mass to actions rejected by `legal_mask`.

Migration: current readers retain v1 and v2 support for legacy generated data.
New mixed-small-board book records use v3; all field shapes and batch layouts
are unchanged.

## 2. Board planes (index order)

0 MyStones · 1 OpponentStones · 2 Empty · 3 BoundaryCorner · 4 BoundaryEdge ·
5 NonTrivialIllegal (empty-but-illegal under the generating rules).
Perspective = side to move. Must match `Engine/src/encoder.rs` bit-for-bit.

## 3. Rule-feature encoding (10 floats)

[0:4] scoring one-hot (area, area_ancient_chinese, territory, territory_with_seki_score) ·
[4:6] ko one-hot (simple_ko, positional_superko) ·
[6:8] suicide one-hot (allowed, forbidden) ·
[8] mover-signed komi ÷ area, clamped ±1 (negative when Black to move and komi favors White) ·
[9] (my captures − opponent captures) ÷ area, clamped ±1.
Source of truth: `sakigo/rulesets.py` `RulesetSpec.rule_features` + `validate_rule_features`.

## 4. Model forward contract

`model(board, rules) -> dict[str, Tensor]` with
`board: [B, 6, N, N]` float, `rules: [B, 10]` float, and outputs:

| Key | Shape |
|---|---|
| `wdl_logits` | [B, 4] |
| `score` | [B, 1] |
| `policy_logits` | [B, N²+1] |
| `budget_logits` | [B, N²+1] |

Row-major cells; pass logit is index N² (last).
Training does not mask illegal moves; masking is an inference-time precaution only.

## 5. Loss semantics

Per-head masks; averaging by `mask.float().sum().clamp_min(1.0)` (branchless).
wdl/policy/budget: soft cross-entropy vs distributions. score: smooth-L1.
Total = `wdl_weight · wdl + score_weight · N² · score +
policy_weight · policy + budget_weight · budget`; the active suite uses
base weight 1 for every head, so score's effective coefficient is 49/64/81.

## 6. Train/val split

Prepared-data format v3 splits on a canonical identity of the exact model
input: the little-endian float32 bytes of `board_planes` followed by
`rule_features`, then blake2b over `(seed, board_size, ruleset_key,
canonical_input_key)`. `position_key` remains record provenance and is not
trusted as the canonical split key. This
keeps transposed move sequences that reach the same model-visible position in
the same split.

Prepared manifests enumerate every validation `(board_size, ruleset_key)`
cohort. Validation batching never crosses a cohort boundary and a positive
`val_batches` cap must allocate at least one batch to every cohort. Runs record
aggregate metrics in `metrics.csv` and long-form per-cohort loss curves in
`validation_metrics.csv`; the same curves are written under `val_groups/` in
TensorBoard. Training batches remain grouped only by board size and preserve
the natural ruleset mixture.

## 7. Checkpoint payload (rebuilt trainer)

`torch.save` dict, loadable with `weights_only=True`:
`model_state`, `optimizer_state`, `scheduler_state`, `step`,
`model_config` (plain dict), `run_config` (plain dict),
`rng` (python/torch/cuda states), `sampler_state`,
`augmentation_state`, `sampler_state_exact`, `checkpoint_schema_version` (=8).
Written atomically (tmp + rename) as `checkpoints/step_%06d.pt`.
Exact sampler, augmentation, and RNG continuation requires checkpoints produced
and resumed with `num_workers=0`; the trainer rejects changed trajectory
properties, prepared-data identity, optimizer state, and prefetched sampler
states. Backend kernels may still be numerically non-bit-reproducible, so this
is batch/control-state exactness rather than a promise of bit-identical weights.

Schema 8 retains schema 7's exact sampler and augmentation state and binds
checkpoints to the board-area-scaled score objective. Earlier model and
optimizer states are intentionally rejected rather than partially migrated.

## 8. Run directory layout

```
runs/<name>/
  config.json          # resolved run config
  tb/                  # TensorBoard event files
  metrics.csv          # thin mirror for the HTML viewer
  validation_metrics.csv # long-form board-size/ruleset validation curves
  checkpoints/step_%06d.pt
  status.json          # heartbeat for orchestration
```

For multi-stage suite runs, use `python -m sakigo.train.suite`; keep each
trainer run under `train/<spec>/` and keep orchestration artifacts out of the
root: `data/`, `prepared/`, `generation/`, `logs/`, `sweeps/`, and `scripts/`.
