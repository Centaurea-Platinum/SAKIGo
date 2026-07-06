# SAKIGo contracts (frozen 2026-07-06, P0)

Stable interfaces every module must honor. Changing anything here requires a
deliberate schema-version bump and a migration note in AI/Decisions.md.

## 1. Training record schema (v1, JSONL / .jsonl.zst)

One JSON object per line. Fields (targets may nest under `targets`):

| Field | Type / shape | Notes |
|---|---|---|
| `schema_version` | int = 1 | |
| `board_size` | int N | square boards only |
| `ply` | int | |
| `position_key` | str | sha1[:20] of moves + to_move |
| `ruleset` | object | `RulesetSpec.metadata()`: name, katago_rules, katago_ko, katago_suicide, saki_scoring, saki_ko, saki_suicide, komi |
| `board_planes` | float[6·N²] flat | plane order below, **mover perspective** |
| `rule_features` | float[10] | encoding below |
| `wdl` | float[4] | win/draw/loss/no_result, **mover perspective** |
| `score` | float | mover-perspective lead **÷ board area** |
| `ownership` | float[N²] ∈ [−1,1] | mover perspective (+1 = mine) |
| `policy` | float[N²+1] | one-hot teacher top-1; **pass = last index** |
| `budget` | float[N²+1] | full teacher policy renormalized over legal moves |
| `legal_mask` | bool[N²+1] | pass last, always true |
| `source` | object | provenance, not consumed by training |

Every target is optional per record; absent targets get mask=False in batches.

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
| `ownership_logits` | [B, N²] |
| `policy_logits` | [B, N²+1] |
| `budget_logits` | [B, N²+1] |

Row-major cells; pass logit is index N² (last). Ownership convention: logit ≥ 0 ⇔ mine.
Training does not mask illegal moves; masking is an inference-time precaution only.

## 5. Loss semantics

Per-head masks; averaging by `mask.float().sum().clamp_min(1.0)` (branchless).
wdl/policy/budget: soft cross-entropy vs distributions. score: smooth-L1.
ownership: BCE-with-logits on (t+1)/2. Total = Σ weight_h · loss_h.

## 6. Train/val split

Deterministic hash split by position: blake2b over
`(seed, board_size, ruleset_key, position_key)` — byte-compatible with
`Training/data.py::split_for_position`. Never change the key material.

## 7. Checkpoint payload (rebuilt trainer)

`torch.save` dict, loadable with `weights_only=True`:
`model_state`, `optimizer_state`, `scheduler_state`, `step`,
`model_config` (plain dict), `run_config` (plain dict),
`rng` (python/torch/cuda states), `schema_version`.
Written atomically (tmp + rename) as `checkpoints/step_%06d.pt`.

## 8. Run directory layout

```
runs/<name>/
  config.json          # resolved run config
  tb/                  # TensorBoard event files
  metrics.csv          # thin mirror for the HTML viewer
  checkpoints/step_%06d.pt
  status.json          # heartbeat for orchestration
```
