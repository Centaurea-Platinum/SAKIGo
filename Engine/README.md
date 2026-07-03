# SAKIGo Engine

This folder contains the Rust rules and feature-encoding engine for SAKIGo. It is
kept separate from the model code so self-play, search, legality, and training
data generation can all share one deterministic source of truth.

## Rationale

The engine owns the parts of Go that should not be learned or duplicated:

- Board topology: size checks, row-major point indexing, neighbor lookup, groups,
  and liberties.
- Rule legality: occupancy, suicide, simple ko, positional superko, captures, and
  pass handling.
- Stable position state: side to move, captures, simple-ko point, move number,
  and seen-position hashes.
- Model input encoding: a compact board tensor plus global rule features that
  match the design notes.

Keeping these pieces in a small dependency-free crate gives the project a few
useful properties:

- Search can ask the same legality questions the training pipeline uses.
- The model does not need history planes just to know non-trivial illegality; the
  engine computes that from the full game state before encoding.
- Experiments can change model architecture without reimplementing captures or ko.
- Tests can target rules and encodings directly, before any neural code is in the
  loop.

## Mechanism

The public API is re-exported from `src/lib.rs`.

### Board representation

`Board` stores `size` and a flat `Vec<Option<Color>>` in row-major order. `Point`
converts between `(row, col)` and a flat index, which keeps tensor encoding and
rule logic aligned.

Groups are discovered with a breadth-first search over same-colored neighbors.
During traversal, adjacent empty points are collected as liberties. Captures are
then just removal of every stone in a group with zero liberties.

Supported board sizes are `1..=32`.

### Rule state and moves

`GameState` wraps the board with:

- `Ruleset`: scoring rule, ko rule, suicide rule, and komi.
- `to_move`: the player whose perspective is used for move generation and
  encoding.
- capture counts for both colors.
- the current simple-ko point, if any.
- position hashes seen so far, plus a position history.
- the move number.

Move application is analyze-first, commit-second:

1. Reject out-of-bounds, occupied, and immediate simple-ko moves.
2. Clone the board and place the current player's stone.
3. Remove any adjacent opponent groups with no liberties.
4. Check the played stone's own group. If it has no liberties, reject the move
   when suicide is forbidden; otherwise remove the self-captured group.
5. Hash the resulting board and reject it under positional superko if that hash
   has already appeared.
6. Compute the next simple-ko point only for the classic one-stone capture shape.
7. Commit the analyzed board, captures, ko state, hash history, side to move, and
   move number.

Passing clears simple ko, records the unchanged board hash, flips the side to
move, and advances the move number.

### Hashing

`hash_board` uses a deterministic 64-bit FNV-style hash over the board size and
every cell state: empty, black, or white. This is intended for fast repetition
tracking inside the engine, not for cryptographic identity.

### Encoding

`EncodedPosition::from_state` produces:

- `board_planes`: `6 * board_area` floats in plane-major layout.
- `rule_features`: 10 global floats.

The board planes are:

1. current player's stones.
2. opponent stones.
3. empty positions.
4. corners.
5. non-corner edges.
6. non-trivial illegal empty points.

The illegal plane is deliberately sparse: occupied points are already described
by the stone planes, so it marks only empty points that fail `would_be_legal`.

The rule features are:

1. four scoring-rule one-hots.
2. two ko-rule one-hots.
3. two suicide-rule one-hots.
4. signed komi divided by board area, from the current player's perspective.
5. capture difference divided by board area, from the current player's
   perspective.

This mirrors the design goal of keeping board inputs minimal while feeding global
rule context as compact scalar features.

## Current scope

Implemented:

- board construction and validation.
- group and liberty detection.
- captures.
- forbidden and allowed suicide.
- simple ko.
- positional superko.
- pass moves.
- capture accounting.
- board and rule feature encoding.

Not implemented here yet:

- final scoring.
- end-of-game adjudication.
- search.
- neural network inference.
- training targets.

Those layers should build on this crate instead of duplicating its rule logic.

## Development

Run the engine tests from this folder:

```powershell
cargo test
```
