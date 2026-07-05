pub mod board;
pub mod encoder;
pub mod game;
pub mod hash;
// `python.rs` (PyO3 bindings) is parked and deliberately not compiled until
// the crates registry is reachable; see AI/Issues.md for the wiring steps.
pub mod rules;

pub use board::{Board, BoardError, Color, Group, Point, MAX_BOARD_SIZE, MIN_BOARD_SIZE};
pub use encoder::{
    encode_board_planes, encode_rule_features, BoardPlane, EncodedPosition, BOARD_PLANE_COUNT,
    RULE_FEATURE_COUNT,
};
pub use game::{GameState, GoMove, IllegalMove, MoveOutcome};
pub use hash::{hash_board, hash_state, PositionHash, StateHash};
pub use rules::{KoRule, Ruleset, ScoringRule, SuicideRule};
