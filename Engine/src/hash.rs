use crate::board::{Board, Color, Point, MAX_BOARD_SIZE};
use crate::rules::{KoRule, Ruleset, ScoringRule, SuicideRule};

/// 128-bit Zobrist position hash (KataGo-style width).
///
/// The hash covers the board configuration plus a board-size constant; it is
/// deliberately independent of side to move, capture counts, and move number,
/// because positional superko compares board positions only. (Folding capture
/// counts in would break superko outright: any play that recreates an earlier
/// board captured stones in between, so counts strictly increase around every
/// cycle and the hash would never repeat.) For a metadata-aware key, see
/// [`StateHash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PositionHash(pub u128);

impl PositionHash {
    /// XOR-toggles one stone in or out of the hash. Zobrist updates are their
    /// own inverse, so the same call adds a missing stone or removes a
    /// present one.
    #[must_use]
    pub fn toggle_stone(self, color: Color, point: Point) -> Self {
        Self(self.0 ^ stone_key(color, point))
    }
}

const MAX_CELLS: usize = MAX_BOARD_SIZE * MAX_BOARD_SIZE;

/// SplitMix64 step used to derive the Zobrist tables deterministically at
/// compile time (no runtime init, no dependencies).
const fn splitmix64(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (state, z ^ (z >> 31))
}

const fn next_u128(state: u64) -> (u64, u128) {
    let (state, high) = splitmix64(state);
    let (state, low) = splitmix64(state);
    (state, ((high as u128) << 64) | low as u128)
}

const fn build_stone_table() -> [[u128; MAX_CELLS]; 2] {
    let mut table = [[0u128; MAX_CELLS]; 2];
    let mut state = 0x0DD5_AC5E_ED12_8B17u64;
    let mut color = 0;
    while color < 2 {
        let mut cell = 0;
        while cell < MAX_CELLS {
            let (next_state, value) = next_u128(state);
            state = next_state;
            table[color][cell] = value;
            cell += 1;
        }
        color += 1;
    }
    table
}

const fn build_size_table() -> [u128; MAX_BOARD_SIZE + 1] {
    let mut table = [0u128; MAX_BOARD_SIZE + 1];
    let mut state = 0xB0A2_D512_E5EE_D001u64;
    let mut size = 1;
    while size <= MAX_BOARD_SIZE {
        let (next_state, value) = next_u128(state);
        state = next_state;
        table[size] = value;
        size += 1;
    }
    table
}

static STONE_TABLE: [[u128; MAX_CELLS]; 2] = build_stone_table();
static SIZE_TABLE: [u128; MAX_BOARD_SIZE + 1] = build_size_table();

fn stone_key(color: Color, point: Point) -> u128 {
    STONE_TABLE[color.index()][point.row * MAX_BOARD_SIZE + point.col]
}

/// Full recomputation from scratch: O(area). Used at game construction and as
/// the ground truth that incremental updates are tested against.
pub fn hash_board(board: &Board) -> PositionHash {
    let size = board.size();
    let mut hash = SIZE_TABLE[size];
    for (index, cell) in board.cells().iter().enumerate() {
        if let Some(color) = cell {
            hash ^= stone_key(*color, Point::from_index(size, index));
        }
    }
    PositionHash(hash)
}

/// 128-bit hash of the full game state: board position plus side to move,
/// simple-ko point, capture counts, and ruleset (including komi).
///
/// This is the transposition-table / NN-cache key (KataGo's
/// `getSituationRulesAndKoHash` analogue). It is NOT a repetition-rule hash:
/// superko checks must keep using [`PositionHash`], whose contents are fixed
/// by the ko rule itself. Capture counts belong here because SAKIGo's network
/// input includes the capture difference, so states differing only in
/// captures evaluate differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateHash(pub u128);

const PLAYER_KEYS: [u128; 2] = {
    let (state, black) = next_u128(0x7057_0D0E_5EED_0001u64);
    let (_, white) = next_u128(state);
    [black, white]
};

const fn build_ko_point_table() -> [u128; MAX_CELLS] {
    let mut table = [0u128; MAX_CELLS];
    let mut state = 0x4B00_0170_5EED_0002u64;
    let mut cell = 0;
    while cell < MAX_CELLS {
        let (next_state, value) = next_u128(state);
        state = next_state;
        table[cell] = value;
        cell += 1;
    }
    table
}

static KO_POINT_TABLE: [u128; MAX_CELLS] = build_ko_point_table();

fn player_key(color: Color) -> u128 {
    PLAYER_KEYS[color.index()]
}

pub fn hash_state(
    position: PositionHash,
    to_move: Color,
    simple_ko: Option<Point>,
    captures: [usize; 2],
    rules: &Ruleset,
) -> StateHash {
    let mut hash = position.0;
    hash ^= player_key(to_move);
    if let Some(point) = simple_ko {
        hash ^= KO_POINT_TABLE[point.row * MAX_BOARD_SIZE + point.col];
    }
    hash ^= mix_metadata(0xCA97, captures[0] as u64);
    hash ^= mix_metadata(0xCA98, captures[1] as u64);
    // Komi discretized to quarter points (KataGo discretizes similarly).
    hash ^= mix_metadata(0x4B0A, (rules.komi * 4.0).round() as i64 as u64);
    hash ^= mix_metadata(
        0x521E,
        match rules.scoring {
            ScoringRule::Area => 0,
            ScoringRule::AreaAncientChinese => 1,
            ScoringRule::Territory => 2,
            ScoringRule::TerritoryWithSekiScore => 3,
        },
    );
    hash ^= mix_metadata(
        0x4B01,
        match rules.ko {
            KoRule::SimpleKo => 0,
            KoRule::PositionalSuperKo => 1,
        },
    );
    hash ^= mix_metadata(
        0x5C1D,
        match rules.suicide {
            SuicideRule::Allowed => 0,
            SuicideRule::Forbidden => 1,
        },
    );
    StateHash(hash)
}

/// Domain-separated SplitMix mixing for scalar metadata: equal values in
/// different fields (e.g. black vs white captures) get unrelated keys.
fn mix_metadata(domain: u64, value: u64) -> u128 {
    let seed = domain
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(value.wrapping_mul(0xD134_2543_DE82_EF95));
    let (_, mixed) = next_u128(seed);
    mixed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggling_stones_matches_full_recompute() {
        let mut board = Board::new(5).unwrap();
        let mut hash = hash_board(&board);
        let stones = [
            (Point::new(0, 0), Color::Black),
            (Point::new(2, 3), Color::White),
            (Point::new(4, 4), Color::Black),
        ];
        for (point, color) in stones {
            board.set(point, Some(color)).unwrap();
            hash = hash.toggle_stone(color, point);
            assert_eq!(hash, hash_board(&board));
        }
        // Removal is the same toggle.
        board.set(Point::new(2, 3), None).unwrap();
        hash = hash.toggle_stone(Color::White, Point::new(2, 3));
        assert_eq!(hash, hash_board(&board));
    }

    #[test]
    fn hash_distinguishes_color_point_and_board_size() {
        let empty5 = hash_board(&Board::new(5).unwrap());
        let empty9 = hash_board(&Board::new(9).unwrap());
        assert_ne!(empty5, empty9);

        let point = Point::new(1, 1);
        let black = empty5.toggle_stone(Color::Black, point);
        let white = empty5.toggle_stone(Color::White, point);
        assert_ne!(black, white);
        assert_ne!(black, empty5);
        assert_ne!(black, empty5.toggle_stone(Color::Black, Point::new(1, 2)));
    }

    #[test]
    fn state_hash_distinguishes_metadata_position_hash_does_not() {
        let board = Board::new(5).unwrap();
        let position = hash_board(&board);
        let rules = Ruleset::default();

        let base = hash_state(position, Color::Black, None, [0, 0], &rules);
        // Side to move.
        assert_ne!(
            base,
            hash_state(position, Color::White, None, [0, 0], &rules)
        );
        // Capture counts, per color.
        assert_ne!(
            base,
            hash_state(position, Color::Black, None, [1, 0], &rules)
        );
        assert_ne!(
            hash_state(position, Color::Black, None, [1, 0], &rules),
            hash_state(position, Color::Black, None, [0, 1], &rules)
        );
        // Simple-ko point.
        assert_ne!(
            base,
            hash_state(position, Color::Black, Some(Point::new(2, 2)), [0, 0], &rules)
        );
        // Komi and rule variations.
        let komi6 = Ruleset::new(rules.scoring, rules.ko, rules.suicide, 6.5);
        assert_ne!(base, hash_state(position, Color::Black, None, [0, 0], &komi6));
        let psk = Ruleset::new(rules.scoring, KoRule::PositionalSuperKo, rules.suicide, rules.komi);
        assert_ne!(base, hash_state(position, Color::Black, None, [0, 0], &psk));
        // Same inputs reproduce the same hash.
        assert_eq!(
            base,
            hash_state(position, Color::Black, None, [0, 0], &rules)
        );
    }
}
