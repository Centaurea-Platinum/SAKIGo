use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::board::{Board, BoardError, Color, Point};
use crate::hash::{hash_board, hash_state, PositionHash, StateHash};
use crate::rules::{KoRule, Ruleset, SuicideRule};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoMove {
    Play(Point),
    Pass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveOutcome {
    pub played: GoMove,
    pub color: Color,
    pub captured_opponent: usize,
    pub captured_self: usize,
    pub next_player: Color,
    pub move_number: usize,
    pub position_hash: PositionHash,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IllegalMove {
    PointOutOfBounds { point: Point, board_size: usize },
    Occupied { point: Point },
    SimpleKo { point: Point },
    Suicide { point: Point },
    SuperKo { point: Point },
}

impl fmt::Display for IllegalMove {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PointOutOfBounds { point, board_size } => write!(
                formatter,
                "point ({}, {}) is outside a {board_size}x{board_size} board",
                point.row, point.col
            ),
            Self::Occupied { point } => {
                write!(
                    formatter,
                    "point ({}, {}) is already occupied",
                    point.row, point.col
                )
            }
            Self::SimpleKo { point } => write!(
                formatter,
                "point ({}, {}) is forbidden by simple ko",
                point.row, point.col
            ),
            Self::Suicide { point } => {
                write!(formatter, "point ({}, {}) is suicide", point.row, point.col)
            }
            Self::SuperKo { point } => write!(
                formatter,
                "point ({}, {}) repeats a previous board position",
                point.row, point.col
            ),
        }
    }
}

impl Error for IllegalMove {}

#[derive(Debug, Clone)]
pub struct GameState {
    board: Board,
    rules: Ruleset,
    to_move: Color,
    captures: [usize; 2],
    simple_ko: Option<Point>,
    position_hash: PositionHash,
    seen_positions: HashSet<PositionHash>,
    position_history: Vec<PositionHash>,
    move_number: usize,
}

impl GameState {
    pub fn new(board_size: usize, rules: Ruleset) -> Result<Self, BoardError> {
        let board = Board::new(board_size)?;
        Ok(Self::from_board(board, rules, Color::Black, [0, 0]))
    }

    pub fn from_board(board: Board, rules: Ruleset, to_move: Color, captures: [usize; 2]) -> Self {
        let initial_hash = hash_board(&board);
        Self {
            board,
            rules,
            to_move,
            captures,
            simple_ko: None,
            position_hash: initial_hash,
            seen_positions: HashSet::from([initial_hash]),
            position_history: vec![initial_hash],
            move_number: 0,
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn rules(&self) -> Ruleset {
        self.rules
    }

    pub fn to_move(&self) -> Color {
        self.to_move
    }

    pub fn move_number(&self) -> usize {
        self.move_number
    }

    pub fn simple_ko_point(&self) -> Option<Point> {
        self.simple_ko
    }

    pub fn captured_by(&self, color: Color) -> usize {
        self.captures[color.index()]
    }

    pub fn capture_diff_for(&self, perspective: Color) -> isize {
        self.captured_by(perspective) as isize - self.captured_by(perspective.opponent()) as isize
    }

    pub fn position_hash(&self) -> PositionHash {
        self.position_hash
    }

    /// Metadata-aware key for transposition tables and NN caches: position
    /// plus side to move, simple-ko point, captures, and rules. Never used
    /// for superko checks (those are rule-defined; see `PositionHash`).
    pub fn state_hash(&self) -> StateHash {
        hash_state(
            self.position_hash,
            self.to_move,
            self.simple_ko,
            self.captures,
            &self.rules,
        )
    }

    pub fn position_history(&self) -> &[PositionHash] {
        &self.position_history
    }

    pub fn is_legal_point(&self, point: Point) -> bool {
        self.analyze_play(point).is_ok()
    }

    pub fn legal_points(&self) -> Vec<Point> {
        self.board
            .points()
            .filter(|point| self.is_legal_point(*point))
            .collect()
    }

    pub fn would_be_legal(&self, point: Point) -> Result<(), IllegalMove> {
        self.analyze_play(point).map(|_| ())
    }

    pub fn play(&mut self, action: GoMove) -> Result<MoveOutcome, IllegalMove> {
        match action {
            GoMove::Pass => Ok(self.pass()),
            GoMove::Play(point) => self.play_point(point),
        }
    }

    fn pass(&mut self) -> MoveOutcome {
        let color = self.to_move;
        let position_hash = self.position_hash;
        self.simple_ko = None;
        self.seen_positions.insert(position_hash);
        self.position_history.push(position_hash);
        self.to_move = self.to_move.opponent();
        self.move_number += 1;
        MoveOutcome {
            played: GoMove::Pass,
            color,
            captured_opponent: 0,
            captured_self: 0,
            next_player: self.to_move,
            move_number: self.move_number,
            position_hash,
        }
    }

    fn play_point(&mut self, point: Point) -> Result<MoveOutcome, IllegalMove> {
        let color = self.to_move;
        let analysis = self.analyze_play(point)?;

        self.board = analysis.board;
        self.captures[color.index()] += analysis.captured_opponent;
        self.captures[color.opponent().index()] += analysis.captured_self;
        self.simple_ko = analysis.next_simple_ko;
        self.position_hash = analysis.position_hash;
        self.seen_positions.insert(analysis.position_hash);
        self.position_history.push(analysis.position_hash);
        self.to_move = color.opponent();
        self.move_number += 1;

        Ok(MoveOutcome {
            played: GoMove::Play(point),
            color,
            captured_opponent: analysis.captured_opponent,
            captured_self: analysis.captured_self,
            next_player: self.to_move,
            move_number: self.move_number,
            position_hash: analysis.position_hash,
        })
    }

    fn analyze_play(&self, point: Point) -> Result<AnalyzedMove, IllegalMove> {
        if !self.board.contains(point) {
            return Err(IllegalMove::PointOutOfBounds {
                point,
                board_size: self.board.size(),
            });
        }
        if !self.board.is_empty(point) {
            return Err(IllegalMove::Occupied { point });
        }
        if self.rules.ko == KoRule::SimpleKo && self.simple_ko == Some(point) {
            return Err(IllegalMove::SimpleKo { point });
        }

        let color = self.to_move;
        let opponent = color.opponent();
        let mut next_board = self.board.clone();
        next_board
            .set(point, Some(color))
            .expect("point was already checked to be in bounds");

        let mut position_hash = self.position_hash.toggle_stone(color, point);
        let mut captured_points = Vec::new();
        for neighbor in self.board.neighbors(point) {
            if next_board.get(neighbor) != Some(opponent) {
                continue;
            }
            let group = next_board
                .group_at(neighbor)
                .expect("neighbor was checked to contain an opponent stone");
            if group.has_no_liberties() {
                let stones = group.stones().to_vec();
                next_board.remove_points(&stones);
                captured_points.extend(stones);
            }
        }
        for &captured in &captured_points {
            position_hash = position_hash.toggle_stone(opponent, captured);
        }

        let mut self_captured_points = Vec::new();
        if next_board.get(point) == Some(color) {
            let own_group = next_board
                .group_at(point)
                .expect("played stone should still be on the board");
            if own_group.has_no_liberties() {
                if self.rules.suicide == SuicideRule::Forbidden {
                    return Err(IllegalMove::Suicide { point });
                }
                self_captured_points = own_group.stones().to_vec();
                next_board.remove_points(&self_captured_points);
                for &captured in &self_captured_points {
                    position_hash = position_hash.toggle_stone(color, captured);
                }
            }
        }

        if self.rules.ko == KoRule::PositionalSuperKo
            && self.seen_positions.contains(&position_hash)
        {
            return Err(IllegalMove::SuperKo { point });
        }

        let next_simple_ko =
            self.next_simple_ko(point, &captured_points, &self_captured_points, &next_board);

        Ok(AnalyzedMove {
            board: next_board,
            captured_opponent: captured_points.len(),
            captured_self: self_captured_points.len(),
            next_simple_ko,
            position_hash,
        })
    }

    fn next_simple_ko(
        &self,
        point: Point,
        captured_points: &[Point],
        self_captured_points: &[Point],
        next_board: &Board,
    ) -> Option<Point> {
        if self.rules.ko != KoRule::SimpleKo
            || captured_points.len() != 1
            || !self_captured_points.is_empty()
        {
            return None;
        }
        let own_group = next_board.group_at(point)?;
        let captured_point = captured_points[0];
        if own_group.stone_count() == 1
            && own_group.liberty_count() == 1
            && own_group.liberties().contains(&captured_point)
        {
            Some(captured_point)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct AnalyzedMove {
    board: Board,
    captured_opponent: usize,
    captured_self: usize,
    next_simple_ko: Option<Point>,
    position_hash: PositionHash,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{KoRule, ScoringRule, SuicideRule};

    fn simple_rules() -> Ruleset {
        Ruleset::new(
            ScoringRule::Area,
            KoRule::SimpleKo,
            SuicideRule::Forbidden,
            7.5,
        )
    }

    #[test]
    fn play_captures_adjacent_group() {
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(0, 2), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 0), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 1), Some(Color::Black)).unwrap();
        let mut state = GameState::from_board(board, simple_rules(), Color::Black, [0, 0]);

        let outcome = state.play(GoMove::Play(Point::new(0, 0))).unwrap();
        assert_eq!(outcome.captured_opponent, 1);
        assert_eq!(state.board().get(Point::new(0, 1)), None);
        assert_eq!(state.captured_by(Color::Black), 1);
    }

    #[test]
    fn suicide_is_rejected_when_forbidden() {
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::White)).unwrap();
        board.set(Point::new(1, 2), Some(Color::White)).unwrap();
        board.set(Point::new(2, 1), Some(Color::White)).unwrap();
        let state = GameState::from_board(board, simple_rules(), Color::Black, [0, 0]);

        assert_eq!(
            state.would_be_legal(Point::new(1, 1)),
            Err(IllegalMove::Suicide {
                point: Point::new(1, 1)
            })
        );
    }

    #[test]
    fn suicide_can_remove_own_group_when_allowed() {
        let rules = Ruleset::new(
            ScoringRule::Area,
            KoRule::SimpleKo,
            SuicideRule::Allowed,
            7.5,
        );
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::White)).unwrap();
        board.set(Point::new(1, 2), Some(Color::White)).unwrap();
        board.set(Point::new(2, 1), Some(Color::White)).unwrap();
        let mut state = GameState::from_board(board, rules, Color::Black, [0, 0]);

        let outcome = state.play(GoMove::Play(Point::new(1, 1))).unwrap();
        assert_eq!(outcome.captured_self, 1);
        assert_eq!(state.board().get(Point::new(1, 1)), None);
        assert_eq!(state.captured_by(Color::White), 1);
    }

    #[test]
    fn positional_superko_blocks_board_repetition() {
        let rules = Ruleset::new(
            ScoringRule::Area,
            KoRule::PositionalSuperKo,
            SuicideRule::Forbidden,
            7.5,
        );
        let mut board = Board::new(4).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::White)).unwrap();
        board.set(Point::new(1, 2), Some(Color::White)).unwrap();
        board.set(Point::new(2, 1), Some(Color::White)).unwrap();
        board.set(Point::new(0, 2), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 3), Some(Color::Black)).unwrap();
        board.set(Point::new(2, 2), Some(Color::Black)).unwrap();
        let mut state = GameState::from_board(board, rules, Color::Black, [0, 0]);

        let outcome = state.play(GoMove::Play(Point::new(1, 1))).unwrap();
        assert_eq!(outcome.captured_opponent, 1);
        // White's recapture would recreate the initial board position.
        assert_eq!(
            state.would_be_legal(Point::new(1, 2)),
            Err(IllegalMove::SuperKo {
                point: Point::new(1, 2)
            })
        );
    }

    #[test]
    fn superko_counts_initial_position() {
        let rules = Ruleset::new(
            ScoringRule::Area,
            KoRule::PositionalSuperKo,
            SuicideRule::Allowed,
            7.5,
        );
        let state = GameState::new(1, rules).unwrap();
        // On 1x1 the first stone is an allowed single-stone suicide, which
        // would recreate the empty initial board: a repeated position.
        assert_eq!(
            state.would_be_legal(Point::new(0, 0)),
            Err(IllegalMove::SuperKo {
                point: Point::new(0, 0)
            })
        );
    }

    #[test]
    fn pass_clears_simple_ko_and_updates_state() {
        let mut board = Board::new(4).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::White)).unwrap();
        board.set(Point::new(1, 2), Some(Color::White)).unwrap();
        board.set(Point::new(2, 1), Some(Color::White)).unwrap();
        board.set(Point::new(0, 2), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 3), Some(Color::Black)).unwrap();
        board.set(Point::new(2, 2), Some(Color::Black)).unwrap();
        let mut state = GameState::from_board(board, simple_rules(), Color::Black, [0, 0]);

        state.play(GoMove::Play(Point::new(1, 1))).unwrap();
        assert_eq!(state.simple_ko_point(), Some(Point::new(1, 2)));
        let move_number = state.move_number();
        let hash_before = *state.position_history().last().unwrap();

        let outcome = state.play(GoMove::Pass).unwrap();
        assert_eq!(outcome.position_hash, hash_before);
        assert_eq!(state.simple_ko_point(), None);
        assert_eq!(state.to_move(), Color::Black);
        assert_eq!(state.move_number(), move_number + 1);

        state.play(GoMove::Pass).unwrap();
        // With the ko cleared by the passes, White may now recapture.
        assert_eq!(state.to_move(), Color::White);
        let outcome = state.play(GoMove::Play(Point::new(1, 2))).unwrap();
        assert_eq!(outcome.captured_opponent, 1);
    }

    #[test]
    fn incremental_hash_matches_full_recompute_through_captures() {
        let rules = Ruleset::new(
            ScoringRule::Area,
            KoRule::PositionalSuperKo,
            SuicideRule::Allowed,
            7.5,
        );
        let mut state = GameState::new(3, rules).unwrap();
        let moves = [
            GoMove::Play(Point::new(0, 1)), // B
            GoMove::Play(Point::new(0, 0)), // W (will be captured)
            GoMove::Play(Point::new(1, 0)), // B captures W(0,0)
            GoMove::Play(Point::new(2, 2)), // W
            GoMove::Pass,                   // B
            GoMove::Play(Point::new(2, 1)), // W
        ];
        for go_move in moves {
            let outcome = state.play(go_move).unwrap();
            assert_eq!(state.position_hash(), hash_board(state.board()));
            assert_eq!(outcome.position_hash, state.position_hash());
        }
    }

    #[test]
    fn one_move_captures_multiple_groups() {
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(0, 0), Some(Color::White)).unwrap();
        board.set(Point::new(0, 2), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 2), Some(Color::Black)).unwrap();
        let mut state = GameState::from_board(board, simple_rules(), Color::Black, [0, 0]);

        let outcome = state.play(GoMove::Play(Point::new(0, 1))).unwrap();
        assert_eq!(outcome.captured_opponent, 2);
        assert_eq!(state.board().get(Point::new(0, 0)), None);
        assert_eq!(state.board().get(Point::new(0, 2)), None);
        assert_eq!(state.captured_by(Color::Black), 2);
    }

    #[test]
    fn simple_ko_blocks_immediate_recapture() {
        let mut board = Board::new(4).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::White)).unwrap();
        board.set(Point::new(1, 2), Some(Color::White)).unwrap();
        board.set(Point::new(2, 1), Some(Color::White)).unwrap();
        board.set(Point::new(0, 2), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 3), Some(Color::Black)).unwrap();
        board.set(Point::new(2, 2), Some(Color::Black)).unwrap();
        let mut state = GameState::from_board(board, simple_rules(), Color::Black, [0, 0]);

        let outcome = state.play(GoMove::Play(Point::new(1, 1))).unwrap();
        assert_eq!(outcome.captured_opponent, 1);
        assert_eq!(state.simple_ko_point(), Some(Point::new(1, 2)));
        assert_eq!(
            state.would_be_legal(Point::new(1, 2)),
            Err(IllegalMove::SimpleKo {
                point: Point::new(1, 2)
            })
        );
    }
}
