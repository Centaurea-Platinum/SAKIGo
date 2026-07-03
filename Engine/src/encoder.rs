use crate::board::{Color, Point};
use crate::game::GameState;
use crate::rules::{KoRule, ScoringRule, SuicideRule};

pub const BOARD_PLANE_COUNT: usize = 6;
pub const RULE_FEATURE_COUNT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum BoardPlane {
    MyStones = 0,
    OpponentStones = 1,
    EmptyPositions = 2,
    Corner = 3,
    Edge = 4,
    NonTrivialIllegal = 5,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedPosition {
    pub board_size: usize,
    pub board_planes: Vec<f32>,
    pub rule_features: [f32; RULE_FEATURE_COUNT],
}

impl EncodedPosition {
    pub fn from_state(state: &GameState) -> Self {
        Self {
            board_size: state.board().size(),
            board_planes: encode_board_planes(state),
            rule_features: encode_rule_features(state),
        }
    }

    pub fn plane_offset(&self, plane: BoardPlane, point: Point) -> usize {
        plane as usize * self.board_size * self.board_size + point.index(self.board_size)
    }

    pub fn plane_value(&self, plane: BoardPlane, point: Point) -> f32 {
        self.board_planes[self.plane_offset(plane, point)]
    }
}

pub fn encode_board_planes(state: &GameState) -> Vec<f32> {
    let board = state.board();
    let size = board.size();
    let area = board.area();
    let perspective = state.to_move();
    let mut planes = vec![0.0; BOARD_PLANE_COUNT * area];

    for point in board.points() {
        let cell_index = point.index(size);
        match board.get(point) {
            Some(color) if color == perspective => {
                planes[offset(BoardPlane::MyStones, area, cell_index)] = 1.0;
            }
            Some(_) => {
                planes[offset(BoardPlane::OpponentStones, area, cell_index)] = 1.0;
            }
            None => {
                planes[offset(BoardPlane::EmptyPositions, area, cell_index)] = 1.0;
                if state.would_be_legal(point).is_err() {
                    planes[offset(BoardPlane::NonTrivialIllegal, area, cell_index)] = 1.0;
                }
            }
        }

        if is_corner(point, size) {
            planes[offset(BoardPlane::Corner, area, cell_index)] = 1.0;
        } else if is_edge(point, size) {
            planes[offset(BoardPlane::Edge, area, cell_index)] = 1.0;
        }
    }

    planes
}

pub fn encode_rule_features(state: &GameState) -> [f32; RULE_FEATURE_COUNT] {
    let mut features = [0.0; RULE_FEATURE_COUNT];
    let rules = state.rules();
    match rules.scoring {
        ScoringRule::Area => features[0] = 1.0,
        ScoringRule::AreaAncientChinese => features[1] = 1.0,
        ScoringRule::Territory => features[2] = 1.0,
        ScoringRule::TerritoryWithSekiScore => features[3] = 1.0,
    }
    match rules.ko {
        KoRule::SimpleKo => features[4] = 1.0,
        KoRule::PositionalSuperKo => features[5] = 1.0,
    }
    match rules.suicide {
        SuicideRule::Allowed => features[6] = 1.0,
        SuicideRule::Forbidden => features[7] = 1.0,
    }

    let area = state.board().area() as f32;
    let perspective = state.to_move();
    let signed_komi = match perspective {
        Color::Black => -rules.komi,
        Color::White => rules.komi,
    };
    features[8] = normalized(signed_komi, area);
    features[9] = normalized(state.capture_diff_for(perspective) as f32, area);
    features
}

fn offset(plane: BoardPlane, area: usize, cell_index: usize) -> usize {
    plane as usize * area + cell_index
}

fn normalized(value: f32, area: f32) -> f32 {
    (value / area).clamp(-1.0, 1.0)
}

fn is_corner(point: Point, size: usize) -> bool {
    let last = size - 1;
    (point.row == 0 || point.row == last) && (point.col == 0 || point.col == last)
}

fn is_edge(point: Point, size: usize) -> bool {
    point.row == 0 || point.col == 0 || point.row + 1 == size || point.col + 1 == size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color};
    use crate::game::GameState;
    use crate::rules::{KoRule, Ruleset, ScoringRule, SuicideRule};

    #[test]
    fn encodes_current_player_board_planes_in_plane_major_layout() {
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(0, 0), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 1), Some(Color::White)).unwrap();
        let state = GameState::from_board(board, Ruleset::default(), Color::White, [0, 0]);

        let encoded = EncodedPosition::from_state(&state);
        assert_eq!(encoded.board_size, 3);
        assert_eq!(encoded.board_planes.len(), BOARD_PLANE_COUNT * 9);
        assert_eq!(
            encoded.plane_value(BoardPlane::MyStones, Point::new(1, 1)),
            1.0
        );
        assert_eq!(
            encoded.plane_value(BoardPlane::OpponentStones, Point::new(0, 0)),
            1.0
        );
        assert_eq!(
            encoded.plane_value(BoardPlane::EmptyPositions, Point::new(0, 1)),
            1.0
        );
        assert_eq!(
            encoded.plane_value(BoardPlane::Corner, Point::new(0, 0)),
            1.0
        );
        assert_eq!(encoded.plane_value(BoardPlane::Edge, Point::new(0, 1)), 1.0);
        assert_eq!(encoded.plane_value(BoardPlane::Edge, Point::new(0, 0)), 0.0);
    }

    #[test]
    fn encodes_suicide_as_non_trivial_illegal() {
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();
        board.set(Point::new(1, 0), Some(Color::White)).unwrap();
        board.set(Point::new(1, 2), Some(Color::White)).unwrap();
        board.set(Point::new(2, 1), Some(Color::White)).unwrap();
        let state = GameState::from_board(board, Ruleset::default(), Color::Black, [0, 0]);

        let encoded = EncodedPosition::from_state(&state);
        assert_eq!(
            encoded.plane_value(BoardPlane::NonTrivialIllegal, Point::new(1, 1)),
            1.0
        );
    }

    #[test]
    fn encodes_rules_as_one_hots_and_perspective_scalars() {
        let rules = Ruleset::new(
            ScoringRule::TerritoryWithSekiScore,
            KoRule::PositionalSuperKo,
            SuicideRule::Allowed,
            6.5,
        );
        let board = Board::new(5).unwrap();
        let state = GameState::from_board(board, rules, Color::White, [2, 5]);

        let features = encode_rule_features(&state);
        assert_eq!(&features[0..4], &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(&features[4..6], &[0.0, 1.0]);
        assert_eq!(&features[6..8], &[1.0, 0.0]);
        assert_eq!(features[8], 6.5 / 25.0);
        assert_eq!(features[9], 3.0 / 25.0);
    }
}
