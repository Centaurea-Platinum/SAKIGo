use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

use crate::board::{Board, Color};
use crate::rules::{Ruleset, ScoringRule};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringError {
    UnsupportedRule(ScoringRule),
}

impl fmt::Display for ScoringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRule(rule) => {
                write!(formatter, "unsupported final scoring rule {rule:?}")
            }
        }
    }
}

impl Error for ScoringError {}

/// Tromp-Taylor/Chinese-style area score from Black's perspective.
pub fn final_area_score(board: &Board, rules: Ruleset) -> Result<f32, ScoringError> {
    if rules.scoring != ScoringRule::Area {
        return Err(ScoringError::UnsupportedRule(rules.scoring));
    }
    let mut black_area = 0usize;
    let mut white_area = 0usize;
    let mut visited = vec![false; board.area()];

    for point in board.points() {
        match board.get(point) {
            Some(Color::Black) => black_area += 1,
            Some(Color::White) => white_area += 1,
            None if !visited[point.index(board.size())] => {
                let mut queue = VecDeque::from([point]);
                let mut region_size = 0usize;
                let mut touches_black = false;
                let mut touches_white = false;
                visited[point.index(board.size())] = true;
                while let Some(empty) = queue.pop_front() {
                    region_size += 1;
                    for neighbor in board.neighbors(empty) {
                        match board.get(neighbor) {
                            Some(Color::Black) => touches_black = true,
                            Some(Color::White) => touches_white = true,
                            None => {
                                let index = neighbor.index(board.size());
                                if !visited[index] {
                                    visited[index] = true;
                                    queue.push_back(neighbor);
                                }
                            }
                        }
                    }
                }
                if touches_black && !touches_white {
                    black_area += region_size;
                } else if touches_white && !touches_black {
                    white_area += region_size;
                }
            }
            None => {}
        }
    }
    Ok(black_area as f32 - white_area as f32 - rules.komi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Point;
    use crate::rules::{KoRule, SuicideRule};

    #[test]
    fn scores_empty_and_single_color_boards() {
        let rules = Ruleset::new(
            ScoringRule::Area,
            KoRule::PositionalSuperKo,
            SuicideRule::Allowed,
            7.5,
        );
        let mut board = Board::new(5).unwrap();
        assert_eq!(final_area_score(&board, rules).unwrap(), -7.5);
        board.set(Point::new(2, 2), Some(Color::Black)).unwrap();
        assert_eq!(final_area_score(&board, rules).unwrap(), 17.5);
    }

    #[test]
    fn rejects_non_area_rules() {
        let rules = Ruleset::new(
            ScoringRule::AreaAncientChinese,
            KoRule::PositionalSuperKo,
            SuicideRule::Allowed,
            7.5,
        );
        assert!(matches!(
            final_area_score(&Board::new(5).unwrap(), rules),
            Err(ScoringError::UnsupportedRule(
                ScoringRule::AreaAncientChinese
            ))
        ));
    }
}
