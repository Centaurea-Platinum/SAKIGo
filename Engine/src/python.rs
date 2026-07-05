//! PyO3 bindings (PARKED, not compiled): expose the engine's game state to
//! Python as `sakigo_engine.Game`, matching the vocabulary of the Phase 1
//! generator's pure-Python `Game` (SAKIGo rule-name strings, BLACK=1/WHITE=-1
//! ints, row-major point indices, pass = index `area`).
//!
//! This file is intentionally unreferenced from `lib.rs`: declaring the
//! optional `pyo3` dependency breaks offline `cargo test` (lock resolution
//! needs the registry), and the crates mirror was unreachable when this was
//! written (2026-07-06). Wiring steps to activate it are in `AI/Issues.md`.
//! NOTE: written against the pyo3 0.23 Bound API but never compiled - expect
//! to fix small API mismatches on first build.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::board::{Color, Point};
use crate::game::{GameState, GoMove};
use crate::rules::{KoRule, Ruleset, ScoringRule, SuicideRule};

fn parse_scoring(raw: &str) -> PyResult<ScoringRule> {
    match raw {
        "area" => Ok(ScoringRule::Area),
        "area_ancient_chinese" => Ok(ScoringRule::AreaAncientChinese),
        "territory" => Ok(ScoringRule::Territory),
        "territory_with_seki_score" => Ok(ScoringRule::TerritoryWithSekiScore),
        _ => Err(PyValueError::new_err(format!("unknown scoring rule {raw:?}"))),
    }
}

fn parse_ko(raw: &str) -> PyResult<KoRule> {
    match raw {
        "simple_ko" => Ok(KoRule::SimpleKo),
        "positional_superko" => Ok(KoRule::PositionalSuperKo),
        _ => Err(PyValueError::new_err(format!("unknown ko rule {raw:?}"))),
    }
}

fn parse_suicide(raw: &str) -> PyResult<SuicideRule> {
    match raw {
        "allowed" => Ok(SuicideRule::Allowed),
        "forbidden" => Ok(SuicideRule::Forbidden),
        _ => Err(PyValueError::new_err(format!("unknown suicide rule {raw:?}"))),
    }
}

fn color_to_int(color: Color) -> i8 {
    match color {
        Color::Black => 1,
        Color::White => -1,
    }
}

#[pyclass(name = "Game")]
struct PyGame {
    state: GameState,
    board_size: usize,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (board_size, scoring, ko, suicide, komi=7.5))]
    fn new(board_size: usize, scoring: &str, ko: &str, suicide: &str, komi: f32) -> PyResult<Self> {
        let rules = Ruleset::new(
            parse_scoring(scoring)?,
            parse_ko(ko)?,
            parse_suicide(suicide)?,
            komi,
        );
        let state = GameState::new(board_size, rules)
            .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        Ok(Self { state, board_size })
    }

    #[getter]
    fn board_size(&self) -> usize {
        self.board_size
    }

    #[getter]
    fn to_move(&self) -> i8 {
        color_to_int(self.state.to_move())
    }

    /// `[black_captures, white_captures]`, stones captured BY that color.
    #[getter]
    fn captures(&self) -> (usize, usize) {
        (
            self.state.captured_by(Color::Black),
            self.state.captured_by(Color::White),
        )
    }

    /// Row-major cells: 1 = black, -1 = white, 0 = empty.
    fn board(&self) -> Vec<i8> {
        self.state
            .board()
            .cells()
            .iter()
            .map(|cell| cell.map_or(0, color_to_int))
            .collect()
    }

    /// Length `area + 1`; the final entry is pass (always legal).
    fn legal_mask(&self) -> Vec<bool> {
        let size = self.board_size;
        let mut mask: Vec<bool> = (0..size * size)
            .map(|index| self.state.is_legal_point(Point::from_index(size, index)))
            .collect();
        mask.push(true);
        mask
    }

    /// Play a row-major point index, or `area` for pass.
    fn play(&mut self, action: usize) -> PyResult<()> {
        let go_move = if action == self.board_size * self.board_size {
            GoMove::Pass
        } else {
            GoMove::Play(Point::from_index(self.board_size, action))
        };
        self.state
            .play(go_move)
            .map(|_| ())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Simple-ko banned point index, if any.
    #[getter]
    fn simple_ko(&self) -> Option<usize> {
        self.state
            .simple_ko_point()
            .map(|point| point.index(self.board_size))
    }

    /// Board-only repetition hash (positional superko semantics).
    fn position_hash(&self) -> u128 {
        self.state.position_hash().0
    }

    /// Metadata-aware transposition/NN-cache key.
    fn state_hash(&self) -> u128 {
        self.state.state_hash().0
    }
}

#[pymodule]
fn sakigo_engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGame>()?;
    Ok(())
}
