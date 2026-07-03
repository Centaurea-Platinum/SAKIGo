use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::fmt;

pub const MIN_BOARD_SIZE: usize = 1;
pub const MAX_BOARD_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Black => 0,
            Self::White => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub row: usize,
    pub col: usize,
}

impl Point {
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    pub fn index(self, board_size: usize) -> usize {
        self.row * board_size + self.col
    }

    pub fn from_index(board_size: usize, index: usize) -> Self {
        Self {
            row: index / board_size,
            col: index % board_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoardError {
    SizeOutOfRange { size: usize },
    WrongCellCount { expected: usize, actual: usize },
    PointOutOfBounds { point: Point, board_size: usize },
}

impl fmt::Display for BoardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOutOfRange { size } => write!(
                formatter,
                "board size {size} is outside the supported range {MIN_BOARD_SIZE}..={MAX_BOARD_SIZE}"
            ),
            Self::WrongCellCount { expected, actual } => {
                write!(formatter, "expected {expected} board cells, got {actual}")
            }
            Self::PointOutOfBounds { point, board_size } => write!(
                formatter,
                "point ({}, {}) is outside a {board_size}x{board_size} board",
                point.row, point.col
            ),
        }
    }
}

impl Error for BoardError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    size: usize,
    cells: Vec<Option<Color>>,
}

impl Board {
    pub fn new(size: usize) -> Result<Self, BoardError> {
        if !(MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&size) {
            return Err(BoardError::SizeOutOfRange { size });
        }
        Ok(Self {
            size,
            cells: vec![None; size * size],
        })
    }

    pub fn from_cells(size: usize, cells: Vec<Option<Color>>) -> Result<Self, BoardError> {
        if !(MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&size) {
            return Err(BoardError::SizeOutOfRange { size });
        }
        let expected = size * size;
        let actual = cells.len();
        if actual != expected {
            return Err(BoardError::WrongCellCount { expected, actual });
        }
        Ok(Self { size, cells })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn area(&self) -> usize {
        self.cells.len()
    }

    pub fn cells(&self) -> &[Option<Color>] {
        &self.cells
    }

    pub fn contains(&self, point: Point) -> bool {
        point.row < self.size && point.col < self.size
    }

    pub fn get(&self, point: Point) -> Option<Color> {
        if !self.contains(point) {
            return None;
        }
        self.cells[point.index(self.size)]
    }

    pub fn set(&mut self, point: Point, value: Option<Color>) -> Result<(), BoardError> {
        if !self.contains(point) {
            return Err(BoardError::PointOutOfBounds {
                point,
                board_size: self.size,
            });
        }
        let index = point.index(self.size);
        self.cells[index] = value;
        Ok(())
    }

    pub fn is_empty(&self, point: Point) -> bool {
        self.contains(point) && self.get(point).is_none()
    }

    pub fn points(&self) -> impl Iterator<Item = Point> + '_ {
        (0..self.area()).map(move |index| Point::from_index(self.size, index))
    }

    pub fn neighbors(&self, point: Point) -> Vec<Point> {
        let mut neighbors = Vec::with_capacity(4);
        if point.row > 0 {
            neighbors.push(Point::new(point.row - 1, point.col));
        }
        if point.col > 0 {
            neighbors.push(Point::new(point.row, point.col - 1));
        }
        if point.row + 1 < self.size {
            neighbors.push(Point::new(point.row + 1, point.col));
        }
        if point.col + 1 < self.size {
            neighbors.push(Point::new(point.row, point.col + 1));
        }
        neighbors
    }

    pub fn group_at(&self, start: Point) -> Option<Group> {
        let color = self.get(start)?;
        let mut queue = VecDeque::from([start]);
        let mut visited = vec![false; self.area()];
        let mut stones = Vec::new();
        let mut liberties = HashSet::new();
        visited[start.index(self.size)] = true;

        while let Some(point) = queue.pop_front() {
            stones.push(point);
            for neighbor in self.neighbors(point) {
                match self.get(neighbor) {
                    Some(neighbor_color) if neighbor_color == color => {
                        let neighbor_index = neighbor.index(self.size);
                        if !visited[neighbor_index] {
                            visited[neighbor_index] = true;
                            queue.push_back(neighbor);
                        }
                    }
                    None => {
                        liberties.insert(neighbor);
                    }
                    Some(_) => {}
                }
            }
        }

        Some(Group {
            color,
            stones,
            liberties,
        })
    }

    pub fn remove_points(&mut self, points: &[Point]) -> usize {
        let mut removed = 0;
        for point in points {
            if self.contains(*point) {
                let index = point.index(self.size);
                if self.cells[index].is_some() {
                    self.cells[index] = None;
                    removed += 1;
                }
            }
        }
        removed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    color: Color,
    stones: Vec<Point>,
    liberties: HashSet<Point>,
}

impl Group {
    pub fn color(&self) -> Color {
        self.color
    }

    pub fn stones(&self) -> &[Point] {
        &self.stones
    }

    pub fn liberties(&self) -> &HashSet<Point> {
        &self.liberties
    }

    pub fn stone_count(&self) -> usize {
        self.stones.len()
    }

    pub fn liberty_count(&self) -> usize {
        self.liberties.len()
    }

    pub fn has_no_liberties(&self) -> bool {
        self.liberties.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_track_stones_and_liberties() {
        let mut board = Board::new(3).unwrap();
        board.set(Point::new(1, 1), Some(Color::Black)).unwrap();
        board.set(Point::new(1, 2), Some(Color::Black)).unwrap();
        board.set(Point::new(0, 1), Some(Color::White)).unwrap();

        let group = board.group_at(Point::new(1, 1)).unwrap();
        assert_eq!(group.color(), Color::Black);
        assert_eq!(group.stone_count(), 2);
        assert_eq!(group.liberty_count(), 4);
        assert!(group.liberties().contains(&Point::new(1, 0)));
    }
}
