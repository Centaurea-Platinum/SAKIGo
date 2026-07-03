use crate::board::{Board, Color};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PositionHash(pub u64);

pub fn hash_board(board: &Board) -> PositionHash {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn write(mut hash: u64, value: u64) -> u64 {
        hash ^= value;
        hash.wrapping_mul(FNV_PRIME)
    }

    let mut hash = write(FNV_OFFSET, board.size() as u64);
    for cell in board.cells() {
        let value = match cell {
            None => 0,
            Some(Color::Black) => 1,
            Some(Color::White) => 2,
        };
        hash = write(hash, value);
    }
    PositionHash(hash)
}
