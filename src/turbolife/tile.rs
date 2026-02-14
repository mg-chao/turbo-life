//! Tile data structures and helpers for TurboLife.
//!
//! Storage is split into parallel arrays on the arena:
//! - Cell buffers: two separate `Vec<[u64; 64]>` with a global phase bit
//!   (halves the working set during kernel — only current is read, only next is written)
//! - `TileMeta`: packed flags in a single u8 bitfield for cache density
//! - `BorderData`: corners packed into a u8 bitfield

pub const TILE_SIZE: usize = 64;
pub const POPULATION_UNKNOWN: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileIdx(pub u32);

impl TileIdx {
    #[inline(always)]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The 8 cardinal and intercardinal directions for neighbor addressing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    North = 0,
    South = 1,
    West = 2,
    East = 3,
    NW = 4,
    NE = 5,
    SW = 6,
    SE = 7,
}

impl Direction {
    #[allow(dead_code)]
    pub const ALL: [Direction; 8] = [
        Direction::North,
        Direction::South,
        Direction::West,
        Direction::East,
        Direction::NW,
        Direction::NE,
        Direction::SW,
        Direction::SE,
    ];

    #[inline(always)]
    #[allow(dead_code)]
    pub const fn offset(self) -> (i64, i64) {
        match self {
            Direction::North => (0, 1),
            Direction::South => (0, -1),
            Direction::West => (-1, 0),
            Direction::East => (1, 0),
            Direction::NW => (-1, 1),
            Direction::NE => (1, 1),
            Direction::SW => (-1, -1),
            Direction::SE => (1, -1),
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub const fn reverse(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
            Direction::East => Direction::West,
            Direction::NW => Direction::SE,
            Direction::NE => Direction::SW,
            Direction::SW => Direction::NE,
            Direction::SE => Direction::NW,
        }
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Sentinel value for "no neighbor".
///
/// Index 0 is reserved for the arena sentinel tile, so missing neighbors map
/// directly to slot 0 without remapping in the hot kernel path.
pub const NO_NEIGHBOR: u32 = 0;
const _: [(); 1] = [(); (NO_NEIGHBOR == 0) as usize];

pub type Neighbors = [u32; 8];
pub const EMPTY_NEIGHBORS: Neighbors = [NO_NEIGHBOR; 8];

/// Pre-extracted border data from a tile's current buffer.
/// Corners packed into a u8 bitfield: bit0=nw, bit1=ne, bit2=sw, bit3=se.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BorderData {
    pub north: u64,
    pub south: u64,
    pub west: u64,
    pub east: u64,
    pub corners: u8,
    /// Bitset for fast frontier checks:
    /// 0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE.
    pub live_mask: u8,
}

impl Default for BorderData {
    #[inline(always)]
    fn default() -> Self {
        Self {
            north: 0,
            south: 0,
            west: 0,
            east: 0,
            corners: 0,
            live_mask: 0,
        }
    }
}

impl BorderData {
    pub const CORNER_NW: u8 = 1 << 0;
    pub const CORNER_NE: u8 = 1 << 1;
    pub const CORNER_SW: u8 = 1 << 2;
    pub const CORNER_SE: u8 = 1 << 3;

    #[inline(always)]
    pub const fn compute_live_mask(
        north: u64,
        south: u64,
        west: u64,
        east: u64,
        corners: u8,
    ) -> u8 {
        ((north != 0) as u8)
            | (((south != 0) as u8) << 1)
            | (((west != 0) as u8) << 2)
            | (((east != 0) as u8) << 3)
            | (((corners & Self::CORNER_NW != 0) as u8) << 4)
            | (((corners & Self::CORNER_NE != 0) as u8) << 5)
            | (((corners & Self::CORNER_SW != 0) as u8) << 6)
            | (((corners & Self::CORNER_SE != 0) as u8) << 7)
    }

    #[inline(always)]
    pub fn refresh_live_mask(&mut self) {
        self.live_mask =
            Self::compute_live_mask(self.north, self.south, self.west, self.east, self.corners);
    }

    #[inline(always)]
    pub fn from_parts(north: u64, south: u64, west: u64, east: u64, corners: u8) -> Self {
        Self {
            north,
            south,
            west,
            east,
            corners,
            live_mask: Self::compute_live_mask(north, south, west, east, corners),
        }
    }

    #[inline(always)]
    pub fn nw(self) -> bool {
        self.corners & Self::CORNER_NW != 0
    }
    #[inline(always)]
    pub fn ne(self) -> bool {
        self.corners & Self::CORNER_NE != 0
    }
    #[inline(always)]
    pub fn sw(self) -> bool {
        self.corners & Self::CORNER_SW != 0
    }
    #[inline(always)]
    pub fn se(self) -> bool {
        self.corners & Self::CORNER_SE != 0
    }
}

/// One-cell halo gathered from 8 neighbors before compute.
#[derive(Clone, Copy, Debug, Default)]
pub struct GhostZone {
    pub north: u64,
    pub south: u64,
    pub west: u64,
    pub east: u64,
    pub nw: bool,
    pub ne: bool,
    pub sw: bool,
    pub se: bool,
}

// ── TileMeta with packed flags ──────────────────────────────────────────

const FLAG_CHANGED: u8 = 1 << 0;
const FLAG_OCCUPIED: u8 = 1 << 1;
const FLAG_IN_CHANGED_LIST: u8 = 1 << 2;
const FLAG_HAS_LIVE: u8 = 1 << 3;
pub const MISSING_ALL_NEIGHBORS: u8 = 0xFF;

/// Per-tile metadata with flags packed into a single byte.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TileMeta {
    pub active_epoch: u32,
    pub population: u16,
    /// Bit i = 1 means neighbor direction i is currently missing.
    /// Direction layout matches `Neighbors` / `Direction` indices.
    pub missing_mask: u8,
    pub flags: u8,
}

impl TileMeta {
    #[inline(always)]
    fn set_flag(&mut self, flag: u8, v: bool) {
        let mask = ((v as u8).wrapping_neg()) & flag;
        self.flags = (self.flags & !flag) | mask;
    }

    #[inline(always)]
    pub fn changed(self) -> bool {
        self.flags & FLAG_CHANGED != 0
    }
    #[inline(always)]
    pub fn occupied(self) -> bool {
        self.flags & FLAG_OCCUPIED != 0
    }
    #[inline(always)]
    pub fn in_changed_list(self) -> bool {
        self.flags & FLAG_IN_CHANGED_LIST != 0
    }
    #[inline(always)]
    pub fn has_live(self) -> bool {
        self.flags & FLAG_HAS_LIVE != 0
    }

    #[inline(always)]
    pub fn set_changed(&mut self, v: bool) {
        self.set_flag(FLAG_CHANGED, v);
    }
    #[inline(always)]
    #[allow(dead_code)]
    pub fn set_occupied(&mut self, v: bool) {
        self.set_flag(FLAG_OCCUPIED, v);
    }
    #[inline(always)]
    pub fn set_in_changed_list(&mut self, v: bool) {
        self.set_flag(FLAG_IN_CHANGED_LIST, v);
    }
    #[inline(always)]
    pub fn set_has_live(&mut self, v: bool) {
        self.set_flag(FLAG_HAS_LIVE, v);
    }

    pub fn empty() -> Self {
        Self {
            active_epoch: 0,
            population: 0,
            missing_mask: MISSING_ALL_NEIGHBORS,
            flags: FLAG_CHANGED | FLAG_OCCUPIED,
        }
    }

    pub fn released() -> Self {
        Self {
            active_epoch: 0,
            population: 0,
            missing_mask: MISSING_ALL_NEIGHBORS,
            flags: 0,
        }
    }
}

// ── Cell buffer helpers (no longer a struct — arena owns two Vec<CellBuf>) ──

/// A single 64×64 cell buffer, cache-line aligned.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct CellBuf(pub [u64; TILE_SIZE]);

impl CellBuf {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self([0u64; TILE_SIZE])
    }
}

impl Default for CellBuf {
    fn default() -> Self {
        Self::empty()
    }
}

/// Compute population of a cell buffer.
#[inline]
pub fn compute_population(buf: &[u64; TILE_SIZE]) -> u16 {
    let pop: u32 = buf.iter().map(|row| row.count_ones()).sum();
    debug_assert!(pop <= u16::MAX as u32);
    pop as u16
}

/// Recompute border data from a cell buffer.
/// Also returns whether the buffer has any live cells (fused into the same loop).
#[inline]
pub fn recompute_border_and_has_live(buf: &[u64; TILE_SIZE]) -> (BorderData, bool) {
    let mut west = 0u64;
    let mut east = 0u64;
    let mut any_live = 0u64;
    for (row_index, &row) in buf.iter().enumerate() {
        west |= (row & 1) << row_index;
        east |= ((row >> 63) & 1) << row_index;
        any_live |= row;
    }
    let mut corners = 0u8;
    if (buf[63] & 1) != 0 {
        corners |= BorderData::CORNER_NW;
    }
    if ((buf[63] >> 63) & 1) != 0 {
        corners |= BorderData::CORNER_NE;
    }
    if (buf[0] & 1) != 0 {
        corners |= BorderData::CORNER_SW;
    }
    if ((buf[0] >> 63) & 1) != 0 {
        corners |= BorderData::CORNER_SE;
    }
    (
        BorderData::from_parts(buf[63], buf[0], west, east, corners),
        any_live != 0,
    )
}

/// Recompute border data from a cell buffer.
#[inline]
#[allow(dead_code)]
pub fn recompute_border(buf: &[u64; TILE_SIZE]) -> BorderData {
    recompute_border_and_has_live(buf).0
}

/// Set a cell in a buffer.
#[inline]
#[allow(dead_code)]
pub fn set_local_cell(buf: &mut [u64; TILE_SIZE], local_x: usize, local_y: usize, alive: bool) {
    let mask = 1u64 << local_x;
    if alive {
        buf[local_y] |= mask;
    } else {
        buf[local_y] &= !mask;
    }
}

/// Get a cell from a buffer.
#[inline(always)]
pub fn get_local_cell(buf: &[u64; TILE_SIZE], local_x: usize, local_y: usize) -> bool {
    (buf[local_y] >> local_x) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::TileMeta;

    #[test]
    fn tile_meta_is_compact() {
        assert_eq!(std::mem::size_of::<TileMeta>(), 8);
    }
}
