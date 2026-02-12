//! Tile data structures and helpers for TurboLife.
//!
//! Storage is split into parallel arrays on the arena:
//! - Cell buffers: two separate `Vec<[u64; 64]>` with a global phase bit
//!   (halves the working set during kernel — only current is read, only next is written)
//! - `TileMeta`: packed flags in a single u8 bitfield for cache density
//! - `BorderData`: corners packed into a u8 bitfield

pub const TILE_SIZE: usize = 64;
pub const POPULATION_UNKNOWN: u32 = u32::MAX;

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
pub const NO_NEIGHBOR: u32 = u32::MAX;

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
        }
    }
}

impl BorderData {
    pub const CORNER_NW: u8 = 1 << 0;
    pub const CORNER_NE: u8 = 1 << 1;
    pub const CORNER_SW: u8 = 1 << 2;
    pub const CORNER_SE: u8 = 1 << 3;

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

/// Per-tile metadata with flags packed into a single byte.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TileMeta {
    pub population: u32,
    pub active_epoch: u32,
    pub flags: u8,
}

impl TileMeta {
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
        if v {
            self.flags |= FLAG_CHANGED
        } else {
            self.flags &= !FLAG_CHANGED
        }
    }
    #[inline(always)]
    #[allow(dead_code)]
    pub fn set_occupied(&mut self, v: bool) {
        if v {
            self.flags |= FLAG_OCCUPIED
        } else {
            self.flags &= !FLAG_OCCUPIED
        }
    }
    #[inline(always)]
    pub fn set_in_changed_list(&mut self, v: bool) {
        if v {
            self.flags |= FLAG_IN_CHANGED_LIST
        } else {
            self.flags &= !FLAG_IN_CHANGED_LIST
        }
    }
    #[inline(always)]
    pub fn set_has_live(&mut self, v: bool) {
        if v {
            self.flags |= FLAG_HAS_LIVE
        } else {
            self.flags &= !FLAG_HAS_LIVE
        }
    }

    pub fn empty() -> Self {
        Self {
            population: 0,
            active_epoch: 0,
            flags: FLAG_CHANGED | FLAG_OCCUPIED,
        }
    }

    pub fn released() -> Self {
        Self {
            population: 0,
            active_epoch: 0,
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
pub fn compute_population(buf: &[u64; TILE_SIZE]) -> u32 {
    buf.iter().map(|row| row.count_ones()).sum()
}

/// Recompute border data from a cell buffer.
#[inline]
pub fn recompute_border(buf: &[u64; TILE_SIZE]) -> BorderData {
    let mut west = 0u64;
    let mut east = 0u64;
    for (row_index, &row) in buf.iter().enumerate() {
        west |= (row & 1) << row_index;
        east |= ((row >> 63) & 1) << row_index;
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
    BorderData {
        north: buf[63],
        south: buf[0],
        west,
        east,
        corners,
    }
}

/// Set a cell in a buffer.
#[inline]
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
