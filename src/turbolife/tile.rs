//! Tile data structures and helpers for TurboLife.
//!
//! Storage is split into parallel arrays on the arena:
//! - `TileCells`: double-buffered cell data (hot path for the kernel)
//! - `TileMeta`: flags and cached population
//! - `BorderData`: pre-extracted border rows/columns for ghost zone fills

pub const TILE_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileIdx(pub u32);

impl TileIdx {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The 8 cardinal and intercardinal directions for neighbor addressing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    North = 0, // (tx, ty+1)
    South = 1, // (tx, ty-1)
    West  = 2, // (tx-1, ty)
    East  = 3, // (tx+1, ty)
    NW    = 4, // (tx-1, ty+1)
    NE    = 5, // (tx+1, ty+1)
    SW    = 6, // (tx-1, ty-1)
    SE    = 7, // (tx+1, ty-1)
}

impl Direction {
    pub const ALL: [Direction; 8] = [
        Direction::North, Direction::South,
        Direction::West,  Direction::East,
        Direction::NW,    Direction::NE,
        Direction::SW,    Direction::SE,
    ];

    /// The coordinate offset for this direction.
    #[inline]
    pub const fn offset(self) -> (i64, i64) {
        match self {
            Direction::North => (0, 1),
            Direction::South => (0, -1),
            Direction::West  => (-1, 0),
            Direction::East  => (1, 0),
            Direction::NW    => (-1, 1),
            Direction::NE    => (1, 1),
            Direction::SW    => (-1, -1),
            Direction::SE    => (1, -1),
        }
    }

    /// The reverse direction (for bidirectional linking).
    #[inline]
    pub const fn reverse(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::West  => Direction::East,
            Direction::East  => Direction::West,
            Direction::NW    => Direction::SE,
            Direction::NE    => Direction::SW,
            Direction::SW    => Direction::NE,
            Direction::SE    => Direction::NW,
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Sentinel value for "no neighbor".
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Pre-bound neighbor indices for a tile. Indexed by `Direction`.
///
/// Uses raw u32 indices with `NO_NEIGHBOR` sentinel to keep the structure
/// compact and branch-light on hot paths.
pub type Neighbors = [u32; 8];

/// An empty neighbor array (all `NO_NEIGHBOR`).
pub const EMPTY_NEIGHBORS: Neighbors = [NO_NEIGHBOR; 8];

/// Pre-extracted border data from a tile's current buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct BorderData {
    pub north: u64,
    pub south: u64,
    pub west: u64,
    pub east: u64,
    pub nw: bool,
    pub ne: bool,
    pub sw: bool,
    pub se: bool,
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

/// Double-buffered cell data for a single tile.
/// Aligned to 64 bytes for cache-line friendliness.
#[derive(Clone, Debug)]
#[repr(C, align(64))]
pub struct TileCells {
    pub cells: [[u64; TILE_SIZE]; 2],
    pub phase: u8,
}

/// Per-tile metadata: flags and cached population.
#[derive(Clone, Debug)]
pub struct TileMeta {
    pub population: Option<u32>,
    /// Epoch when this tile was last marked active.
    pub active_epoch: u32,
    pub changed: bool,
    pub occupied: bool,
    pub in_changed_list: bool,
    pub has_live: bool,
}

impl TileMeta {
    pub fn empty() -> Self {
        Self {
            changed: true,
            active_epoch: 0,
            population: Some(0),
            occupied: true,
            in_changed_list: false,
            has_live: false,
        }
    }

    pub fn released() -> Self {
        Self {
            changed: false,
            active_epoch: 0,
            population: Some(0),
            occupied: false,
            in_changed_list: false,
            has_live: false,
        }
    }
}


impl TileCells {
    pub fn empty() -> Self {
        Self {
            cells: [[0; TILE_SIZE]; 2],
            phase: 0,
        }
    }

    #[inline]
    pub fn current(&self) -> &[u64; TILE_SIZE] {
        &self.cells[self.phase as usize]
    }

    #[inline]
    pub fn current_mut(&mut self) -> &mut [u64; TILE_SIZE] {
        &mut self.cells[self.phase as usize]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn next_mut(&mut self) -> &mut [u64; TILE_SIZE] {
        &mut self.cells[1 - self.phase as usize]
    }

    #[inline]
    pub fn current_and_next_mut(&mut self) -> (&[u64; TILE_SIZE], &mut [u64; TILE_SIZE]) {
        let (a, b) = self.cells.split_at_mut(1);
        if self.phase == 0 {
            (&a[0], &mut b[0])
        } else {
            (&b[0], &mut a[0])
        }
    }

    #[inline]
    pub fn swap(&mut self) {
        self.phase ^= 1;
    }

    #[inline]
    #[allow(dead_code)]
    pub fn has_live_cells(&self) -> bool {
        self.current().iter().any(|&row| row != 0)
    }

    #[inline]
    pub fn compute_population(&self) -> u32 {
        self.current().iter().map(|row| row.count_ones()).sum()
    }

    pub fn set_local_cell(&mut self, local_x: usize, local_y: usize, alive: bool) {
        let mask = 1u64 << local_x;
        let row = &mut self.current_mut()[local_y];
        if alive {
            *row |= mask;
        } else {
            *row &= !mask;
        }
    }

    #[inline]
    pub fn get_local_cell(&self, local_x: usize, local_y: usize) -> bool {
        (self.current()[local_y] >> local_x) & 1 == 1
    }

    /// Recompute border data from the current buffer.
    /// Called after set_cell mutations (not in the hot path).
    pub fn recompute_border(&self) -> BorderData {
        let current = self.current();
        let mut west = 0u64;
        let mut east = 0u64;
        for (row_index, row) in current.iter().enumerate() {
            west |= (row & 1) << row_index;
            east |= ((row >> 63) & 1) << row_index;
        }
        BorderData {
            north: current[63],
            south: current[0],
            west,
            east,
            nw: (current[63] & 1) != 0,
            ne: ((current[63] >> 63) & 1) != 0,
            sw: (current[0] & 1) != 0,
            se: ((current[0] >> 63) & 1) != 0,
        }
    }
}
