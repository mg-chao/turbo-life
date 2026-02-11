//! Tile arena with split storage for TurboLife.
//!
//! Cell data is stored as two separate `Vec<CellBuf>` with a global phase bit,
//! halving the kernel working set. Borders are double-buffered.
//! Slot 0 is reserved as a sentinel (all zeros) so ghost-zone gathers
//! can use unconditional loads — NO_NEIGHBOR maps to the sentinel.

use rustc_hash::FxHashMap;

use super::tile::{
    BorderData, CellBuf, Direction, Neighbors, EMPTY_NEIGHBORS,
    TileIdx, TileMeta, TILE_SIZE, NO_NEIGHBOR,
};

/// Index of the sentinel slot (always zeroed).
pub const SENTINEL_IDX: usize = 0;

pub struct TileArena {
    /// Two cell buffers: `cell_bufs[phase]` = current (read), `cell_bufs[1-phase]` = next (write).
    pub cell_bufs: [Vec<CellBuf>; 2],
    /// Global cell phase. Flipped after each step.
    pub cell_phase: usize,
    pub meta: Vec<TileMeta>,
    pub neighbors: Vec<Neighbors>,
    pub coords: Vec<(i64, i64)>,
    /// Double-buffered borders. `borders[border_phase]` = current gen (read).
    pub borders: [Vec<BorderData>; 2],
    pub border_phase: usize,
    pub coord_to_idx: FxHashMap<(i64, i64), TileIdx>,
    pub free_list: Vec<TileIdx>,
    pub changed_list: Vec<TileIdx>,

    pub active_epoch: u32,
    pub active_set: Vec<TileIdx>,
    pub expand_buf: Vec<(i64, i64)>,
    pub prune_buf: Vec<TileIdx>,
    pub changed_scratch: Vec<TileIdx>,
}

impl TileArena {
    pub fn new() -> Self {
        // Slot 0 is the sentinel — always zeroed, never used for real tiles.
        let sentinel_cells = CellBuf::empty();
        let sentinel_meta = TileMeta::released();
        let sentinel_border = BorderData::default();
        let sentinel_neighbors = EMPTY_NEIGHBORS;
        let sentinel_coord = (i64::MIN, i64::MIN);

        Self {
            cell_bufs: [vec![sentinel_cells.clone()], vec![sentinel_cells]],
            cell_phase: 0,
            meta: vec![sentinel_meta],
            neighbors: vec![sentinel_neighbors],
            coords: vec![sentinel_coord],
            borders: [vec![sentinel_border], vec![sentinel_border]],
            border_phase: 0,
            coord_to_idx: FxHashMap::default(),
            free_list: Vec::new(),
            changed_list: Vec::new(),
            active_epoch: 1,
            active_set: Vec::new(),
            expand_buf: Vec::new(),
            prune_buf: Vec::new(),
            changed_scratch: Vec::new(),
        }
    }

    /// Current cell buffer (read side).
    #[inline(always)]
    pub fn current_buf(&self, idx: TileIdx) -> &[u64; TILE_SIZE] {
        &self.cell_bufs[self.cell_phase][idx.index()].0
    }

    /// Next cell buffer (write side).
    #[inline(always)]
    #[allow(dead_code)]
    pub fn next_buf_mut(&mut self, idx: TileIdx) -> &mut [u64; TILE_SIZE] {
        &mut self.cell_bufs[1 - self.cell_phase][idx.index()].0
    }

    /// Mutable ref to current cell buffer (for set_cell).
    #[inline(always)]
    pub fn current_buf_mut(&mut self, idx: TileIdx) -> &mut [u64; TILE_SIZE] {
        &mut self.cell_bufs[self.cell_phase][idx.index()].0
    }

    /// Flip cell phase (swap current/next).
    #[inline(always)]
    pub fn flip_cells(&mut self) {
        self.cell_phase = 1 - self.cell_phase;
    }

    /// Current-gen border for a tile.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn border(&self, idx: TileIdx) -> &BorderData {
        &self.borders[self.border_phase][idx.index()]
    }

    /// Mutable ref to current-gen border.
    #[inline(always)]
    pub fn border_mut(&mut self, idx: TileIdx) -> &mut BorderData {
        &mut self.borders[self.border_phase][idx.index()]
    }

    /// Flip border phase.
    #[inline(always)]
    pub fn flip_borders(&mut self) {
        self.border_phase = 1 - self.border_phase;
    }

    #[inline(always)]
    pub fn idx_at(&self, coord: (i64, i64)) -> Option<TileIdx> {
        self.coord_to_idx.get(&coord).copied()
    }

    #[inline(always)]
    pub fn meta(&self, idx: TileIdx) -> &TileMeta {
        &self.meta[idx.index()]
    }

    #[inline(always)]
    pub fn meta_mut(&mut self, idx: TileIdx) -> &mut TileMeta {
        &mut self.meta[idx.index()]
    }

    #[inline]
    pub fn mark_changed(&mut self, idx: TileIdx) {
        let m = &mut self.meta[idx.index()];
        m.set_changed(true);
        if !m.in_changed_list() {
            m.set_in_changed_list(true);
            self.changed_list.push(idx);
        }
    }

    pub fn allocate(&mut self, coord: (i64, i64)) -> TileIdx {
        if let Some(existing) = self.idx_at(coord) {
            return existing;
        }

        let idx = if let Some(recycled) = self.free_list.pop() {
            let i = recycled.index();
            self.cell_bufs[0][i] = CellBuf::empty();
            self.cell_bufs[1][i] = CellBuf::empty();
            self.meta[i] = TileMeta::empty();
            self.neighbors[i] = EMPTY_NEIGHBORS;
            self.coords[i] = coord;
            self.borders[0][i] = BorderData::default();
            self.borders[1][i] = BorderData::default();
            recycled
        } else {
            let idx = TileIdx(self.cell_bufs[0].len() as u32);
            self.cell_bufs[0].push(CellBuf::empty());
            self.cell_bufs[1].push(CellBuf::empty());
            self.meta.push(TileMeta::empty());
            self.neighbors.push(EMPTY_NEIGHBORS);
            self.coords.push(coord);
            self.borders[0].push(BorderData::default());
            self.borders[1].push(BorderData::default());

            // Keep sentinel's neighbor mapping valid: NO_NEIGHBOR (u32::MAX)
            // won't index into our vecs, but sentinel is at index 0 and
            // all neighbor entries for real tiles pointing to NO_NEIGHBOR
            // will be remapped to SENTINEL_IDX in the ghost gather.
            idx
        };

        self.coord_to_idx.insert(coord, idx);

        for dir in Direction::ALL {
            let (dx, dy) = dir.offset();
            let neighbor_coord = (coord.0 + dx, coord.1 + dy);
            if let Some(neighbor_idx) = self.coord_to_idx.get(&neighbor_coord).copied() {
                self.neighbors[idx.index()][dir.index()] = neighbor_idx.0;
                self.neighbors[neighbor_idx.index()][dir.reverse().index()] = idx.0;
            }
        }

        idx
    }

    pub fn release(&mut self, idx: TileIdx) {
        let i = idx.index();
        if !self.meta[i].occupied() {
            return;
        }

        for dir in Direction::ALL {
            let neighbor_raw = self.neighbors[i][dir.index()];
            if neighbor_raw != NO_NEIGHBOR {
                let neighbor_idx = TileIdx(neighbor_raw);
                self.neighbors[neighbor_idx.index()][dir.reverse().index()] = NO_NEIGHBOR;
            }
        }
        self.neighbors[i] = EMPTY_NEIGHBORS;

        let coord = self.coords[i];
        self.coord_to_idx.remove(&coord);
        self.cell_bufs[0][i] = CellBuf::empty();
        self.cell_bufs[1][i] = CellBuf::empty();
        self.borders[0][i] = BorderData::default();
        self.borders[1][i] = BorderData::default();
        self.meta[i] = TileMeta::released();
        self.free_list.push(idx);
    }

    #[inline]
    pub fn ensure_neighbor(&mut self, tx: i64, ty: i64) {
        let coord = (tx, ty);
        if self.idx_at(coord).is_some() {
            return;
        }
        let idx = self.allocate(coord);
        let m = &mut self.meta[idx.index()];
        m.population = 0;
        m.set_has_live(false);
        self.mark_changed(idx);
    }
}
