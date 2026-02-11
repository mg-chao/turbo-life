//! Tile arena with split storage for TurboLife.
//!
//! Cell data, metadata, and borders are stored in separate parallel `Vec`s.
//! Borders are double-buffered: `borders[border_phase]` holds current-gen
//! borders (read by ghost fills), `borders[1 - border_phase]` is written
//! by the kernel. After the parallel phase, `border_phase` flips.
//! This eliminates the post-parallel fixup loop entirely.

use rustc_hash::FxHashMap;

use super::tile::{BorderData, Direction, Neighbors, EMPTY_NEIGHBORS, TileCells, TileIdx, TileMeta};

pub struct TileArena {
    pub cell_data: Vec<TileCells>,
    pub meta: Vec<TileMeta>,
    pub neighbors: Vec<Neighbors>,
    /// Tile coordinates, parallel to cell_data/meta.
    pub coords: Vec<(i64, i64)>,
    /// Double-buffered borders. `borders[border_phase]` = current gen (read),
    /// `borders[1 - border_phase]` = next gen (written by kernel).
    pub borders: [Vec<BorderData>; 2],
    /// Which border buffer is current. Flipped after each parallel step.
    pub border_phase: usize,
    pub coord_to_idx: FxHashMap<(i64, i64), TileIdx>,
    pub free_list: Vec<TileIdx>,
    pub changed_list: Vec<TileIdx>,

    /// Epoch counter for active-set tracking. Incremented each step.
    /// A tile is active iff `meta[i].active_epoch == active_epoch`.
    pub active_epoch: u8,

    pub active_set: Vec<TileIdx>,
    pub expand_buf: Vec<(i64, i64)>,
    pub prune_buf: Vec<TileIdx>,
    pub changed_scratch: Vec<TileIdx>,
}

impl TileArena {
    pub fn new() -> Self {
        Self {
            cell_data: Vec::new(),
            meta: Vec::new(),
            neighbors: Vec::new(),
            coords: Vec::new(),
            borders: [Vec::new(), Vec::new()],
            border_phase: 0,
            coord_to_idx: FxHashMap::default(),
            free_list: Vec::new(),
            changed_list: Vec::new(),
            active_epoch: 1, // Start at 1 so default meta (epoch=0) is inactive
            active_set: Vec::new(),
            expand_buf: Vec::new(),
            prune_buf: Vec::new(),
            changed_scratch: Vec::new(),
        }
    }

    /// Current-gen border for a tile (read by ghost fills).
    #[inline]
    #[allow(dead_code)]
    pub fn border(&self, idx: TileIdx) -> &BorderData {
        &self.borders[self.border_phase][idx.index()]
    }

    /// Mutable ref to current-gen border (for set_cell mutations).
    #[inline]
    pub fn border_mut(&mut self, idx: TileIdx) -> &mut BorderData {
        &mut self.borders[self.border_phase][idx.index()]
    }

    /// Next-gen border buffer (written by kernel during parallel phase).
    #[inline]
    #[allow(dead_code)]
    pub fn next_border_mut(&mut self, idx: TileIdx) -> &mut BorderData {
        &mut self.borders[1 - self.border_phase][idx.index()]
    }

    /// Flip border phase after parallel compute completes.
    #[inline]
    pub fn flip_borders(&mut self) {
        self.border_phase = 1 - self.border_phase;
    }

    #[inline]
    #[allow(dead_code)]
    pub fn neighbors(&self, idx: TileIdx) -> &Neighbors {
        &self.neighbors[idx.index()]
    }

    #[inline]
    pub fn idx_at(&self, coord: (i64, i64)) -> Option<TileIdx> {
        self.coord_to_idx.get(&coord).copied()
    }

    #[inline]
    pub fn cells(&self, idx: TileIdx) -> &TileCells {
        &self.cell_data[idx.index()]
    }

    #[inline]
    pub fn cells_mut(&mut self, idx: TileIdx) -> &mut TileCells {
        &mut self.cell_data[idx.index()]
    }

    #[inline]
    pub fn meta(&self, idx: TileIdx) -> &TileMeta {
        &self.meta[idx.index()]
    }

    #[inline]
    pub fn meta_mut(&mut self, idx: TileIdx) -> &mut TileMeta {
        &mut self.meta[idx.index()]
    }

    #[inline]
    pub fn mark_changed(&mut self, idx: TileIdx) {
        let m = &mut self.meta[idx.index()];
        m.changed = true;
        if !m.in_changed_list {
            m.in_changed_list = true;
            self.changed_list.push(idx);
        }
    }

    pub fn allocate(&mut self, coord: (i64, i64)) -> TileIdx {
        if let Some(existing) = self.idx_at(coord) {
            return existing;
        }

        let idx = if let Some(recycled) = self.free_list.pop() {
            self.cell_data[recycled.index()] = TileCells::empty();
            self.meta[recycled.index()] = TileMeta::empty();
            self.neighbors[recycled.index()] = EMPTY_NEIGHBORS;
            self.coords[recycled.index()] = coord;
            self.borders[0][recycled.index()] = BorderData::default();
            self.borders[1][recycled.index()] = BorderData::default();
            recycled
        } else {
            let idx = TileIdx(self.cell_data.len() as u32);
            self.cell_data.push(TileCells::empty());
            self.meta.push(TileMeta::empty());
            self.neighbors.push(EMPTY_NEIGHBORS);
            self.coords.push(coord);
            self.borders[0].push(BorderData::default());
            self.borders[1].push(BorderData::default());
            idx
        };

        self.coord_to_idx.insert(coord, idx);

        for dir in Direction::ALL {
            let (dx, dy) = dir.offset();
            let neighbor_coord = (coord.0 + dx, coord.1 + dy);
            if let Some(neighbor_idx) = self.coord_to_idx.get(&neighbor_coord).copied() {
                self.neighbors[idx.index()][dir.index()] = Some(neighbor_idx);
                self.neighbors[neighbor_idx.index()][dir.reverse().index()] = Some(idx);
            }
        }

        idx
    }

    pub fn release(&mut self, idx: TileIdx) {
        if !self.meta[idx.index()].occupied {
            return;
        }

        for dir in Direction::ALL {
            if let Some(neighbor_idx) = self.neighbors[idx.index()][dir.index()] {
                self.neighbors[neighbor_idx.index()][dir.reverse().index()] = None;
            }
        }
        self.neighbors[idx.index()] = EMPTY_NEIGHBORS;

        let coord = self.coords[idx.index()];
        self.coord_to_idx.remove(&coord);
        self.cell_data[idx.index()].cells = [[0; 64]; 2];
        self.borders[0][idx.index()] = BorderData::default();
        self.borders[1][idx.index()] = BorderData::default();
        self.meta[idx.index()] = TileMeta::released();
        self.free_list.push(idx);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn slot_count(&self) -> usize {
        self.cell_data.len()
    }

    #[inline]
    pub fn ensure_neighbor(&mut self, tx: i64, ty: i64) {
        let coord = (tx, ty);
        if self.idx_at(coord).is_some() {
            return;
        }
        let idx = self.allocate(coord);
        self.meta[idx.index()].population = Some(0);
        self.mark_changed(idx);
    }
}
