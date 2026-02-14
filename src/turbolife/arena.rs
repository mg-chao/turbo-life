//! Tile arena with split storage for TurboLife.
//!
//! Cell data is stored as two separate `Vec<CellBuf>` with a global phase bit,
//! halving the kernel working set. Borders are double-buffered.
//! Slot 0 is reserved as a sentinel (all zeros) so ghost-zone gathers
//! can use unconditional loads — NO_NEIGHBOR maps to the sentinel.

use super::tile::{
    BorderData, CellBuf, EMPTY_NEIGHBORS, NO_NEIGHBOR, Neighbors, TILE_SIZE, TileIdx, TileMeta,
};
use super::tilemap::TileMap;

const INITIAL_TILE_CAPACITY: usize = 256;
const MIN_GROW_TILES: usize = 256;
const ACTIVE_SORT_RADIX_BUCKETS: usize = 1 << 16;

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
    /// Cached `BorderData::live_mask` for each border phase.
    pub border_live_masks: [Vec<u8>; 2],
    pub border_phase: usize,
    pub coord_to_idx: TileMap,
    pub free_list: Vec<TileIdx>,
    pub changed_list: Vec<TileIdx>,
    changed_bits: Vec<u64>,
    changed_bitmap_synced: bool,
    pub occupied_count: usize,
    pub occupied_bits: Vec<u64>,

    pub active_set: Vec<TileIdx>,
    active_epoch: u16,
    active_tags: Vec<u16>,
    pub active_sort_scratch: Vec<TileIdx>,
    pub active_sort_counts: Vec<u32>,
    /// Pending directional frontier-expansion candidates.
    /// Packed as `((src_idx as u32) << 3) | dir`.
    pub expand_buf: Vec<u32>,
    pub prune_buf: Vec<TileIdx>,
    pub prune_marks: Vec<u8>,
    pub prune_marks_words: Vec<u64>,
    pub changed_scratch: Vec<TileIdx>,
}

impl TileArena {
    pub fn new() -> Self {
        // Slot 0 is the sentinel – always zeroed, never used for real tiles.
        let sentinel_cells = CellBuf::empty();
        let sentinel_meta = TileMeta::released();
        let sentinel_border = BorderData::default();
        let sentinel_neighbors = EMPTY_NEIGHBORS;
        let sentinel_coord = (i64::MIN, i64::MIN);

        let mut cell_bufs0 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut cell_bufs1 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut meta = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut neighbors = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut coords = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut borders0 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut borders1 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut border_live_masks0 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut border_live_masks1 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);

        cell_bufs0.push(sentinel_cells.clone());
        cell_bufs1.push(sentinel_cells);
        meta.push(sentinel_meta);
        neighbors.push(sentinel_neighbors);
        coords.push(sentinel_coord);
        borders0.push(sentinel_border);
        borders1.push(sentinel_border);
        border_live_masks0.push(0);
        border_live_masks1.push(0);

        Self {
            cell_bufs: [cell_bufs0, cell_bufs1],
            cell_phase: 0,
            meta,
            neighbors,
            coords,
            borders: [borders0, borders1],
            border_live_masks: [border_live_masks0, border_live_masks1],
            border_phase: 0,
            coord_to_idx: TileMap::with_capacity(INITIAL_TILE_CAPACITY),
            free_list: Vec::new(),
            changed_list: Vec::new(),
            changed_bits: vec![0],
            changed_bitmap_synced: true,
            occupied_count: 0,
            occupied_bits: vec![0],
            active_set: Vec::new(),
            active_epoch: 1,
            active_tags: vec![0],
            active_sort_scratch: Vec::new(),
            active_sort_counts: vec![0u32; ACTIVE_SORT_RADIX_BUCKETS],
            expand_buf: Vec::new(),
            prune_buf: Vec::new(),
            prune_marks: Vec::new(),
            prune_marks_words: Vec::new(),
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

    /// Write current-gen border and keep cached live-mask in sync.
    #[inline(always)]
    pub fn set_current_border(&mut self, idx: TileIdx, border: BorderData) {
        let i = idx.index();
        self.borders[self.border_phase][i] = border;
        self.border_live_masks[self.border_phase][i] = border.live_mask();
    }

    /// Refresh cached live-mask from current-gen border after in-place edits.
    #[inline(always)]
    pub fn sync_current_border_live_mask(&mut self, idx: TileIdx) {
        let i = idx.index();
        self.border_live_masks[self.border_phase][i] =
            self.borders[self.border_phase][i].live_mask();
    }

    /// Flip border phase.
    #[inline(always)]
    pub fn flip_borders(&mut self) {
        self.border_phase = 1 - self.border_phase;
    }

    #[inline(always)]
    pub fn idx_at(&self, coord: (i64, i64)) -> Option<TileIdx> {
        self.coord_to_idx.get(coord.0, coord.1)
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
        self.sync_changed_bitmap_if_needed();
        let i = idx.index();
        if !self.changed_test_and_set(i) {
            self.changed_list.push(idx);
        }
    }

    #[inline(always)]
    pub(crate) fn mark_changed_new_unique(&mut self, idx: TileIdx) {
        if self.changed_bitmap_synced {
            let duplicate = self.changed_test_and_set(idx.index());
            debug_assert!(
                !duplicate,
                "mark_changed_new_unique received a duplicate tile index"
            );
            if duplicate {
                return;
            }
        }
        self.changed_list.push(idx);
    }

    #[inline(always)]
    pub fn push_changed_from_kernel(&mut self, idx: TileIdx) {
        self.changed_list.push(idx);
        self.changed_bitmap_synced = false;
    }

    #[inline(always)]
    pub(crate) fn mark_changed_bitmap_unsynced(&mut self) {
        self.changed_bitmap_synced = false;
    }

    #[inline]
    pub(crate) fn reserve_additional_tiles(&mut self, additional: usize) {
        if additional == 0 {
            return;
        }
        self.cell_bufs[0].reserve(additional);
        self.cell_bufs[1].reserve(additional);
        self.meta.reserve(additional);
        self.neighbors.reserve(additional);
        self.coords.reserve(additional);
        self.borders[0].reserve(additional);
        self.borders[1].reserve(additional);
        self.border_live_masks[0].reserve(additional);
        self.border_live_masks[1].reserve(additional);
        self.coord_to_idx.reserve(additional);
        let target_slots = self.meta.len().saturating_add(additional);
        let target_words = target_slots.div_ceil(64);
        if self.occupied_bits.len() < target_words {
            self.occupied_bits
                .reserve(target_words - self.occupied_bits.len());
        }
        if self.changed_bits.len() < target_words {
            self.changed_bits
                .reserve(target_words - self.changed_bits.len());
        }
        if self.active_tags.len() < target_slots {
            self.active_tags
                .reserve(target_slots - self.active_tags.len());
        }
    }

    #[inline(always)]
    fn ensure_changed_bit_capacity(&mut self, idx: usize) {
        let word = idx >> 6;
        if word >= self.changed_bits.len() {
            self.changed_bits.resize(word + 1, 0);
        }
    }

    #[inline(always)]
    fn changed_test_and_set(&mut self, idx: usize) -> bool {
        self.ensure_changed_bit_capacity(idx);
        let word = idx >> 6;
        let bit = 1u64 << (idx & 63);
        let old = self.changed_bits[word];
        self.changed_bits[word] = old | bit;
        (old & bit) != 0
    }

    #[inline(always)]
    fn sync_changed_bitmap_if_needed(&mut self) {
        if self.changed_bitmap_synced {
            return;
        }
        let len = self.changed_list.len();
        for i in 0..len {
            let idx = self.changed_list[i].index();
            self.ensure_changed_bit_capacity(idx);
            let word = idx >> 6;
            self.changed_bits[word] |= 1u64 << (idx & 63);
        }
        self.changed_bitmap_synced = true;
    }

    #[inline(always)]
    pub(crate) fn clear_changed_mark(&mut self, idx: usize) {
        let word = idx >> 6;
        if word >= self.changed_bits.len() {
            return;
        }
        self.changed_bits[word] &= !(1u64 << (idx & 63));
    }

    #[inline(always)]
    fn ensure_active_tag_capacity(&mut self, idx: usize) {
        if idx >= self.active_tags.len() {
            self.active_tags.resize(idx + 1, 0);
        }
    }

    #[inline(always)]
    pub(crate) fn begin_active_rebuild(&mut self) {
        self.active_epoch = self.active_epoch.wrapping_add(1);
        if self.active_epoch == 0 {
            self.active_epoch = 1;
            self.active_tags.fill(0);
        }
    }

    #[inline(always)]
    pub(crate) fn begin_active_rebuild_with_capacity(&mut self, required_slots: usize) {
        self.begin_active_rebuild();
        if self.active_tags.len() < required_slots {
            self.active_tags.resize(required_slots, 0);
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn active_test_and_set(&mut self, idx: usize) -> bool {
        self.ensure_active_tag_capacity(idx);
        let epoch = self.active_epoch;
        let tag = &mut self.active_tags[idx];
        if *tag == epoch {
            true
        } else {
            *tag = epoch;
            false
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn active_test_and_set_unchecked(&mut self, idx: usize) -> bool {
        debug_assert!(idx < self.active_tags.len());
        let epoch = self.active_epoch;
        let tag = unsafe { self.active_tags.get_unchecked_mut(idx) };
        if *tag == epoch {
            true
        } else {
            *tag = epoch;
            false
        }
    }

    #[inline]
    pub(crate) fn begin_changed_rebuild(&mut self) -> bool {
        let was_synced = self.changed_bitmap_synced;
        self.changed_bitmap_synced = true;
        was_synced
    }

    #[inline]
    fn ensure_growth_capacity(&mut self) {
        let len = self.cell_bufs[0].len();
        if len < self.cell_bufs[0].capacity() {
            return;
        }
        let grow_by = (len / 2).max(MIN_GROW_TILES);
        self.reserve_additional_tiles(grow_by);
    }

    #[inline(always)]
    fn set_occupied_bit(&mut self, idx: usize) {
        let word = idx >> 6;
        if word >= self.occupied_bits.len() {
            self.occupied_bits.resize(word + 1, 0);
        }
        self.occupied_bits[word] |= 1u64 << (idx & 63);
    }

    #[inline(always)]
    fn clear_occupied_bit(&mut self, idx: usize) {
        let word = idx >> 6;
        debug_assert!(word < self.occupied_bits.len());
        if let Some(bits) = self.occupied_bits.get_mut(word) {
            *bits &= !(1u64 << (idx & 63));
        }
    }

    #[inline]
    fn allocate_slot(&mut self, coord: (i64, i64)) -> TileIdx {
        if let Some(recycled) = self.free_list.pop() {
            let i = recycled.index();
            // Use ptr::write_bytes for fast zeroing of cell buffers (512 bytes each).
            unsafe {
                std::ptr::write_bytes(self.cell_bufs[0].as_mut_ptr().add(i), 0, 1);
                std::ptr::write_bytes(self.cell_bufs[1].as_mut_ptr().add(i), 0, 1);
            }
            self.meta[i] = TileMeta::empty();
            self.neighbors[i] = EMPTY_NEIGHBORS;
            self.coords[i] = coord;
            self.borders[0][i] = BorderData::default();
            self.borders[1][i] = BorderData::default();
            self.border_live_masks[0][i] = 0;
            self.border_live_masks[1][i] = 0;
            self.set_occupied_bit(i);
            self.ensure_active_tag_capacity(i);
            self.active_tags[i] = 0;
            recycled
        } else {
            self.ensure_growth_capacity();
            let idx = TileIdx(self.cell_bufs[0].len() as u32);
            self.cell_bufs[0].push(CellBuf::empty());
            self.cell_bufs[1].push(CellBuf::empty());
            self.meta.push(TileMeta::empty());
            self.neighbors.push(EMPTY_NEIGHBORS);
            self.coords.push(coord);
            self.borders[0].push(BorderData::default());
            self.borders[1].push(BorderData::default());
            self.border_live_masks[0].push(0);
            self.border_live_masks[1].push(0);
            self.active_tags.push(0);
            self.set_occupied_bit(idx.index());
            idx
        }
    }

    #[inline]
    fn link_neighbors(&mut self, idx: TileIdx, coord: (i64, i64)) {
        let (cx, cy) = coord;
        let idx_val = idx.0;
        let i = idx.index();
        let neighbors_ptr = self.neighbors.as_mut_ptr();
        let meta_ptr = self.meta.as_mut_ptr();

        // Unrolled direction offsets: (dx, dy, dir_index, reverse_dir_index)
        const DIRS: [(i64, i64, usize, usize); 8] = [
            (0, 1, 0, 1),   // North -> reverse South
            (0, -1, 1, 0),  // South -> reverse North
            (-1, 0, 2, 3),  // West -> reverse East
            (1, 0, 3, 2),   // East -> reverse West
            (-1, 1, 4, 7),  // NW -> reverse SE
            (1, 1, 5, 6),   // NE -> reverse SW
            (-1, -1, 6, 5), // SW -> reverse NE
            (1, -1, 7, 4),  // SE -> reverse NW
        ];

        for &(dx, dy, dir_idx, rev_idx) in &DIRS {
            if let Some(neighbor_idx) = self.coord_to_idx.get(cx + dx, cy + dy) {
                let ni = neighbor_idx.index();
                // SAFETY: idx and neighbor_idx are valid arena indices.
                unsafe {
                    (*neighbors_ptr.add(i))[dir_idx] = neighbor_idx.0;
                    (*neighbors_ptr.add(ni))[rev_idx] = idx_val;
                    (*meta_ptr.add(i)).missing_mask &= !(1u8 << dir_idx);
                    (*meta_ptr.add(ni)).missing_mask &= !(1u8 << rev_idx);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn allocate_absent(&mut self, coord: (i64, i64)) -> TileIdx {
        debug_assert!(self.coord_to_idx.get(coord.0, coord.1).is_none());
        let idx = self.allocate_slot(coord);
        self.coord_to_idx.insert(coord.0, coord.1, idx);
        self.occupied_count += 1;

        self.link_neighbors(idx, coord);
        idx
    }

    #[allow(dead_code)]
    pub fn allocate(&mut self, coord: (i64, i64)) -> TileIdx {
        if let Some(existing) = self.idx_at(coord) {
            return existing;
        }

        self.allocate_absent(coord)
    }

    pub fn release(&mut self, idx: TileIdx) {
        let i = idx.index();
        if !self.meta[i].occupied() {
            return;
        }
        self.clear_occupied_bit(i);
        self.clear_changed_mark(i);
        self.ensure_active_tag_capacity(i);
        self.active_tags[i] = 0;

        // Unlink neighbors using raw pointer access.
        let neighbors_ptr = self.neighbors.as_mut_ptr();
        let meta_ptr = self.meta.as_mut_ptr();
        unsafe {
            let nb = &*neighbors_ptr.add(i);
            for dir_idx in 0..8u8 {
                let neighbor_raw = nb[dir_idx as usize];
                if neighbor_raw != NO_NEIGHBOR {
                    // Reverse direction index lookup table.
                    const REV: [usize; 8] = [1, 0, 3, 2, 7, 6, 5, 4];
                    let rev = REV[dir_idx as usize];
                    let ni = neighbor_raw as usize;
                    (*neighbors_ptr.add(ni))[rev] = NO_NEIGHBOR;
                    (*meta_ptr.add(ni)).missing_mask |= 1u8 << rev;
                }
            }
            (*neighbors_ptr.add(i)) = EMPTY_NEIGHBORS;
        }

        let coord = self.coords[i];
        self.coord_to_idx.remove(coord.0, coord.1);
        // Use ptr::write_bytes for fast zeroing of cell buffers.
        unsafe {
            std::ptr::write_bytes(self.cell_bufs[0].as_mut_ptr().add(i), 0, 1);
            std::ptr::write_bytes(self.cell_bufs[1].as_mut_ptr().add(i), 0, 1);
        }
        self.borders[0][i] = BorderData::default();
        self.borders[1][i] = BorderData::default();
        self.border_live_masks[0][i] = 0;
        self.border_live_masks[1][i] = 0;
        self.meta[i] = TileMeta::released();
        self.free_list.push(idx);
        self.occupied_count = self.occupied_count.saturating_sub(1);
    }

    #[inline]
    pub fn ensure_neighbor(&mut self, tx: i64, ty: i64) {
        let coord = (tx, ty);
        if self.idx_at(coord).is_some() {
            return;
        }
        let idx = self.allocate_absent(coord);
        let m = &mut self.meta[idx.index()];
        m.population = 0;
        m.set_has_live(false);
        self.mark_changed(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::TileArena;

    #[test]
    fn mark_changed_dedupes_after_kernel_queue_becomes_unsynced() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        arena.push_changed_from_kernel(idx);
        assert!(!arena.changed_bitmap_synced);
        assert_eq!(arena.changed_list.len(), 1);

        arena.mark_changed(idx);
        assert_eq!(arena.changed_list.len(), 1);
        assert!(arena.changed_bitmap_synced);
    }

    #[test]
    fn mark_changed_new_unique_preserves_synced_bitmap_when_possible() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        arena.mark_changed_new_unique(idx);
        assert!(arena.changed_bitmap_synced);
        assert_eq!(arena.changed_list.len(), 1);

        arena.mark_changed(idx);
        assert_eq!(arena.changed_list.len(), 1);
    }

    #[test]
    fn active_tags_wrap_epoch_and_reset() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        let i = idx.index();

        arena.active_epoch = u16::MAX;
        arena.active_tags[i] = u16::MAX;

        arena.begin_active_rebuild();
        assert_eq!(arena.active_epoch, 1);
        assert_eq!(arena.active_tags[i], 0);
        assert!(!arena.active_test_and_set(i));
        assert!(arena.active_test_and_set(i));
    }

    #[test]
    fn release_clears_active_tag_before_recycling_slot() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        let i = idx.index();

        arena.active_epoch = 9;
        arena.active_tags[i] = 9;
        arena.release(idx);
        let recycled = arena.allocate((1, 0));

        assert_eq!(recycled, idx);
        assert_eq!(arena.active_tags[recycled.index()], 0);
    }
}
