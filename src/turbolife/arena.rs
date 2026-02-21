//! Tile arena with split storage for TurboLife.
//!
//! Cell data is stored as two separate `Vec<CellBuf>` with a global phase bit,
//! halving the kernel working set. Borders are double-buffered.
//! Slot 0 is reserved as a sentinel (all zeros) so ghost-zone gathers
//! can use unconditional loads — NO_NEIGHBOR maps to the sentinel.

use super::tile::{
    BorderData, CellBuf, EMPTY_NEIGHBORS, MAX_NEIGHBOR_INDEX, NO_NEIGHBOR, NeighborIdx, Neighbors,
    TILE_SIZE, TileIdx, TileMeta,
};
use super::tilemap::{
    TILE_HASH_X_STEP, TILE_HASH_Y_STEP, TileMap, tile_hash, tile_hash_from_lanes, tile_hash_lanes,
};

const INITIAL_TILE_CAPACITY: usize = 256;
const MIN_GROW_TILES: usize = 256;
const ACTIVE_SORT_RADIX_BUCKETS: usize = 1 << 16;
const COORD_LOOKUP_CACHE_ENABLED: bool = false;
const COORD_LOOKUP_CACHE_SETS: usize = 1 << 14;
const COORD_LOOKUP_CACHE_MASK: usize = COORD_LOOKUP_CACHE_SETS - 1;
const COORD_LOOKUP_CACHE_REPLACEMENT_SHIFT: u32 = 17;
pub(crate) const CHANGED_INFLUENCE_ALL: u8 = 0xFF;
const DIR_OFFSETS: [(i64, i64); 8] = [
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
];
const DIR_REVERSE: [usize; 8] = [1, 0, 3, 2, 7, 6, 5, 4];
const DIR_HASH_X_OFFSETS: [u64; 8] = [
    0,
    0,
    TILE_HASH_X_STEP.wrapping_neg(),
    TILE_HASH_X_STEP,
    TILE_HASH_X_STEP.wrapping_neg(),
    TILE_HASH_X_STEP,
    TILE_HASH_X_STEP.wrapping_neg(),
    TILE_HASH_X_STEP,
];
const DIR_HASH_Y_OFFSETS: [u64; 8] = [
    TILE_HASH_Y_STEP,
    TILE_HASH_Y_STEP.wrapping_neg(),
    0,
    0,
    TILE_HASH_Y_STEP,
    TILE_HASH_Y_STEP,
    TILE_HASH_Y_STEP.wrapping_neg(),
    TILE_HASH_Y_STEP.wrapping_neg(),
];
const UNKNOWN_HINT: i8 = -1;

const fn direction_index_from_offset(dx: i64, dy: i64) -> i8 {
    let mut dir = 0usize;
    while dir < DIR_OFFSETS.len() {
        let (odx, ody) = DIR_OFFSETS[dir];
        if odx == dx && ody == dy {
            return dir as i8;
        }
        dir += 1;
    }
    UNKNOWN_HINT
}

const fn build_expand_neighbor_hints() -> [[i8; 8]; 8] {
    let mut table = [[UNKNOWN_HINT; 8]; 8];
    let mut expand_dir = 0usize;
    while expand_dir < 8 {
        let (ex, ey) = DIR_OFFSETS[expand_dir];
        let mut target_neighbor_dir = 0usize;
        while target_neighbor_dir < 8 {
            let (nx, ny) = DIR_OFFSETS[target_neighbor_dir];
            let sx = ex + nx;
            let sy = ey + ny;
            table[expand_dir][target_neighbor_dir] = direction_index_from_offset(sx, sy);
            target_neighbor_dir += 1;
        }
        expand_dir += 1;
    }
    table
}

const EXPAND_NEIGHBOR_HINTS: [[i8; 8]; 8] = build_expand_neighbor_hints();
const UNKNOWN_TWO_HOP_HINT: [i8; 2] = [UNKNOWN_HINT, UNKNOWN_HINT];

const fn build_expand_neighbor_two_hop_hints() -> [[[i8; 2]; 8]; 8] {
    let mut table = [[UNKNOWN_TWO_HOP_HINT; 8]; 8];
    let mut expand_dir = 0usize;
    while expand_dir < 8 {
        let mut target_neighbor_dir = 0usize;
        while target_neighbor_dir < 8 {
            if EXPAND_NEIGHBOR_HINTS[expand_dir][target_neighbor_dir] == UNKNOWN_HINT {
                let (ex, ey) = DIR_OFFSETS[expand_dir];
                let (tx, ty) = DIR_OFFSETS[target_neighbor_dir];
                let want_x = ex + tx;
                let want_y = ey + ty;
                let mut via_dir = 0usize;
                while via_dir < 8 {
                    if via_dir != expand_dir {
                        let (vx, vy) = DIR_OFFSETS[via_dir];
                        let mut via_neighbor_dir = 0usize;
                        while via_neighbor_dir < 8 {
                            let (nx, ny) = DIR_OFFSETS[via_neighbor_dir];
                            if vx + nx == want_x && vy + ny == want_y {
                                table[expand_dir][target_neighbor_dir] =
                                    [via_dir as i8, via_neighbor_dir as i8];
                                via_dir = 8;
                                break;
                            }
                            via_neighbor_dir += 1;
                        }
                    }
                    via_dir += 1;
                }
            }
            target_neighbor_dir += 1;
        }
        expand_dir += 1;
    }
    table
}

const EXPAND_NEIGHBOR_TWO_HOP_HINTS: [[[i8; 2]; 8]; 8] = build_expand_neighbor_two_hop_hints();

#[inline(always)]
fn encode_neighbor_idx(idx: TileIdx) -> NeighborIdx {
    NeighborIdx::try_from(idx.0).expect("tile index exceeds NeighborIdx capacity")
}

#[inline(always)]
fn encode_neighbor_usize(idx: usize) -> NeighborIdx {
    NeighborIdx::try_from(idx).expect("tile index exceeds NeighborIdx capacity")
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CoordLookupCacheEntry {
    x: i64,
    y: i64,
    idx: u32,
}

impl CoordLookupCacheEntry {
    const EMPTY: Self = Self { x: 0, y: 0, idx: 0 };
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CoordLookupCacheSet {
    a: CoordLookupCacheEntry,
    b: CoordLookupCacheEntry,
}

impl CoordLookupCacheSet {
    const EMPTY: Self = Self {
        a: CoordLookupCacheEntry::EMPTY,
        b: CoordLookupCacheEntry::EMPTY,
    };
}

#[derive(Clone)]
pub struct BorderPlanes {
    north: Vec<u64>,
    south: Vec<u64>,
    west: Vec<u64>,
    east: Vec<u64>,
}

impl BorderPlanes {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            north: Vec::with_capacity(capacity),
            south: Vec::with_capacity(capacity),
            west: Vec::with_capacity(capacity),
            east: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    fn push_zero(&mut self) {
        self.north.push(0);
        self.south.push(0);
        self.west.push(0);
        self.east.push(0);
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.north.reserve(additional);
        self.south.reserve(additional);
        self.west.reserve(additional);
        self.east.reserve(additional);
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.north.len(), self.south.len());
        debug_assert_eq!(self.north.len(), self.west.len());
        debug_assert_eq!(self.north.len(), self.east.len());
        self.north.len()
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> BorderData {
        BorderData::from_edges(
            self.north[idx],
            self.south[idx],
            self.west[idx],
            self.east[idx],
        )
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, border: BorderData) {
        self.north[idx] = border.north;
        self.south[idx] = border.south;
        self.west[idx] = border.west;
        self.east[idx] = border.east;
    }

    #[inline(always)]
    pub fn clear(&mut self, idx: usize) {
        self.north[idx] = 0;
        self.south[idx] = 0;
        self.west[idx] = 0;
        self.east[idx] = 0;
    }

    #[inline(always)]
    pub fn north_ptr(&self) -> *const u64 {
        self.north.as_ptr()
    }

    #[inline(always)]
    pub fn south_ptr(&self) -> *const u64 {
        self.south.as_ptr()
    }

    #[inline(always)]
    pub fn west_ptr(&self) -> *const u64 {
        self.west.as_ptr()
    }

    #[inline(always)]
    pub fn east_ptr(&self) -> *const u64 {
        self.east.as_ptr()
    }

    #[inline(always)]
    pub fn north_mut_ptr(&mut self) -> *mut u64 {
        self.north.as_mut_ptr()
    }

    #[inline(always)]
    pub fn south_mut_ptr(&mut self) -> *mut u64 {
        self.south.as_mut_ptr()
    }

    #[inline(always)]
    pub fn west_mut_ptr(&mut self) -> *mut u64 {
        self.west.as_mut_ptr()
    }

    #[inline(always)]
    pub fn east_mut_ptr(&mut self) -> *mut u64 {
        self.east.as_mut_ptr()
    }
}

pub struct TileArena {
    /// Two cell buffers: `cell_bufs[phase]` = current (read), `cell_bufs[1-phase]` = next (write).
    pub cell_bufs: [Vec<CellBuf>; 2],
    /// Global cell phase. Flipped after each step.
    pub cell_phase: usize,
    pub meta: Vec<TileMeta>,
    pub neighbors: Vec<Neighbors>,
    pub coords: Vec<(i64, i64)>,
    /// Double-buffered border planes. `borders[border_phase]` = current gen (read).
    pub borders: [BorderPlanes; 2],
    /// Cached `BorderData::live_mask` for each border phase.
    pub border_live_masks: [Vec<u8>; 2],
    pub border_phase: usize,
    coord_to_idx: TileMap,
    coord_lookup_cache: Box<[CoordLookupCacheSet]>,
    pub free_list: Vec<TileIdx>,
    pub changed_list: Vec<TileIdx>,
    pub changed_influence: Vec<u8>,
    changed_influence_uniform_all: bool,
    changed_bits: Vec<u64>,
    changed_bitmap_synced: bool,
    pub occupied_count: usize,
    pub occupied_bits: Vec<u64>,

    pub active_set: Vec<TileIdx>,
    pub active_set_dense_contiguous: bool,
    active_epoch: u16,
    active_tags: Vec<u16>,
    pub active_marks_words: Vec<u64>,
    pub active_sort_scratch: Vec<TileIdx>,
    pub active_sort_counts: Vec<u32>,
    /// Pending directional frontier-expansion candidates.
    /// Packed as `((src_idx as u32) << 3) | dir`.
    pub expand_buf: Vec<u32>,
    pub prune_candidates_verified: bool,
    pub prune_buf: Vec<TileIdx>,
    pub prune_marks: Vec<u8>,
    pub prune_marks_words: Vec<u64>,
    pub changed_scratch: Vec<TileIdx>,
    pub changed_influence_scratch: Vec<u8>,
}

impl TileArena {
    pub fn new() -> Self {
        // Slot 0 is the sentinel – always zeroed, never used for real tiles.
        let sentinel_cells = CellBuf::empty();
        let sentinel_meta = TileMeta::released();
        let sentinel_neighbors = EMPTY_NEIGHBORS;
        let sentinel_coord = (i64::MIN, i64::MIN);

        let mut cell_bufs0 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut cell_bufs1 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut meta = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut neighbors = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut coords = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut borders0 = BorderPlanes::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut borders1 = BorderPlanes::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut border_live_masks0 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);
        let mut border_live_masks1 = Vec::with_capacity(INITIAL_TILE_CAPACITY + 1);

        cell_bufs0.push(sentinel_cells.clone());
        cell_bufs1.push(sentinel_cells);
        meta.push(sentinel_meta);
        neighbors.push(sentinel_neighbors);
        coords.push(sentinel_coord);
        borders0.push_zero();
        borders1.push_zero();
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
            coord_lookup_cache: if COORD_LOOKUP_CACHE_ENABLED {
                vec![CoordLookupCacheSet::EMPTY; COORD_LOOKUP_CACHE_SETS].into_boxed_slice()
            } else {
                vec![CoordLookupCacheSet::EMPTY; 1].into_boxed_slice()
            },
            free_list: Vec::new(),
            changed_list: Vec::new(),
            changed_influence: Vec::new(),
            changed_influence_uniform_all: false,
            changed_bits: vec![0],
            changed_bitmap_synced: true,
            occupied_count: 0,
            occupied_bits: vec![0],
            active_set: Vec::new(),
            active_set_dense_contiguous: false,
            active_epoch: 1,
            active_tags: vec![0],
            active_marks_words: vec![0],
            active_sort_scratch: Vec::new(),
            active_sort_counts: vec![0u32; ACTIVE_SORT_RADIX_BUCKETS],
            expand_buf: Vec::new(),
            prune_candidates_verified: false,
            prune_buf: Vec::new(),
            prune_marks: Vec::new(),
            prune_marks_words: Vec::new(),
            changed_scratch: Vec::new(),
            changed_influence_scratch: Vec::new(),
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
    pub fn border(&self, idx: TileIdx) -> BorderData {
        self.borders[self.border_phase].get(idx.index())
    }

    /// Write current-gen border and keep cached live-mask in sync.
    #[inline(always)]
    pub fn set_current_border(&mut self, idx: TileIdx, border: BorderData) {
        let i = idx.index();
        self.borders[self.border_phase].set(i, border);
        self.border_live_masks[self.border_phase][i] = border.live_mask();
    }

    /// Flip border phase.
    #[inline(always)]
    pub fn flip_borders(&mut self) {
        self.border_phase = 1 - self.border_phase;
    }

    #[inline(always)]
    fn coord_lookup_cache_slot(hash: u64) -> usize {
        hash as usize & COORD_LOOKUP_CACHE_MASK
    }

    #[inline(always)]
    fn coord_lookup_cache_get(&self, coord: (i64, i64), hash: u64) -> Option<TileIdx> {
        if !COORD_LOOKUP_CACHE_ENABLED {
            return None;
        }
        let set = unsafe {
            self.coord_lookup_cache
                .get_unchecked(Self::coord_lookup_cache_slot(hash))
        };
        if set.a.idx != 0 && set.a.x == coord.0 && set.a.y == coord.1 {
            return Some(TileIdx(set.a.idx));
        }
        if set.b.idx != 0 && set.b.x == coord.0 && set.b.y == coord.1 {
            return Some(TileIdx(set.b.idx));
        }
        None
    }

    #[inline(always)]
    fn coord_lookup_cache_set(&mut self, coord: (i64, i64), idx: TileIdx, hash: u64) {
        if !COORD_LOOKUP_CACHE_ENABLED {
            return;
        }
        let set = unsafe {
            self.coord_lookup_cache
                .get_unchecked_mut(Self::coord_lookup_cache_slot(hash))
        };
        if set.a.idx != 0 && set.a.x == coord.0 && set.a.y == coord.1 {
            set.a.idx = idx.0;
            return;
        }
        if set.b.idx != 0 && set.b.x == coord.0 && set.b.y == coord.1 {
            set.b.idx = idx.0;
            return;
        }
        if set.a.idx == 0 {
            set.a.x = coord.0;
            set.a.y = coord.1;
            set.a.idx = idx.0;
            return;
        }
        if set.b.idx == 0 {
            set.b.x = coord.0;
            set.b.y = coord.1;
            set.b.idx = idx.0;
            return;
        }
        if ((hash >> COORD_LOOKUP_CACHE_REPLACEMENT_SHIFT) & 1) == 0 {
            set.a.x = coord.0;
            set.a.y = coord.1;
            set.a.idx = idx.0;
        } else {
            set.b.x = coord.0;
            set.b.y = coord.1;
            set.b.idx = idx.0;
        }
    }

    #[inline(always)]
    fn coord_lookup_cache_clear(&mut self, coord: (i64, i64), idx: TileIdx, hash: u64) {
        if !COORD_LOOKUP_CACHE_ENABLED {
            return;
        }
        let set = unsafe {
            self.coord_lookup_cache
                .get_unchecked_mut(Self::coord_lookup_cache_slot(hash))
        };
        if set.a.idx == idx.0 && set.a.x == coord.0 && set.a.y == coord.1 {
            set.a.idx = 0;
        }
        if set.b.idx == idx.0 && set.b.x == coord.0 && set.b.y == coord.1 {
            set.b.idx = 0;
        }
    }

    #[inline(always)]
    pub fn idx_at(&self, coord: (i64, i64)) -> Option<TileIdx> {
        let idx = self.coord_to_idx.get(coord.0, coord.1)?;
        #[cfg(any(test, debug_assertions))]
        {
            if self.idx_matches_live_coord(idx, coord) {
                Some(idx)
            } else {
                None
            }
        }
        #[cfg(not(any(test, debug_assertions)))]
        {
            Some(idx)
        }
    }

    #[inline(always)]
    pub fn idx_at_cached(&mut self, coord: (i64, i64)) -> Option<TileIdx> {
        let hash = tile_hash(coord.0, coord.1);
        self.idx_at_cached_hashed(coord, hash)
    }

    #[inline(always)]
    fn idx_at_cached_hashed(&mut self, coord: (i64, i64), hash: u64) -> Option<TileIdx> {
        #[cfg(not(any(test, debug_assertions)))]
        {
            if COORD_LOOKUP_CACHE_ENABLED
                && let Some(idx) = self.coord_lookup_cache_get(coord, hash)
            {
                return Some(idx);
            }
            let idx = self.coord_to_idx.get_hashed(coord.0, coord.1, hash)?;
            if COORD_LOOKUP_CACHE_ENABLED {
                self.coord_lookup_cache_set(coord, idx, hash);
            }
            return Some(idx);
        }

        #[cfg(any(test, debug_assertions))]
        {
            if COORD_LOOKUP_CACHE_ENABLED
                && let Some(idx) = self.coord_lookup_cache_get(coord, hash)
            {
                if self.idx_matches_live_coord(idx, coord) {
                    return Some(idx);
                }
                self.coord_lookup_cache_clear(coord, idx, hash);
            }
            let idx = self.coord_to_idx.get_hashed(coord.0, coord.1, hash)?;
            if !self.idx_matches_live_coord(idx, coord) {
                // Defensive cleanup in case a stale coordinate mapping survives.
                self.coord_to_idx.remove_hashed(coord.0, coord.1, hash);
                if COORD_LOOKUP_CACHE_ENABLED {
                    self.coord_lookup_cache_clear(coord, idx, hash);
                }
                return None;
            }
            if COORD_LOOKUP_CACHE_ENABLED {
                self.coord_lookup_cache_set(coord, idx, hash);
            }
            Some(idx)
        }
    }

    #[cfg(any(test, debug_assertions))]
    #[inline(always)]
    fn idx_matches_live_coord(&self, idx: TileIdx, coord: (i64, i64)) -> bool {
        let i = idx.index();
        self.meta[i].occupied() && self.coords[i] == coord
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
        self.mark_changed_with_influence(idx, CHANGED_INFLUENCE_ALL);
    }

    #[inline(always)]
    fn debug_assert_changed_influence_layout(&self) {
        debug_assert!(
            (self.changed_influence_uniform_all && self.changed_influence.is_empty())
                || (!self.changed_influence_uniform_all
                    && self.changed_list.len() == self.changed_influence.len())
        );
    }

    #[inline]
    fn materialize_changed_influence_if_uniform(&mut self) {
        if self.changed_influence_uniform_all {
            self.changed_influence
                .resize(self.changed_list.len(), CHANGED_INFLUENCE_ALL);
            self.changed_influence_uniform_all = false;
        }
    }

    #[inline]
    pub fn mark_changed_with_influence(&mut self, idx: TileIdx, influence_mask: u8) {
        if influence_mask != CHANGED_INFLUENCE_ALL {
            self.materialize_changed_influence_if_uniform();
        }
        self.sync_changed_bitmap_if_needed();
        let i = idx.index();
        if !self.changed_test_and_set(i) {
            self.changed_list.push(idx);
            if self.changed_influence_uniform_all {
                debug_assert_eq!(influence_mask, CHANGED_INFLUENCE_ALL);
            } else {
                self.changed_influence.push(influence_mask);
            }
        } else {
            self.merge_changed_influence_for_existing(idx, influence_mask);
        }
    }

    #[inline]
    fn merge_changed_influence_for_existing(&mut self, idx: TileIdx, influence_mask: u8) {
        if influence_mask == 0 {
            return;
        }
        if self.changed_influence_uniform_all {
            if influence_mask == CHANGED_INFLUENCE_ALL {
                return;
            }
            self.materialize_changed_influence_if_uniform();
        }
        if let Some(pos) = self.changed_list.iter().rposition(|&queued| queued == idx) {
            self.changed_influence[pos] |= influence_mask;
        } else {
            debug_assert!(
                false,
                "changed bitmap contained an index that was missing from changed_list"
            );
        }
    }

    #[inline(always)]
    pub(crate) fn mark_changed_new_unique(&mut self, idx: TileIdx) {
        self.mark_changed_new_unique_with_influence(idx, CHANGED_INFLUENCE_ALL);
    }

    #[inline(always)]
    pub(crate) fn mark_changed_new_unique_with_influence(
        &mut self,
        idx: TileIdx,
        influence_mask: u8,
    ) {
        if influence_mask != CHANGED_INFLUENCE_ALL {
            self.materialize_changed_influence_if_uniform();
        }
        if self.changed_bitmap_synced {
            let duplicate = self.changed_test_and_set(idx.index());
            debug_assert!(
                !duplicate,
                "mark_changed_new_unique received a duplicate tile index"
            );
            if duplicate {
                self.merge_changed_influence_for_existing(idx, influence_mask);
                return;
            }
        }
        self.changed_list.push(idx);
        if self.changed_influence_uniform_all {
            debug_assert_eq!(influence_mask, CHANGED_INFLUENCE_ALL);
        } else {
            self.changed_influence.push(influence_mask);
        }
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_changed_from_kernel(&mut self, idx: TileIdx) {
        self.changed_list.push(idx);
        self.changed_influence.clear();
        self.changed_influence_uniform_all = true;
        self.changed_bitmap_synced = false;
    }

    #[inline(always)]
    pub(crate) fn mark_changed_bitmap_unsynced(&mut self) {
        self.materialize_changed_influence_if_uniform();
        self.changed_bitmap_synced = false;
        self.debug_assert_changed_influence_layout();
    }

    #[inline(always)]
    pub(crate) fn mark_changed_bitmap_unsynced_uniform_all(&mut self) {
        self.changed_bitmap_synced = false;
        self.changed_influence_uniform_all = true;
        self.changed_influence.clear();
        self.debug_assert_changed_influence_layout();
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
        if self.active_marks_words.len() < target_words {
            self.active_marks_words
                .reserve(target_words - self.active_marks_words.len());
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
        self.debug_assert_changed_influence_layout();
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
    pub(crate) fn clear_all_changed_marks(&mut self) {
        self.changed_bits.fill(0);
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
        if self.active_tags.is_empty() {
            self.active_tags.push(0);
        }
        // Reserve slot 0 (NO_NEIGHBOR sentinel) so branchless neighbor fan-out
        // can mark/test it without ever pushing it to active_set.
        self.active_tags[0] = self.active_epoch;
    }

    #[inline(always)]
    pub(crate) fn begin_active_rebuild_with_capacity(&mut self, required_slots: usize) {
        self.begin_active_rebuild();
        let required_slots = required_slots.max(1);
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
    pub(crate) fn begin_changed_rebuild(&mut self) -> (bool, bool) {
        self.debug_assert_changed_influence_layout();
        let was_synced = self.changed_bitmap_synced;
        let uniform_influence = self.changed_influence_uniform_all;
        self.changed_bitmap_synced = true;
        self.changed_influence_uniform_all = false;
        (was_synced, uniform_influence)
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
        self.active_set_dense_contiguous = false;
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
            self.borders[0].clear(i);
            self.borders[1].clear(i);
            self.border_live_masks[0][i] = 0;
            self.border_live_masks[1][i] = 0;
            self.set_occupied_bit(i);
            self.ensure_active_tag_capacity(i);
            self.active_tags[i] = 0;
            recycled
        } else {
            self.ensure_growth_capacity();
            let next_index = self.cell_bufs[0].len();
            assert!(
                next_index <= MAX_NEIGHBOR_INDEX,
                "TurboLife neighbor index overflow: {} exceeds {}",
                next_index,
                MAX_NEIGHBOR_INDEX
            );
            let idx = TileIdx(next_index as u32);
            self.cell_bufs[0].push(CellBuf::empty());
            self.cell_bufs[1].push(CellBuf::empty());
            self.meta.push(TileMeta::empty());
            self.neighbors.push(EMPTY_NEIGHBORS);
            self.coords.push(coord);
            self.borders[0].push_zero();
            self.borders[1].push_zero();
            self.border_live_masks[0].push(0);
            self.border_live_masks[1].push(0);
            self.active_tags.push(0);
            self.set_occupied_bit(idx.index());
            idx
        }
    }

    #[inline(always)]
    fn recycle_uncommitted_slot(&mut self, idx: TileIdx) {
        let i = idx.index();
        self.neighbors[i] = EMPTY_NEIGHBORS;
        self.meta[i] = TileMeta::released();
        self.clear_occupied_bit(i);
        self.clear_changed_mark(i);
        self.ensure_active_tag_capacity(i);
        self.active_tags[i] = 0;
        self.free_list.push(idx);
    }

    #[inline]
    unsafe fn link_neighbor_pair_raw(
        neighbors_ptr: *mut Neighbors,
        meta_ptr: *mut TileMeta,
        tile_i: usize,
        dir_idx: usize,
        neighbor_i: usize,
    ) {
        let reverse_dir = DIR_REVERSE[dir_idx];
        let tile_raw = encode_neighbor_usize(tile_i);
        let neighbor_raw = encode_neighbor_usize(neighbor_i);
        unsafe {
            (&mut *neighbors_ptr.add(tile_i))[dir_idx] = neighbor_raw;
            (&mut *neighbors_ptr.add(neighbor_i))[reverse_dir] = tile_raw;
            (*meta_ptr.add(tile_i)).missing_mask &= !(1u8 << dir_idx);
            (*meta_ptr.add(neighbor_i)).missing_mask &= !(1u8 << reverse_dir);
        }
    }

    #[inline]
    fn link_neighbors(&mut self, idx: TileIdx, coord: (i64, i64)) {
        let (cx, cy) = coord;
        let (center_hash_x, center_hash_y) = tile_hash_lanes(cx, cy);
        let i = idx.index();
        let neighbors_ptr = self.neighbors.as_mut_ptr();
        let meta_ptr = self.meta.as_mut_ptr();

        for (dir_idx, (dx, dy)) in DIR_OFFSETS.iter().copied().enumerate() {
            let neighbor_coord = (cx + dx, cy + dy);
            let neighbor_hash = tile_hash_from_lanes(
                center_hash_x.wrapping_add(DIR_HASH_X_OFFSETS[dir_idx]),
                center_hash_y.wrapping_add(DIR_HASH_Y_OFFSETS[dir_idx]),
            );
            if let Some(neighbor_idx) = self.idx_at_cached_hashed(neighbor_coord, neighbor_hash) {
                let ni = neighbor_idx.index();
                // SAFETY: idx and neighbor_idx are valid arena indices.
                unsafe {
                    Self::link_neighbor_pair_raw(neighbors_ptr, meta_ptr, i, dir_idx, ni);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn allocate_absent(&mut self, coord: (i64, i64)) -> TileIdx {
        let hash = tile_hash(coord.0, coord.1);
        debug_assert!(self.idx_at(coord).is_none());
        let idx = self.allocate_slot(coord);
        self.coord_to_idx.insert_hashed(coord.0, coord.1, idx, hash);
        self.coord_lookup_cache_set(coord, idx, hash);
        self.occupied_count += 1;

        self.link_neighbors(idx, coord);
        idx
    }

    #[inline]
    pub(crate) fn allocate_absent_neighbor_from(
        &mut self,
        src: TileIdx,
        dir_idx: usize,
    ) -> (TileIdx, bool) {
        debug_assert!(dir_idx < 8);
        let src_i = src.index();
        let existing_neighbor = self.neighbors[src_i][dir_idx];
        if existing_neighbor != NO_NEIGHBOR {
            return (TileIdx(existing_neighbor as u32), false);
        }

        let (sx, sy) = self.coords[src_i];
        let (src_hash_x, src_hash_y) = tile_hash_lanes(sx, sy);
        let (dx, dy) = DIR_OFFSETS[dir_idx];
        let coord = (sx + dx, sy + dy);
        let coord_hash_x = src_hash_x.wrapping_add(DIR_HASH_X_OFFSETS[dir_idx]);
        let coord_hash_y = src_hash_y.wrapping_add(DIR_HASH_Y_OFFSETS[dir_idx]);
        let hash = tile_hash_from_lanes(coord_hash_x, coord_hash_y);

        #[cfg(any(test, debug_assertions))]
        if let Some(existing_idx) = self.idx_at_cached_hashed(coord, hash) {
            unsafe {
                Self::link_neighbor_pair_raw(
                    self.neighbors.as_mut_ptr(),
                    self.meta.as_mut_ptr(),
                    src_i,
                    dir_idx,
                    existing_idx.index(),
                );
            }
            return (existing_idx, false);
        }

        let idx = self.allocate_slot(coord);
        if let Err(existing_idx) = self
            .coord_to_idx
            .insert_unique_hashed(coord.0, coord.1, idx, hash)
        {
            self.coord_lookup_cache_set(coord, existing_idx, hash);
            self.recycle_uncommitted_slot(idx);
            unsafe {
                Self::link_neighbor_pair_raw(
                    self.neighbors.as_mut_ptr(),
                    self.meta.as_mut_ptr(),
                    src_i,
                    dir_idx,
                    existing_idx.index(),
                );
            }
            return (existing_idx, false);
        }
        self.coord_lookup_cache_set(coord, idx, hash);
        self.occupied_count += 1;

        let idx_i = idx.index();
        let src_neighbors = self.neighbors[src_i];
        let neighbors_ptr = self.neighbors.as_mut_ptr();
        let meta_ptr = self.meta.as_mut_ptr();

        // Direct source-to-target link.
        unsafe {
            Self::link_neighbor_pair_raw(
                neighbors_ptr,
                meta_ptr,
                idx_i,
                DIR_REVERSE[dir_idx],
                src_i,
            );
        }

        // Resolve and stitch remaining target-neighbor links.
        for target_neighbor_dir in 0..8usize {
            if target_neighbor_dir == DIR_REVERSE[dir_idx] {
                continue;
            }

            let hint = EXPAND_NEIGHBOR_HINTS[dir_idx][target_neighbor_dir];
            let neighbor_raw = if hint >= 0 {
                let hinted_raw = src_neighbors[hint as usize];
                if hinted_raw != NO_NEIGHBOR {
                    #[cfg(any(test, debug_assertions))]
                    {
                        // Tests intentionally break hint invariants; keep a
                        // correctness fallback in non-release builds.
                        let (ndx, ndy) = DIR_OFFSETS[target_neighbor_dir];
                        let neighbor_coord = (coord.0 + ndx, coord.1 + ndy);
                        let hinted_i = hinted_raw as usize;
                        if self.meta[hinted_i].occupied() && self.coords[hinted_i] == neighbor_coord
                        {
                            hinted_raw
                        } else {
                            let neighbor_hash = tile_hash_from_lanes(
                                coord_hash_x.wrapping_add(DIR_HASH_X_OFFSETS[target_neighbor_dir]),
                                coord_hash_y.wrapping_add(DIR_HASH_Y_OFFSETS[target_neighbor_dir]),
                            );
                            self.idx_at_cached_hashed(neighbor_coord, neighbor_hash)
                                .map_or(NO_NEIGHBOR, encode_neighbor_idx)
                        }
                    }
                    #[cfg(not(any(test, debug_assertions)))]
                    {
                        hinted_raw
                    }
                } else {
                    #[cfg(any(test, debug_assertions))]
                    {
                        let (ndx, ndy) = DIR_OFFSETS[target_neighbor_dir];
                        let neighbor_coord = (coord.0 + ndx, coord.1 + ndy);
                        let neighbor_hash = tile_hash_from_lanes(
                            coord_hash_x.wrapping_add(DIR_HASH_X_OFFSETS[target_neighbor_dir]),
                            coord_hash_y.wrapping_add(DIR_HASH_Y_OFFSETS[target_neighbor_dir]),
                        );
                        self.idx_at_cached_hashed(neighbor_coord, neighbor_hash)
                            .map_or(NO_NEIGHBOR, encode_neighbor_idx)
                    }
                    #[cfg(not(any(test, debug_assertions)))]
                    {
                        // If the source has no neighbor in the hinted direction,
                        // the target-neighbor coordinate cannot be occupied.
                        NO_NEIGHBOR
                    }
                }
            } else {
                let mut candidate = NO_NEIGHBOR;
                let two_hop = EXPAND_NEIGHBOR_TWO_HOP_HINTS[dir_idx][target_neighbor_dir];
                if two_hop[0] != UNKNOWN_HINT {
                    let via_raw = src_neighbors[two_hop[0] as usize];
                    if via_raw != NO_NEIGHBOR {
                        let via_neighbor_raw =
                            self.neighbors[via_raw as usize][two_hop[1] as usize];
                        if via_neighbor_raw != NO_NEIGHBOR {
                            candidate = via_neighbor_raw;
                        }
                    }
                }

                #[cfg(any(test, debug_assertions))]
                if candidate != NO_NEIGHBOR {
                    let (ndx, ndy) = DIR_OFFSETS[target_neighbor_dir];
                    let expected_coord = (coord.0 + ndx, coord.1 + ndy);
                    let candidate_i = candidate as usize;
                    if !self.meta[candidate_i].occupied()
                        || self.coords[candidate_i] != expected_coord
                    {
                        candidate = NO_NEIGHBOR;
                    }
                }

                if candidate != NO_NEIGHBOR {
                    candidate
                } else {
                    let (ndx, ndy) = DIR_OFFSETS[target_neighbor_dir];
                    let neighbor_coord = (coord.0 + ndx, coord.1 + ndy);
                    let neighbor_hash = tile_hash_from_lanes(
                        coord_hash_x.wrapping_add(DIR_HASH_X_OFFSETS[target_neighbor_dir]),
                        coord_hash_y.wrapping_add(DIR_HASH_Y_OFFSETS[target_neighbor_dir]),
                    );
                    self.idx_at_cached_hashed(neighbor_coord, neighbor_hash)
                        .map_or(NO_NEIGHBOR, encode_neighbor_idx)
                }
            };

            if neighbor_raw == NO_NEIGHBOR {
                continue;
            }

            unsafe {
                Self::link_neighbor_pair_raw(
                    neighbors_ptr,
                    meta_ptr,
                    idx_i,
                    target_neighbor_dir,
                    neighbor_raw as usize,
                );
            }
        }

        (idx, true)
    }

    #[allow(dead_code)]
    pub fn allocate(&mut self, coord: (i64, i64)) -> TileIdx {
        if let Some(existing) = self.idx_at_cached(coord) {
            return existing;
        }

        self.allocate_absent(coord)
    }

    pub fn release(&mut self, idx: TileIdx) {
        let i = idx.index();
        if !self.meta[i].occupied() {
            return;
        }
        self.active_set_dense_contiguous = false;
        self.clear_occupied_bit(i);
        self.clear_changed_mark(i);
        self.ensure_active_tag_capacity(i);
        self.active_tags[i] = 0;

        // Unlink neighbors using raw pointer access.
        let neighbors_ptr = self.neighbors.as_mut_ptr();
        let meta_ptr = self.meta.as_mut_ptr();
        unsafe {
            let nb = *neighbors_ptr.add(i);
            for dir_idx in 0..8u8 {
                let neighbor_raw = nb[dir_idx as usize];
                if neighbor_raw != NO_NEIGHBOR {
                    let rev = DIR_REVERSE[dir_idx as usize];
                    let ni = neighbor_raw as usize;
                    (&mut *neighbors_ptr.add(ni))[rev] = NO_NEIGHBOR;
                    (*meta_ptr.add(ni)).missing_mask |= 1u8 << rev;
                }
            }
            (*neighbors_ptr.add(i)) = EMPTY_NEIGHBORS;
        }

        let coord = self.coords[i];
        let hash = tile_hash(coord.0, coord.1);
        self.coord_to_idx.remove_hashed(coord.0, coord.1, hash);
        self.coord_lookup_cache_clear(coord, idx, hash);
        self.meta[i] = TileMeta::released();
        self.free_list.push(idx);
        self.occupied_count = self.occupied_count.saturating_sub(1);
    }

    #[inline]
    pub fn ensure_neighbor(&mut self, tx: i64, ty: i64) {
        let coord = (tx, ty);
        if self.idx_at_cached(coord).is_some() {
            return;
        }
        let idx = self.allocate_absent(coord);
        let m = &mut self.meta[idx.index()];
        m.set_has_live(false);
        m.set_alt_phase_dirty(false);
        self.mark_changed(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::{CHANGED_INFLUENCE_ALL, TileArena};
    use crate::turbolife::tile::{BorderData, NO_NEIGHBOR, NeighborIdx};

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
    fn mark_changed_with_influence_merges_duplicate_masks() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        arena.mark_changed_with_influence(idx, 1 << 0);
        arena.mark_changed_with_influence(idx, (1 << 2) | (1 << 4));

        assert_eq!(arena.changed_list, vec![idx]);
        assert_eq!(
            arena.changed_influence,
            vec![(1 << 0) | (1 << 2) | (1 << 4)]
        );
    }

    #[test]
    fn mark_changed_with_influence_merges_duplicate_masks_after_unsynced_queue() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        arena.mark_changed_with_influence(idx, 1 << 1);
        arena.mark_changed_bitmap_unsynced();
        arena.mark_changed_with_influence(idx, 1 << 3);

        assert_eq!(arena.changed_list, vec![idx]);
        assert_eq!(arena.changed_influence, vec![(1 << 1) | (1 << 3)]);
        assert!(arena.changed_bitmap_synced);
    }

    #[test]
    fn mark_changed_bitmap_unsynced_materializes_uniform_influence_layout() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        arena.mark_changed(idx);
        arena.mark_changed_bitmap_unsynced_uniform_all();
        assert!(arena.changed_influence_uniform_all);
        assert!(arena.changed_influence.is_empty());

        arena.mark_changed_bitmap_unsynced();

        assert!(!arena.changed_influence_uniform_all);
        assert_eq!(arena.changed_list, vec![idx]);
        assert_eq!(arena.changed_influence, vec![CHANGED_INFLUENCE_ALL]);
        assert!(!arena.changed_bitmap_synced);
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
    fn begin_active_rebuild_keeps_no_neighbor_slot_reserved() {
        let mut arena = TileArena::new();

        arena.begin_active_rebuild();

        assert!(arena.active_test_and_set(NO_NEIGHBOR as usize));
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

    #[test]
    fn release_removes_coord_map_entry() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        assert_eq!(arena.coord_to_idx.len(), 1);

        arena.release(idx);

        assert_eq!(arena.coord_to_idx.len(), 0);
        assert!(arena.idx_at((0, 0)).is_none());
    }

    #[test]
    fn idx_at_rejects_stale_coord_mapping() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        arena.coords[idx.index()] = (9, 9);

        assert!(arena.idx_at((0, 0)).is_none());
    }

    #[test]
    fn idx_at_cached_prunes_stale_coord_mapping() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        assert_eq!(arena.idx_at_cached((0, 0)), Some(idx));
        arena.coords[idx.index()] = (9, 9);

        assert!(arena.idx_at_cached((0, 0)).is_none());
        assert!(arena.coord_to_idx.get(0, 0).is_none());
        assert!(arena.idx_at_cached((0, 0)).is_none());
    }

    #[test]
    fn structural_slot_changes_invalidate_dense_active_cache_flag() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));

        arena.active_set_dense_contiguous = true;
        let _ = arena.allocate((1, 0));
        assert!(!arena.active_set_dense_contiguous);

        arena.active_set_dense_contiguous = true;
        arena.release(idx);
        assert!(!arena.active_set_dense_contiguous);
    }

    #[test]
    fn recycled_slot_resets_buffers_and_border_cache() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        let i = idx.index();

        arena.cell_bufs[0][i].0[0] = u64::MAX;
        arena.cell_bufs[1][i].0[1] = u64::MAX;
        arena.borders[0].set(i, BorderData::from_edges(1, 2, 3, 4));
        arena.borders[1].set(i, BorderData::from_edges(5, 6, 7, 8));
        arena.border_live_masks[0][i] = 0xFF;
        arena.border_live_masks[1][i] = 0xFF;

        arena.release(idx);
        let recycled = arena.allocate((1, 0));
        let ri = recycled.index();

        assert_eq!(recycled, idx);
        assert!(arena.cell_bufs[0][ri].0.iter().all(|&row| row == 0));
        assert!(arena.cell_bufs[1][ri].0.iter().all(|&row| row == 0));
        let border0 = arena.borders[0].get(ri);
        assert_eq!(border0.north, 0);
        assert_eq!(border0.south, 0);
        assert_eq!(border0.west, 0);
        assert_eq!(border0.east, 0);
        let border1 = arena.borders[1].get(ri);
        assert_eq!(border1.north, 0);
        assert_eq!(border1.south, 0);
        assert_eq!(border1.west, 0);
        assert_eq!(border1.east, 0);
        assert_eq!(arena.border_live_masks[0][ri], 0);
        assert_eq!(arena.border_live_masks[1][ri], 0);
    }

    #[test]
    fn allocate_absent_neighbor_from_falls_back_when_hint_is_missing() {
        let mut arena = TileArena::new();
        let src = arena.allocate((0, 0));
        let west = arena.allocate((-1, 0));

        let src_i = src.index();
        let west_i = west.index();
        arena.neighbors[src_i][2] = NO_NEIGHBOR;
        arena.meta[src_i].missing_mask |= 1u8 << 2;
        assert_eq!(arena.neighbors[west_i][3] as u32, src.0);

        let (north, allocated) = arena.allocate_absent_neighbor_from(src, 0);
        assert!(allocated);
        assert_eq!(arena.coords[north.index()], (0, 1));
        assert_eq!(arena.neighbors[north.index()][6] as u32, west.0);
        assert_eq!(arena.neighbors[west_i][5] as u32, north.0);
    }

    #[test]
    fn allocate_absent_neighbor_from_ignores_stale_hint_coordinates() {
        let mut arena = TileArena::new();
        let src = arena.allocate((0, 0));
        let west = arena.allocate((-1, 0));
        let wrong = arena.allocate((42, 42));

        let src_i = src.index();
        arena.neighbors[src_i][2] = wrong.0 as NeighborIdx;
        arena.meta[src_i].missing_mask &= !(1u8 << 2);

        let (north, allocated) = arena.allocate_absent_neighbor_from(src, 0);
        assert!(allocated);
        assert_eq!(arena.coords[north.index()], (0, 1));
        assert_eq!(arena.neighbors[north.index()][6] as u32, west.0);
        assert_eq!(arena.neighbors[west.index()][5] as u32, north.0);
        assert_eq!(arena.neighbors[wrong.index()][5], NO_NEIGHBOR);
    }
}
