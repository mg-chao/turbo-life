use super::kernel::TileAdvanceResult;
use super::kernel::{
    advance_core_empty_with_clear, ghost_is_empty_from_live_masks_ptr, tile_is_empty,
};
use super::tile::{BorderData, CellBuf, GhostZone, MISSING_ALL_NEIGHBORS, TILE_SIZE};

const CACHE_SIZE: usize = 1 << 12;
const CACHE_MASK: usize = CACHE_SIZE - 1;
const ADAPT_WINDOW: u32 = 256;
const MIN_HIT_RATE_PCT: u32 = 20;
const DISABLE_THRESHOLD: u32 = 1;
const REPROBE_INTERVAL: u32 = 32;
const LIVE_NW: u8 = 1 << 4;
const LIVE_NE: u8 = 1 << 5;
const LIVE_SW: u8 = 1 << 6;
const LIVE_SE: u8 = 1 << 7;

#[inline(always)]
fn no_track_hint(changed: bool, prune_ready: bool) -> u8 {
    if changed {
        u8::MAX
    } else if prune_ready {
        1
    } else {
        0
    }
}

#[repr(C, align(64))]
#[derive(Clone)]
struct CacheEntry {
    key_lo: u64,
    key_hi: u64,
    input_cells: [u64; TILE_SIZE],
    input_ghost: GhostZone,
    next_cells: [u64; TILE_SIZE],
    border: BorderData,
    changed: bool,
    has_live: bool,
    live_mask: u8,
    neighbor_influence_mask: u8,
}

impl CacheEntry {
    const EMPTY: Self = Self {
        key_lo: 0,
        key_hi: 0,
        input_cells: [0u64; TILE_SIZE],
        input_ghost: GhostZone {
            north: 0,
            south: 0,
            west: 0,
            east: 0,
            nw: false,
            ne: false,
            sw: false,
            se: false,
        },
        next_cells: [0u64; TILE_SIZE],
        border: BorderData {
            north: 0,
            south: 0,
            west: 0,
            east: 0,
        },
        changed: false,
        has_live: false,
        live_mask: 0,
        neighbor_influence_mask: 0,
    };
}

#[inline(always)]
fn ghost_eq(a: &GhostZone, b: &GhostZone) -> bool {
    a.north == b.north
        && a.south == b.south
        && a.west == b.west
        && a.east == b.east
        && a.nw == b.nw
        && a.ne == b.ne
        && a.sw == b.sw
        && a.se == b.se
}

#[inline(always)]
fn hash_input(cells: &[u64; TILE_SIZE], ghost: &GhostZone) -> (u64, u64) {
    const K0: u64 = 0xa076_1d64_78bd_642f;
    const K1: u64 = 0xe703_7ed1_a0b4_28db;
    const K2: u64 = 0x8ebc_6af0_9c88_c6e3;
    const K3: u64 = 0x5899_65cc_7537_4cc3;
    const K4: u64 = 0x1d8e_4e27_c47d_124f;
    const K5: u64 = 0x94d0_49bb_1331_11eb;
    const K6: u64 = 0x2545_f491_4f6c_dd1d;
    const K7: u64 = 0x9e6c_63d0_676a_9a99;
    let mut h0: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h1: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let mut i = 0;
    while i < TILE_SIZE {
        let a = cells[i];
        let b = cells[i + 1];
        let c = cells[i + 2];
        let d = cells[i + 3];

        let lo0 = (a ^ K0).wrapping_mul(b ^ K1);
        let hi0 = (c ^ K2).wrapping_mul(d ^ K3);
        h0 = (h0 ^ lo0 ^ hi0).wrapping_mul(K0);

        let lo1 = (a ^ K4).wrapping_mul(c ^ K5);
        let hi1 = (b ^ K6).wrapping_mul(d ^ K7);
        h1 ^= lo1 ^ hi1;
        h1 = h1.rotate_left(29).wrapping_mul(K4);
        i += 4;
    }
    h0 ^= ghost.north.wrapping_mul(K1);
    h0 ^= ghost.south.wrapping_mul(K2);
    h0 ^= ghost.west.wrapping_mul(K3);
    h0 ^= ghost.east.wrapping_mul(K0);

    h1 ^= ghost.north.rotate_left(13).wrapping_mul(K5);
    h1 ^= ghost.south.rotate_left(17).wrapping_mul(K6);
    h1 ^= ghost.west.rotate_left(23).wrapping_mul(K7);
    h1 ^= ghost.east.rotate_left(31).wrapping_mul(K4);
    let c = (ghost.nw as u64)
        | ((ghost.ne as u64) << 1)
        | ((ghost.sw as u64) << 2)
        | ((ghost.se as u64) << 3);
    h0 ^= c.wrapping_mul(K1);
    h1 ^= c.wrapping_mul(K6);

    h0 ^= h0 >> 32;
    h0 = h0.wrapping_mul(K0);
    h0 ^= h0 >> 29;

    h1 ^= h1 >> 33;
    h1 = h1.wrapping_mul(K5);
    h1 ^= h1 >> 31;

    (h0 | 1, h1 | 1)
}

#[inline(always)]
unsafe fn neighbor_influence_mask_for_result(
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    idx: usize,
    border: &BorderData,
) -> u8 {
    let prev_north = unsafe { *borders_north_read_ptr.add(idx) };
    let prev_south = unsafe { *borders_south_read_ptr.add(idx) };
    let prev_west = unsafe { *borders_west_read_ptr.add(idx) };
    let prev_east = unsafe { *borders_east_read_ptr.add(idx) };
    BorderData::compute_live_mask(
        border.north ^ prev_north,
        border.south ^ prev_south,
        border.west ^ prev_west,
        border.east ^ prev_east,
    )
}

pub struct TileCache {
    entries: Box<[CacheEntry]>,
    window_hits: u32,
    window_total: u32,
    low_streak: u32,
    enabled: bool,
    disabled_steps: u32,
}

impl TileCache {
    pub fn new() -> Self {
        Self {
            entries: vec![CacheEntry::EMPTY; CACHE_SIZE].into_boxed_slice(),
            window_hits: 0,
            window_total: 0,
            low_streak: 0,
            enabled: true,
            disabled_steps: 0,
        }
    }

    #[inline]
    pub fn begin_step(&mut self) {
        if !self.enabled {
            self.disabled_steps += 1;
            if self.disabled_steps >= REPROBE_INTERVAL {
                self.enabled = true;
                self.disabled_steps = 0;
                self.window_hits = 0;
                self.window_total = 0;
                self.low_streak = 0;
            }
        }
    }

    #[inline(always)]
    fn record_hit(&mut self) {
        self.window_hits += 1;
        self.window_total += 1;
        if self.window_total >= ADAPT_WINDOW {
            self.evaluate_window();
        }
    }

    #[inline(always)]
    fn record_miss(&mut self) {
        self.window_total += 1;
        if self.window_total >= ADAPT_WINDOW {
            self.evaluate_window();
        }
    }

    #[inline]
    fn evaluate_window(&mut self) {
        let pct = (self.window_hits as u64 * 100 / self.window_total.max(1) as u64) as u32;
        if pct < MIN_HIT_RATE_PCT {
            self.low_streak += 1;
            if self.low_streak >= DISABLE_THRESHOLD {
                self.enabled = false;
                self.disabled_steps = 0;
            }
        } else {
            self.low_streak = 0;
        }
        self.window_hits = 0;
        self.window_total = 0;
    }
}

/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
unsafe fn advance_core_const<const USE_AVX2: bool>(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    if USE_AVX2 {
        #[cfg(target_arch = "x86_64")]
        {
            return unsafe { super::kernel::advance_core_avx2(current, next, ghost) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!("AVX2 backend selected on non-x86 target");
        }
    }
    super::kernel::advance_core_scalar(current, next, ghost)
}

/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn advance_tile_cached_impl<const USE_AVX2: bool, const TRACK_NEIGHBOR_INFLUENCE: bool>(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> TileAdvanceResult {
    let nb = unsafe { &*neighbors_ptr.add(idx) };
    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta = unsafe { &mut *meta_ptr.add(idx) };
    let missing_mask = meta.missing_mask;
    let tile_has_live = meta.has_live();

    if !tile_has_live {
        if missing_mask == MISSING_ALL_NEIGHBORS {
            debug_assert!(tile_is_empty(current));
            unsafe {
                if meta.alt_phase_dirty() {
                    std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
                } else {
                    debug_assert!(tile_is_empty(next));
                }
                *next_borders_north_ptr.add(idx) = 0;
                *next_borders_south_ptr.add(idx) = 0;
                *next_borders_west_ptr.add(idx) = 0;
                *next_borders_east_ptr.add(idx) = 0;
                *next_live_masks_ptr.add(idx) = 0;
            }
            meta.update_after_step(false, false);
            let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
                0
            } else {
                no_track_hint(false, true)
            };
            return TileAdvanceResult::new(
                false,
                false,
                missing_mask,
                0,
                neighbor_influence_mask,
                true,
            );
        }

        // Ultra-fast path: metadata says current tile is empty and halo has no
        // incoming live cells.
        let ghost_empty = unsafe { ghost_is_empty_from_live_masks_ptr(live_masks_read_ptr, nb) };

        if ghost_empty {
            debug_assert!(tile_is_empty(current));
            unsafe {
                if meta.alt_phase_dirty() {
                    std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
                } else {
                    debug_assert!(tile_is_empty(next));
                }
                *next_borders_north_ptr.add(idx) = 0;
                *next_borders_south_ptr.add(idx) = 0;
                *next_borders_west_ptr.add(idx) = 0;
                *next_borders_east_ptr.add(idx) = 0;
                *next_live_masks_ptr.add(idx) = 0;
            }
            meta.update_after_step(false, false);
            let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
                0
            } else {
                no_track_hint(false, false)
            };
            return TileAdvanceResult::new(
                false,
                false,
                missing_mask,
                0,
                neighbor_influence_mask,
                false,
            );
        }
    }

    let north_i = nb[0] as usize;
    let south_i = nb[1] as usize;
    let west_i = nb[2] as usize;
    let east_i = nb[3] as usize;
    let nw_i = nb[4] as usize;
    let ne_i = nb[5] as usize;
    let sw_i = nb[6] as usize;
    let se_i = nb[7] as usize;

    let ghost = GhostZone {
        north: unsafe { *borders_south_read_ptr.add(north_i) },
        south: unsafe { *borders_north_read_ptr.add(south_i) },
        west: unsafe { *borders_east_read_ptr.add(west_i) },
        east: unsafe { *borders_west_read_ptr.add(east_i) },
        nw: unsafe { (*live_masks_read_ptr.add(nw_i) & LIVE_SE) != 0 },
        ne: unsafe { (*live_masks_read_ptr.add(ne_i) & LIVE_SW) != 0 },
        sw: unsafe { (*live_masks_read_ptr.add(sw_i) & LIVE_NE) != 0 },
        se: unsafe { (*live_masks_read_ptr.add(se_i) & LIVE_NW) != 0 },
    };

    if !tile_has_live {
        debug_assert!(tile_is_empty(current));
        let (changed, border, has_live) =
            advance_core_empty_with_clear(next, &ghost, meta.alt_phase_dirty());
        let live_mask = border.live_mask();
        unsafe {
            *next_borders_north_ptr.add(idx) = border.north;
            *next_borders_south_ptr.add(idx) = border.south;
            *next_borders_west_ptr.add(idx) = border.west;
            *next_borders_east_ptr.add(idx) = border.east;
            *next_live_masks_ptr.add(idx) = live_mask;
        }
        let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
            if changed {
                unsafe {
                    neighbor_influence_mask_for_result(
                        borders_north_read_ptr,
                        borders_south_read_ptr,
                        borders_west_read_ptr,
                        borders_east_read_ptr,
                        idx,
                        &border,
                    )
                }
            } else {
                0
            }
        } else {
            no_track_hint(changed, false)
        };
        meta.update_after_step(changed, has_live);
        return TileAdvanceResult::new(
            changed,
            has_live,
            missing_mask,
            live_mask,
            neighbor_influence_mask,
            false,
        );
    }

    let cr = unsafe { &mut *cache };

    if cr.enabled {
        let (key_lo, key_hi) = hash_input(current, &ghost);
        let slot = key_lo as usize & CACHE_MASK;
        let entry = unsafe { cr.entries.get_unchecked(slot) };
        if entry.key_lo == key_lo
            && entry.key_hi == key_hi
            && entry.input_cells == *current
            && ghost_eq(&entry.input_ghost, &ghost)
        {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    entry.next_cells.as_ptr(),
                    next.as_mut_ptr(),
                    TILE_SIZE,
                );
                *next_borders_north_ptr.add(idx) = entry.border.north;
                *next_borders_south_ptr.add(idx) = entry.border.south;
                *next_borders_west_ptr.add(idx) = entry.border.west;
                *next_borders_east_ptr.add(idx) = entry.border.east;
                *next_live_masks_ptr.add(idx) = entry.live_mask;
            }
            let changed = entry.changed;
            let has_live = entry.has_live;
            let live_mask = entry.live_mask;
            let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
                if changed {
                    entry.neighbor_influence_mask
                } else {
                    0
                }
            } else {
                no_track_hint(changed, false)
            };
            meta.update_after_step(changed, has_live);
            cr.record_hit();
            return TileAdvanceResult::new(
                changed,
                has_live,
                missing_mask,
                live_mask,
                neighbor_influence_mask,
                false,
            );
        }
        let (changed, border, has_live) =
            unsafe { advance_core_const::<USE_AVX2>(current, next, &ghost) };
        let live_mask = border.live_mask();
        unsafe {
            *next_borders_north_ptr.add(idx) = border.north;
            *next_borders_south_ptr.add(idx) = border.south;
            *next_borders_west_ptr.add(idx) = border.west;
            *next_borders_east_ptr.add(idx) = border.east;
            *next_live_masks_ptr.add(idx) = live_mask;
        }
        let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
            if changed {
                unsafe {
                    neighbor_influence_mask_for_result(
                        borders_north_read_ptr,
                        borders_south_read_ptr,
                        borders_west_read_ptr,
                        borders_east_read_ptr,
                        idx,
                        &border,
                    )
                }
            } else {
                0
            }
        } else {
            no_track_hint(changed, false)
        };
        meta.update_after_step(changed, has_live);
        let em = unsafe { cr.entries.get_unchecked_mut(slot) };
        em.key_lo = key_lo;
        em.key_hi = key_hi;
        unsafe {
            std::ptr::copy_nonoverlapping(current.as_ptr(), em.input_cells.as_mut_ptr(), TILE_SIZE);
        }
        em.input_ghost = ghost;
        unsafe {
            std::ptr::copy_nonoverlapping(next.as_ptr(), em.next_cells.as_mut_ptr(), TILE_SIZE);
        }
        em.border = border;
        em.changed = changed;
        em.has_live = has_live;
        em.live_mask = live_mask;
        em.neighbor_influence_mask = neighbor_influence_mask;
        cr.record_miss();
        return TileAdvanceResult::new(
            changed,
            has_live,
            missing_mask,
            live_mask,
            neighbor_influence_mask,
            false,
        );
    }

    // Cache disabled path.
    let (changed, border, has_live) =
        unsafe { advance_core_const::<USE_AVX2>(current, next, &ghost) };
    let live_mask = border.live_mask();
    unsafe {
        *next_borders_north_ptr.add(idx) = border.north;
        *next_borders_south_ptr.add(idx) = border.south;
        *next_borders_west_ptr.add(idx) = border.west;
        *next_borders_east_ptr.add(idx) = border.east;
        *next_live_masks_ptr.add(idx) = live_mask;
    }
    let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
        if changed {
            unsafe {
                neighbor_influence_mask_for_result(
                    borders_north_read_ptr,
                    borders_south_read_ptr,
                    borders_west_read_ptr,
                    borders_east_read_ptr,
                    idx,
                    &border,
                )
            }
        } else {
            0
        }
    } else {
        no_track_hint(changed, false)
    };
    meta.update_after_step(changed, has_live);
    TileAdvanceResult::new(
        changed,
        has_live,
        missing_mask,
        live_mask,
        neighbor_influence_mask,
        false,
    )
}

/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_cached_scalar_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_cached_impl::<false, true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
            cache,
        )
    }
}

/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_cached_scalar_no_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_cached_impl::<false, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
            cache,
        )
    }
}

#[cfg(target_arch = "x86_64")]
/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_cached_avx2_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_cached_impl::<true, true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
            cache,
        )
    }
}

#[cfg(target_arch = "x86_64")]
/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_cached_avx2_no_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_cached_impl::<true, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
            cache,
        )
    }
}
