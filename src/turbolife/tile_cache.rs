use super::kernel::{ghost_is_empty_from_live_masks, tile_is_empty};
use super::tile::{BorderData, CellBuf, GhostZone, MISSING_ALL_NEIGHBORS, TILE_SIZE};

const CACHE_SIZE: usize = 1 << 13;
const CACHE_MASK: usize = CACHE_SIZE - 1;
const ADAPT_WINDOW: u32 = 512;
const MIN_HIT_RATE_PCT: u32 = 20;
const DISABLE_THRESHOLD: u32 = 3;
const REPROBE_INTERVAL: u32 = 8;

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
            corners: 0,
            live_mask: 0,
        },
        changed: false,
        has_live: false,
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
unsafe fn advance_tile_cached_impl<const USE_AVX2: bool>(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> bool {
    let nb = unsafe { &*neighbors_ptr.add(idx) };
    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta = unsafe { &mut *meta_ptr.add(idx) };

    if !meta.has_live() {
        if meta.missing_mask == MISSING_ALL_NEIGHBORS {
            debug_assert!(tile_is_empty(current));
            unsafe {
                std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
                *next_borders_ptr.add(idx) = BorderData::default();
                *next_live_masks_ptr.add(idx) = 0;
            }
            meta.update_after_step(false, false);
            return false;
        }

        // Ultra-fast path: metadata says current tile is empty and halo has no
        // incoming live cells.
        let ghost_empty = ghost_is_empty_from_live_masks([
            unsafe { *live_masks_read_ptr.add(nb[0] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[1] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[2] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[3] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[4] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[5] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[6] as usize) },
            unsafe { *live_masks_read_ptr.add(nb[7] as usize) },
        ]);

        if ghost_empty {
            debug_assert!(tile_is_empty(current));
            unsafe {
                std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
                *next_borders_ptr.add(idx) = BorderData::default();
                *next_live_masks_ptr.add(idx) = 0;
            }
            meta.update_after_step(false, false);
            return false;
        }
    }

    let north_b = unsafe { &*borders_read_ptr.add(nb[0] as usize) };
    let south_b = unsafe { &*borders_read_ptr.add(nb[1] as usize) };
    let west_b = unsafe { &*borders_read_ptr.add(nb[2] as usize) };
    let east_b = unsafe { &*borders_read_ptr.add(nb[3] as usize) };
    let nw_b = unsafe { &*borders_read_ptr.add(nb[4] as usize) };
    let ne_b = unsafe { &*borders_read_ptr.add(nb[5] as usize) };
    let sw_b = unsafe { &*borders_read_ptr.add(nb[6] as usize) };
    let se_b = unsafe { &*borders_read_ptr.add(nb[7] as usize) };

    let ghost = GhostZone {
        north: north_b.south,
        south: south_b.north,
        west: west_b.east,
        east: east_b.west,
        nw: nw_b.se(),
        ne: ne_b.sw(),
        sw: sw_b.ne(),
        se: se_b.nw(),
    };

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
                *next_borders_ptr.add(idx) = entry.border;
                *next_live_masks_ptr.add(idx) = entry.border.live_mask;
            }
            let changed = entry.changed;
            meta.update_after_step(changed, entry.has_live);
            cr.record_hit();
            return changed;
        }
        let (changed, border, has_live) =
            unsafe { advance_core_const::<USE_AVX2>(current, next, &ghost) };
        unsafe {
            *next_borders_ptr.add(idx) = border;
            *next_live_masks_ptr.add(idx) = border.live_mask;
        }
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
        cr.record_miss();
        return changed;
    }

    // Cache disabled path.
    let (changed, border, has_live) =
        unsafe { advance_core_const::<USE_AVX2>(current, next, &ghost) };
    unsafe {
        *next_borders_ptr.add(idx) = border;
        *next_live_masks_ptr.add(idx) = border.live_mask;
    }
    meta.update_after_step(changed, has_live);
    changed
}

/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_cached_scalar(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> bool {
    unsafe {
        advance_tile_cached_impl::<false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_ptr,
            borders_read_ptr,
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
pub unsafe fn advance_tile_cached_avx2(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    cache: *mut TileCache,
) -> bool {
    unsafe {
        advance_tile_cached_impl::<true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_ptr,
            borders_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
            cache,
        )
    }
}
