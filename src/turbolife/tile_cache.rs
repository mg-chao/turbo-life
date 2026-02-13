use super::tile::{BorderData, CellBuf, GhostZone, TILE_SIZE};

const CACHE_SIZE: usize = 1 << 16;
const CACHE_MASK: usize = CACHE_SIZE - 1;
const ADAPT_WINDOW: u32 = 512;
const MIN_HIT_RATE_PCT: u32 = 20;
const DISABLE_THRESHOLD: u32 = 3;
const REPROBE_INTERVAL: u32 = 8;

#[repr(C, align(64))]
#[derive(Clone)]
struct CacheEntry {
    key: u64,
    next_cells: [u64; TILE_SIZE],
    border: BorderData,
    changed: bool,
    has_live: bool,
}

impl CacheEntry {
    const EMPTY: Self = Self {
        key: 0,
        next_cells: [0u64; TILE_SIZE],
        border: BorderData { north: 0, south: 0, west: 0, east: 0, corners: 0 },
        changed: false,
        has_live: false,
    };
}

#[inline(always)]
fn hash_input(cells: &[u64; TILE_SIZE], ghost: &GhostZone) -> u64 {
    const K0: u64 = 0xa076_1d64_78bd_642f;
    const K1: u64 = 0xe703_7ed1_a0b4_28db;
    const K2: u64 = 0x8ebc_6af0_9c88_c6e3;
    const K3: u64 = 0x5899_65cc_7537_4cc3;
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut i = 0;
    while i < TILE_SIZE {
        let lo = (cells[i] ^ K0).wrapping_mul(cells[i + 1] ^ K1);
        let hi = (cells[i + 2] ^ K2).wrapping_mul(cells[i + 3] ^ K3);
        h = h ^ lo ^ hi;
        h = h.wrapping_mul(K0);
        i += 4;
    }
    h ^= ghost.north.wrapping_mul(K1);
    h ^= ghost.south.wrapping_mul(K2);
    h ^= ghost.west.wrapping_mul(K3);
    h ^= ghost.east.wrapping_mul(K0);
    let c = (ghost.nw as u64) | ((ghost.ne as u64) << 1)
        | ((ghost.sw as u64) << 2) | ((ghost.se as u64) << 3);
    h ^= c.wrapping_mul(K1);
    h ^= h >> 32;
    h = h.wrapping_mul(K0);
    h ^= h >> 29;
    h | 1
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
        if self.window_total >= ADAPT_WINDOW { self.evaluate_window(); }
    }

    #[inline(always)]
    fn record_miss(&mut self) {
        self.window_total += 1;
        if self.window_total >= ADAPT_WINDOW { self.evaluate_window(); }
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
pub unsafe fn advance_tile_cached(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut super::tile::TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    idx: usize,
    backend: super::kernel::KernelBackend,
    cache: *mut TileCache,
) -> bool {
    use super::arena::SENTINEL_IDX;
    use super::tile::{NO_NEIGHBOR, POPULATION_UNKNOWN};
    debug_assert_eq!(SENTINEL_IDX, 0);

    let nb = unsafe { &*neighbors_ptr.add(idx) };

    #[inline(always)]
    unsafe fn sentinel_or(raw: u32) -> usize {
        let present_mask = (raw != NO_NEIGHBOR) as usize;
        (raw as usize) & present_mask.wrapping_neg()
    }

    let north_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[0])) };
    let south_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[1])) };
    let west_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[2])) };
    let east_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[3])) };
    let nw_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[4])) };
    let ne_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[5])) };
    let sw_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[6])) };
    let se_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[7])) };

    let ghost = GhostZone {
        north: north_b.south, south: south_b.north,
        west: west_b.east, east: east_b.west,
        nw: nw_b.se(), ne: ne_b.sw(), sw: sw_b.ne(), se: se_b.nw(),
    };

    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta = unsafe { &mut *meta_ptr.add(idx) };

    // Ultra-fast path: empty tile + empty ghost.
    if ghost.north | ghost.south | ghost.west | ghost.east == 0
        && !ghost.nw && !ghost.ne && !ghost.sw && !ghost.se
    {
        let mut any = 0u64;
        let mut i = 0;
        while i < TILE_SIZE {
            any |= current[i] | current[i+1] | current[i+2] | current[i+3];
            if any != 0 { break; }
            i += 4;
        }
        if any == 0 {
            unsafe {
                std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
                *next_borders_ptr.add(idx) = BorderData {
                    north: 0, south: 0, west: 0, east: 0, corners: 0,
                };
            }
            meta.set_changed(false);
            return false;
        }
    }

    let cr = unsafe { &mut *cache };

    if cr.enabled {
        let key = hash_input(current, &ghost);
        let slot = key as usize & CACHE_MASK;
        let entry = unsafe { cr.entries.get_unchecked(slot) };
        if entry.key == key {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    entry.next_cells.as_ptr(), next.as_mut_ptr(), TILE_SIZE,
                );
                *next_borders_ptr.add(idx) = entry.border;
            }
            let changed = entry.changed;
            meta.set_changed(changed);
            if changed {
                meta.population = POPULATION_UNKNOWN;
                meta.set_has_live(entry.has_live);
            }
            cr.record_hit();
            return changed;
        }
        let (changed, border, has_live) =
            super::kernel::advance_core(current, next, &ghost, backend);
        unsafe { *next_borders_ptr.add(idx) = border; }
        meta.set_changed(changed);
        if changed {
            meta.population = POPULATION_UNKNOWN;
            meta.set_has_live(has_live);
        }
        let em = unsafe { cr.entries.get_unchecked_mut(slot) };
        em.key = key;
        unsafe {
            std::ptr::copy_nonoverlapping(
                next.as_ptr(), em.next_cells.as_mut_ptr(), TILE_SIZE,
            );
        }
        em.border = border;
        em.changed = changed;
        em.has_live = has_live;
        cr.record_miss();
        return changed;
    }

    // Cache disabled path.
    let (changed, border, has_live) =
        super::kernel::advance_core(current, next, &ghost, backend);
    unsafe { *next_borders_ptr.add(idx) = border; }
    meta.set_changed(changed);
    if changed {
        meta.population = POPULATION_UNKNOWN;
        meta.set_has_live(has_live);
    }
    changed
}
