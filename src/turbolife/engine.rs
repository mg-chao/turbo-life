//! TurboLife engine core.
//!
//! Split-buffer architecture: cell_bufs[phase] is read, cell_bufs[1-phase] is written.
//! After the parallel phase, flip_cells() + flip_borders() makes next gen current.
//! Changed tracking uses a pre-allocated atomic byte array to eliminate per-chunk
//! Vec allocations. Prefetch depth is 3 tiles ahead.

use rayon::prelude::*;
use std::sync::OnceLock;

use super::activity::{prune_and_expand, rebuild_active_set};
use super::arena::TileArena;
use super::kernel::{advance_tile_split, advance_core};
use super::sync::gather_ghost_zone;
use super::tile::{self, POPULATION_UNKNOWN};

struct SendPtr<T> {
    _inner: *mut T,
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> { fn clone(&self) -> Self { *self } }
impl<T> SendPtr<T> {
    #[inline(always)]
    fn new(ptr: *mut T) -> Self { Self { _inner: ptr } }
    #[inline(always)]
    fn get(&self) -> *mut T { self._inner }
}

struct SendConstPtr<T> {
    _inner: *const T,
}
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
impl<T> Copy for SendConstPtr<T> {}
impl<T> Clone for SendConstPtr<T> { fn clone(&self) -> Self { *self } }
impl<T> SendConstPtr<T> {
    #[inline(always)]
    fn new(ptr: *const T) -> Self { Self { _inner: ptr } }
    #[inline(always)]
    fn get(&self) -> *const T { self._inner }
}

const TILE_SIZE_I64: i64 = 64;
const PARALLEL_KERNEL_MIN_ACTIVE: usize = 128;
const PARALLEL_KERNEL_TILES_PER_THREAD: usize = 16;
const PARALLEL_KERNEL_MIN_CHUNKS: usize = 2;
const KERNEL_CHUNK_MIN: usize = 32;
const KERNEL_CHUNK_MAX: usize = 256;
const ENV_OVERRIDE_THREADS: &str = "TURBOLIFE_NUM_THREADS";
const ENV_MAX_THREADS: &str = "TURBOLIFE_MAX_THREADS";
const PREFETCH_DISTANCE: usize = 5;

static TURBOLIFE_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
static PHYSICAL_CORES: OnceLock<usize> = OnceLock::new();

#[inline]
fn parse_positive_env(var: &str) -> Option<usize> {
    let raw = std::env::var(var).ok()?;
    let parsed = raw.trim().parse::<usize>().ok()?;
    (parsed > 0).then_some(parsed)
}

#[inline]
fn physical_core_count() -> usize {
    *PHYSICAL_CORES.get_or_init(|| num_cpus::get_physical().max(1))
}

fn desired_thread_count() -> usize {
    let mut threads = parse_positive_env(ENV_OVERRIDE_THREADS)
        .or_else(|| parse_positive_env("RAYON_NUM_THREADS"))
        .unwrap_or_else(physical_core_count);
    if let Some(max_threads) = parse_positive_env(ENV_MAX_THREADS) {
        threads = threads.min(max_threads);
    }
    threads.max(1)
}

#[inline]
fn install_in_pool<T>(f: impl FnOnce() -> T + Send) -> T
where T: Send,
{
    let pool = TURBOLIFE_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(desired_thread_count())
            .build()
            .expect("failed to build TurboLife rayon thread pool")
    });
    pool.install(f)
}

#[inline]
fn kernel_chunk_size(active_len: usize, threads: usize) -> usize {
    // Target more chunks than threads for better work-stealing balance.
    // On many-core machines (24+), over-decomposition helps hide latency.
    let multiplier = if threads >= 16 { 8 } else { 4 };
    let target_chunks = threads.saturating_mul(multiplier).max(1);
    let size = active_len.div_ceil(target_chunks);
    size.clamp(KERNEL_CHUNK_MIN, KERNEL_CHUNK_MAX)
}

#[inline]
fn effective_parallel_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 || active_len < PARALLEL_KERNEL_MIN_ACTIVE {
        return 1;
    }
    let by_work = active_len / PARALLEL_KERNEL_TILES_PER_THREAD;
    let by_chunks = active_len / KERNEL_CHUNK_MIN;
    let effective = by_work.min(by_chunks).min(thread_count).max(1);
    if effective < PARALLEL_KERNEL_MIN_CHUNKS { 1 } else { effective }
}

#[inline]
fn tuned_parallel_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 { return 1; }
    let mut effective = effective_parallel_threads(active_len, thread_count);
    if effective <= 1 { return 1; }
    // Scale caps with core count — the old caps (2/4/12) starved many-core machines.
    // On a 24-core machine the old cap of 12 for <4096 tiles left half the cores idle.
    let cores = thread_count;
    let tuned_cap = if active_len < 256 { (cores / 4).max(2) }
        else if active_len < 512 { (cores / 3).max(4) }
        else if active_len < 1_024 { (cores / 2).max(6) }
        else if active_len < 4_096 { (cores * 3 / 4).max(8) }
        else { cores };
    effective = effective.min(tuned_cap).min(thread_count);
    effective.max(1)
}

pub struct TurboLife {
    arena: TileArena,
    generation: u64,
    population_cache: Option<u64>,
    /// Plain byte array for changed flags. No atomics needed because each tile
    /// in the active_set appears exactly once, and par_chunks gives non-overlapping
    /// slices — so each index is written by exactly one thread.
    changed_flags: Vec<u8>,
}

impl TurboLife {
    pub fn new() -> Self {
        Self {
            arena: TileArena::new(),
            generation: 0,
            population_cache: Some(0),
            changed_flags: vec![0u8],
        }
    }

    /// Ensure changed_flags has enough slots for all tile indices.
    #[inline]
    fn ensure_changed_flags(&mut self) {
        let needed = self.arena.cell_bufs[0].len();
        if self.changed_flags.len() < needed {
            self.changed_flags.resize(needed, 0u8);
        }
    }

    /// Set a changed flag via raw pointer (safe because each index is written by one thread).
    #[inline(always)]
    unsafe fn set_changed_flag(flags: *mut u8, idx: usize) {
        unsafe { *flags.add(idx) = 1; }
    }

    pub fn set_cell(&mut self, x: i64, y: i64, alive: bool) {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        if !alive && self.arena.idx_at(tile_coord).is_none() {
            return;
        }

        let idx = self.arena.allocate(tile_coord);
        self.ensure_changed_flags();

        let buf = self.arena.current_buf_mut(idx);
        tile::set_local_cell(buf, local_x, local_y, alive);
        let border = tile::recompute_border(buf);
        let has_live = buf.iter().any(|&row| row != 0);

        *self.arena.border_mut(idx) = border;
        {
            let meta = self.arena.meta_mut(idx);
            meta.population = POPULATION_UNKNOWN;
            meta.set_has_live(has_live);
        }
        self.population_cache = None;
        self.arena.mark_changed(idx);

        if alive {
            let (tx, ty) = tile_coord;
            let on_south = local_y == 0;
            let on_north = local_y == 63;
            let on_west = local_x == 0;
            let on_east = local_x == 63;

            if on_north { self.arena.ensure_neighbor(tx, ty + 1); }
            if on_south { self.arena.ensure_neighbor(tx, ty - 1); }
            if on_west  { self.arena.ensure_neighbor(tx - 1, ty); }
            if on_east  { self.arena.ensure_neighbor(tx + 1, ty); }
            if on_north && on_west { self.arena.ensure_neighbor(tx - 1, ty + 1); }
            if on_north && on_east { self.arena.ensure_neighbor(tx + 1, ty + 1); }
            if on_south && on_west { self.arena.ensure_neighbor(tx - 1, ty - 1); }
            if on_south && on_east { self.arena.ensure_neighbor(tx + 1, ty - 1); }

            self.ensure_changed_flags();
        }
    }

    pub fn get_cell(&self, x: i64, y: i64) -> bool {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        self.arena
            .idx_at(tile_coord)
            .map(|idx| {
                self.arena.meta(idx).occupied()
                    && tile::get_local_cell(self.arena.current_buf(idx), local_x, local_y)
            })
            .unwrap_or(false)
    }

    fn step_impl(&mut self) {
        rebuild_active_set(&mut self.arena);

        if self.arena.active_set.is_empty() {
            self.generation += 1;
            return;
        }

        self.ensure_changed_flags();

        let active_len = self.arena.active_set.len();
        let bp = self.arena.border_phase;
        let cp = self.arena.cell_phase;
        let thread_count = rayon::current_num_threads().max(1);
        let effective_threads = tuned_parallel_threads(active_len, thread_count);
        let run_parallel = effective_threads > 1;

        // Split borrows for cell_bufs and borders using split_at_mut.
        let (cb_lo, cb_hi) = self.arena.cell_bufs.split_at_mut(1);
        let (current_vec, next_vec) = if cp == 0 {
            (&cb_lo[0], &mut cb_hi[0])
        } else {
            (&cb_hi[0], &mut cb_lo[0])
        };
        let (bd_lo, bd_hi) = self.arena.borders.split_at_mut(1);
        let (borders_read, next_borders_vec) = if bp == 0 {
            (bd_lo[0].as_slice(), &mut bd_hi[0])
        } else {
            (bd_hi[0].as_slice(), &mut bd_lo[0])
        };

        if !run_parallel {
            // Serial path
            let neighbors = &self.arena.neighbors;

            for i in 0..active_len {
                let idx = self.arena.active_set[i];
                let ghost = gather_ghost_zone(idx, borders_read, neighbors);

                let current = &current_vec[idx.index()].0;
                let next = &mut next_vec[idx.index()].0;
                let next_border = &mut next_borders_vec[idx.index()];
                let meta = &mut self.arena.meta[idx.index()];

                let (changed, border, has_live) = advance_core(current, next, &ghost);

                *next_border = border;
                meta.set_changed(changed);
                meta.population = POPULATION_UNKNOWN;
                meta.set_has_live(has_live);

                if changed {
                    if !meta.in_changed_list() {
                        meta.set_in_changed_list(true);
                        self.arena.changed_list.push(idx);
                    }
                }
            }
        } else {
            // Parallel path: plain byte flags (no atomics — each tile written by one thread).
            let active_set = &self.arena.active_set;
            let current_ptr = SendConstPtr::new(current_vec.as_ptr());
            let next_ptr = SendPtr::new(next_vec.as_mut_ptr());
            let meta_ptr = SendPtr::new(self.arena.meta.as_mut_ptr());
            let neighbors = &self.arena.neighbors;
            let next_borders_ptr = SendPtr::new(next_borders_vec.as_mut_ptr());
            let chunk_size = kernel_chunk_size(active_len, effective_threads);

            // Clear flags for active tiles.
            for &idx in active_set {
                self.changed_flags[idx.index()] = 0;
            }

            let flags_ptr = SendPtr::new(self.changed_flags.as_mut_ptr());

            active_set.par_chunks(chunk_size).for_each({
                let current_ptr = current_ptr;
                let next_ptr = next_ptr;
                let meta_ptr = meta_ptr;
                let next_borders_ptr = next_borders_ptr;
                let flags_ptr = flags_ptr;
                move |chunk| {
                    for (ci, &idx) in chunk.iter().enumerate() {
                        // Deep prefetch: read-side and write-side for upcoming tiles.
                        if ci + PREFETCH_DISTANCE < chunk.len() {
                            let pf_idx = chunk[ci + PREFETCH_DISTANCE];
                            unsafe {
                                let cell_ptr = current_ptr.get().add(pf_idx.index()) as *const i8;
                                let nb_ptr = neighbors.as_ptr().add(pf_idx.index()) as *const i8;
                                let border_ptr = borders_read.as_ptr().add(pf_idx.index()) as *const i8;
                                let next_cell_ptr = next_ptr.get().add(pf_idx.index()) as *const i8;
                                let next_meta_ptr = meta_ptr.get().add(pf_idx.index()) as *const i8;
                                let next_border_ptr = next_borders_ptr.get().add(pf_idx.index()) as *const i8;
                                #[cfg(target_arch = "x86_64")]
                                {
                                    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                                    _mm_prefetch(cell_ptr, _MM_HINT_T0);
                                    _mm_prefetch(cell_ptr.add(64), _MM_HINT_T0);
                                    _mm_prefetch(cell_ptr.add(128), _MM_HINT_T0);
                                    _mm_prefetch(cell_ptr.add(192), _MM_HINT_T0);
                                    _mm_prefetch(nb_ptr, _MM_HINT_T0);
                                    _mm_prefetch(border_ptr, _MM_HINT_T0);
                                    _mm_prefetch(next_cell_ptr, _MM_HINT_T0);
                                    _mm_prefetch(next_cell_ptr.add(64), _MM_HINT_T0);
                                    _mm_prefetch(next_cell_ptr.add(128), _MM_HINT_T0);
                                    _mm_prefetch(next_cell_ptr.add(192), _MM_HINT_T0);
                                    _mm_prefetch(next_meta_ptr, _MM_HINT_T0);
                                    _mm_prefetch(next_border_ptr, _MM_HINT_T0);
                                }
                            }
                        }

                        let ghost = gather_ghost_zone(idx, borders_read, neighbors);

                        unsafe {
                            let tile_changed = advance_tile_split(
                                current_ptr.get(),
                                next_ptr.get(),
                                meta_ptr.get(),
                                next_borders_ptr.get(),
                                idx.index(),
                                &ghost,
                            );

                            if tile_changed {
                                Self::set_changed_flag(flags_ptr.get(), idx.index());
                            }
                        }
                    }
                }
            });

            // Collect changed tiles from flags (serial, very fast).
            for &idx in active_set {
                if self.changed_flags[idx.index()] != 0 {
                    let m = &mut self.arena.meta[idx.index()];
                    if !m.in_changed_list() {
                        m.set_in_changed_list(true);
                        self.arena.changed_list.push(idx);
                    }
                }
            }
        }

        self.arena.flip_borders();
        self.arena.flip_cells();

        prune_and_expand(&mut self.arena);

        self.population_cache = None;
        self.generation += 1;
    }

    pub fn step(&mut self) {
        install_in_pool(|| {
            self.step_impl();
        });
    }

    pub fn step_n(&mut self, n: u64) {
        install_in_pool(|| {
            for _ in 0..n {
                self.step_impl();
            }
        });
    }

    pub fn population(&mut self) -> u64 {
        if let Some(cached) = self.population_cache {
            return cached;
        }

        let cp = self.arena.cell_phase;
        let mut total = 0u64;
        for (buf, meta) in self.arena.cell_bufs[cp].iter().zip(self.arena.meta.iter_mut()) {
            if !meta.occupied() { continue; }
            if !meta.has_live() {
                meta.population = 0;
                continue;
            }
            let pop = if meta.population != POPULATION_UNKNOWN {
                meta.population
            } else {
                let computed = tile::compute_population(&buf.0);
                meta.population = computed;
                computed
            };
            total += pop as u64;
        }

        self.population_cache = Some(total);
        total
    }

    pub fn is_empty(&mut self) -> bool {
        self.population() == 0
    }

    pub fn bounds(&self) -> Option<(i64, i64, i64, i64)> {
        let mut min_x = i64::MAX;
        let mut min_y = i64::MAX;
        let mut max_x = i64::MIN;
        let mut max_y = i64::MIN;
        let mut seen = false;

        self.for_each_live(|x, y| {
            seen = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        });

        seen.then_some((min_x, min_y, max_x, max_y))
    }

    pub fn for_each_live<F: FnMut(i64, i64)>(&self, mut f: F) {
        let cp = self.arena.cell_phase;
        for i in 0..self.arena.cell_bufs[cp].len() {
            let meta = &self.arena.meta[i];
            if !meta.occupied() { continue; }

            let current = &self.arena.cell_bufs[cp][i].0;
            let coord = self.arena.coords[i];
            let base_x = coord.0 * TILE_SIZE_I64;
            let base_y = coord.1 * TILE_SIZE_I64;

            for (row_index, &row) in current.iter().enumerate() {
                let mut bits = row;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as i64;
                    f(base_x + bit, base_y + row_index as i64);
                    bits &= bits - 1;
                }
            }
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::TurboLife;

    #[test]
    fn empty_universe_advances_generation() {
        let mut engine = TurboLife::new();
        assert_eq!(engine.generation(), 0);
        engine.step();
        assert_eq!(engine.generation(), 1);
        assert_eq!(engine.population(), 0);
    }
}
