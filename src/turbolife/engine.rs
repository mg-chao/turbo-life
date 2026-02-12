use rayon::prelude::*;
use std::sync::OnceLock;
use std::time::Instant;

use super::activity::{prune_and_expand, rebuild_active_set};
use super::arena::TileArena;
use super::kernel::{KernelBackend, advance_core, advance_tile_split};
use super::sync::gather_ghost_zone_raw;
use super::tile::{self, POPULATION_UNKNOWN};

struct SendPtr<T> {
    inner: *mut T,
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn new(ptr: *mut T) -> Self {
        Self { inner: ptr }
    }
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.inner
    }
}

struct SendConstPtr<T> {
    inner: *const T,
}
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
impl<T> Copy for SendConstPtr<T> {}
impl<T> Clone for SendConstPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> SendConstPtr<T> {
    #[inline(always)]
    fn new(ptr: *const T) -> Self {
        Self { inner: ptr }
    }
    #[inline(always)]
    fn get(&self) -> *const T {
        self.inner
    }
}

const TILE_SIZE_I64: i64 = 64;
const PARALLEL_KERNEL_MIN_ACTIVE: usize = 128;
const PARALLEL_KERNEL_TILES_PER_THREAD: usize = 16;
const PARALLEL_KERNEL_MIN_CHUNKS: usize = 2;
const KERNEL_CHUNK_MIN: usize = 32;

static PHYSICAL_CORES: OnceLock<usize> = OnceLock::new();

#[inline]
fn detect_kernel_backend() -> KernelBackend {
    calibrate_auto_backend()
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn next_seed(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

#[cfg(target_arch = "x86_64")]
fn benchmark_backend(backend: KernelBackend) -> u128 {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut current = [0u64; tile::TILE_SIZE];
    let mut next = [0u64; tile::TILE_SIZE];
    for row in &mut current {
        *row = next_seed(&mut seed);
    }

    let ghost = tile::GhostZone {
        north: next_seed(&mut seed),
        south: next_seed(&mut seed),
        west: next_seed(&mut seed),
        east: next_seed(&mut seed),
        nw: (next_seed(&mut seed) & 1) != 0,
        ne: (next_seed(&mut seed) & 1) != 0,
        sw: (next_seed(&mut seed) & 1) != 0,
        se: (next_seed(&mut seed) & 1) != 0,
    };

    let rounds = 1536usize;
    let start = Instant::now();
    let mut checksum = 0u64;
    for round in 0..rounds {
        let (changed, border, has_live) = advance_core(&current, &mut next, &ghost, backend);
        checksum ^= border.north
            ^ border.south
            ^ border.west
            ^ border.east
            ^ border.corners as u64
            ^ changed as u64
            ^ has_live as u64;

        if round & 1 == 0 {
            current = next;
            next = [0u64; tile::TILE_SIZE];
        }
    }
    std::hint::black_box(checksum);
    start.elapsed().as_nanos()
}

#[cfg(target_arch = "x86_64")]
fn calibrate_auto_backend() -> KernelBackend {
    if !std::is_x86_feature_detected!("avx2") {
        return KernelBackend::Scalar;
    }

    let scalar_ns = benchmark_backend(KernelBackend::Scalar);
    let avx2_ns = benchmark_backend(KernelBackend::Avx2);
    if avx2_ns <= scalar_ns {
        KernelBackend::Avx2
    } else {
        KernelBackend::Scalar
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn calibrate_auto_backend() -> KernelBackend {
    KernelBackend::Scalar
}

#[inline]
fn physical_core_count() -> usize {
    *PHYSICAL_CORES.get_or_init(|| num_cpus::get_physical().max(1))
}

#[inline]
fn effective_parallel_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 || active_len < PARALLEL_KERNEL_MIN_ACTIVE {
        return 1;
    }
    let by_work = active_len / PARALLEL_KERNEL_TILES_PER_THREAD;
    let by_chunks = active_len / KERNEL_CHUNK_MIN;
    let effective = by_work.min(by_chunks).min(thread_count).max(1);
    if effective < PARALLEL_KERNEL_MIN_CHUNKS {
        1
    } else {
        effective
    }
}

#[inline]
fn memory_parallel_cap(thread_count: usize) -> usize {
    if thread_count <= 8 {
        thread_count
    } else {
        thread_count.saturating_mul(2).div_ceil(3).max(8)
    }
}

#[inline]
fn tuned_parallel_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 {
        return 1;
    }
    let mut effective = effective_parallel_threads(active_len, thread_count);
    if effective <= 1 {
        return 1;
    }
    let bw_cap = memory_parallel_cap(thread_count);
    let tuned_cap = if active_len < 256 {
        bw_cap.min(2)
    } else if active_len < 512 {
        bw_cap.min(4)
    } else if active_len < 2_048 {
        bw_cap.min(6)
    } else {
        bw_cap
    };
    effective = effective.min(tuned_cap).min(thread_count);
    effective.max(1)
}

/// Resolve the thread count from a config, falling back to auto-detect.
fn resolve_thread_count(config: &TurboLifeConfig) -> usize {
    let mut threads = config
        .thread_count
        .unwrap_or_else(|| memory_parallel_cap(physical_core_count()));
    if let Some(cap) = config.max_threads {
        threads = threads.min(cap);
    }
    threads.max(1)
}

/// Resolve the kernel backend from a config, falling back to auto-calibrate.
fn resolve_kernel_backend(config: &TurboLifeConfig) -> KernelBackend {
    if let Some(backend) = config.kernel {
        // Validate AVX2 availability at runtime.
        if backend == KernelBackend::Avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    return KernelBackend::Avx2;
                }
            }
            return KernelBackend::Scalar;
        }
        return backend;
    }
    detect_kernel_backend()
}

/// Configuration for a TurboLife engine instance.
///
/// Use `TurboLifeConfig::default()` for auto-tuned defaults, or customise
/// individual knobs via the builder methods.
#[derive(Clone, Debug)]
pub struct TurboLifeConfig {
    /// Number of threads for the compute pool.
    /// `None` means auto-detect (physical cores, memory-bandwidth capped).
    pub thread_count: Option<usize>,
    /// Hard upper bound on threads regardless of auto-detection.
    /// `None` means no additional cap beyond `thread_count`.
    pub max_threads: Option<usize>,
    /// Kernel backend selection.
    /// `None` means auto-calibrate (benchmark Scalar vs AVX2 at startup).
    pub kernel: Option<KernelBackend>,
}

impl Default for TurboLifeConfig {
    fn default() -> Self {
        Self {
            thread_count: None,
            max_threads: None,
            kernel: None,
        }
    }
}

impl TurboLifeConfig {
    /// Set an explicit thread count for the compute pool.
    pub fn thread_count(mut self, n: usize) -> Self {
        self.thread_count = Some(n.max(1));
        self
    }

    /// Set a hard upper bound on threads.
    pub fn max_threads(mut self, n: usize) -> Self {
        self.max_threads = Some(n.max(1));
        self
    }

    /// Force a specific kernel backend.
    pub fn kernel(mut self, backend: KernelBackend) -> Self {
        self.kernel = Some(backend);
        self
    }
}

pub struct TurboLife {
    arena: TileArena,
    generation: u64,
    population_cache: Option<u64>,
    pool: rayon::ThreadPool,
    backend: KernelBackend,
    /// Reusable per-tile marker for batch dedup in `set_cells`.
    /// Indexed by `TileIdx.index()`, grown on demand.
    touched_flags: Vec<bool>,
}

impl TurboLife {
    pub fn new() -> Self {
        Self::with_config(TurboLifeConfig::default())
    }

    /// Create a TurboLife engine with explicit configuration.
    pub fn with_config(config: TurboLifeConfig) -> Self {
        let threads = resolve_thread_count(&config);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build TurboLife rayon thread pool");
        let backend = resolve_kernel_backend(&config);

        Self {
            arena: TileArena::new(),
            generation: 0,
            population_cache: Some(0),
            pool,
            backend,
            touched_flags: Vec::new(),
        }
    }

    pub fn set_cell(&mut self, x: i64, y: i64, alive: bool) {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        let idx = match self.arena.idx_at(tile_coord) {
            Some(existing) => existing,
            None if alive => self.arena.allocate_absent(tile_coord),
            None => return,
        };

        let buf = self.arena.current_buf_mut(idx);
        tile::set_local_cell(buf, local_x, local_y, alive);
        let (border, has_live) = tile::recompute_border_and_has_live(buf);

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

            if on_north {
                self.arena.ensure_neighbor(tx, ty + 1);
            }
            if on_south {
                self.arena.ensure_neighbor(tx, ty - 1);
            }
            if on_west {
                self.arena.ensure_neighbor(tx - 1, ty);
            }
            if on_east {
                self.arena.ensure_neighbor(tx + 1, ty);
            }
            if on_north && on_west {
                self.arena.ensure_neighbor(tx - 1, ty + 1);
            }
            if on_north && on_east {
                self.arena.ensure_neighbor(tx + 1, ty + 1);
            }
            if on_south && on_west {
                self.arena.ensure_neighbor(tx - 1, ty - 1);
            }
            if on_south && on_east {
                self.arena.ensure_neighbor(tx + 1, ty - 1);
            }
        }
    }

    /// Batch-update many cells, amortizing per-tile metadata recompute.
    pub fn set_cells<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = (i64, i64, bool)>,
    {
        let mut touched_tiles = Vec::new();

        for (x, y, alive) in cells {
            let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
            let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
            let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

            let idx = match self.arena.idx_at(tile_coord) {
                Some(existing) => existing,
                None if alive => self.arena.allocate_absent(tile_coord),
                None => continue,
            };

            let changed = {
                let buf = self.arena.current_buf_mut(idx);
                let mask = 1u64 << local_x;
                let row = &mut buf[local_y];
                let was_alive = (*row & mask) != 0;
                if was_alive == alive {
                    false
                } else {
                    if alive {
                        *row |= mask;
                    } else {
                        *row &= !mask;
                    }
                    true
                }
            };

            if !changed {
                continue;
            }

            // O(1) dedup via direct-indexed flag array — no hashing needed.
            // Grow the flags vec on demand to cover any newly allocated tiles.
            let i = idx.index();
            if i >= self.touched_flags.len() {
                self.touched_flags.resize(self.arena.meta.len(), false);
            }
            if !self.touched_flags[i] {
                self.touched_flags[i] = true;
                touched_tiles.push(idx);
                self.arena.mark_changed(idx);
            }

            if alive {
                let (tx, ty) = tile_coord;
                let on_south = local_y == 0;
                let on_north = local_y == 63;
                let on_west = local_x == 0;
                let on_east = local_x == 63;

                if on_north {
                    self.arena.ensure_neighbor(tx, ty + 1);
                }
                if on_south {
                    self.arena.ensure_neighbor(tx, ty - 1);
                }
                if on_west {
                    self.arena.ensure_neighbor(tx - 1, ty);
                }
                if on_east {
                    self.arena.ensure_neighbor(tx + 1, ty);
                }
                if on_north && on_west {
                    self.arena.ensure_neighbor(tx - 1, ty + 1);
                }
                if on_north && on_east {
                    self.arena.ensure_neighbor(tx + 1, ty + 1);
                }
                if on_south && on_west {
                    self.arena.ensure_neighbor(tx - 1, ty - 1);
                }
                if on_south && on_east {
                    self.arena.ensure_neighbor(tx + 1, ty - 1);
                }
            }
        }

        if touched_tiles.is_empty() {
            return;
        }

        for idx in &touched_tiles {
            let buf = self.arena.current_buf(*idx);
            let (border, has_live) = tile::recompute_border_and_has_live(buf);

            *self.arena.border_mut(*idx) = border;
            {
                let meta = self.arena.meta_mut(*idx);
                meta.population = POPULATION_UNKNOWN;
                meta.set_has_live(has_live);
            }
        }

        // Clear touched flags for reuse (O(touched) not O(arena)).
        for idx in &touched_tiles {
            self.touched_flags[idx.index()] = false;
        }

        self.population_cache = None;
    }

    /// Batch-set many live cells.
    pub fn set_cells_alive<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = (i64, i64)>,
    {
        self.set_cells(cells.into_iter().map(|(x, y)| (x, y, true)));
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

        let active_len = self.arena.active_set.len();
        let bp = self.arena.border_phase;
        let cp = self.arena.cell_phase;
        let backend = self.backend;
        let thread_count = rayon::current_num_threads().max(1);
        let effective_threads = tuned_parallel_threads(active_len, thread_count);
        let run_parallel = effective_threads > 1;

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

        debug_assert_eq!(self.arena.meta.len(), current_vec.len());
        debug_assert_eq!(self.arena.meta.len(), next_vec.len());
        debug_assert_eq!(self.arena.meta.len(), borders_read.len());
        debug_assert_eq!(self.arena.meta.len(), next_borders_vec.len());
        debug_assert_eq!(self.arena.meta.len(), self.arena.neighbors.len());

        if !run_parallel {
            // Serial path: use raw pointers to eliminate bounds checks in the hot loop.
            let neighbors_ptr = self.arena.neighbors.as_ptr();
            let borders_read_ptr = borders_read.as_ptr();
            let current_ptr = current_vec.as_ptr();
            let next_ptr = next_vec.as_mut_ptr();
            let next_borders_ptr = next_borders_vec.as_mut_ptr();
            let meta_ptr = self.arena.meta.as_mut_ptr();

            for i in 0..active_len {
                let idx = self.arena.active_set[i];
                let ii = idx.index();
                let ghost = unsafe {
                    gather_ghost_zone_raw(ii, borders_read_ptr, neighbors_ptr)
                };

                let (changed, _border, has_live) = unsafe {
                    let current = &(*current_ptr.add(ii)).0;
                    let next = &mut (*next_ptr.add(ii)).0;
                    let result = advance_core(current, next, &ghost, backend);
                    *next_borders_ptr.add(ii) = result.1;
                    result
                };

                unsafe {
                    let meta = &mut *meta_ptr.add(ii);
                    meta.set_changed(changed);
                    if changed {
                        meta.population = POPULATION_UNKNOWN;
                        meta.set_has_live(has_live);
                        if !meta.in_changed_list() {
                            meta.set_in_changed_list(true);
                            self.arena.changed_list.push(idx);
                        }
                    }
                }
            }
        } else {
            // === Optimized parallel kernel ===
            // Instead of per-chunk Vec allocations + reduce + serial merge,
            // we use a pre-allocated AtomicBool array indexed by tile slot.
            // Each worker writes its changed flag directly via raw pointer
            // (advance_tile_split already writes meta), and we set the
            // atomic flag for changed tiles. After the parallel phase,
            // a single serial scan of the active set collects changed tiles.
            let active_set = &self.arena.active_set;
            let current_ptr = SendConstPtr::new(current_vec.as_ptr());
            let next_ptr = SendPtr::new(next_vec.as_mut_ptr());
            let meta_ptr = SendPtr::new(self.arena.meta.as_mut_ptr());
            let neighbors_ptr = SendConstPtr::new(self.arena.neighbors.as_ptr());
            let next_borders_ptr = SendPtr::new(next_borders_vec.as_mut_ptr());
            let borders_read_ptr = SendConstPtr::new(borders_read.as_ptr());
            let worker_count = effective_threads.min(active_len);
            let chunk_size = active_len.div_ceil(worker_count);

            active_set
                .par_chunks(chunk_size)
                .for_each(move |chunk| {
                    for &idx in chunk {
                        let ghost = unsafe {
                            gather_ghost_zone_raw(
                                idx.index(),
                                borders_read_ptr.get(),
                                neighbors_ptr.get(),
                            )
                        };

                        unsafe {
                            advance_tile_split(
                                current_ptr.get(),
                                next_ptr.get(),
                                meta_ptr.get(),
                                next_borders_ptr.get(),
                                idx.index(),
                                &ghost,
                                backend,
                            );
                        }
                    }
                });

            // Serial scan: advance_tile_split already set meta.changed() for
            // each tile. We just need to collect those into the changed_list.
            // This is cheaper than the old approach (Vec alloc per chunk + reduce).
            for &idx in active_set {
                if self.arena.meta[idx.index()].changed() {
                    let meta = &mut self.arena.meta[idx.index()];
                    if !meta.in_changed_list() {
                        meta.set_in_changed_list(true);
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
        // Split borrow: take a shared ref to the pool before mutably borrowing self.
        let pool = &self.pool as *const rayon::ThreadPool;
        // SAFETY: `pool` is not modified during `install`, and `step_impl` only
        // mutates arena/generation/population_cache — never the pool itself.
        unsafe { &*pool }.install(|| {
            self.step_impl();
        });
    }

    pub fn step_n(&mut self, n: u64) {
        let pool = &self.pool as *const rayon::ThreadPool;
        unsafe { &*pool }.install(|| {
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
        for (buf, meta) in self.arena.cell_bufs[cp]
            .iter()
            .zip(self.arena.meta.iter_mut())
        {
            if !meta.occupied() {
                continue;
            }
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
            if !meta.occupied() {
                continue;
            }

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
