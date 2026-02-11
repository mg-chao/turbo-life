//! TurboLife engine core.
//!
//! Tile computation is parallelized via rayon with chunked dispatch.
//! Ghost reads use `borders[border_phase]` (current gen, immutable).
//! Kernel writes to `borders[1 - border_phase]` (next gen).
//! After dispatch, `flip_borders()` makes next gen current.
//! Changed-tile indices are collected alongside the kernel computation.

use rayon::prelude::*;
use std::sync::OnceLock;

use super::activity::{prune_and_expand, rebuild_active_set};
use super::arena::TileArena;
use super::kernel::{advance_tile_fused, advance_core as advance_tile_core};
use super::sync::gather_ghost_zone;
use super::tile::POPULATION_UNKNOWN;

struct SendPtr<T> {
    _inner: *mut T,
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> SendPtr<T> {
    fn new(ptr: *mut T) -> Self { Self { _inner: ptr } }
    fn get(&self) -> *mut T { self._inner }
}

const TILE_SIZE_I64: i64 = 64;
const PARALLEL_KERNEL_MIN_ACTIVE: usize = 384;
const PARALLEL_KERNEL_TILES_PER_THREAD: usize = 48;
const PARALLEL_KERNEL_MIN_CHUNKS: usize = 2;
const KERNEL_CHUNK_MIN: usize = 64;
const KERNEL_CHUNK_MAX: usize = 1024;
const ENV_OVERRIDE_THREADS: &str = "TURBOLIFE_NUM_THREADS";
const ENV_MAX_THREADS: &str = "TURBOLIFE_MAX_THREADS";

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
where
    T: Send,
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
    let target_chunks = threads.saturating_mul(4).max(1);
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

    if effective < PARALLEL_KERNEL_MIN_CHUNKS {
        1
    } else {
        effective
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

    let tuned_cap = if active_len < 1_024 {
        2
    } else if active_len < 2_048 {
        4
    } else if active_len < 8_192 {
        8
    } else if active_len < 32_768 {
        12
    } else {
        thread_count
    };

    effective = effective
        .min(tuned_cap)
        .min(thread_count);
    effective.max(1)
}

pub struct TurboLife {
    arena: TileArena,
    generation: u64,
    population_cache: Option<u64>,
}

impl TurboLife {
    pub fn new() -> Self {
        Self {
            arena: TileArena::new(),
            generation: 0,
            population_cache: Some(0),
        }
    }

    pub fn set_cell(&mut self, x: i64, y: i64, alive: bool) {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        if !alive && self.arena.idx_at(tile_coord).is_none() {
            return;
        }

        let idx = self.arena.allocate(tile_coord);
        let (border, has_live) = {
            let cells = self.arena.cells_mut(idx);
            cells.set_local_cell(local_x, local_y, alive);
            let border = cells.recompute_border();
            let has_live = cells.current().iter().any(|&row| row != 0);
            (border, has_live)
        };
        *self.arena.border_mut(idx) = border;
        {
            let meta = self.arena.meta_mut(idx);
            meta.population = POPULATION_UNKNOWN;
            meta.has_live = has_live;
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
        }
    }

    pub fn get_cell(&self, x: i64, y: i64) -> bool {
        let tile_coord = (x.div_euclid(TILE_SIZE_I64), y.div_euclid(TILE_SIZE_I64));
        let local_x = x.rem_euclid(TILE_SIZE_I64) as usize;
        let local_y = y.rem_euclid(TILE_SIZE_I64) as usize;

        self.arena
            .idx_at(tile_coord)
            .map(|idx| {
                self.arena.meta(idx).occupied
                    && self.arena.cells(idx).get_local_cell(local_x, local_y)
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
        let thread_count = rayon::current_num_threads().max(1);
        let effective_threads = tuned_parallel_threads(active_len, thread_count);
        let run_parallel = effective_threads > 1;

        if !run_parallel {
            // Serial path: avoid rayon overhead for small active sets.
            let (b0, b1) = self.arena.borders.split_at_mut(1);
            let (borders_read, next_borders) = if bp == 0 {
                (b0[0].as_slice(), b1[0].as_mut_slice())
            } else {
                (b1[0].as_slice(), b0[0].as_mut_slice())
            };

            for i in 0..active_len {
                let idx = self.arena.active_set[i];
                let ghost = gather_ghost_zone(idx, borders_read, &self.arena.neighbors);

                let cells = &mut self.arena.cell_data[idx.index()];
                let meta = &mut self.arena.meta[idx.index()];

                let (current, next) = cells.current_and_next_mut();

                let (changed, border, has_live) = advance_tile_core(current, next, &ghost);

                cells.swap();
                next_borders[idx.index()] = border;
                meta.changed = changed;
                meta.population = POPULATION_UNKNOWN;
                meta.has_live = has_live;

                if changed {
                    if !meta.in_changed_list {
                        meta.in_changed_list = true;
                        self.arena.changed_list.push(idx);
                    }
                }
            }
        } else {
            // Parallel path for large active sets.
            let active_set = &self.arena.active_set;
            let cell_mut_ptr = SendPtr::new(self.arena.cell_data.as_mut_ptr());
            let meta_ptr = SendPtr::new(self.arena.meta.as_mut_ptr());
            let neighbors = &self.arena.neighbors;
            let (b0, b1) = self.arena.borders.split_at_mut(1);
            let (borders_read, next_borders_ptr) = if bp == 0 {
                (b0[0].as_slice(), SendPtr::new(b1[0].as_mut_ptr()))
            } else {
                (b1[0].as_slice(), SendPtr::new(b0[0].as_mut_ptr()))
            };
            let chunk_size = kernel_chunk_size(active_len, effective_threads);

            let changed_chunks: Vec<Vec<super::tile::TileIdx>> = active_set.par_chunks(chunk_size).map({
                let cell_mut_ptr = cell_mut_ptr;
                let meta_ptr = meta_ptr;
                let next_borders_ptr = next_borders_ptr;
                move |chunk| {
                    let mut changed = Vec::with_capacity(chunk.len() / 2 + 4);
                    for (ci, &idx) in chunk.iter().enumerate() {
                        if ci + 1 < chunk.len() {
                            let next_idx = chunk[ci + 1];
                            unsafe {
                                let ptr = cell_mut_ptr.get().add(next_idx.index()) as *const u8;
                                let nb_ptr = neighbors.as_ptr().add(next_idx.index()) as *const u8;
                                #[cfg(target_arch = "x86_64")]
                                {
                                    std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                    std::arch::x86_64::_mm_prefetch(nb_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                                }
                            }
                        }

                        let ghost = gather_ghost_zone(idx, borders_read, neighbors);

                        unsafe {
                            let tile_changed = advance_tile_fused(
                                cell_mut_ptr.get(),
                                meta_ptr.get(),
                                next_borders_ptr.get(),
                                idx.index(),
                                &ghost,
                            );

                            if tile_changed {
                                changed.push(idx);
                            }
                        }
                    }
                    changed
                }
            }).collect();

            for chunk_changed in changed_chunks {
                for idx in chunk_changed {
                    let m = &mut self.arena.meta[idx.index()];
                    if !m.in_changed_list {
                        m.in_changed_list = true;
                        self.arena.changed_list.push(idx);
                    }
                }
            }
        }

        self.arena.flip_borders();

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

        let mut total = 0u64;
        for (cells, meta) in self.arena.cell_data.iter().zip(self.arena.meta.iter_mut()) {
            if !meta.occupied { continue; }
            if !meta.has_live {
                meta.population = 0;
                continue;
            }
            let pop = if meta.population != POPULATION_UNKNOWN {
                meta.population
            } else {
                let computed = cells.compute_population();
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
        for i in 0..self.arena.cell_data.len() {
            let meta = &self.arena.meta[i];
            if !meta.occupied { continue; }

            let cells = &self.arena.cell_data[i];
            let coord = self.arena.coords[i];
            let current = cells.current();
            let base_x = coord.0 * TILE_SIZE_I64;
            let base_y = coord.1 * TILE_SIZE_I64;

            for (row_index, row) in current.iter().enumerate() {
                let mut bits = *row;
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
