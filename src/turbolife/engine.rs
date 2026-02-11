//! TurboLife engine core.
//!
//! Tile computation is parallelized via rayon with chunked dispatch.
//! Ghost reads use `borders[border_phase]` (current gen, immutable).
//! Kernel writes to `borders[1 - border_phase]` (next gen).
//! After dispatch, `flip_borders()` makes next gen current.
//! Changed-tile indices are collected alongside the kernel computation.

use rayon::prelude::*;

use super::activity::{prune_and_expand, rebuild_active_set};
use super::arena::TileArena;
use super::kernel::{advance_tile_fused, advance_core as advance_tile_core};
use super::sync::gather_ghost_zone;

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
        let cells = self.arena.cells_mut(idx);
        cells.set_local_cell(local_x, local_y, alive);
        let border = cells.recompute_border();
        *self.arena.border_mut(idx) = border;
        self.arena.meta_mut(idx).population = None;
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

    /// Threshold: below this many active tiles, run kernel serially.
    const PARALLEL_KERNEL_THRESHOLD: usize = 128;

    pub fn step(&mut self) {
        rebuild_active_set(&mut self.arena);

        if self.arena.active_set.is_empty() {
            self.generation += 1;
            return;
        }

        let active_len = self.arena.active_set.len();
        let bp = self.arena.border_phase;

        if active_len < Self::PARALLEL_KERNEL_THRESHOLD {
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

                let current = *cells.current();
                let next = cells.next_mut();

                let (changed, border) = advance_tile_core(&current, next, &ghost);

                cells.swap();
                next_borders[idx.index()] = border;
                meta.changed = changed;
                meta.population = None;

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

            let changed_chunks: Vec<Vec<super::tile::TileIdx>> = active_set.par_chunks(64).map({
                let cell_mut_ptr = cell_mut_ptr;
                let meta_ptr = meta_ptr;
                let next_borders_ptr = next_borders_ptr;
                move |chunk| {
                    let mut changed = Vec::new();
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
                            advance_tile_fused(
                                cell_mut_ptr.get(),
                                meta_ptr.get(),
                                next_borders_ptr.get(),
                                idx.index(),
                                &ghost,
                            );

                            let meta = &*meta_ptr.get().add(idx.index());
                            if meta.changed {
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

    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    pub fn population(&mut self) -> u64 {
        if let Some(cached) = self.population_cache {
            return cached;
        }

        let mut total = 0u64;
        for (cells, meta) in self.arena.cell_data.iter().zip(self.arena.meta.iter_mut()) {
            if !meta.occupied { continue; }
            let pop = if let Some(cached) = meta.population {
                cached
            } else {
                let computed = cells.compute_population();
                meta.population = Some(computed);
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
