//! Active set and prune/expand logic for TurboLife.

use rayon::prelude::*;

use super::arena::TileArena;
use super::tile::{BorderData, NO_NEIGHBOR, TileIdx, TileMeta};

const PARALLEL_PRUNE_MIN_ACTIVE: usize = 192;
const PARALLEL_PRUNE_TILES_PER_THREAD: usize = 24;
const PARALLEL_PRUNE_MIN_CHUNKS: usize = 2;
const PRUNE_CHUNK_MIN: usize = 64;
const PRUNE_CHUNK_MAX: usize = 512;
const EXPAND_OFFSETS: [(i64, i64); 8] = [
    (0, 1),   // N
    (0, -1),  // S
    (-1, 0),  // W
    (1, 0),   // E
    (-1, 1),  // NW
    (1, 1),   // NE
    (-1, -1), // SW
    (1, -1),  // SE
];

#[inline(always)]
fn pack_expand_candidate(idx: TileIdx, dir: usize) -> u64 {
    ((idx.0 as u64) << 3) | dir as u64
}

#[inline(always)]
fn unpack_expand_candidate(candidate: u64) -> (TileIdx, usize) {
    (
        TileIdx((candidate >> 3) as u32),
        (candidate & 0b111) as usize,
    )
}

#[inline]
fn prune_chunk_size(active_len: usize, threads: usize) -> usize {
    let target_chunks = threads.saturating_mul(4).max(1);
    let size = active_len.div_ceil(target_chunks);
    size.clamp(PRUNE_CHUNK_MIN, PRUNE_CHUNK_MAX)
}

#[inline]
fn effective_prune_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 || active_len < PARALLEL_PRUNE_MIN_ACTIVE {
        return 1;
    }
    let by_work = active_len / PARALLEL_PRUNE_TILES_PER_THREAD;
    let by_chunks = active_len / PRUNE_CHUNK_MIN;
    let effective = by_work.min(by_chunks).min(thread_count).max(1);
    if effective < PARALLEL_PRUNE_MIN_CHUNKS {
        1
    } else {
        effective
    }
}

#[inline]
fn tuned_prune_threads(active_len: usize, thread_count: usize) -> usize {
    if thread_count <= 1 {
        return 1;
    }
    let mut effective = effective_prune_threads(active_len, thread_count);
    if effective <= 1 {
        return 1;
    }
    let cores = thread_count;
    let tuned_cap = if active_len < 256 {
        (cores / 4).max(2)
    } else if active_len < 512 {
        (cores / 3).max(4)
    } else if active_len < 1_024 {
        (cores / 2).max(6)
    } else if active_len < 4_096 {
        (cores * 3 / 4).max(8)
    } else {
        cores
    };
    effective = effective.min(tuned_cap).min(thread_count);
    effective.max(1)
}

/// Rebuild the active set from the changed list in O(changed * 9).
///
/// Uses unsafe raw pointer access to meta/neighbors for the hot inner loop
/// to eliminate bounds checks on every neighbor lookup.
pub fn rebuild_active_set(arena: &mut TileArena) {
    arena.active_epoch = arena.active_epoch.wrapping_add(1);
    if arena.active_epoch == 0 {
        arena.active_epoch = 1;
        for m in arena.meta.iter_mut() {
            m.active_epoch = 0;
        }
    }
    let epoch = arena.active_epoch;

    arena.active_set.clear();
    arena.changed_scratch.clear();
    // Swap instead of copy to avoid O(n) memcpy.
    std::mem::swap(&mut arena.changed_scratch, &mut arena.changed_list);

    let changed_count = arena.changed_scratch.len();
    if changed_count == 0 {
        return;
    }

    let dense_rebuild = arena.occupied_count >= 4096
        && changed_count.saturating_mul(100) >= arena.occupied_count.saturating_mul(95);
    if dense_rebuild {
        for &idx in &arena.changed_scratch {
            arena.meta[idx.index()].set_in_changed_list(false);
        }
        arena.active_set.reserve(arena.occupied_count);
        for i in 1..arena.meta.len() {
            if arena.meta[i].occupied() {
                arena.meta[i].active_epoch = epoch;
                arena.active_set.push(TileIdx(i as u32));
            }
        }
        return;
    }

    // Hot loop: epoch-stamp changed tiles + their 8 neighbors.
    // Use raw pointers to eliminate bounds checks on every neighbor access.
    // Reserve for worst case: each changed tile + up to 8 unique neighbors.
    // Use 9x estimate (tile + 8 neighbors) to avoid mid-loop reallocation.
    arena
        .active_set
        .reserve(changed_count.saturating_mul(9).min(arena.occupied_count));
    let meta_ptr = arena.meta.as_mut_ptr();
    let neighbors_ptr = arena.neighbors.as_ptr();
    let meta_len = arena.meta.len();

    for &idx in &arena.changed_scratch {
        let i = idx.index();
        debug_assert!(i < meta_len);
        // SAFETY: i < meta_len guaranteed by arena invariants (idx came from changed_list).
        unsafe {
            let m = &mut *meta_ptr.add(i);
            m.set_in_changed_list(false);
            if m.occupied() && m.active_epoch != epoch {
                m.active_epoch = epoch;
                arena.active_set.push(idx);
            }
            let nb = &*neighbors_ptr.add(i);
            for &ni_raw in nb.iter() {
                if ni_raw != NO_NEIGHBOR {
                    let ni_i = ni_raw as usize;
                    debug_assert!(ni_i < meta_len);
                    let nm = &mut *meta_ptr.add(ni_i);
                    if nm.occupied() && nm.active_epoch != epoch {
                        nm.active_epoch = epoch;
                        arena.active_set.push(TileIdx(ni_raw));
                    }
                }
            }
        }
    }

    // Sort active set by index for better cache locality during kernel execution.
    // The kernel accesses cell_bufs, borders, meta, and neighbors arrays sequentially
    // by tile index - sorting ensures we walk these arrays in order.
    // Only sort when the set is small enough that sort cost < cache benefit.
    if arena.active_set.len() <= 8192 {
        arena.active_set.sort_unstable_by_key(|idx| idx.0);
    }
}

#[inline]
#[allow(dead_code)]
fn neighbor_has_changed(neighbors: &[[u32; 8]], meta: &[TileMeta], idx: TileIdx) -> bool {
    let nb = &neighbors[idx.index()];
    for &ni_raw in nb.iter().take(8) {
        if ni_raw != NO_NEIGHBOR && meta[ni_raw as usize].changed() {
            return true;
        }
    }
    false
}

/// Scan a tile for prune/expand using raw pointers to avoid bounds checks.
///
/// # Safety
/// All pointers must be valid and `idx.index()` must be within bounds.
/// Neighbor indices in `neighbors_ptr[idx]` must be valid or NO_NEIGHBOR.
#[inline]
unsafe fn scan_tile_prune_expand_raw(
    idx: TileIdx,
    meta_ptr: *const TileMeta,
    borders_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    expand: &mut Vec<u64>,
) -> bool {
    unsafe {
        let i_idx = idx.index();
        let meta = &*meta_ptr.add(i_idx);
        if !meta.occupied() {
            return false;
        }

        let nb = &*neighbors_ptr.add(i_idx);
        let border = &*borders_ptr.add(i_idx);

        // Fast path: when no neighbors are missing, there is no frontier expansion.
        let missing = meta.missing_mask;

        if missing != 0 {
            let live_border: u8 = ((border.north != 0) as u8)
                | (((border.south != 0) as u8) << 1)
                | (((border.west != 0) as u8) << 2)
                | (((border.east != 0) as u8) << 3)
                | (((border.corners & BorderData::CORNER_NW != 0) as u8) << 4)
                | (((border.corners & BorderData::CORNER_NE != 0) as u8) << 5)
                | (((border.corners & BorderData::CORNER_SW != 0) as u8) << 6)
                | (((border.corners & BorderData::CORNER_SE != 0) as u8) << 7);

            let mut bits = missing & live_border;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                expand.push(pack_expand_candidate(idx, bit));
                bits &= bits - 1;
            }
        }

        if meta.changed() || meta.has_live() {
            return false;
        }
        // Check if any neighbor changed using raw pointer for meta access.
        for &ni_raw in nb.iter() {
            if ni_raw != NO_NEIGHBOR && (*meta_ptr.add(ni_raw as usize)).changed() {
                return false;
            }
        }
        true
    }
}

/// Prune dead tiles and expand the frontier.
pub fn prune_and_expand(arena: &mut TileArena) {
    arena.expand_buf.clear();
    arena.prune_buf.clear();

    let bp = arena.border_phase;
    let active_len = arena.active_set.len();
    let thread_count = rayon::current_num_threads().max(1);

    if active_len == 0 {
        return;
    }

    let effective_threads = tuned_prune_threads(active_len, thread_count);
    let run_parallel = effective_threads > 1;

    // Get raw pointers for bounds-check-free access in both serial and parallel paths.
    let meta_ptr = arena.meta.as_ptr();
    let borders_ptr = arena.borders[bp].as_ptr();
    let neighbors_ptr = arena.neighbors.as_ptr();

    if !run_parallel {
        for i in 0..active_len {
            let idx = arena.active_set[i];
            let should_prune = unsafe {
                scan_tile_prune_expand_raw(
                    idx,
                    meta_ptr,
                    borders_ptr,
                    neighbors_ptr,
                    &mut arena.expand_buf,
                )
            };
            if should_prune {
                arena.prune_buf.push(idx);
            }
        }
    } else {
        // Wrap raw pointers for Send/Sync using usize transmutation.
        // SAFETY: pointers are valid for the duration of the parallel phase
        // (arena is not modified during par_chunks).
        let s_meta = meta_ptr as usize;
        let s_borders = borders_ptr as usize;
        let s_neighbors = neighbors_ptr as usize;

        let chunk_size = prune_chunk_size(active_len, effective_threads);
        let (expand_all, prune_all) = arena
            .active_set
            .par_chunks(chunk_size)
            .fold(
                || (Vec::new(), Vec::new()),
                move |mut acc, chunk| {
                    let mp = s_meta as *const TileMeta;
                    let bp = s_borders as *const BorderData;
                    let np = s_neighbors as *const [u32; 8];
                    for &idx in chunk {
                        let should_prune =
                            unsafe { scan_tile_prune_expand_raw(idx, mp, bp, np, &mut acc.0) };
                        if should_prune {
                            acc.1.push(idx);
                        }
                    }
                    acc
                },
            )
            .reduce(
                || (Vec::new(), Vec::new()),
                |mut left, mut right| {
                    left.0.append(&mut right.0);
                    left.1.append(&mut right.1);
                    left
                },
            );
        arena.expand_buf.extend(expand_all);
        arena.prune_buf.extend(prune_all);
    }
    if !arena.expand_buf.is_empty() {
        arena.reserve_additional_tiles(arena.expand_buf.len());
    }
    for i in 0..arena.expand_buf.len() {
        let (src_idx, dir) = unpack_expand_candidate(arena.expand_buf[i]);
        let src_i = src_idx.index();
        if !arena.meta[src_i].occupied() {
            continue;
        }
        if arena.neighbors[src_i][dir] != NO_NEIGHBOR {
            continue;
        }
        let (tx, ty) = arena.coords[src_i];
        let (dx, dy) = EXPAND_OFFSETS[dir];
        let coord = (tx + dx, ty + dy);
        let idx = arena.allocate_absent(coord);
        arena.meta[idx.index()].set_changed(true);
        arena.meta[idx.index()].active_epoch = 0;
        arena.meta[idx.index()].population = 0;
        arena.mark_changed(idx);
    }

    for i in 0..arena.prune_buf.len() {
        let idx = arena.prune_buf[i];
        arena.release(idx);
    }
}
