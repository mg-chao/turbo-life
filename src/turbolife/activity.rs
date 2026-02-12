//! Active set and prune/expand logic for TurboLife.

use rayon::prelude::*;

use super::arena::TileArena;
use super::tile::{BorderData, Direction, NO_NEIGHBOR, TileIdx, TileMeta};

const PARALLEL_PRUNE_MIN_ACTIVE: usize = 192;
const PARALLEL_PRUNE_TILES_PER_THREAD: usize = 24;
const PARALLEL_PRUNE_MIN_CHUNKS: usize = 2;
const PRUNE_CHUNK_MIN: usize = 64;
const PRUNE_CHUNK_MAX: usize = 512;

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
/// Uses parallel candidate gathering when the changed list is large enough.
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
    arena.changed_scratch.extend_from_slice(&arena.changed_list);
    for &idx in &arena.changed_list {
        arena.meta[idx.index()].set_in_changed_list(false);
    }
    arena.changed_list.clear();

    let changed_count = arena.changed_scratch.len();
    if changed_count == 0 {
        return;
    }

    let dense_rebuild = arena.occupied_count >= 4096
        && changed_count.saturating_mul(100) >= arena.occupied_count.saturating_mul(95);
    if dense_rebuild {
        arena.active_set.reserve(arena.occupied_count);
        for i in 1..arena.meta.len() {
            if arena.meta[i].occupied() {
                arena.meta[i].active_epoch = epoch;
                arena.active_set.push(TileIdx(i as u32));
            }
        }
        return;
    }

    // The epoch-stamping must be serial (it's a dedup mechanism), but we can
    // at least avoid branch mispredictions by pre-reserving.
    arena.active_set.reserve(changed_count * 4);
    for &idx in &arena.changed_scratch {
        let i = idx.index();
        if arena.meta[i].occupied() && arena.meta[i].active_epoch != epoch {
            arena.meta[i].active_epoch = epoch;
            arena.active_set.push(idx);
        }
        let nb = &arena.neighbors[i];
        for &ni_raw in nb.iter().take(8) {
            if ni_raw != NO_NEIGHBOR {
                let ni_i = ni_raw as usize;
                if arena.meta[ni_i].occupied() && arena.meta[ni_i].active_epoch != epoch {
                    arena.meta[ni_i].active_epoch = epoch;
                    arena.active_set.push(TileIdx(ni_raw));
                }
            }
        }
    }
}

#[inline]
fn neighbor_has_changed(neighbors: &[[u32; 8]], meta: &[TileMeta], idx: TileIdx) -> bool {
    let nb = &neighbors[idx.index()];
    for &ni_raw in nb.iter().take(8) {
        if ni_raw != NO_NEIGHBOR && meta[ni_raw as usize].changed() {
            return true;
        }
    }
    false
}

#[inline]
fn scan_tile_prune_expand(
    idx: TileIdx,
    meta: &[TileMeta],
    borders: &[BorderData],
    neighbors: &[[u32; 8]],
    coords: &[(i64, i64)],
    expand: &mut Vec<(i64, i64)>,
) -> bool {
    let i_idx = idx.index();
    if !meta[i_idx].occupied() {
        return false;
    }

    let (tx, ty) = coords[i_idx];
    let border = &borders[i_idx];
    let nb = &neighbors[i_idx];

    let has_missing_neighbor = nb[0] == NO_NEIGHBOR
        || nb[1] == NO_NEIGHBOR
        || nb[2] == NO_NEIGHBOR
        || nb[3] == NO_NEIGHBOR
        || nb[4] == NO_NEIGHBOR
        || nb[5] == NO_NEIGHBOR
        || nb[6] == NO_NEIGHBOR
        || nb[7] == NO_NEIGHBOR;

    if has_missing_neighbor {
        if border.north != 0 && nb[Direction::North.index()] == NO_NEIGHBOR {
            expand.push((tx, ty + 1));
        }
        if border.south != 0 && nb[Direction::South.index()] == NO_NEIGHBOR {
            expand.push((tx, ty - 1));
        }
        if border.west != 0 && nb[Direction::West.index()] == NO_NEIGHBOR {
            expand.push((tx - 1, ty));
        }
        if border.east != 0 && nb[Direction::East.index()] == NO_NEIGHBOR {
            expand.push((tx + 1, ty));
        }
        if border.nw() && nb[Direction::NW.index()] == NO_NEIGHBOR {
            expand.push((tx - 1, ty + 1));
        }
        if border.ne() && nb[Direction::NE.index()] == NO_NEIGHBOR {
            expand.push((tx + 1, ty + 1));
        }
        if border.sw() && nb[Direction::SW.index()] == NO_NEIGHBOR {
            expand.push((tx - 1, ty - 1));
        }
        if border.se() && nb[Direction::SE.index()] == NO_NEIGHBOR {
            expand.push((tx + 1, ty - 1));
        }
    }

    let changed = meta[i_idx].changed();
    if changed {
        return false;
    }
    if neighbor_has_changed(neighbors, meta, idx) {
        return false;
    }
    !meta[i_idx].has_live()
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

    if !run_parallel {
        for i in 0..active_len {
            let idx = arena.active_set[i];
            let should_prune = scan_tile_prune_expand(
                idx,
                &arena.meta,
                &arena.borders[bp],
                &arena.neighbors,
                &arena.coords,
                &mut arena.expand_buf,
            );
            if should_prune {
                arena.prune_buf.push(idx);
            }
        }
    } else {
        let chunk_size = prune_chunk_size(active_len, effective_threads);
        let (expand_all, prune_all) = arena
            .active_set
            .par_chunks(chunk_size)
            .fold(
                || (Vec::new(), Vec::new()),
                |mut acc, chunk| {
                    for &idx in chunk {
                        let should_prune = scan_tile_prune_expand(
                            idx,
                            &arena.meta,
                            &arena.borders[bp],
                            &arena.neighbors,
                            &arena.coords,
                            &mut acc.0,
                        );
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

    if arena.expand_buf.len() > 16 {
        arena.expand_buf.sort_unstable();
        arena.expand_buf.dedup();
    }

    for i in 0..arena.expand_buf.len() {
        let coord = arena.expand_buf[i];
        if arena.idx_at(coord).is_some() {
            continue;
        }
        let idx = arena.allocate(coord);
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
