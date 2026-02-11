//! Active set and prune/expand logic for TurboLife.
//!
//! The active set is built each step from the changed list: a tile is active
//! if it changed or is an occupied neighbor of a changed tile. This is
//! O(changed * 9) instead of O(total_slots), avoiding a full scan of the
//! arena. Epoch-based tracking deactivates all tiles in O(1).
//!
//! Prune/expand scanning uses adaptive parallelism: serial for small active
//! sets, parallel with chunked dispatch for large ones.

use rayon::prelude::*;

use super::arena::TileArena;
use super::tile::{Direction, TileIdx, NO_NEIGHBOR};

/// Minimum active tiles before enabling parallel prune/expand.
const PARALLEL_PRUNE_MIN_ACTIVE: usize = 384;

/// Minimum active tiles per worker before enabling parallel prune/expand.
const PARALLEL_PRUNE_TILES_PER_THREAD: usize = 48;

/// Minimum chunk size for parallel prune/expand.
const PRUNE_CHUNK_MIN: usize = 128;

/// Maximum chunk size for parallel prune/expand.
const PRUNE_CHUNK_MAX: usize = 1024;

#[inline]
fn prune_chunk_size(active_len: usize, threads: usize) -> usize {
    let target_chunks = threads.saturating_mul(4).max(1);
    let size = active_len.div_ceil(target_chunks);
    size.clamp(PRUNE_CHUNK_MIN, PRUNE_CHUNK_MAX)
}

/// Rebuild the active set from the changed list in O(changed * 9).
///
/// A tile is active if it is occupied and either:
/// - it changed this step, or
/// - one of its 8 neighbors changed this step.
///
/// We walk `changed_scratch` and for each changed tile, add it and its
/// occupied neighbors to the active set, using `active_epoch` for O(1) dedup.
pub fn rebuild_active_set(arena: &mut TileArena) {
    arena.active_epoch = arena.active_epoch.wrapping_add(1);
    if arena.active_epoch == 0 {
        arena.active_epoch = 1;
        for i in 0..arena.meta.len() {
            arena.meta[i].active_epoch = 0;
        }
    }
    let epoch = arena.active_epoch;

    arena.active_set.clear();

    arena.changed_scratch.clear();
    arena.changed_scratch.extend_from_slice(&arena.changed_list);
    for &idx in &arena.changed_list {
        arena.meta[idx.index()].in_changed_list = false;
    }
    arena.changed_list.clear();

    let changed_count = arena.changed_scratch.len();
    if changed_count == 0 {
        return;
    }

    // Build active set directly from changed_scratch: O(changed * 9).
    // Each changed tile contributes itself + up to 8 occupied neighbors.
    // Epoch-based dedup: each tile is added at most once per step.
    // This is always serial — the per-tile work is ~9 integer comparisons,
    // which is too lightweight to benefit from parallelism.
    arena.active_set.reserve(changed_count * 4);
    for &idx in &arena.changed_scratch {
        let i = idx.index();
        if arena.meta[i].occupied && arena.meta[i].active_epoch != epoch {
            arena.meta[i].active_epoch = epoch;
            arena.active_set.push(idx);
        }
        let nb = &arena.neighbors[i];
        for dir_idx in 0..8 {
            let ni_raw = nb[dir_idx];
            if ni_raw != NO_NEIGHBOR {
                let ni_i = ni_raw as usize;
                if arena.meta[ni_i].occupied && arena.meta[ni_i].active_epoch != epoch {
                    arena.meta[ni_i].active_epoch = epoch;
                    arena.active_set.push(TileIdx(ni_raw));
                }
            }
        }
    }
}

/// Check if any neighbor has its `changed` flag set.
#[inline]
fn neighbor_has_changed(
    neighbors: &[[u32; 8]],
    meta: &[super::tile::TileMeta],
    idx: TileIdx,
) -> bool {
    let nb = &neighbors[idx.index()];
    for dir_idx in 0..8 {
        let ni_raw = nb[dir_idx];
        if ni_raw != NO_NEIGHBOR && meta[ni_raw as usize].changed {
            return true;
        }
    }
    false
}

/// Scan a single tile for expand/prune candidates.
#[inline]
fn scan_tile_prune_expand(
    idx: TileIdx,
    meta: &[super::tile::TileMeta],
    borders: &[super::tile::BorderData],
    neighbors: &[[u32; 8]],
    coords: &[(i64, i64)],
    expand: &mut Vec<(i64, i64)>,
) -> bool {
    let i_idx = idx.index();
    if !meta[i_idx].occupied {
        return false;
    }

    let (tx, ty) = coords[i_idx];
    let border = &borders[i_idx];
    let nb = &neighbors[i_idx];

    let has_missing_neighbor = nb[0] == NO_NEIGHBOR || nb[1] == NO_NEIGHBOR
        || nb[2] == NO_NEIGHBOR || nb[3] == NO_NEIGHBOR
        || nb[4] == NO_NEIGHBOR || nb[5] == NO_NEIGHBOR
        || nb[6] == NO_NEIGHBOR || nb[7] == NO_NEIGHBOR;

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
        if border.nw && nb[Direction::NW.index()] == NO_NEIGHBOR {
            expand.push((tx - 1, ty + 1));
        }
        if border.ne && nb[Direction::NE.index()] == NO_NEIGHBOR {
            expand.push((tx + 1, ty + 1));
        }
        if border.sw && nb[Direction::SW.index()] == NO_NEIGHBOR {
            expand.push((tx - 1, ty - 1));
        }
        if border.se && nb[Direction::SE.index()] == NO_NEIGHBOR {
            expand.push((tx + 1, ty - 1));
        }
    }

    let changed = meta[i_idx].changed;
    if changed {
        return false;
    }
    if neighbor_has_changed(neighbors, meta, idx) {
        return false;
    }
    let has_live = meta[i_idx].has_live;
    !has_live
}

/// Prune dead tiles and expand the frontier.
///
/// Uses adaptive parallelism: serial for small active sets to avoid
/// rayon overhead, parallel with chunked dispatch for large ones.
pub fn prune_and_expand(arena: &mut TileArena) {
    arena.expand_buf.clear();
    arena.prune_buf.clear();

    let bp = arena.border_phase;
    let active_len = arena.active_set.len();
    let thread_count = rayon::current_num_threads().max(1);

    if active_len == 0 {
        return;
    }

    let run_parallel = active_len >= PARALLEL_PRUNE_MIN_ACTIVE
        && active_len >= thread_count * PARALLEL_PRUNE_TILES_PER_THREAD;

    if !run_parallel {
        // Serial path for small active sets.
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
        // Parallel path for large active sets.
        let chunk_size = prune_chunk_size(active_len, thread_count);
        let (expand_all, prune_all) = arena.active_set
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
        arena.meta[idx.index()].changed = true;
        arena.meta[idx.index()].active_epoch = 0;
        arena.meta[idx.index()].population = Some(0);
        arena.mark_changed(idx);
    }

    for i in 0..arena.prune_buf.len() {
        let idx = arena.prune_buf[i];
        arena.release(idx);
    }
}
