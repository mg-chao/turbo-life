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
use super::tile::{Direction, TileIdx};

/// Threshold: below this active set size, prune/expand runs serially.
const PARALLEL_PRUNE_THRESHOLD: usize = 256;

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
    let epoch = arena.active_epoch;

    arena.active_set.clear();

    // Build changed_scratch from changed_list or full scan.
    let occupied_count = arena.coord_to_idx.len();
    let use_changed_list =
        occupied_count > 0 && arena.changed_list.len() <= occupied_count / 4;

    arena.changed_scratch.clear();
    if use_changed_list {
        arena.changed_scratch.extend_from_slice(&arena.changed_list);
        for &idx in &arena.changed_list {
            arena.meta[idx.index()].in_changed_list = false;
        }
        arena.changed_list.clear();
    } else {
        for &idx in &arena.changed_list {
            arena.meta[idx.index()].in_changed_list = false;
        }
        arena.changed_list.clear();
        // Fallback: scan all slots linearly for changed+occupied tiles.
        // Linear scan has better cache locality than hashmap iteration.
        // This path is taken when changed_list > 25% of occupied, so
        // most slots will match anyway.
        for i in 0..arena.meta.len() {
            if arena.meta[i].occupied && arena.meta[i].changed {
                arena.changed_scratch.push(TileIdx(i as u32));
            }
        }
    }

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
            if let Some(ni) = nb[dir_idx] {
                let ni_i = ni.index();
                if arena.meta[ni_i].occupied && arena.meta[ni_i].active_epoch != epoch {
                    arena.meta[ni_i].active_epoch = epoch;
                    arena.active_set.push(ni);
                }
            }
        }
    }
}

/// Check if any neighbor has its `changed` flag set.
#[inline]
fn neighbor_has_changed(
    neighbors: &[[Option<TileIdx>; 8]],
    meta: &[super::tile::TileMeta],
    idx: TileIdx,
) -> bool {
    let nb = &neighbors[idx.index()];
    for dir_idx in 0..8 {
        if let Some(ni) = nb[dir_idx] {
            if meta[ni.index()].changed {
                return true;
            }
        }
    }
    false
}

/// Scan a single tile for expand/prune candidates.
#[inline]
fn scan_tile_prune_expand(
    idx: TileIdx,
    cell_data: &[super::tile::TileCells],
    meta: &[super::tile::TileMeta],
    borders: &[super::tile::BorderData],
    neighbors: &[[Option<TileIdx>; 8]],
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

    let has_missing_neighbor = nb[0].is_none() || nb[1].is_none()
        || nb[2].is_none() || nb[3].is_none()
        || nb[4].is_none() || nb[5].is_none()
        || nb[6].is_none() || nb[7].is_none();

    if has_missing_neighbor {
        if border.north != 0 && nb[Direction::North.index()].is_none() {
            expand.push((tx, ty + 1));
        }
        if border.south != 0 && nb[Direction::South.index()].is_none() {
            expand.push((tx, ty - 1));
        }
        if border.west != 0 && nb[Direction::West.index()].is_none() {
            expand.push((tx - 1, ty));
        }
        if border.east != 0 && nb[Direction::East.index()].is_none() {
            expand.push((tx + 1, ty));
        }
        if border.nw && nb[Direction::NW.index()].is_none() {
            expand.push((tx - 1, ty + 1));
        }
        if border.ne && nb[Direction::NE.index()].is_none() {
            expand.push((tx + 1, ty + 1));
        }
        if border.sw && nb[Direction::SW.index()].is_none() {
            expand.push((tx - 1, ty - 1));
        }
        if border.se && nb[Direction::SE.index()].is_none() {
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
    let has_live = match meta[i_idx].population {
        Some(p) => p > 0,
        None => cell_data[i_idx].compute_population() > 0,
    };
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

    if active_len == 0 {
        return;
    }

    if active_len < PARALLEL_PRUNE_THRESHOLD {
        // Serial path for small active sets.
        for i in 0..active_len {
            let idx = arena.active_set[i];
            let should_prune = scan_tile_prune_expand(
                idx,
                &arena.cell_data,
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
        let results: Vec<(Vec<(i64, i64)>, Vec<TileIdx>)> = arena.active_set
            .par_chunks(256)
            .map(|chunk| {
                let mut expand = Vec::new();
                let mut prune = Vec::new();
                for &idx in chunk {
                    let should_prune = scan_tile_prune_expand(
                        idx,
                        &arena.cell_data,
                        &arena.meta,
                        &arena.borders[bp],
                        &arena.neighbors,
                        &arena.coords,
                        &mut expand,
                    );
                    if should_prune {
                        prune.push(idx);
                    }
                }
                (expand, prune)
            })
            .collect();

        for (expand, prune) in results {
            arena.expand_buf.extend(expand);
            arena.prune_buf.extend(prune);
        }
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
