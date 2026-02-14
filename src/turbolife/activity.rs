//! Active set rebuild and post-kernel prune/expand for TurboLife.

use rayon::prelude::*;

use super::arena::TileArena;
use super::kernel::ghost_is_empty_from_live_masks;
use super::tile::{NO_NEIGHBOR, TileIdx};

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
const PARALLEL_PRUNE_CANDIDATES_MIN: usize = 256;
const PRUNE_FILTER_CHUNK_MIN: usize = 64;
const PRUNE_FILTER_CHUNK_MAX: usize = 512;
const ACTIVE_SORT_STD_MAX: usize = 8_192;
const ACTIVE_SORT_RADIX_MIN: usize = 32_768;

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
    fn get(self) -> *const T {
        self.inner
    }
}

#[inline]
fn prune_filter_chunk_size(candidate_len: usize, threads: usize) -> usize {
    let target_chunks = threads.saturating_mul(4).max(1);
    let size = candidate_len.div_ceil(target_chunks);
    size.clamp(PRUNE_FILTER_CHUNK_MIN, PRUNE_FILTER_CHUNK_MAX)
}

#[inline]
fn radix_sort_active_set(arena: &mut TileArena) {
    let len = arena.active_set.len();
    if len <= 1 {
        return;
    }

    debug_assert!(len <= u32::MAX as usize);
    debug_assert_eq!(arena.active_sort_counts.len(), 1 << 16);

    arena.active_sort_scratch.clear();
    arena.active_sort_scratch.resize(len, TileIdx(0));
    let counts = &mut arena.active_sort_counts;
    counts.fill(0);

    for &idx in arena.active_set.iter() {
        counts[(idx.0 & 0xFFFF) as usize] += 1;
    }

    let mut prefix = 0u32;
    for count in counts.iter_mut() {
        let next = prefix + *count;
        *count = prefix;
        prefix = next;
    }

    for &idx in arena.active_set.iter() {
        let bucket = (idx.0 & 0xFFFF) as usize;
        let dst = counts[bucket] as usize;
        unsafe {
            *arena.active_sort_scratch.get_unchecked_mut(dst) = idx;
        }
        counts[bucket] += 1;
    }

    counts.fill(0);
    for &idx in arena.active_sort_scratch.iter() {
        counts[(idx.0 >> 16) as usize] += 1;
    }

    prefix = 0;
    for count in counts.iter_mut() {
        let next = prefix + *count;
        *count = prefix;
        prefix = next;
    }

    for &idx in arena.active_sort_scratch.iter() {
        let bucket = (idx.0 >> 16) as usize;
        let dst = counts[bucket] as usize;
        unsafe {
            *arena.active_set.get_unchecked_mut(dst) = idx;
        }
        counts[bucket] += 1;
    }
}

struct ExpandMaskTable {
    dirs: [[u8; 8]; 256],
    len: [u8; 256],
}

const fn build_expand_mask_table() -> ExpandMaskTable {
    let mut table = ExpandMaskTable {
        dirs: [[0u8; 8]; 256],
        len: [0u8; 256],
    };
    let mut mask = 0usize;
    while mask < 256 {
        let mut count = 0u8;
        let mut dir = 0usize;
        while dir < 8 {
            if (mask & (1usize << dir)) != 0 {
                table.dirs[mask][count as usize] = dir as u8;
                count += 1;
            }
            dir += 1;
        }
        table.len[mask] = count;
        mask += 1;
    }
    table
}

const EXPAND_MASK_TABLE: ExpandMaskTable = build_expand_mask_table();

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

#[inline(always)]
pub(crate) fn append_expand_candidates(
    expand: &mut Vec<u64>,
    idx: TileIdx,
    missing_mask: u8,
    live_mask: u8,
) {
    let expand_mask = (missing_mask & live_mask) as usize;
    if expand_mask == 0 {
        return;
    }
    let dirs = &EXPAND_MASK_TABLE.dirs[expand_mask];
    let count = EXPAND_MASK_TABLE.len[expand_mask] as usize;
    for &dir in dirs[..count].iter() {
        expand.push(pack_expand_candidate(idx, dir as usize));
    }
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
            let meta = &mut arena.meta[idx.index()];
            if meta.in_changed_list() {
                meta.set_in_changed_list(false);
            }
        }
        arena.active_set.reserve(arena.occupied_count);
        if arena.free_list.is_empty() && arena.occupied_count + 1 == arena.meta.len() {
            for i in 1..arena.meta.len() {
                if arena.meta[i].occupied() {
                    arena.meta[i].active_epoch = epoch;
                    arena.active_set.push(TileIdx(i as u32));
                }
            }
        } else {
            for (word_idx, &word) in arena.occupied_bits.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let i = (word_idx << 6) + bit;
                    if i == 0 || i >= arena.meta.len() {
                        bits &= bits - 1;
                        continue;
                    }
                    arena.meta[i].active_epoch = epoch;
                    arena.active_set.push(TileIdx(i as u32));
                    bits &= bits - 1;
                }
            }
        }
        return;
    }

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
            if m.in_changed_list() {
                m.set_in_changed_list(false);
            }
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
    // Small sets use std sort, very large sets use a stable two-pass radix sort.
    // Mid-sized sets skip sorting to keep rebuild costs bounded.
    let active_len = arena.active_set.len();
    if active_len <= ACTIVE_SORT_STD_MAX {
        arena.active_set.sort_unstable_by_key(|idx| idx.0);
    } else if active_len >= ACTIVE_SORT_RADIX_MIN {
        radix_sort_active_set(arena);
    }
}

/// Apply expansion and pruning from candidates collected during kernel execution.
pub fn finalize_prune_and_expand(arena: &mut TileArena) {
    if !arena.expand_buf.is_empty() {
        arena.reserve_additional_tiles(arena.expand_buf.len());
    }

    for i in 0..arena.expand_buf.len() {
        let (src_idx, dir) = unpack_expand_candidate(arena.expand_buf[i]);
        let src_i = src_idx.index();
        let src_meta = arena.meta[src_i];
        debug_assert!(src_meta.occupied());
        if !src_meta.occupied() {
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

    let prune_len = arena.prune_buf.len();
    if prune_len == 0 {
        arena.expand_buf.clear();
        arena.prune_buf.clear();
        return;
    }

    let thread_count = rayon::current_num_threads().max(1);
    let run_parallel = thread_count > 1 && prune_len >= PARALLEL_PRUNE_CANDIDATES_MIN;
    let borders = &arena.borders[arena.border_phase];

    let mut to_prune = if run_parallel {
        let chunk_size = prune_filter_chunk_size(prune_len, thread_count);
        let meta_ptr = SendConstPtr::new(arena.meta.as_ptr());
        let neighbors_ptr = SendConstPtr::new(arena.neighbors.as_ptr());
        let borders_ptr = SendConstPtr::new(borders.as_ptr());
        let prune_candidates = &arena.prune_buf;
        prune_candidates
            .par_chunks(chunk_size)
            .fold(Vec::new, move |mut acc, chunk| {
                let mp = meta_ptr.get();
                let np = neighbors_ptr.get();
                let bp = borders_ptr.get();
                for &idx in chunk {
                    let ii = idx.index();
                    // SAFETY: pointers are stable and only read during this phase.
                    unsafe {
                        let meta = *mp.add(ii);
                        if !meta.occupied() || meta.changed() || meta.has_live() {
                            continue;
                        }

                        let nb = *np.add(ii);
                        let ghost_empty = ghost_is_empty_from_live_masks([
                            (*bp.add(nb[0] as usize)).live_mask,
                            (*bp.add(nb[1] as usize)).live_mask,
                            (*bp.add(nb[2] as usize)).live_mask,
                            (*bp.add(nb[3] as usize)).live_mask,
                            (*bp.add(nb[4] as usize)).live_mask,
                            (*bp.add(nb[5] as usize)).live_mask,
                            (*bp.add(nb[6] as usize)).live_mask,
                            (*bp.add(nb[7] as usize)).live_mask,
                        ]);
                        if ghost_empty {
                            acc.push(idx);
                        }
                    }
                }
                acc
            })
            .reduce(Vec::new, |mut left, mut right| {
                left.append(&mut right);
                left
            })
    } else {
        let mut serial = Vec::with_capacity(prune_len);
        for i in 0..prune_len {
            let idx = arena.prune_buf[i];
            let ii = idx.index();
            let meta = arena.meta[ii];
            if !meta.occupied() || meta.changed() || meta.has_live() {
                continue;
            }

            let nb = arena.neighbors[ii];
            let ghost_empty = ghost_is_empty_from_live_masks([
                borders[nb[0] as usize].live_mask,
                borders[nb[1] as usize].live_mask,
                borders[nb[2] as usize].live_mask,
                borders[nb[3] as usize].live_mask,
                borders[nb[4] as usize].live_mask,
                borders[nb[5] as usize].live_mask,
                borders[nb[6] as usize].live_mask,
                borders[nb[7] as usize].live_mask,
            ]);
            if ghost_empty {
                serial.push(idx);
            }
        }
        serial
    };

    for idx in to_prune.drain(..) {
        arena.release(idx);
    }

    arena.expand_buf.clear();
    arena.prune_buf.clear();
}
