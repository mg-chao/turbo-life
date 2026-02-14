//! Active set rebuild and post-kernel prune/expand for TurboLife.

use rayon::prelude::*;

use super::arena::TileArena;
use super::kernel::ghost_is_empty_from_live_masks_ptr;
use super::tile::{BorderData, MISSING_ALL_NEIGHBORS, NO_NEIGHBOR, TileIdx};

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
const PARALLEL_PRUNE_BITMAP_MIN: usize = 65_536;
const ACTIVE_SORT_STD_MAX: usize = 8_192;
const ACTIVE_SORT_RADIX_MIN: usize = 32_768;
const ACTIVE_SORT_LOW_CHURN_PCT: usize = 20;
const ACTIVE_SORT_SKIP_MIN: usize = 2_048;
const ACTIVE_SORT_SKIP_MAX: usize = 8_192;
const ACTIVE_SORT_PROBE_CHURN_PCT: usize = 40;
const DIRECTIONAL_FILTER_ALWAYS_ON_MAX_CHANGED: usize = 64;
const DIRECTIONAL_FILTER_PROBE_MAX_CHANGED: usize = 256;
const DIRECTIONAL_FILTER_SAMPLE_MAX: usize = 256;
const DIRECTIONAL_FILTER_DENSE_CHANGED_PCT: usize = 80;
const DIRECTIONAL_FILTER_MAX_AVG_DIRS_X10: usize = 45;
const _: [(); 1] = [(); (ACTIVE_SORT_SKIP_MIN <= ACTIVE_SORT_SKIP_MAX) as usize];
const _: [(); 1] = [(); (ACTIVE_SORT_SKIP_MAX <= ACTIVE_SORT_STD_MAX) as usize];
const _: [(); 1] = [(); (ACTIVE_SORT_LOW_CHURN_PCT <= 100) as usize];
const _: [(); 1] = [(); (ACTIVE_SORT_PROBE_CHURN_PCT <= 100) as usize];
const _: [(); 1] = [(); (ACTIVE_SORT_LOW_CHURN_PCT < ACTIVE_SORT_PROBE_CHURN_PCT) as usize];
const _: [(); 1] = [(); (DIRECTIONAL_FILTER_ALWAYS_ON_MAX_CHANGED
    <= DIRECTIONAL_FILTER_PROBE_MAX_CHANGED) as usize];
const _: [(); 1] =
    [(); (DIRECTIONAL_FILTER_PROBE_MAX_CHANGED <= DIRECTIONAL_FILTER_SAMPLE_MAX) as usize];
const _: [(); 1] = [(); (DIRECTIONAL_FILTER_DENSE_CHANGED_PCT <= 100) as usize];
const _: [(); 1] = [(); (DIRECTIONAL_FILTER_MAX_AVG_DIRS_X10 <= 80) as usize];

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

#[inline(always)]
fn should_skip_std_active_sort(active_len: usize, changed_count: usize) -> bool {
    let in_skip_window = (ACTIVE_SORT_SKIP_MIN..=ACTIVE_SORT_SKIP_MAX).contains(&active_len);
    if !in_skip_window {
        return false;
    }

    changed_count.saturating_mul(100) <= active_len.saturating_mul(ACTIVE_SORT_LOW_CHURN_PCT)
}

#[inline(always)]
fn should_probe_std_active_sort(active_len: usize, changed_count: usize) -> bool {
    let in_probe_window = (ACTIVE_SORT_SKIP_MIN..=ACTIVE_SORT_SKIP_MAX).contains(&active_len);
    if !in_probe_window {
        return false;
    }

    changed_count.saturating_mul(100) <= active_len.saturating_mul(ACTIVE_SORT_PROBE_CHURN_PCT)
}

#[inline(always)]
fn active_set_is_sorted(active_set: &[TileIdx]) -> bool {
    let len = active_set.len();
    if len <= 1 {
        return true;
    }

    let ptr = active_set.as_ptr();
    let mut i = 1usize;
    // SAFETY: `ptr` originates from `active_set`; `i` starts at 1 and is
    // incremented while `i < len`, so all dereferences are in-bounds.
    unsafe {
        let mut prev = (*ptr).0;
        while i < len {
            let cur = (*ptr.add(i)).0;
            if cur < prev {
                return false;
            }
            prev = cur;
            i += 1;
        }
    }
    true
}

#[derive(Clone, Copy)]
struct BorderEdges {
    north: u64,
    south: u64,
    west: u64,
    east: u64,
}

#[inline(always)]
unsafe fn load_border_edges(
    north_ptr: *const u64,
    south_ptr: *const u64,
    west_ptr: *const u64,
    east_ptr: *const u64,
    idx: usize,
) -> BorderEdges {
    BorderEdges {
        north: unsafe { *north_ptr.add(idx) },
        south: unsafe { *south_ptr.add(idx) },
        west: unsafe { *west_ptr.add(idx) },
        east: unsafe { *east_ptr.add(idx) },
    }
}

#[inline(always)]
fn border_neighbor_influence_mask(current: BorderEdges, previous: BorderEdges) -> u8 {
    BorderData::compute_live_mask(
        current.north ^ previous.north,
        current.south ^ previous.south,
        current.west ^ previous.west,
        current.east ^ previous.east,
    )
}

#[inline]
fn should_use_directional_neighbor_filter(
    arena: &TileArena,
    changed_tiles: &[TileIdx],
    border_phase: usize,
) -> bool {
    let changed_len = changed_tiles.len();
    if changed_len <= DIRECTIONAL_FILTER_ALWAYS_ON_MAX_CHANGED {
        return true;
    }
    if changed_len > DIRECTIONAL_FILTER_PROBE_MAX_CHANGED {
        return false;
    }
    if arena.occupied_count > 0
        && changed_len.saturating_mul(100)
            >= arena
                .occupied_count
                .saturating_mul(DIRECTIONAL_FILTER_DENSE_CHANGED_PCT)
    {
        return false;
    }

    let sample_len = changed_len.min(DIRECTIONAL_FILTER_SAMPLE_MAX);
    let borders_current = &arena.borders[border_phase];
    let borders_prev = &arena.borders[1 - border_phase];
    let current_north_ptr = borders_current.north_ptr();
    let current_south_ptr = borders_current.south_ptr();
    let current_west_ptr = borders_current.west_ptr();
    let current_east_ptr = borders_current.east_ptr();
    let prev_north_ptr = borders_prev.north_ptr();
    let prev_south_ptr = borders_prev.south_ptr();
    let prev_west_ptr = borders_prev.west_ptr();
    let prev_east_ptr = borders_prev.east_ptr();
    let changed_ptr = changed_tiles.as_ptr();
    let mut influence_dirs = 0usize;

    for i in 0..sample_len {
        let idx = unsafe { *changed_ptr.add(i) }.index();
        let influence = unsafe {
            border_neighbor_influence_mask(
                load_border_edges(
                    current_north_ptr,
                    current_south_ptr,
                    current_west_ptr,
                    current_east_ptr,
                    idx,
                ),
                load_border_edges(
                    prev_north_ptr,
                    prev_south_ptr,
                    prev_west_ptr,
                    prev_east_ptr,
                    idx,
                ),
            )
        };
        influence_dirs += influence.count_ones() as usize;
    }

    influence_dirs.saturating_mul(10)
        <= sample_len.saturating_mul(DIRECTIONAL_FILTER_MAX_AVG_DIRS_X10)
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
fn pack_expand_candidate(idx: TileIdx, dir: usize) -> u32 {
    (idx.0 << 3) | dir as u32
}

#[inline(always)]
fn prune_candidate_invalid(meta: super::tile::TileMeta) -> bool {
    !meta.occupied() || meta.has_live()
}

#[inline(always)]
pub(crate) fn append_expand_candidates(
    expand: &mut Vec<u32>,
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
    arena.active_set.clear();
    arena.changed_scratch.clear();
    // Swap instead of copy to avoid O(n) memcpy.
    std::mem::swap(&mut arena.changed_scratch, &mut arena.changed_list);
    let had_synced_changed_bits = arena.begin_changed_rebuild();

    let changed_count = arena.changed_scratch.len();
    if changed_count == 0 {
        return;
    }
    if had_synced_changed_bits {
        for i in 0..changed_count {
            let idx = arena.changed_scratch[i];
            arena.clear_changed_mark(idx.index());
        }
    }
    let meta_len = arena.meta.len();
    arena.begin_active_rebuild_with_capacity(meta_len);

    let dense_rebuild = arena.occupied_count >= 4096
        && changed_count.saturating_mul(100) >= arena.occupied_count.saturating_mul(95);
    if dense_rebuild {
        arena.active_set.reserve(arena.occupied_count);
        if arena.free_list.is_empty() && arena.occupied_count + 1 == arena.meta.len() {
            for i in 1..arena.meta.len() {
                if arena.meta[i].occupied() {
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
    let neighbors_ptr = arena.neighbors.as_ptr();
    let meta_ptr = arena.meta.as_ptr();
    let border_phase = arena.border_phase;
    let borders_current = &arena.borders[border_phase];
    let borders_prev = &arena.borders[1 - border_phase];
    let current_north_ptr = borders_current.north_ptr();
    let current_south_ptr = borders_current.south_ptr();
    let current_west_ptr = borders_current.west_ptr();
    let current_east_ptr = borders_current.east_ptr();
    let prev_north_ptr = borders_prev.north_ptr();
    let prev_south_ptr = borders_prev.south_ptr();
    let prev_west_ptr = borders_prev.west_ptr();
    let prev_east_ptr = borders_prev.east_ptr();
    let directional_neighbor_filter =
        should_use_directional_neighbor_filter(arena, &arena.changed_scratch, border_phase);

    let changed_ptr = arena.changed_scratch.as_ptr();
    for changed_i in 0..changed_count {
        let idx = unsafe { *changed_ptr.add(changed_i) };
        let i = idx.index();
        debug_assert!(i < meta_len);
        debug_assert!(unsafe { (*meta_ptr.add(i)).occupied() });
        if unsafe { !arena.active_test_and_set_unchecked(i) } {
            arena.active_set.push(idx);
        }

        if directional_neighbor_filter {
            let influence_mask = unsafe {
                border_neighbor_influence_mask(
                    load_border_edges(
                        current_north_ptr,
                        current_south_ptr,
                        current_west_ptr,
                        current_east_ptr,
                        i,
                    ),
                    load_border_edges(
                        prev_north_ptr,
                        prev_south_ptr,
                        prev_west_ptr,
                        prev_east_ptr,
                        i,
                    ),
                )
            };
            if influence_mask == 0 {
                continue;
            }

            // SAFETY: i < meta_len guaranteed by arena invariants
            // (idx came from changed_list).
            unsafe {
                let nb = *neighbors_ptr.add(i);
                let dirs = &EXPAND_MASK_TABLE.dirs[influence_mask as usize];
                let count = EXPAND_MASK_TABLE.len[influence_mask as usize] as usize;
                for &dir in dirs[..count].iter() {
                    let ni_raw = nb[dir as usize];
                    if ni_raw == NO_NEIGHBOR {
                        continue;
                    }
                    let ni_i = ni_raw as usize;
                    debug_assert!(ni_i < meta_len);
                    debug_assert!((*meta_ptr.add(ni_i)).occupied());
                    if !arena.active_test_and_set_unchecked(ni_i) {
                        arena.active_set.push(TileIdx(ni_raw));
                    }
                }
            }
        } else {
            // SAFETY: i < meta_len guaranteed by arena invariants
            // (idx came from changed_list).
            unsafe {
                let nb = *neighbors_ptr.add(i);
                for &ni_raw in nb.iter() {
                    if ni_raw == NO_NEIGHBOR {
                        continue;
                    }
                    let ni_i = ni_raw as usize;
                    debug_assert!(ni_i < meta_len);
                    debug_assert!((*meta_ptr.add(ni_i)).occupied());
                    if !arena.active_test_and_set_unchecked(ni_i) {
                        arena.active_set.push(TileIdx(ni_raw));
                    }
                }
            }
        }
    }

    // Sort active set by index for better cache locality during kernel execution.
    // Small sets use std sort unless the 2k-8k window opts out:
    // very low churn skips immediately, and moderate churn probes sortedness.
    // Very large sets use a stable two-pass radix sort.
    // The 8k-32k band still skips sorting to keep rebuild costs bounded.
    let active_len = arena.active_set.len();
    if active_len <= ACTIVE_SORT_STD_MAX {
        let skip_for_low_churn = should_skip_std_active_sort(active_len, changed_count);
        let probe_sorted =
            !skip_for_low_churn && should_probe_std_active_sort(active_len, changed_count);
        let skip_for_sorted = probe_sorted && active_set_is_sorted(&arena.active_set);
        if !(skip_for_low_churn || skip_for_sorted) {
            arena.active_set.sort_unstable_by_key(|idx| idx.0);
        }
    } else if active_len >= ACTIVE_SORT_RADIX_MIN {
        radix_sort_active_set(arena);
    }
}

/// Apply expansion and pruning from candidates collected during kernel execution.
pub fn finalize_prune_and_expand(arena: &mut TileArena) {
    if !arena.expand_buf.is_empty() {
        arena.reserve_additional_tiles(arena.expand_buf.len());
        for i in 0..arena.expand_buf.len() {
            let candidate = arena.expand_buf[i];
            let src_i = (candidate >> 3) as usize;
            let dir = (candidate & 0b111) as usize;
            debug_assert!(arena.meta[src_i].occupied());
            if arena.neighbors[src_i][dir] != NO_NEIGHBOR {
                continue;
            }
            let (tx, ty) = arena.coords[src_i];
            let (dx, dy) = EXPAND_OFFSETS[dir];
            let idx = arena.allocate_absent((tx + dx, ty + dy));
            arena.meta[idx.index()].population = 0;
            arena.mark_changed_new_unique(idx);
        }
    }

    let prune_len = arena.prune_buf.len();
    if prune_len == 0 {
        arena.expand_buf.clear();
        arena.prune_buf.clear();
        return;
    }

    let border_phase = arena.border_phase;
    let thread_count = rayon::current_num_threads().max(1);
    let run_parallel = thread_count > 1 && prune_len >= PARALLEL_PRUNE_CANDIDATES_MIN;
    let live_masks_ptr = arena.border_live_masks[border_phase].as_ptr();
    debug_assert_eq!(
        arena.border_live_masks[border_phase].len(),
        arena.borders[border_phase].len()
    );
    let mut used_bitmap_marks = false;

    if run_parallel {
        let candidate_chunk_size = prune_filter_chunk_size(prune_len, thread_count);
        let meta_ptr = SendConstPtr::new(arena.meta.as_ptr());
        let neighbors_ptr = SendConstPtr::new(arena.neighbors.as_ptr());
        let live_masks_ptr = SendConstPtr::new(live_masks_ptr);
        if prune_len >= PARALLEL_PRUNE_BITMAP_MIN {
            used_bitmap_marks = true;
            let words_len = prune_len.div_ceil(64);
            arena.prune_marks_words.resize(words_len, 0);
            let word_chunk_size = candidate_chunk_size.div_ceil(64).max(1);
            let prune_candidates_ptr = SendConstPtr::new(arena.prune_buf.as_ptr());
            let prune_marks_words = &mut arena.prune_marks_words;
            prune_marks_words
                .par_chunks_mut(word_chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, marks_words)| {
                    let word_start = chunk_idx * word_chunk_size;
                    let mp = meta_ptr.get();
                    let np = neighbors_ptr.get();
                    let lp = live_masks_ptr.get();
                    let pp = prune_candidates_ptr.get();
                    for (word_offset, marks_word) in marks_words.iter_mut().enumerate() {
                        let word_idx = word_start + word_offset;
                        let start = word_idx << 6;
                        let end = (start + 64).min(prune_len);
                        let mut word = 0u64;
                        for offset in 0..(end - start) {
                            let idx = unsafe { *pp.add(start + offset) };
                            let ii = idx.index();
                            // SAFETY: pointers are valid for the entire prune phase.
                            let should_prune = unsafe {
                                let meta = *mp.add(ii);
                                if prune_candidate_invalid(meta) {
                                    false
                                } else {
                                    let missing_mask = meta.missing_mask;
                                    missing_mask == MISSING_ALL_NEIGHBORS
                                        || ghost_is_empty_from_live_masks_ptr(lp, &*np.add(ii))
                                }
                            };
                            word |= (should_prune as u64) << offset;
                        }
                        *marks_word = word;
                    }
                });
        } else {
            arena.prune_marks.resize(prune_len, 0);
            let prune_candidates = &arena.prune_buf;
            let prune_marks = &mut arena.prune_marks;
            prune_candidates
                .par_chunks(candidate_chunk_size)
                .zip(prune_marks.par_chunks_mut(candidate_chunk_size))
                .for_each(|(chunk, marks)| {
                    let mp = meta_ptr.get();
                    let np = neighbors_ptr.get();
                    let lp = live_masks_ptr.get();
                    for (mark, &idx) in marks.iter_mut().zip(chunk.iter()) {
                        let ii = idx.index();
                        // SAFETY: pointers are valid for the entire prune phase.
                        let should_prune = unsafe {
                            let meta = *mp.add(ii);
                            if prune_candidate_invalid(meta) {
                                false
                            } else {
                                let missing_mask = meta.missing_mask;
                                missing_mask == MISSING_ALL_NEIGHBORS
                                    || ghost_is_empty_from_live_masks_ptr(lp, &*np.add(ii))
                            }
                        };
                        *mark = should_prune as u8;
                    }
                });
        }
    } else {
        for i in 0..prune_len {
            let idx = arena.prune_buf[i];
            let ii = idx.index();
            let meta = arena.meta[ii];
            if prune_candidate_invalid(meta) {
                continue;
            }
            let should_prune = meta.missing_mask == MISSING_ALL_NEIGHBORS
                || unsafe {
                    ghost_is_empty_from_live_masks_ptr(live_masks_ptr, &arena.neighbors[ii])
                };
            if should_prune {
                arena.release(idx);
            }
        }
        arena.expand_buf.clear();
        arena.prune_buf.clear();
        return;
    }

    if used_bitmap_marks {
        let prune_word_len = arena.prune_marks_words.len();
        for word_idx in 0..prune_word_len {
            let mut bits = arena.prune_marks_words[word_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let idx = arena.prune_buf[(word_idx << 6) + bit];
                arena.release(idx);
                bits &= bits - 1;
            }
        }
    } else {
        for i in 0..prune_len {
            if arena.prune_marks[i] != 0 {
                let idx = arena.prune_buf[i];
                arena.release(idx);
            }
        }
    }

    arena.expand_buf.clear();
    arena.prune_buf.clear();
}

#[cfg(test)]
mod tests {
    use super::{
        BorderEdges, PARALLEL_PRUNE_BITMAP_MIN, PARALLEL_PRUNE_CANDIDATES_MIN, TileArena,
        active_set_is_sorted, border_neighbor_influence_mask, finalize_prune_and_expand,
        rebuild_active_set, should_probe_std_active_sort, should_skip_std_active_sort,
    };
    use crate::turbolife::tile::{BorderData, MISSING_ALL_NEIGHBORS, TileIdx};

    #[test]
    fn rebuild_active_set_stays_empty_without_changes() {
        let mut arena = TileArena::new();

        rebuild_active_set(&mut arena);

        assert!(arena.active_set.is_empty());
    }

    #[test]
    fn rebuild_active_set_clears_active_tags_between_generations() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        arena.mark_changed(idx);

        rebuild_active_set(&mut arena);
        assert_eq!(arena.active_set, vec![idx]);

        rebuild_active_set(&mut arena);
        assert!(arena.active_set.is_empty());

        arena.mark_changed(idx);
        rebuild_active_set(&mut arena);
        assert_eq!(arena.active_set, vec![idx]);
    }

    #[test]
    fn rebuild_active_set_keeps_other_tiles_markable() {
        let mut arena = TileArena::new();
        let first = arena.allocate((0, 0));
        let second = arena.allocate((8, 8));

        arena.mark_changed(first);
        rebuild_active_set(&mut arena);
        assert!(arena.changed_list.is_empty());

        arena.mark_changed(second);
        assert_eq!(arena.changed_list, vec![second]);
    }

    #[test]
    fn finalize_prune_and_expand_skips_live_candidates() {
        let mut arena = TileArena::new();
        let live_idx = arena.allocate((0, 0));
        arena.meta_mut(live_idx).set_has_live(true);
        arena.prune_buf.push(live_idx);

        finalize_prune_and_expand(&mut arena);

        assert!(arena.meta(live_idx).occupied());
        assert!(arena.meta(live_idx).has_live());
        assert_eq!(arena.idx_at((0, 0)), Some(live_idx));
    }

    #[test]
    fn finalize_prune_and_expand_prunes_large_empty_candidate_sets() {
        let mut arena = TileArena::new();
        let mut prune_candidates = Vec::new();
        for x in 0..PARALLEL_PRUNE_CANDIDATES_MIN {
            let idx = arena.allocate((x as i64, 0));
            let meta = arena.meta_mut(idx);
            meta.set_has_live(false);
            meta.missing_mask = MISSING_ALL_NEIGHBORS;
            prune_candidates.push(idx);
        }
        arena.prune_buf.extend(prune_candidates.iter().copied());

        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("failed to build rayon test pool")
            .install(|| finalize_prune_and_expand(&mut arena));

        for idx in prune_candidates {
            assert!(!arena.meta(idx).occupied());
        }
        assert!(arena.prune_buf.is_empty());
    }

    #[test]
    fn finalize_prune_and_expand_bitmap_marks_prune_candidates() {
        let mut arena = TileArena::new();
        let idx = arena.allocate((0, 0));
        let meta = arena.meta_mut(idx);
        meta.set_has_live(false);
        meta.missing_mask = MISSING_ALL_NEIGHBORS;
        arena.prune_buf.resize(PARALLEL_PRUNE_BITMAP_MIN, idx);

        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("failed to build rayon test pool")
            .install(|| finalize_prune_and_expand(&mut arena));

        assert!(!arena.meta(idx).occupied());
        assert!(arena.prune_buf.is_empty());
    }

    #[test]
    fn active_set_is_sorted_detects_monotonic_sequences() {
        assert!(active_set_is_sorted(&[]));
        assert!(active_set_is_sorted(&[TileIdx(1)]));
        assert!(active_set_is_sorted(&[
            TileIdx(1),
            TileIdx(2),
            TileIdx(2),
            TileIdx(9)
        ]));
        assert!(!active_set_is_sorted(&[
            TileIdx(1),
            TileIdx(4),
            TileIdx(3),
            TileIdx(9),
        ]));
    }

    #[test]
    fn low_churn_skip_window_extends_to_std_sort_limit() {
        assert!(!should_skip_std_active_sort(2_047, 300));
        assert!(should_skip_std_active_sort(8_192, 1_638));
        assert!(!should_skip_std_active_sort(8_192, 1_639));
        assert!(!should_skip_std_active_sort(8_193, 1_000));
    }

    #[test]
    fn std_sort_probe_window_caps_at_forty_percent_churn() {
        assert!(!should_probe_std_active_sort(2_047, 700));
        assert!(should_probe_std_active_sort(2_048, 819));
        assert!(!should_probe_std_active_sort(2_048, 820));
        assert!(should_probe_std_active_sort(8_192, 3_276));
        assert!(!should_probe_std_active_sort(8_192, 3_277));
        assert!(!should_probe_std_active_sort(8_193, 100));
    }

    #[test]
    fn border_neighbor_influence_mask_tracks_edges_and_corners() {
        let prev = BorderData::default();
        let cur = BorderData::from_edges(1u64 << 5, 1u64 << 63, 0, 0);
        let mask = border_neighbor_influence_mask(
            BorderEdges {
                north: cur.north,
                south: cur.south,
                west: cur.west,
                east: cur.east,
            },
            BorderEdges {
                north: prev.north,
                south: prev.south,
                west: prev.west,
                east: prev.east,
            },
        );
        assert_eq!(mask & (1 << 0), 1 << 0); // N
        assert_eq!(mask & (1 << 1), 1 << 1); // S
        assert_eq!(mask & (1 << 5), 0); // NE corner unchanged
        assert_eq!(mask & (1 << 7), 1 << 7); // SE corner changed
    }

    #[test]
    fn rebuild_active_set_skips_neighbors_when_border_is_unchanged() {
        let mut arena = TileArena::new();
        let center = arena.allocate((0, 0));
        let _north = arena.allocate((0, 1));
        let _south = arena.allocate((0, -1));
        let _west = arena.allocate((-1, 0));
        let _east = arena.allocate((1, 0));
        let _nw = arena.allocate((-1, 1));
        let _ne = arena.allocate((1, 1));
        let _sw = arena.allocate((-1, -1));
        let _se = arena.allocate((1, -1));

        let bp = arena.border_phase;
        arena.borders[bp].set(center.index(), BorderData::default());
        arena.borders[1 - bp].set(center.index(), BorderData::default());
        arena.mark_changed(center);

        rebuild_active_set(&mut arena);

        assert_eq!(arena.active_set, vec![center]);
    }

    #[test]
    fn rebuild_active_set_only_activates_impacted_neighbors() {
        let mut arena = TileArena::new();
        let center = arena.allocate((0, 0));
        let north = arena.allocate((0, 1));
        let _south = arena.allocate((0, -1));
        let _west = arena.allocate((-1, 0));
        let _east = arena.allocate((1, 0));
        let nw = arena.allocate((-1, 1));
        let _ne = arena.allocate((1, 1));
        let _sw = arena.allocate((-1, -1));
        let _se = arena.allocate((1, -1));

        let bp = arena.border_phase;
        arena.borders[bp].set(center.index(), BorderData::from_edges(1, 0, 0, 0));
        arena.borders[1 - bp].set(center.index(), BorderData::default());
        arena.mark_changed(center);

        rebuild_active_set(&mut arena);

        let mut active = arena.active_set.clone();
        active.sort_unstable_by_key(|idx| idx.0);
        assert_eq!(active, vec![center, north, nw]);
    }
}
