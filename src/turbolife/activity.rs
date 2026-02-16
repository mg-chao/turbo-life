//! Active set rebuild and post-kernel prune/expand for TurboLife.

use rayon::prelude::*;

use super::arena::TileArena;
use super::kernel::ghost_is_empty_from_live_masks_ptr;
use super::tile::{MISSING_ALL_NEIGHBORS, NO_NEIGHBOR, TileIdx};

const PARALLEL_PRUNE_CANDIDATES_MIN: usize = 512;
const PRUNE_FILTER_CHUNK_MIN: usize = 64;
const PRUNE_FILTER_CHUNK_MAX: usize = 512;
const PARALLEL_PRUNE_BITMAP_MIN: usize = 4_096;
const ACTIVE_SORT_STD_MAX: usize = 8_192;
const ACTIVE_SORT_RADIX_MIN: usize = 32_768;
const ACTIVE_SORT_RADIX_BUCKETS: usize = 1 << 16;
const ACTIVE_BITMAP_REBUILD_MIN_OCCUPIED: usize = 2_048;
const ACTIVE_BITMAP_REBUILD_MAX_OCCUPIED: usize = 8_192;
const ACTIVE_BITMAP_REBUILD_MIN_CHANGED: usize = 1_024;
const ACTIVE_BITMAP_REBUILD_DENSE_CHANGED_PCT: usize = 45;
const _: [(); 1] = [(); (ACTIVE_SORT_STD_MAX > 0) as usize];
const _: [(); 1] = [(); (ACTIVE_SORT_STD_MAX < ACTIVE_SORT_RADIX_MIN) as usize];
const _: [(); 1] = [(); (ACTIVE_BITMAP_REBUILD_DENSE_CHANGED_PCT <= 100) as usize];
const _: [(); 1] = [(); (ACTIVE_BITMAP_REBUILD_MIN_CHANGED > 0) as usize];
const _: [(); 1] =
    [(); (ACTIVE_BITMAP_REBUILD_MIN_OCCUPIED <= ACTIVE_BITMAP_REBUILD_MAX_OCCUPIED) as usize];

struct SendConstPtr<T> {
    inner: *const T,
}
unsafe impl<T: Sync> Send for SendConstPtr<T> {}
unsafe impl<T: Sync> Sync for SendConstPtr<T> {}
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
fn should_use_bitmap_active_rebuild(occupied_count: usize, changed_count: usize) -> bool {
    (ACTIVE_BITMAP_REBUILD_MIN_OCCUPIED..=ACTIVE_BITMAP_REBUILD_MAX_OCCUPIED)
        .contains(&occupied_count)
        && changed_count >= ACTIVE_BITMAP_REBUILD_MIN_CHANGED
        && changed_count.saturating_mul(100)
            >= occupied_count.saturating_mul(ACTIVE_BITMAP_REBUILD_DENSE_CHANGED_PCT)
}

#[inline(always)]
fn active_set_is_sorted(active_set: &[TileIdx]) -> bool {
    let len = active_set.len();
    if len < 2 {
        return true;
    }
    // SAFETY: len >= 2 and `ptr` comes from `active_set`, so `ptr.add(i)` is
    // valid for every i in 0..len.
    unsafe {
        let ptr = active_set.as_ptr();
        let mut prev = (*ptr).0;
        let mut i = 1usize;
        while i < len {
            let curr = (*ptr.add(i)).0;
            if prev > curr {
                return false;
            }
            prev = curr;
            i += 1;
        }
    }
    true
}

#[inline]
fn radix_sort_active_set(arena: &mut TileArena) {
    let len = arena.active_set.len();
    if len <= 1 {
        return;
    }

    if len > u32::MAX as usize || arena.active_sort_counts.len() != ACTIVE_SORT_RADIX_BUCKETS {
        arena.active_set.sort_unstable_by_key(|idx| idx.0);
        return;
    }

    debug_assert_eq!(arena.active_sort_counts.len(), ACTIVE_SORT_RADIX_BUCKETS);

    arena.active_sort_scratch.clear();
    arena.active_sort_scratch.resize(len, TileIdx(0));
    let counts = &mut arena.active_sort_counts;
    counts.fill(0);

    // SAFETY: `src_ptr` comes from `active_set` and we only read 0..len.
    // Bucket indices are masked to 16 bits and fit in `counts`.
    unsafe {
        let src_ptr = arena.active_set.as_ptr();
        let mut i = 0usize;
        while i < len {
            let idx = (*src_ptr.add(i)).0;
            *counts.get_unchecked_mut((idx & 0xFFFF) as usize) += 1;
            i += 1;
        }
    }

    let mut prefix = 0u32;
    let mut bucket = 0usize;
    while bucket < counts.len() {
        let count = unsafe { counts.get_unchecked_mut(bucket) };
        let next = prefix + *count;
        *count = prefix;
        prefix = next;
        bucket += 1;
    }

    // SAFETY: `src_ptr` and `dst_ptr` come from vectors sized to `len`.
    // Prefix counts produce destinations in 0..len.
    unsafe {
        let src_ptr = arena.active_set.as_ptr();
        let dst_ptr = arena.active_sort_scratch.as_mut_ptr();
        let mut i = 0usize;
        while i < len {
            let idx = *src_ptr.add(i);
            let bucket = (idx.0 & 0xFFFF) as usize;
            let dst = *counts.get_unchecked(bucket) as usize;
            dst_ptr.add(dst).write(idx);
            *counts.get_unchecked_mut(bucket) = (dst + 1) as u32;
            i += 1;
        }
    }

    counts.fill(0);
    // SAFETY: `src_ptr` comes from `active_sort_scratch` and we only read 0..len.
    // High 16-bit buckets are always in range for `counts`.
    unsafe {
        let src_ptr = arena.active_sort_scratch.as_ptr();
        let mut i = 0usize;
        while i < len {
            let idx = (*src_ptr.add(i)).0;
            *counts.get_unchecked_mut((idx >> 16) as usize) += 1;
            i += 1;
        }
    }

    prefix = 0;
    bucket = 0;
    while bucket < counts.len() {
        let count = unsafe { counts.get_unchecked_mut(bucket) };
        let next = prefix + *count;
        *count = prefix;
        prefix = next;
        bucket += 1;
    }

    // SAFETY: `src_ptr` and `dst_ptr` come from vectors sized to `len`.
    // Prefix counts produce destinations in 0..len.
    unsafe {
        let src_ptr = arena.active_sort_scratch.as_ptr();
        let dst_ptr = arena.active_set.as_mut_ptr();
        let mut i = 0usize;
        while i < len {
            let idx = *src_ptr.add(i);
            let bucket = (idx.0 >> 16) as usize;
            let dst = *counts.get_unchecked(bucket) as usize;
            dst_ptr.add(dst).write(idx);
            *counts.get_unchecked_mut(bucket) = (dst + 1) as u32;
            i += 1;
        }
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

/// # Safety
/// Caller must ensure `expand` has enough spare capacity for all candidates in
/// `missing_mask & live_mask` before calling.
#[inline(always)]
pub(crate) unsafe fn append_expand_candidates_unchecked(
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

    let len = expand.len();
    debug_assert!(len.saturating_add(count) <= expand.capacity());
    unsafe {
        let mut write_ptr = expand.as_mut_ptr().add(len);
        for &dir in dirs[..count].iter() {
            write_ptr.write(pack_expand_candidate(idx, dir as usize));
            write_ptr = write_ptr.add(1);
        }
        expand.set_len(len + count);
    }
}

/// Rebuild the active set from the changed list in O(changed * 9).
///
/// Uses unsafe raw pointer access to meta/neighbors for the hot inner loop
/// to eliminate bounds checks on every neighbor lookup.
pub fn rebuild_active_set(arena: &mut TileArena) {
    arena.changed_scratch.clear();
    arena.changed_influence_scratch.clear();
    let (had_synced_changed_bits, changed_influence_uniform_all) = arena.begin_changed_rebuild();
    // Swap instead of copy to avoid O(n) memcpy.
    std::mem::swap(&mut arena.changed_scratch, &mut arena.changed_list);
    if !changed_influence_uniform_all {
        std::mem::swap(
            &mut arena.changed_influence_scratch,
            &mut arena.changed_influence,
        );
    }

    let changed_count = arena.changed_scratch.len();
    if !changed_influence_uniform_all {
        debug_assert_eq!(changed_count, arena.changed_influence_scratch.len());
    }
    if changed_count == 0 {
        arena.active_set.clear();
        arena.active_set_dense_contiguous = false;
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
        let can_reuse_full_contiguous_active = arena.free_list.is_empty()
            && arena.occupied_count + 1 == arena.meta.len()
            && arena.active_set.len() == arena.occupied_count
            && arena.active_set_dense_contiguous;
        if can_reuse_full_contiguous_active {
            return;
        }

        arena.active_set.clear();
        arena.active_set.reserve(arena.occupied_count);
        if arena.free_list.is_empty() && arena.occupied_count + 1 == arena.meta.len() {
            let count = arena.occupied_count;
            debug_assert_eq!(arena.active_set.len(), 0);
            debug_assert!(
                arena.meta[1..].iter().all(|meta| meta.occupied()),
                "dense contiguous rebuild requires occupied metadata for all slots"
            );
            unsafe {
                let ptr = arena.active_set.as_mut_ptr();
                arena.active_set.set_len(count);
                for i in 0..count {
                    ptr.add(i).write(TileIdx((i + 1) as u32));
                }
            }
            arena.active_set_dense_contiguous = true;
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
            arena.active_set_dense_contiguous = false;
        }
        return;
    }

    arena.active_set.clear();
    arena.active_set_dense_contiguous = false;

    let bitmap_rebuild = should_use_bitmap_active_rebuild(arena.occupied_count, changed_count);
    if bitmap_rebuild {
        let word_len = meta_len.div_ceil(64);
        if arena.active_marks_words.len() < word_len {
            arena.active_marks_words.resize(word_len, 0);
        } else {
            arena.active_marks_words[..word_len].fill(0);
        }

        let marks_ptr = arena.active_marks_words.as_mut_ptr();
        let neighbors_ptr = arena.neighbors.as_ptr();
        let changed_ptr = arena.changed_scratch.as_ptr();
        if changed_influence_uniform_all {
            for changed_i in 0..changed_count {
                let idx = unsafe { *changed_ptr.add(changed_i) };
                let i = idx.index();

                unsafe {
                    let word = i >> 6;
                    *marks_ptr.add(word) |= 1u64 << (i & 63);
                    let nb = *neighbors_ptr.add(i);
                    for &ni_raw in nb.iter() {
                        let ni = ni_raw as usize;
                        let ni_word = ni >> 6;
                        *marks_ptr.add(ni_word) |= 1u64 << (ni & 63);
                    }
                }
            }
        } else {
            let influence_ptr = arena.changed_influence_scratch.as_ptr();
            for changed_i in 0..changed_count {
                let idx = unsafe { *changed_ptr.add(changed_i) };
                let influence_mask = unsafe { *influence_ptr.add(changed_i) };
                let i = idx.index();

                unsafe {
                    let word = i >> 6;
                    *marks_ptr.add(word) |= 1u64 << (i & 63);
                    let nb = *neighbors_ptr.add(i);
                    if influence_mask == u8::MAX {
                        for &ni_raw in nb.iter() {
                            let ni = ni_raw as usize;
                            let ni_word = ni >> 6;
                            *marks_ptr.add(ni_word) |= 1u64 << (ni & 63);
                        }
                    } else {
                        let dirs = &EXPAND_MASK_TABLE.dirs[influence_mask as usize];
                        let count = EXPAND_MASK_TABLE.len[influence_mask as usize] as usize;
                        for &dir in dirs[..count].iter() {
                            let ni_raw = nb[dir as usize];
                            let ni = ni_raw as usize;
                            let ni_word = ni >> 6;
                            *marks_ptr.add(ni_word) |= 1u64 << (ni & 63);
                        }
                    }
                }
            }
        }

        if word_len != 0 {
            arena.active_marks_words[0] &= !1u64;
            let tail_bits = meta_len & 63;
            if tail_bits != 0 {
                let tail_mask = (1u64 << tail_bits) - 1;
                let last_word = word_len - 1;
                arena.active_marks_words[last_word] &= tail_mask;
            }
        }

        arena
            .active_set
            .reserve(changed_count.saturating_mul(9).min(arena.occupied_count));
        for word_idx in 0..word_len {
            let mut bits = arena.active_marks_words[word_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let i = (word_idx << 6) + bit;
                debug_assert!(i < meta_len);
                debug_assert!(arena.meta[i].occupied());
                arena.active_set.push(TileIdx(i as u32));
                bits &= bits - 1;
            }
        }
        if arena.free_list.is_empty()
            && arena.occupied_count + 1 == meta_len
            && arena.active_set.len() == arena.occupied_count
            && arena.active_set.first().map(|idx| idx.0) == Some(1)
            && arena.active_set.last().map(|idx| idx.0) == Some(arena.occupied_count as u32)
        {
            arena.active_set_dense_contiguous = true;
        }
        return;
    }

    arena
        .active_set
        .reserve(changed_count.saturating_mul(9).min(arena.occupied_count));
    let neighbors_ptr = arena.neighbors.as_ptr();
    let meta_ptr = arena.meta.as_ptr();
    let changed_ptr = arena.changed_scratch.as_ptr();
    if changed_influence_uniform_all {
        for changed_i in 0..changed_count {
            let idx = unsafe { *changed_ptr.add(changed_i) };
            let i = idx.index();
            debug_assert!(i < meta_len);
            debug_assert!(unsafe { (*meta_ptr.add(i)).occupied() });
            if unsafe { !arena.active_test_and_set_unchecked(i) } {
                arena.active_set.push(idx);
            }

            unsafe {
                let nb = *neighbors_ptr.add(i);
                for &ni_raw in nb.iter() {
                    let ni_i = ni_raw as usize;
                    debug_assert!(ni_i < meta_len);
                    debug_assert!(ni_i == NO_NEIGHBOR as usize || (*meta_ptr.add(ni_i)).occupied());
                    if !arena.active_test_and_set_unchecked(ni_i) {
                        arena.active_set.push(TileIdx(ni_raw));
                    }
                }
            }
        }
    } else {
        let influence_ptr = arena.changed_influence_scratch.as_ptr();
        for changed_i in 0..changed_count {
            let idx = unsafe { *changed_ptr.add(changed_i) };
            let influence_mask = unsafe { *influence_ptr.add(changed_i) };
            let i = idx.index();
            debug_assert!(i < meta_len);
            debug_assert!(unsafe { (*meta_ptr.add(i)).occupied() });
            if unsafe { !arena.active_test_and_set_unchecked(i) } {
                arena.active_set.push(idx);
            }
            if influence_mask == 0 {
                continue;
            }

            // SAFETY: i < meta_len guaranteed by arena invariants
            // (idx came from changed_list).
            unsafe {
                let nb = *neighbors_ptr.add(i);
                if influence_mask == u8::MAX {
                    for &ni_raw in nb.iter() {
                        let ni_i = ni_raw as usize;
                        debug_assert!(ni_i < meta_len);
                        debug_assert!(
                            ni_i == NO_NEIGHBOR as usize || (*meta_ptr.add(ni_i)).occupied()
                        );
                        if !arena.active_test_and_set_unchecked(ni_i) {
                            arena.active_set.push(TileIdx(ni_raw));
                        }
                    }
                } else {
                    let dirs = &EXPAND_MASK_TABLE.dirs[influence_mask as usize];
                    let count = EXPAND_MASK_TABLE.len[influence_mask as usize] as usize;
                    for &dir in dirs[..count].iter() {
                        let ni_raw = nb[dir as usize];
                        let ni_i = ni_raw as usize;
                        debug_assert!(ni_i < meta_len);
                        debug_assert!(
                            ni_i == NO_NEIGHBOR as usize || (*meta_ptr.add(ni_i)).occupied()
                        );
                        if !arena.active_test_and_set_unchecked(ni_i) {
                            arena.active_set.push(TileIdx(ni_raw));
                        }
                    }
                }
            }
        }
    }

    // Sort active set by index for better cache locality during kernel execution.
    // Small and mid-size frontiers always use std sort; very large frontiers
    // use a stable two-pass radix sort. The 8k-32k band still skips sorting
    // to keep rebuild costs bounded.
    let active_len = arena.active_set.len();
    let mut active_set_sorted = false;
    if active_len <= ACTIVE_SORT_STD_MAX {
        if !active_set_is_sorted(&arena.active_set) {
            arena.active_set.sort_unstable_by_key(|idx| idx.0);
        }
        active_set_sorted = true;
    } else if active_len >= ACTIVE_SORT_RADIX_MIN {
        if !active_set_is_sorted(&arena.active_set) {
            radix_sort_active_set(arena);
        }
        active_set_sorted = true;
    }

    if active_set_sorted
        && arena.free_list.is_empty()
        && arena.occupied_count + 1 == arena.meta.len()
        && arena.active_set.len() == arena.occupied_count
        && arena.active_set.first().map(|idx| idx.0) == Some(1)
        && arena.active_set.last().map(|idx| idx.0) == Some(arena.occupied_count as u32)
    {
        arena.active_set_dense_contiguous = true;
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
            let src_idx = TileIdx(src_i as u32);
            let (idx, was_allocated) = arena.allocate_absent_neighbor_from(src_idx, dir);
            if !was_allocated {
                continue;
            }
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
                                if !meta.occupied() || meta.has_live() {
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
                            if !meta.occupied() || meta.has_live() {
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
            if !meta.occupied() || meta.has_live() {
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
    use std::collections::BTreeSet;

    use super::{
        PARALLEL_PRUNE_BITMAP_MIN, PARALLEL_PRUNE_CANDIDATES_MIN, TileArena, active_set_is_sorted,
        finalize_prune_and_expand, rebuild_active_set, should_use_bitmap_active_rebuild,
    };
    use crate::turbolife::tile::{MISSING_ALL_NEIGHBORS, NO_NEIGHBOR, TileIdx};

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
    fn dense_rebuild_contiguous_slots_stay_index_ordered() {
        let mut arena = TileArena::new();
        let tile_count = 4_096usize;

        for x in 0..tile_count {
            let idx = arena.allocate((x as i64, 0));
            arena.mark_changed(idx);
        }

        rebuild_active_set(&mut arena);

        assert_eq!(arena.active_set.len(), tile_count);
        for (i, idx) in arena.active_set.iter().enumerate() {
            assert_eq!(idx.0, (i + 1) as u32);
        }
    }

    #[test]
    fn dense_rebuild_skips_stale_contiguous_cache_after_growth() {
        let mut arena = TileArena::new();
        let tile_count = 4_096usize;

        for x in 0..tile_count {
            let idx = arena.allocate((x as i64, 0));
            arena.mark_changed(idx);
        }
        rebuild_active_set(&mut arena);
        assert_eq!(arena.active_set.len(), tile_count);
        assert!(arena.active_set_dense_contiguous);

        let new_idx = arena.allocate((tile_count as i64, 0));
        for i in 1..=tile_count {
            arena.mark_changed(TileIdx(i as u32));
        }
        arena.mark_changed(new_idx);

        rebuild_active_set(&mut arena);

        assert_eq!(arena.active_set.len(), arena.occupied_count);
        assert_eq!(arena.active_set.first().map(|idx| idx.0), Some(1));
        assert_eq!(
            arena.active_set.last().map(|idx| idx.0),
            Some(arena.occupied_count as u32)
        );
    }

    #[test]
    fn dense_rebuild_repairs_corrupted_cache_even_on_low_churn() {
        let mut arena = TileArena::new();
        let tile_count = 4_096usize;
        let mut tiles = Vec::with_capacity(tile_count);

        for x in 0..tile_count {
            let idx = arena.allocate((x as i64, 0));
            arena.mark_changed(idx);
            tiles.push(idx);
        }

        rebuild_active_set(&mut arena);
        assert_eq!(arena.active_set.len(), tile_count);

        // Simulate stale/corrupted cached ordering that can otherwise satisfy
        // quick first/last checks.
        arena.active_set[1] = TileIdx(1);

        // Keep churn intentionally below the old "skip std-sort" threshold.
        for &idx in tiles.iter().take(800) {
            arena.mark_changed(idx);
        }
        rebuild_active_set(&mut arena);

        for (i, idx) in arena.active_set.iter().enumerate() {
            assert_eq!(idx.0, (i + 1) as u32);
        }
    }

    #[test]
    fn bitmap_rebuild_gate_targets_mid_large_dense_frontiers() {
        assert!(!should_use_bitmap_active_rebuild(2_047, 1_024));
        assert!(!should_use_bitmap_active_rebuild(2_048, 1_023));
        assert!(should_use_bitmap_active_rebuild(2_048, 1_024));
        assert!(!should_use_bitmap_active_rebuild(8_193, 4_096));
        assert!(!should_use_bitmap_active_rebuild(4_096, 1_000));
    }

    #[test]
    fn bitmap_rebuild_matches_reference_neighbor_expansion() {
        let mut arena = TileArena::new();
        let width = 64i64;
        let height = 32i64;
        let mut tiles = Vec::new();

        for y in 0..height {
            for x in 0..width {
                tiles.push(arena.allocate((x, y)));
            }
        }

        let changed = 1_024usize;
        for &idx in tiles.iter().take(changed) {
            arena.mark_changed(idx);
        }

        let mut expected = BTreeSet::new();
        for &idx in tiles.iter().take(changed) {
            expected.insert(idx.0);
            for &ni in arena.neighbors[idx.index()].iter() {
                if ni != NO_NEIGHBOR {
                    expected.insert(ni);
                }
            }
        }

        rebuild_active_set(&mut arena);

        let active_raw: Vec<u32> = arena.active_set.iter().map(|idx| idx.0).collect();
        let mut expected_vec: Vec<u32> = expected.into_iter().collect();
        expected_vec.retain(|&idx| idx != 0);

        assert!(active_set_is_sorted(&arena.active_set));
        assert_eq!(active_raw, expected_vec);
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
    fn rebuild_active_set_respects_zero_influence_masks() {
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

        arena.mark_changed_with_influence(center, 0);
        rebuild_active_set(&mut arena);

        assert_eq!(arena.active_set, vec![center]);
    }

    #[test]
    fn rebuild_active_set_activates_only_masked_neighbors() {
        let mut arena = TileArena::new();
        let center = arena.allocate((0, 0));
        let north = arena.allocate((0, 1));
        let _south = arena.allocate((0, -1));
        let west = arena.allocate((-1, 0));
        let _east = arena.allocate((1, 0));
        let nw = arena.allocate((-1, 1));
        let _ne = arena.allocate((1, 1));
        let _sw = arena.allocate((-1, -1));
        let _se = arena.allocate((1, -1));

        // North + West + NW influence.
        arena.mark_changed_with_influence(center, (1 << 0) | (1 << 2) | (1 << 4));

        rebuild_active_set(&mut arena);

        let mut active = arena.active_set.clone();
        active.sort_unstable_by_key(|idx| idx.0);
        assert_eq!(active, vec![center, north, west, nw]);
    }

    #[test]
    fn bitmap_rebuild_respects_zero_influence_masks() {
        let mut arena = TileArena::new();
        let width = 64i64;
        let height = 32i64;
        let mut tiles = Vec::new();

        for y in 0..height {
            for x in 0..width {
                tiles.push(arena.allocate((x, y)));
            }
        }

        for &idx in tiles.iter().take(1_024) {
            arena.mark_changed_with_influence(idx, 0);
        }

        rebuild_active_set(&mut arena);

        assert_eq!(arena.active_set.len(), 1_024);
        assert!(active_set_is_sorted(&arena.active_set));
    }
}
