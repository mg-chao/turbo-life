//! `TileMap` — a turbolife-specialized open-addressing hashmap for
//! `(i64, i64) → TileIdx` lookups.
//!
//! Design goals:
//! - Flat, cache-friendly layout with compact control metadata.
//! - Robin Hood probing with backward-shift deletion (no tombstones).
//! - Specialized hash for tile coordinates with a throughput-first fast mix.
//! - Fingerprint check in the control word to skip full key comparisons.
//! - Probe distance encoded in the control word (no per-probe hash reload).

use super::tile::TileIdx;

// ── Hash function ───────────────────────────────────────────────────────

/// Two distinct Fibonacci-derived constants for mixing x and y independently.
/// This avoids the systematic collisions that a single-constant sequential
/// hash (like FxHash) produces on grid-aligned coordinate patterns.
const MX: u64 = 0x517c_c1b7_2722_0a95;
const MY: u64 = 0x6c62_272e_07bb_0142;

#[inline(always)]
pub(crate) fn tile_hash(x: i64, y: i64) -> u64 {
    // Throughput-first 2-multiply mix (no final fold). Rotate y's lane so
    // both axes contribute entropy to low bucket bits.
    (x as u64).wrapping_mul(MX) ^ (y as u64).wrapping_mul(MY).rotate_right(31)
}

// ── Slot layout ─────────────────────────────────────────────────────────

const EMPTY: u32 = 0;
const OCCUPIED_BIT: u32 = 0x8000_0000;
const DIST_SHIFT: u32 = 12;
const DIST_MASK: u32 = 0x7fff_f000;
const FP_MASK: u32 = 0x0000_0fff;
const MATCH_MASK: u32 = OCCUPIED_BIT | FP_MASK;
const MAX_DIST: usize = (DIST_MASK >> DIST_SHIFT) as usize;

/// A single slot in the hash table.
///
/// Layout: key_x(8) + key_y(8) + value(4) + ctrl(4) = 24 bytes.
///
/// Control word layout:
/// - bit 31: occupied flag
/// - bits 12..30: probe distance (Robin Hood DIB)
/// - bits 0..11: key fingerprint
#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    key_x: i64,
    key_y: i64,
    value: u32,
    /// High bit = occupied, mid bits = probe distance, low bits = fingerprint.
    /// `0` means empty.
    ctrl: u32,
}

impl Slot {
    const EMPTY: Self = Self {
        key_x: 0,
        key_y: 0,
        value: 0,
        ctrl: EMPTY,
    };

    #[inline(always)]
    fn is_occupied(self) -> bool {
        self.ctrl & OCCUPIED_BIT != 0
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.ctrl == EMPTY
    }

    #[inline(always)]
    fn distance(self) -> usize {
        ((self.ctrl & DIST_MASK) >> DIST_SHIFT) as usize
    }

    #[inline(always)]
    fn set_distance(&mut self, distance: usize) {
        assert!(
            distance <= MAX_DIST,
            "TileMap probe distance overflow (distance={distance}, max={MAX_DIST})"
        );
        self.ctrl = (self.ctrl & !DIST_MASK) | ((distance as u32) << DIST_SHIFT);
    }

    #[inline(always)]
    fn match_ctrl(self) -> u32 {
        self.ctrl & MATCH_MASK
    }
}

// ── TileMap ─────────────────────────────────────────────────────────────

/// Maximum load factor numerator / denominator: 1/2 = 50%.
const LOAD_NUM: usize = 1;
const LOAD_DEN: usize = 2;

/// Open-addressed hashmap specialised for `(i64, i64) → TileIdx`.
pub struct TileMap {
    slots: Vec<Slot>,
    len: usize,
    /// `capacity - 1`.  Capacity is always a power of two.
    mask: usize,
}

impl TileMap {
    /// Create an empty map with room for at least `cap` entries before growing.
    pub fn with_capacity(cap: usize) -> Self {
        // We need capacity such that `cap * LOAD_DEN <= capacity * LOAD_NUM`.
        let min_slots = cap
            .saturating_mul(LOAD_DEN)
            .div_ceil(LOAD_NUM)
            .next_power_of_two()
            .max(16);
        Self {
            slots: vec![Slot::EMPTY; min_slots],
            len: 0,
            mask: min_slots - 1,
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Reserve room for `additional` more entries without rehashing.
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len.saturating_add(additional);
        let min_slots = required
            .saturating_mul(LOAD_DEN)
            .div_ceil(LOAD_NUM)
            .next_power_of_two();
        if min_slots > self.slots.len() {
            self.resize(min_slots);
        }
    }

    // ── Lookup ──────────────────────────────────────────────────────────

    /// Look up a tile index by coordinate.
    ///
    /// Uses Robin Hood displacement for early exit on miss: if the slot's
    /// displacement is less than what our key would have at this position,
    /// the key cannot exist further in the chain.
    #[inline(always)]
    pub fn get(&self, x: i64, y: i64) -> Option<TileIdx> {
        let hash = tile_hash(x, y);
        self.get_hashed(x, y, hash)
    }

    #[inline(always)]
    pub(crate) fn get_hashed(&self, x: i64, y: i64, hash: u64) -> Option<TileIdx> {
        debug_assert_eq!(hash, tile_hash(x, y));
        let target_ctrl = match_ctrl_of(hash);
        let mut pos = hash as usize & self.mask;
        let mask = self.mask;
        let slots_ptr = self.slots.as_ptr();
        let mut our_dist = 0usize;

        // SAFETY: `pos` is always `& self.mask` which is < self.slots.len()
        // (capacity is a power of two, mask = capacity - 1).
        loop {
            let slot = unsafe { *slots_ptr.add(pos) };
            if slot.is_empty() {
                return None;
            }
            // Robin Hood early exit: if this occupied slot is closer to its
            // home than we are to ours, our key was never inserted here.
            let their_dist = slot.distance();
            if our_dist > their_dist {
                return None;
            }
            if slot.match_ctrl() == target_ctrl && slot.key_x == x && slot.key_y == y {
                return Some(TileIdx(slot.value));
            }
            pos = (pos + 1) & mask;
            our_dist += 1;
        }
    }

    // ── Insert ──────────────────────────────────────────────────────────

    /// Insert a mapping.  Returns the previous value if the key was present.
    #[inline]
    #[allow(dead_code)]
    pub fn insert(&mut self, x: i64, y: i64, value: TileIdx) -> Option<TileIdx> {
        let hash = tile_hash(x, y);
        self.insert_hashed(x, y, value, hash)
    }

    #[inline]
    pub(crate) fn insert_hashed(
        &mut self,
        x: i64,
        y: i64,
        value: TileIdx,
        hash: u64,
    ) -> Option<TileIdx> {
        debug_assert_eq!(hash, tile_hash(x, y));
        if self.needs_grow() {
            self.grow();
        }
        self.insert_no_grow(x, y, value, hash)
    }

    /// Insert without checking capacity — caller must ensure space.
    #[inline]
    fn insert_no_grow(&mut self, x: i64, y: i64, value: TileIdx, hash: u64) -> Option<TileIdx> {
        let target_ctrl = match_ctrl_of(hash);
        let mask = self.mask;
        let mut pos = hash as usize & mask;
        let slots_ptr = self.slots.as_mut_ptr();

        let mut ins = Slot {
            key_x: x,
            key_y: y,
            value: value.0,
            ctrl: occupied_ctrl(fingerprint_of(hash), 0),
        };

        loop {
            // SAFETY: pos is always & mask which is < slots.len().
            let slot = unsafe { &mut *slots_ptr.add(pos) };

            if slot.is_empty() {
                *slot = ins;
                self.len += 1;
                return None;
            }

            // Exact match — update in place.
            if slot.match_ctrl() == target_ctrl && slot.key_x == x && slot.key_y == y {
                let old = TileIdx(slot.value);
                slot.value = value.0;
                return Some(old);
            }

            // Robin Hood: if the existing entry is closer to home, steal its spot.
            let ins_dist = ins.distance();
            let existing_dist = slot.distance();
            if ins_dist > existing_dist {
                std::mem::swap(&mut *slot, &mut ins);
            }

            ins.set_distance(ins.distance() + 1);
            pos = (pos + 1) & mask;
        }
    }

    /// Insert during resize — recomputes home position, skips duplicate check.
    #[inline]
    fn insert_rehash(&mut self, mut slot_in: Slot) {
        let mask = self.mask;
        let slots_ptr = self.slots.as_mut_ptr();
        let hash = tile_hash(slot_in.key_x, slot_in.key_y);
        let mut pos = hash as usize & mask;
        slot_in.set_distance(0);
        let mut ins = slot_in;

        loop {
            // SAFETY: pos is always & mask which is < slots.len().
            let slot = unsafe { &mut *slots_ptr.add(pos) };

            if slot.is_empty() {
                *slot = ins;
                self.len += 1;
                return;
            }

            let ins_dist = ins.distance();
            let existing_dist = slot.distance();
            if ins_dist > existing_dist {
                std::mem::swap(&mut *slot, &mut ins);
            }

            ins.set_distance(ins.distance() + 1);
            pos = (pos + 1) & mask;
        }
    }

    // ── Remove ──────────────────────────────────────────────────────────

    /// Remove a mapping.  Returns the value if the key was present.
    #[inline]
    #[allow(dead_code)]
    pub fn remove(&mut self, x: i64, y: i64) -> Option<TileIdx> {
        let hash = tile_hash(x, y);
        self.remove_hashed(x, y, hash)
    }

    #[inline]
    pub(crate) fn remove_hashed(&mut self, x: i64, y: i64, hash: u64) -> Option<TileIdx> {
        debug_assert_eq!(hash, tile_hash(x, y));
        let target_ctrl = match_ctrl_of(hash);
        let mask = self.mask;
        let mut pos = hash as usize & mask;
        let mut our_dist = 0usize;

        loop {
            // SAFETY: pos is always & mask which is < slots.len().
            let slot = unsafe { *self.slots.get_unchecked(pos) };
            if slot.is_empty() {
                return None;
            }
            // Robin Hood early exit on miss.
            let their_dist = slot.distance();
            if our_dist > their_dist {
                return None;
            }
            if slot.match_ctrl() == target_ctrl && slot.key_x == x && slot.key_y == y {
                let old = TileIdx(slot.value);
                self.backward_shift_delete(pos);
                self.len -= 1;
                return Some(old);
            }
            pos = (pos + 1) & mask;
            our_dist += 1;
        }
    }

    /// Backward-shift deletion using stored probe distances.
    fn backward_shift_delete(&mut self, removed: usize) {
        let mask = self.mask;
        let slots_ptr = self.slots.as_mut_ptr();
        let mut gap = removed;
        loop {
            let next = (gap + 1) & mask;
            // SAFETY: next is always & mask which is < slots.len().
            let mut candidate = unsafe { *slots_ptr.add(next) };

            if candidate.is_empty() {
                unsafe {
                    *slots_ptr.add(gap) = Slot::EMPTY;
                }
                return;
            }

            // Distance 0 means the candidate is at home and cannot be shifted.
            let candidate_dist = candidate.distance();
            if candidate_dist == 0 {
                // Candidate is at its home position — doesn't need to shift.
                unsafe {
                    *slots_ptr.add(gap) = Slot::EMPTY;
                }
                return;
            }

            // Shift it back.
            candidate.set_distance(candidate_dist - 1);
            unsafe {
                *slots_ptr.add(gap) = candidate;
            }
            gap = next;
        }
    }

    // ── Resize / grow ───────────────────────────────────────────────────

    #[inline(always)]
    fn needs_grow(&self) -> bool {
        self.len * LOAD_DEN >= self.slots.len() * LOAD_NUM
    }

    fn grow(&mut self) {
        let new_cap = (self.slots.len() * 2).max(16);
        self.resize(new_cap);
    }

    /// Resize and reinsert entries.
    fn resize(&mut self, new_cap: usize) {
        debug_assert!(new_cap.is_power_of_two());
        let old_slots = std::mem::replace(&mut self.slots, vec![Slot::EMPTY; new_cap]);
        self.mask = new_cap - 1;
        self.len = 0;
        for slot in old_slots {
            if slot.is_occupied() {
                self.insert_rehash(slot);
            }
        }
    }

    // ── Iteration ───────────────────────────────────────────────────────

    /// Iterate over all `(coord, TileIdx)` pairs.
    #[inline]
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = ((i64, i64), TileIdx)> + '_ {
        self.slots.iter().filter_map(|slot| {
            if slot.is_occupied() {
                Some(((slot.key_x, slot.key_y), TileIdx(slot.value)))
            } else {
                None
            }
        })
    }
}

// ── Free helpers ────────────────────────────────────────────────────────

/// Extract fingerprint bits from a hash.
#[inline(always)]
fn fingerprint_of(hash: u64) -> u32 {
    // Use high bits (bucket selection uses low bits).
    // Keep at least one bit set so EMPTY remains ctrl == 0.
    ((hash >> 52) as u32 & FP_MASK) | 1
}

#[inline(always)]
fn occupied_ctrl(fp: u32, distance: usize) -> u32 {
    OCCUPIED_BIT | ((distance as u32) << DIST_SHIFT) | fp
}

#[inline(always)]
fn match_ctrl_of(hash: u64) -> u32 {
    OCCUPIED_BIT | fingerprint_of(hash)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut m = TileMap::with_capacity(64);
        assert!(m.get(10, 20).is_none());

        m.insert(10, 20, TileIdx(42));
        assert_eq!(m.get(10, 20), Some(TileIdx(42)));
        assert_eq!(m.len(), 1);

        // Overwrite.
        let old = m.insert(10, 20, TileIdx(99));
        assert_eq!(old, Some(TileIdx(42)));
        assert_eq!(m.get(10, 20), Some(TileIdx(99)));
        assert_eq!(m.len(), 1);

        // Remove.
        let removed = m.remove(10, 20);
        assert_eq!(removed, Some(TileIdx(99)));
        assert!(m.get(10, 20).is_none());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn negative_coords() {
        let mut m = TileMap::with_capacity(16);
        m.insert(-5, -10, TileIdx(1));
        m.insert(0, 0, TileIdx(2));
        m.insert(i64::MIN, i64::MAX, TileIdx(3));

        assert_eq!(m.get(-5, -10), Some(TileIdx(1)));
        assert_eq!(m.get(0, 0), Some(TileIdx(2)));
        assert_eq!(m.get(i64::MIN, i64::MAX), Some(TileIdx(3)));
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn grow_under_pressure() {
        let mut m = TileMap::with_capacity(4);
        for i in 0..1000i64 {
            m.insert(i, i * 3, TileIdx(i as u32));
        }
        assert_eq!(m.len(), 1000);
        for i in 0..1000i64 {
            assert_eq!(m.get(i, i * 3), Some(TileIdx(i as u32)));
        }
    }

    #[test]
    fn remove_does_not_break_chains() {
        let mut m = TileMap::with_capacity(64);
        // Insert entries that are likely to form a probe chain.
        for i in 0..20i64 {
            m.insert(i, 0, TileIdx(i as u32));
        }
        // Remove every other entry.
        for i in (0..20i64).step_by(2) {
            m.remove(i, 0);
        }
        // Remaining entries must still be reachable.
        for i in (1..20i64).step_by(2) {
            assert_eq!(m.get(i, 0), Some(TileIdx(i as u32)));
        }
        assert_eq!(m.len(), 10);
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut m = TileMap::with_capacity(32);
        for i in 0..50i64 {
            m.insert(i, -i, TileIdx(i as u32));
        }
        let mut collected: Vec<_> = m.iter().collect();
        collected.sort_by_key(|&(_, idx)| idx.0);
        assert_eq!(collected.len(), 50);
        for (i, &((x, y), idx)) in collected.iter().enumerate() {
            assert_eq!(x, i as i64);
            assert_eq!(y, -(i as i64));
            assert_eq!(idx.0, i as u32);
        }
    }

    #[test]
    fn slot_size_is_24_bytes() {
        assert_eq!(std::mem::size_of::<Slot>(), 24);
    }

    #[test]
    fn hash_mix_avoids_simple_axis_cancellation_pattern() {
        let mut hashes = std::collections::BTreeSet::new();
        for k in 0..64i64 {
            hashes.insert(tile_hash(-k, k << 32));
        }
        assert_eq!(
            hashes.len(),
            64,
            "tile_hash should not collapse axis-canceling coordinates"
        );
    }

    #[test]
    fn hash_mix_spreads_axis_aligned_coordinates() {
        let mut x_buckets = std::collections::BTreeSet::new();
        let mut y_buckets = std::collections::BTreeSet::new();
        let bucket_mask = (1u64 << 16) - 1;
        for i in 0..(1 << 16) {
            let i = i as i64;
            x_buckets.insert(tile_hash(i, 0) & bucket_mask);
            y_buckets.insert(tile_hash(0, i) & bucket_mask);
        }

        assert!(
            x_buckets.len() >= 40_000,
            "x-axis bucket spread regressed: {}",
            x_buckets.len()
        );
        assert!(
            y_buckets.len() >= 40_000,
            "y-axis bucket spread regressed: {}",
            y_buckets.len()
        );
    }

    #[test]
    fn control_word_keeps_large_probe_distance_headroom() {
        let mut slot = Slot {
            key_x: 0,
            key_y: 0,
            value: 0,
            ctrl: occupied_ctrl(1, 0),
        };
        slot.set_distance(1 << 18);
        assert_eq!(slot.distance(), 1 << 18);
    }

    #[test]
    #[should_panic(expected = "TileMap probe distance overflow")]
    fn slot_distance_overflow_panics() {
        let mut slot = Slot {
            key_x: 0,
            key_y: 0,
            value: 0,
            ctrl: occupied_ctrl(1, 0),
        };
        slot.set_distance(MAX_DIST + 1);
    }
}
