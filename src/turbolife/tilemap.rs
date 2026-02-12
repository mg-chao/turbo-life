//! `TileMap` — a turbolife-specialized open-addressing hashmap for
//! `(i64, i64) → TileIdx` lookups.
//!
//! Design goals:
//! - Flat, cache-friendly layout: 32-byte slots, two per cache line.
//! - Robin Hood probing with backward-shift deletion (no tombstones).
//! - Specialized hash for tile coordinates (distinct multipliers + rotation).
//! - Fingerprint check in the control word to skip full key comparisons.

use super::tile::TileIdx;

// ── Hash function ───────────────────────────────────────────────────────

/// Two distinct Fibonacci-derived constants for mixing x and y independently.
/// This avoids the systematic collisions that a single-constant sequential
/// hash (like FxHash) produces on grid-aligned coordinate patterns.
const MX: u64 = 0x517c_c1b7_2722_0a95;
const MY: u64 = 0x6c62_272e_07bb_0142;

#[inline(always)]
fn tile_hash(x: i64, y: i64) -> u64 {
    let hx = (x as u64).wrapping_mul(MX);
    let hy = (y as u64).wrapping_mul(MY);
    hx ^ hy.rotate_right(32)
}

// ── Slot layout ─────────────────────────────────────────────────────────

const EMPTY: u32 = 0;
const OCCUPIED_BIT: u32 = 0x8000_0000;

/// A single slot in the hash table.
///
/// 32 bytes, naturally aligned so two slots fill one 64-byte cache line.
#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    key_x: i64,
    key_y: i64,
    value: u32,
    /// High bit = occupied flag.  Lower 31 bits = hash fingerprint.
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
    fn fingerprint(self) -> u32 {
        self.ctrl & !OCCUPIED_BIT
    }
}

// ── TileMap ─────────────────────────────────────────────────────────────

/// Maximum load factor numerator / denominator: 3/4 = 75%.
const LOAD_NUM: usize = 3;
const LOAD_DEN: usize = 4;

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
    #[inline(always)]
    pub fn get(&self, x: i64, y: i64) -> Option<TileIdx> {
        let hash = tile_hash(x, y);
        let fp = fingerprint_of(hash);
        let mut pos = hash as usize & self.mask;

        // SAFETY: `pos` is always `& self.mask` which is < self.slots.len()
        // (capacity is a power of two, mask = capacity - 1).
        loop {
            let slot = unsafe { *self.slots.get_unchecked(pos) };
            if slot.is_empty() {
                return None;
            }
            if slot.is_occupied()
                && slot.fingerprint() == fp
                && slot.key_x == x
                && slot.key_y == y
            {
                return Some(TileIdx(slot.value));
            }
            pos = (pos + 1) & self.mask;
        }
    }

    // ── Insert ──────────────────────────────────────────────────────────

    /// Insert a mapping.  Returns the previous value if the key was present.
    #[inline]
    pub fn insert(&mut self, x: i64, y: i64, value: TileIdx) -> Option<TileIdx> {
        if self.needs_grow() {
            self.grow();
        }
        self.insert_no_grow(x, y, value)
    }

    /// Insert without checking capacity — caller must ensure space.
    #[inline]
    fn insert_no_grow(&mut self, x: i64, y: i64, value: TileIdx) -> Option<TileIdx> {
        let hash = tile_hash(x, y);
        let fp = fingerprint_of(hash);
        let ctrl = OCCUPIED_BIT | fp;
        let mut pos = hash as usize & self.mask;

        let mut ins = Slot {
            key_x: x,
            key_y: y,
            value: value.0,
            ctrl,
        };
        let mut ins_home = pos;

        loop {
            let slot = &mut self.slots[pos];

            if slot.is_empty() {
                *slot = ins;
                self.len += 1;
                return None;
            }

            // Exact match — update in place.
            if slot.is_occupied()
                && slot.fingerprint() == fp
                && slot.key_x == x
                && slot.key_y == y
            {
                let old = TileIdx(slot.value);
                slot.value = value.0;
                return Some(old);
            }

            // Robin Hood: if the existing entry is closer to home, steal its spot.
            let existing_home =
                tile_hash(slot.key_x, slot.key_y) as usize & self.mask;
            let existing_dist = displacement(existing_home, pos, self.mask);
            let ins_dist = displacement(ins_home, pos, self.mask);

            if ins_dist > existing_dist {
                let displaced = *slot;
                *slot = ins;
                ins = displaced;
                ins_home = existing_home;
            }

            pos = (pos + 1) & self.mask;
        }
    }

    // ── Remove ──────────────────────────────────────────────────────────

    /// Remove a mapping.  Returns the value if the key was present.
    #[inline]
    pub fn remove(&mut self, x: i64, y: i64) -> Option<TileIdx> {
        let hash = tile_hash(x, y);
        let fp = fingerprint_of(hash);
        let mut pos = hash as usize & self.mask;

        loop {
            let slot = self.slots[pos];
            if slot.is_empty() {
                return None;
            }
            if slot.is_occupied()
                && slot.fingerprint() == fp
                && slot.key_x == x
                && slot.key_y == y
            {
                let old = TileIdx(slot.value);
                self.backward_shift_delete(pos);
                self.len -= 1;
                return Some(old);
            }
            pos = (pos + 1) & self.mask;
        }
    }

    /// Backward-shift deletion: pull subsequent displaced entries back to
    /// fill the gap, keeping probe chains intact without tombstones.
    fn backward_shift_delete(&mut self, removed: usize) {
        let mut gap = removed;
        loop {
            let next = (gap + 1) & self.mask;
            let candidate = self.slots[next];

            if candidate.is_empty() {
                self.slots[gap] = Slot::EMPTY;
                return;
            }

            // If the candidate is at its home position it doesn't need to shift.
            let home = tile_hash(candidate.key_x, candidate.key_y) as usize & self.mask;
            if home == next {
                self.slots[gap] = Slot::EMPTY;
                return;
            }

            // Shift it back.
            self.slots[gap] = candidate;
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

    fn resize(&mut self, new_cap: usize) {
        debug_assert!(new_cap.is_power_of_two());
        let old_slots = std::mem::replace(&mut self.slots, vec![Slot::EMPTY; new_cap]);
        self.mask = new_cap - 1;
        self.len = 0;
        for slot in old_slots {
            if slot.is_occupied() {
                self.insert_no_grow(slot.key_x, slot.key_y, TileIdx(slot.value));
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

/// Extract the 31-bit fingerprint from a hash.
#[inline(always)]
fn fingerprint_of(hash: u64) -> u32 {
    // Use the upper 32 bits (the lower bits select the bucket).
    // Ensure at least one bit is set so fingerprint != 0 (which would
    // collide with EMPTY's ctrl == 0 after masking).
    let raw = (hash >> 32) as u32 & !OCCUPIED_BIT;
    if raw == 0 { 1 } else { raw }
}

/// Displacement (distance from home) in a power-of-two table.
#[inline(always)]
fn displacement(home: usize, pos: usize, mask: usize) -> usize {
    pos.wrapping_sub(home) & mask
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
}
