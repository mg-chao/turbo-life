//! Reusable coordinate deduper for prune/expand frontier generation.
//!
//! This is an open-addressed linear-probing hash set for `(i64, i64)` keys.
//! Slots are lazily cleared with an epoch stamp, so each simulation step can
//! start a fresh set without touching the full backing array.

const LOAD_NUM: usize = 3;
const LOAD_DEN: usize = 4;

#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    x: i64,
    y: i64,
    stamp: u32,
}

impl Slot {
    const EMPTY: Self = Self {
        x: 0,
        y: 0,
        stamp: 0,
    };
}

#[inline(always)]
fn coord_hash(x: i64, y: i64) -> u64 {
    // Reuse the lightweight coordinate mix used by TileMap for speed.
    const MX: u64 = 0x517c_c1b7_2722_0a95;
    const MY: u64 = 0x6c62_272e_07bb_0142;
    let hx = (x as u64).wrapping_mul(MX);
    let hy = (y as u64).wrapping_mul(MY);
    hx ^ hy.rotate_right(32)
}

pub struct CoordSet {
    slots: Vec<Slot>,
    mask: usize,
    stamp: u32,
    len: usize,
}

impl CoordSet {
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let slots = cap
            .saturating_mul(LOAD_DEN)
            .div_ceil(LOAD_NUM)
            .next_power_of_two()
            .max(16);
        Self {
            slots: vec![Slot::EMPTY; slots],
            mask: slots - 1,
            stamp: 1,
            len: 0,
        }
    }

    #[inline]
    pub fn begin_step(&mut self) {
        self.len = 0;
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.stamp = 1;
            for slot in &mut self.slots {
                slot.stamp = 0;
            }
        }
    }

    #[inline]
    pub fn reserve_for(&mut self, keys: usize) {
        if keys == 0 {
            return;
        }
        let needed = keys
            .saturating_mul(LOAD_DEN)
            .div_ceil(LOAD_NUM)
            .next_power_of_two()
            .max(16);
        if needed > self.slots.len() {
            self.resize(needed);
        }
    }

    #[inline(always)]
    fn needs_grow(&self) -> bool {
        self.len * LOAD_DEN >= self.slots.len() * LOAD_NUM
    }

    #[inline]
    fn resize(&mut self, new_slots: usize) {
        debug_assert!(new_slots.is_power_of_two());
        let old_slots = std::mem::replace(&mut self.slots, vec![Slot::EMPTY; new_slots]);
        self.mask = new_slots - 1;
        self.len = 0;

        for slot in old_slots {
            if slot.stamp == self.stamp {
                self.insert_rehash(slot.x, slot.y);
            }
        }
    }

    #[inline(always)]
    fn insert_rehash(&mut self, x: i64, y: i64) {
        let mask = self.mask;
        let mut pos = coord_hash(x, y) as usize & mask;

        loop {
            let slot = unsafe { self.slots.get_unchecked_mut(pos) };
            if slot.stamp != self.stamp {
                *slot = Slot {
                    x,
                    y,
                    stamp: self.stamp,
                };
                self.len += 1;
                return;
            }
            pos = (pos + 1) & mask;
        }
    }

    /// Insert a coordinate.
    /// Returns `true` if newly inserted, `false` if it already existed.
    #[inline]
    pub fn insert(&mut self, x: i64, y: i64) -> bool {
        if self.needs_grow() {
            self.resize((self.slots.len() * 2).max(16));
        }

        let mask = self.mask;
        let mut pos = coord_hash(x, y) as usize & mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked_mut(pos) };
            if slot.stamp != self.stamp {
                *slot = Slot {
                    x,
                    y,
                    stamp: self.stamp,
                };
                self.len += 1;
                return true;
            }
            if slot.x == x && slot.y == y {
                return false;
            }
            pos = (pos + 1) & mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CoordSet;

    #[test]
    fn dedups_within_step_and_resets_across_steps() {
        let mut set = CoordSet::new();
        set.begin_step();
        assert!(set.insert(1, 2));
        assert!(!set.insert(1, 2));
        assert!(set.insert(-5, 9));

        set.begin_step();
        assert!(set.insert(1, 2));
        assert!(!set.insert(1, 2));
    }

    #[test]
    fn reserve_and_insert_many() {
        let mut set = CoordSet::with_capacity(8);
        set.begin_step();
        set.reserve_for(10_000);
        for i in 0..10_000i64 {
            assert!(set.insert(i, -i));
        }
        for i in 0..10_000i64 {
            assert!(!set.insert(i, -i));
        }
    }
}
