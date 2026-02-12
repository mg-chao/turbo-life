//! Scalar bit-parallel kernel for TurboLife.
//!
//! Computes the next generation for a single tile using a full-adder chain.
//! Border extraction is fused into the main loop.
//! Works with split cell buffers (separate current/next vecs).

use super::tile::{BorderData, CellBuf, GhostZone, POPULATION_UNKNOWN, TILE_SIZE, TileMeta};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBackend {
    Scalar,
    Avx2,
}

#[inline(always)]
fn full_add(a: u64, b: u64, c: u64) -> (u64, u64) {
    let sum = a ^ b ^ c;
    let carry = (a & b) | (b & c) | (a & c);
    (sum, carry)
}

#[inline(always)]
fn half_add(a: u64, b: u64) -> (u64, u64) {
    (a ^ b, a & b)
}

#[inline(always)]
fn west_neighbor_plane(word: u64, ghost_w: bool) -> u64 {
    (word << 1) | ghost_w as u64
}

#[inline(always)]
fn east_neighbor_plane(word: u64, ghost_e: bool) -> u64 {
    (word >> 1) | ((ghost_e as u64) << 63)
}

#[inline(always)]
fn ghost_bit(column: u64, row: usize) -> bool {
    ((column >> row) & 1) != 0
}

/// Advance one tile generation and extract its new border data.
/// Returns (changed, border_data, has_live).
#[inline(always)]
pub fn advance_core_scalar(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    let mut changed = false;
    let mut has_live = false;
    let mut border_west = 0u64;
    let mut border_east = 0u64;

    for row in 0..TILE_SIZE {
        let row_above = if row == TILE_SIZE - 1 {
            ghost.north
        } else {
            current[row + 1]
        };
        let row_self = current[row];
        let row_below = if row == 0 {
            ghost.south
        } else {
            current[row - 1]
        };

        let ghost_w_above = if row == TILE_SIZE - 1 {
            ghost.nw
        } else {
            ghost_bit(ghost.west, row + 1)
        };
        let ghost_e_above = if row == TILE_SIZE - 1 {
            ghost.ne
        } else {
            ghost_bit(ghost.east, row + 1)
        };
        let ghost_w_self = ghost_bit(ghost.west, row);
        let ghost_e_self = ghost_bit(ghost.east, row);
        let ghost_w_below = if row == 0 {
            ghost.sw
        } else {
            ghost_bit(ghost.west, row - 1)
        };
        let ghost_e_below = if row == 0 {
            ghost.se
        } else {
            ghost_bit(ghost.east, row - 1)
        };

        let nw = west_neighbor_plane(row_above, ghost_w_above);
        let n = row_above;
        let ne = east_neighbor_plane(row_above, ghost_e_above);
        let w = west_neighbor_plane(row_self, ghost_w_self);
        let e = east_neighbor_plane(row_self, ghost_e_self);
        let sw = west_neighbor_plane(row_below, ghost_w_below);
        let s = row_below;
        let se = east_neighbor_plane(row_below, ghost_e_below);

        let (a0, a1) = full_add(nw, n, ne);
        let (s0, s1) = half_add(w, e);
        let (b0, b1) = full_add(sw, s, se);
        let (t0, t0c) = full_add(a0, s0, b0);
        let (u0, u0c) = full_add(a1, s1, b1);
        let (t1, t1c) = half_add(u0, t0c);
        let (t2, _) = half_add(u0c, t1c);

        let alive_mask = !t2 & t1;
        let next_row = (alive_mask & t0) | (alive_mask & !t0 & row_self);

        next[row] = next_row;
        changed |= next_row != row_self;
        has_live |= next_row != 0;
        border_west |= (next_row & 1) << row;
        border_east |= ((next_row >> 63) & 1) << row;
    }

    let mut corners = 0u8;
    if (next[63] & 1) != 0 {
        corners |= BorderData::CORNER_NW;
    }
    if ((next[63] >> 63) & 1) != 0 {
        corners |= BorderData::CORNER_NE;
    }
    if (next[0] & 1) != 0 {
        corners |= BorderData::CORNER_SW;
    }
    if ((next[0] >> 63) & 1) != 0 {
        corners |= BorderData::CORNER_SE;
    }

    let border = BorderData {
        north: next[63],
        south: next[0],
        west: border_west,
        east: border_east,
        corners,
    };

    (changed, border, has_live)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_set_u64x4(words: [u64; 4]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::_mm256_set_epi64x;

    unsafe {
        _mm256_set_epi64x(
            words[3] as i64,
            words[2] as i64,
            words[1] as i64,
            words[0] as i64,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_store_u64x4(value: std::arch::x86_64::__m256i) -> [u64; 4] {
    use std::arch::x86_64::_mm256_storeu_si256;

    let mut out = [0u64; 4];
    unsafe {
        _mm256_storeu_si256(out.as_mut_ptr() as *mut std::arch::x86_64::__m256i, value);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_full_add(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
    c: std::arch::x86_64::__m256i,
) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::{_mm256_and_si256, _mm256_or_si256, _mm256_xor_si256};

    unsafe {
        let sum = _mm256_xor_si256(_mm256_xor_si256(a, b), c);
        let carry = _mm256_or_si256(
            _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(b, c)),
            _mm256_and_si256(a, c),
        );
        (sum, carry)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_half_add(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::{_mm256_and_si256, _mm256_xor_si256};

    unsafe { (_mm256_xor_si256(a, b), _mm256_and_si256(a, b)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn west_neighbor_plane_avx2(
    word: std::arch::x86_64::__m256i,
    ghost_w: [bool; 4],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::{_mm256_or_si256, _mm256_slli_epi64};

    unsafe {
        let shifted = _mm256_slli_epi64(word, 1);
        let carry = avx2_set_u64x4([
            ghost_w[0] as u64,
            ghost_w[1] as u64,
            ghost_w[2] as u64,
            ghost_w[3] as u64,
        ]);
        _mm256_or_si256(shifted, carry)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn east_neighbor_plane_avx2(
    word: std::arch::x86_64::__m256i,
    ghost_e: [bool; 4],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::{_mm256_or_si256, _mm256_srli_epi64};

    unsafe {
        let shifted = _mm256_srli_epi64(word, 1);
        let carry = avx2_set_u64x4([
            (ghost_e[0] as u64) << 63,
            (ghost_e[1] as u64) << 63,
            (ghost_e[2] as u64) << 63,
            (ghost_e[3] as u64) << 63,
        ]);
        _mm256_or_si256(shifted, carry)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn advance_core_avx2(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    use std::arch::x86_64::{_mm256_and_si256, _mm256_andnot_si256, _mm256_or_si256};

    let mut changed = false;
    let mut has_live = false;
    let mut border_west = 0u64;
    let mut border_east = 0u64;

    for row_base in (0..TILE_SIZE).step_by(4) {
        let mut row_above_words = [0u64; 4];
        let mut row_self_words = [0u64; 4];
        let mut row_below_words = [0u64; 4];

        let mut ghost_w_above = [false; 4];
        let mut ghost_e_above = [false; 4];
        let mut ghost_w_self = [false; 4];
        let mut ghost_e_self = [false; 4];
        let mut ghost_w_below = [false; 4];
        let mut ghost_e_below = [false; 4];

        for lane in 0..4 {
            let row = row_base + lane;
            row_above_words[lane] = if row == TILE_SIZE - 1 {
                ghost.north
            } else {
                current[row + 1]
            };
            row_self_words[lane] = current[row];
            row_below_words[lane] = if row == 0 {
                ghost.south
            } else {
                current[row - 1]
            };

            ghost_w_above[lane] = if row == TILE_SIZE - 1 {
                ghost.nw
            } else {
                ghost_bit(ghost.west, row + 1)
            };
            ghost_e_above[lane] = if row == TILE_SIZE - 1 {
                ghost.ne
            } else {
                ghost_bit(ghost.east, row + 1)
            };
            ghost_w_self[lane] = ghost_bit(ghost.west, row);
            ghost_e_self[lane] = ghost_bit(ghost.east, row);
            ghost_w_below[lane] = if row == 0 {
                ghost.sw
            } else {
                ghost_bit(ghost.west, row - 1)
            };
            ghost_e_below[lane] = if row == 0 {
                ghost.se
            } else {
                ghost_bit(ghost.east, row - 1)
            };
        }

        let row_above = unsafe { avx2_set_u64x4(row_above_words) };
        let row_self = unsafe { avx2_set_u64x4(row_self_words) };
        let row_below = unsafe { avx2_set_u64x4(row_below_words) };

        let nw = unsafe { west_neighbor_plane_avx2(row_above, ghost_w_above) };
        let n = row_above;
        let ne = unsafe { east_neighbor_plane_avx2(row_above, ghost_e_above) };
        let w = unsafe { west_neighbor_plane_avx2(row_self, ghost_w_self) };
        let e = unsafe { east_neighbor_plane_avx2(row_self, ghost_e_self) };
        let sw = unsafe { west_neighbor_plane_avx2(row_below, ghost_w_below) };
        let s = row_below;
        let se = unsafe { east_neighbor_plane_avx2(row_below, ghost_e_below) };

        let (a0, a1) = unsafe { avx2_full_add(nw, n, ne) };
        let (s0, s1) = unsafe { avx2_half_add(w, e) };
        let (b0, b1) = unsafe { avx2_full_add(sw, s, se) };
        let (t0, t0c) = unsafe { avx2_full_add(a0, s0, b0) };
        let (u0, u0c) = unsafe { avx2_full_add(a1, s1, b1) };
        let (t1, t1c) = unsafe { avx2_half_add(u0, t0c) };
        let (t2, _) = unsafe { avx2_half_add(u0c, t1c) };

        let alive_mask = _mm256_andnot_si256(t2, t1);
        let born = _mm256_and_si256(alive_mask, t0);
        let survive = _mm256_and_si256(_mm256_andnot_si256(t0, alive_mask), row_self);
        let next_rows = _mm256_or_si256(born, survive);

        let next_words = unsafe { avx2_store_u64x4(next_rows) };

        for lane in 0..4 {
            let row = row_base + lane;
            let next_row = next_words[lane];
            next[row] = next_row;
            changed |= next_row != row_self_words[lane];
            has_live |= next_row != 0;
            border_west |= (next_row & 1) << row;
            border_east |= ((next_row >> 63) & 1) << row;
        }
    }

    let mut corners = 0u8;
    if (next[63] & 1) != 0 {
        corners |= BorderData::CORNER_NW;
    }
    if ((next[63] >> 63) & 1) != 0 {
        corners |= BorderData::CORNER_NE;
    }
    if (next[0] & 1) != 0 {
        corners |= BorderData::CORNER_SW;
    }
    if ((next[0] >> 63) & 1) != 0 {
        corners |= BorderData::CORNER_SE;
    }

    let border = BorderData {
        north: next[63],
        south: next[0],
        west: border_west,
        east: border_east,
        corners,
    };

    (changed, border, has_live)
}

#[inline(always)]
pub fn advance_core(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
    backend: KernelBackend,
) -> (bool, BorderData, bool) {
    match backend {
        KernelBackend::Scalar => advance_core_scalar(current, next, ghost),
        KernelBackend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                unsafe { advance_core_avx2(current, next, ghost) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                advance_core_scalar(current, next, ghost)
            }
        }
    }
}

/// Advance a tile using split buffer pointers (unsafe parallel path).
///
/// # Safety
/// `current_ptr`, `next_ptr`, `meta_ptr`, and `next_borders_ptr` must point
/// to valid slices, and the caller must ensure exclusive write access to
/// the element at `idx` in the write-side arrays.
#[inline(always)]
pub unsafe fn advance_tile_split(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_ptr: *mut BorderData,
    idx: usize,
    ghost: &GhostZone,
    backend: KernelBackend,
) -> bool {
    debug_assert!(!current_ptr.is_null());
    debug_assert!(!next_ptr.is_null());
    debug_assert!(!meta_ptr.is_null());
    debug_assert!(!next_borders_ptr.is_null());

    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta = unsafe { &mut *meta_ptr.add(idx) };

    let (changed, border, has_live) = advance_core(current, next, ghost, backend);

    unsafe {
        *next_borders_ptr.add(idx) = border;
    }

    meta.set_changed(changed);
    if changed {
        meta.population = POPULATION_UNKNOWN;
        meta.set_has_live(has_live);
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::{GhostZone, TILE_SIZE, advance_core_scalar, ghost_bit};

    #[cfg(target_arch = "x86_64")]
    use super::advance_core_avx2;

    use rand::RngCore;
    use rand::SeedableRng;

    fn random_tile(rng: &mut rand::rngs::StdRng) -> [u64; TILE_SIZE] {
        let mut tile = [0u64; TILE_SIZE];
        for row in &mut tile {
            *row = rng.next_u64();
        }
        tile
    }

    fn random_ghost(rng: &mut rand::rngs::StdRng) -> GhostZone {
        GhostZone {
            north: rng.next_u64(),
            south: rng.next_u64(),
            west: rng.next_u64(),
            east: rng.next_u64(),
            nw: (rng.next_u64() & 1) != 0,
            ne: (rng.next_u64() & 1) != 0,
            sw: (rng.next_u64() & 1) != 0,
            se: (rng.next_u64() & 1) != 0,
        }
    }

    #[test]
    fn ghost_bit_extracts_expected_values() {
        let column = 0b1010u64;
        assert!(!ghost_bit(column, 0));
        assert!(ghost_bit(column, 1));
        assert!(!ghost_bit(column, 2));
        assert!(ghost_bit(column, 3));
    }

    #[test]
    fn scalar_core_handles_empty_tile() {
        let current = [0u64; TILE_SIZE];
        let mut next = [0u64; TILE_SIZE];
        let ghost = GhostZone::default();
        let (changed, border, has_live) = advance_core_scalar(&current, &mut next, &ghost);
        assert!(!changed);
        assert!(!has_live);
        assert_eq!(border.north, 0);
        assert_eq!(border.south, 0);
        assert_eq!(border.west, 0);
        assert_eq!(border.east, 0);
        assert_eq!(border.corners, 0);
    }

    #[test]
    fn avx2_matches_scalar_randomized() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                return;
            }

            let mut rng = rand::rngs::StdRng::seed_from_u64(0xA55A_F00D_1122_3344);
            for _ in 0..2048 {
                let current = random_tile(&mut rng);
                let ghost = random_ghost(&mut rng);
                let mut next_scalar = [0u64; TILE_SIZE];
                let mut next_avx = [0u64; TILE_SIZE];

                let scalar = advance_core_scalar(&current, &mut next_scalar, &ghost);
                let avx = unsafe { advance_core_avx2(&current, &mut next_avx, &ghost) };

                assert_eq!(next_scalar, next_avx);
                assert_eq!(scalar.0, avx.0);
                assert_eq!(scalar.2, avx.2);
                assert_eq!(scalar.1.north, avx.1.north);
                assert_eq!(scalar.1.south, avx.1.south);
                assert_eq!(scalar.1.west, avx.1.west);
                assert_eq!(scalar.1.east, avx.1.east);
                assert_eq!(scalar.1.corners, avx.1.corners);
            }
        }
    }
}
