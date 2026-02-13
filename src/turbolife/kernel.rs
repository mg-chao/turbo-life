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
    let ghost_w = ghost.west;
    let ghost_e = ghost.east;

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
            ghost_bit(ghost_w, row + 1)
        };
        let ghost_e_above = if row == TILE_SIZE - 1 {
            ghost.ne
        } else {
            ghost_bit(ghost_e, row + 1)
        };
        let ghost_w_self = ghost_bit(ghost_w, row);
        let ghost_e_self = ghost_bit(ghost_e, row);
        let ghost_w_below = if row == 0 {
            ghost.sw
        } else {
            ghost_bit(ghost_w, row - 1)
        };
        let ghost_e_below = if row == 0 {
            ghost.se
        } else {
            ghost_bit(ghost_e, row - 1)
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

// ── AVX2 kernel ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_set_u64x4_lane_order(
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::_mm256_set_epi64x;
    unsafe { _mm256_set_epi64x(a3 as i64, a2 as i64, a1 as i64, a0 as i64) }
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

/// Build carry masks for 4 lanes from 4 ghost bits, placing each bit at `shift`.
#[cfg(target_arch = "x86_64")]
const fn avx2_build_carry_table(shift: u32) -> [[u64; 4]; 16] {
    let mut table = [[0u64; 4]; 16];
    let mut bits = 0usize;
    while bits < 16 {
        let b = bits as u64;
        table[bits] = [
            (b & 1) << shift,
            ((b >> 1) & 1) << shift,
            ((b >> 2) & 1) << shift,
            ((b >> 3) & 1) << shift,
        ];
        bits += 1;
    }
    table
}

#[cfg(target_arch = "x86_64")]
const AVX2_CARRY_LO_TABLE: [[u64; 4]; 16] = avx2_build_carry_table(0);

#[cfg(target_arch = "x86_64")]
const AVX2_CARRY_HI_TABLE: [[u64; 4]; 16] = avx2_build_carry_table(63);

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_carry_mask_lo(bits4: u64) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256};
    let row = unsafe { AVX2_CARRY_LO_TABLE.get_unchecked(bits4 as usize) };
    unsafe { _mm256_loadu_si256(row.as_ptr() as *const __m256i) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_carry_mask_hi(bits4: u64) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256};
    let row = unsafe { AVX2_CARRY_HI_TABLE.get_unchecked(bits4 as usize) };
    unsafe { _mm256_loadu_si256(row.as_ptr() as *const __m256i) }
}

/// AVX2 kernel: processes 4 rows at a time using 256-bit SIMD.
///
/// Optimizations over scalar:
/// - 4x throughput on the full-adder chain (4 rows per iteration)
/// - Vectorized changed/has_live detection via `_mm256_testz_si256`
/// - Lookup-table-based ghost bit injection (no per-lane branching)
/// - Direct store to output buffer via `_mm256_storeu_si256`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn advance_core_avx2(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_loadu_si256, _mm256_or_si256,
        _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };

    let mut changed = false;
    let mut has_live = false;
    let mut border_west = 0u64;
    let mut border_east = 0u64;
    let current_ptr = current.as_ptr();

    for row_base in (0..TILE_SIZE).step_by(4) {
        let row_self = unsafe { _mm256_loadu_si256(current_ptr.add(row_base) as *const __m256i) };
        let row_above = if row_base < TILE_SIZE - 4 {
            unsafe { _mm256_loadu_si256(current_ptr.add(row_base + 1) as *const __m256i) }
        } else {
            unsafe { avx2_set_u64x4_lane_order(current[61], current[62], current[63], ghost.north) }
        };
        let row_below = if row_base == 0 {
            unsafe { avx2_set_u64x4_lane_order(ghost.south, current[0], current[1], current[2]) }
        } else {
            unsafe { _mm256_loadu_si256(current_ptr.add(row_base - 1) as *const __m256i) }
        };

        let ghost_w_self = (ghost.west >> row_base) & 0xF;
        let ghost_e_self = (ghost.east >> row_base) & 0xF;
        let ghost_w_above = if row_base < TILE_SIZE - 4 {
            (ghost.west >> (row_base + 1)) & 0xF
        } else {
            ((ghost.west >> 61) & 0x7) | ((ghost.nw as u64) << 3)
        };
        let ghost_e_above = if row_base < TILE_SIZE - 4 {
            (ghost.east >> (row_base + 1)) & 0xF
        } else {
            ((ghost.east >> 61) & 0x7) | ((ghost.ne as u64) << 3)
        };
        let ghost_w_below = if row_base == 0 {
            ((ghost.west & 0x7) << 1) | ghost.sw as u64
        } else {
            (ghost.west >> (row_base - 1)) & 0xF
        };
        let ghost_e_below = if row_base == 0 {
            ((ghost.east & 0x7) << 1) | ghost.se as u64
        } else {
            (ghost.east >> (row_base - 1)) & 0xF
        };

        let nw = _mm256_or_si256(_mm256_slli_epi64(row_above, 1), unsafe {
            avx2_carry_mask_lo(ghost_w_above)
        });
        let n = row_above;
        let ne = _mm256_or_si256(_mm256_srli_epi64(row_above, 1), unsafe {
            avx2_carry_mask_hi(ghost_e_above)
        });
        let w = _mm256_or_si256(_mm256_slli_epi64(row_self, 1), unsafe {
            avx2_carry_mask_lo(ghost_w_self)
        });
        let e = _mm256_or_si256(_mm256_srli_epi64(row_self, 1), unsafe {
            avx2_carry_mask_hi(ghost_e_self)
        });
        let sw = _mm256_or_si256(_mm256_slli_epi64(row_below, 1), unsafe {
            avx2_carry_mask_lo(ghost_w_below)
        });
        let s = row_below;
        let se = _mm256_or_si256(_mm256_srli_epi64(row_below, 1), unsafe {
            avx2_carry_mask_hi(ghost_e_below)
        });

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

        let diff = _mm256_xor_si256(next_rows, row_self);
        changed |= _mm256_testz_si256(diff, diff) == 0;
        has_live |= _mm256_testz_si256(next_rows, next_rows) == 0;

        // Store directly to output and extract border bits.
        unsafe {
            _mm256_storeu_si256(next.as_mut_ptr().add(row_base) as *mut __m256i, next_rows);
        }
        for lane in 0..4usize {
            let row = row_base + lane;
            let next_row = next[row];
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

/// Fused ghost-gather + kernel advance using raw pointers.
///
/// Eliminates the intermediate GhostZone struct by inlining the gather
/// directly into the kernel dispatch. This reduces register pressure
/// and avoids constructing/destructuring the struct.
///
/// # Safety
/// All pointers must be valid. `idx` must be within bounds of all arrays.
/// Caller must ensure exclusive write access to `next_ptr[idx]`,
/// `meta_ptr[idx]`, and `next_borders_ptr[idx]`.
#[inline(always)]
pub unsafe fn advance_tile_fused(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    idx: usize,
    backend: KernelBackend,
) -> bool {
    use super::arena::SENTINEL_IDX;
    use super::tile::NO_NEIGHBOR;

    debug_assert_eq!(SENTINEL_IDX, 0);

    // Inline branchless ghost-zone gather (avoids function call + struct construction).
    let nb = unsafe { &*neighbors_ptr.add(idx) };

    #[inline(always)]
    unsafe fn sentinel_or(raw: u32) -> usize {
        let present_mask = (raw != NO_NEIGHBOR) as usize;
        (raw as usize) & present_mask.wrapping_neg()
    }

    let north_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[0])) };
    let south_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[1])) };
    let west_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[2])) };
    let east_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[3])) };
    let nw_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[4])) };
    let ne_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[5])) };
    let sw_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[6])) };
    let se_b = unsafe { &*borders_read_ptr.add(sentinel_or(nb[7])) };

    let ghost = GhostZone {
        north: north_b.south,
        south: south_b.north,
        west: west_b.east,
        east: east_b.west,
        nw: nw_b.se(),
        ne: ne_b.sw(),
        sw: sw_b.ne(),
        se: se_b.nw(),
    };

    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta = unsafe { &mut *meta_ptr.add(idx) };

    // Ultra-fast path: empty tile + empty ghost zone.
    if ghost.north | ghost.south | ghost.west | ghost.east == 0
        && !ghost.nw && !ghost.ne && !ghost.sw && !ghost.se
    {
        let mut any = 0u64;
        let mut i = 0;
        while i < TILE_SIZE {
            any |= current[i] | current[i+1] | current[i+2] | current[i+3];
            if any != 0 { break; }
            i += 4;
        }
        if any == 0 {
            unsafe {
                std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
                *next_borders_ptr.add(idx) = BorderData {
                    north: 0, south: 0, west: 0, east: 0, corners: 0,
                };
            }
            meta.set_changed(false);
            return false;
        }
    }

    let (changed, border, has_live) = advance_core(current, next, &ghost, backend);

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

/// Advance a tile using split buffer pointers (unsafe parallel path).
///
/// # Safety
/// `current_ptr`, `next_ptr`, `meta_ptr`, and `next_borders_ptr` must point
/// to valid slices, and the caller must ensure exclusive write access to
/// the element at `idx` in the write-side arrays.
#[inline(always)]
#[allow(dead_code)]
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
