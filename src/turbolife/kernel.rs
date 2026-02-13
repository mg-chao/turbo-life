//! Scalar bit-parallel kernel for TurboLife.
//!
//! Computes the next generation for a single tile using a full-adder chain.
//! Border extraction is fused into the main loop.
//! Works with split cell buffers (separate current/next vecs).

use super::tile::{BorderData, CellBuf, GhostZone, POPULATION_UNKNOWN, TILE_SIZE, TileMeta};
const _: [(); 1] = [(); (TILE_SIZE == 64) as usize];

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
fn west_neighbor_plane(word: u64, ghost_w: u64) -> u64 {
    (word << 1) | ghost_w
}

#[inline(always)]
fn east_neighbor_plane(word: u64, ghost_e: u64) -> u64 {
    (word >> 1) | (ghost_e << 63)
}

#[inline(always)]
fn corners_from_rows(north: u64, south: u64) -> u8 {
    ((north & 1 != 0) as u8)
        | (((north >> 63 != 0) as u8) << 1)
        | (((south & 1 != 0) as u8) << 2)
        | (((south >> 63 != 0) as u8) << 3)
}

#[inline(always)]
pub(crate) fn ghost_is_empty(ghost: &GhostZone) -> bool {
    let corners = (ghost.nw as u64)
        | ((ghost.ne as u64) << 1)
        | ((ghost.sw as u64) << 2)
        | ((ghost.se as u64) << 3);
    (ghost.north | ghost.south | ghost.west | ghost.east | corners) == 0
}

#[inline(always)]
pub(crate) fn tile_is_empty(current: &[u64; TILE_SIZE]) -> bool {
    let mut i = 0;
    while i < TILE_SIZE {
        if (current[i] | current[i + 1] | current[i + 2] | current[i + 3]) != 0 {
            return false;
        }
        i += 4;
    }
    true
}

#[derive(Clone, Copy)]
struct RowGhostBits {
    west_above: u64,
    east_above: u64,
    west_self: u64,
    east_self: u64,
    west_below: u64,
    east_below: u64,
}

#[inline(always)]
fn row_ghost_bits_from_windows(west_window: u128, east_window: u128) -> RowGhostBits {
    RowGhostBits {
        west_below: (west_window & 1) as u64,
        west_self: ((west_window >> 1) & 1) as u64,
        west_above: ((west_window >> 2) & 1) as u64,
        east_below: (east_window & 1) as u64,
        east_self: ((east_window >> 1) & 1) as u64,
        east_above: ((east_window >> 2) & 1) as u64,
    }
}

#[inline(always)]
fn advance_row_scalar(row_above: u64, row_self: u64, row_below: u64, ghost: RowGhostBits) -> u64 {
    let nw = west_neighbor_plane(row_above, ghost.west_above);
    let n = row_above;
    let ne = east_neighbor_plane(row_above, ghost.east_above);
    let w = west_neighbor_plane(row_self, ghost.west_self);
    let e = east_neighbor_plane(row_self, ghost.east_self);
    let sw = west_neighbor_plane(row_below, ghost.west_below);
    let s = row_below;
    let se = east_neighbor_plane(row_below, ghost.east_below);

    let (a0, a1) = full_add(nw, n, ne);
    let (s0, s1) = half_add(w, e);
    let (b0, b1) = full_add(sw, s, se);
    let (t0, t0c) = full_add(a0, s0, b0);
    let (u0, u0c) = full_add(a1, s1, b1);
    let (t1, t1c) = half_add(u0, t0c);
    let (t2, _) = half_add(u0c, t1c);

    let alive_mask = !t2 & t1;
    (alive_mask & t0) | (alive_mask & !t0 & row_self)
}

#[cfg(test)]
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
    let mut west_window =
        (ghost.sw as u128) | ((ghost.west as u128) << 1) | ((ghost.nw as u128) << 65);
    let mut east_window =
        (ghost.se as u128) | ((ghost.east as u128) << 1) | ((ghost.ne as u128) << 65);

    macro_rules! process_row {
        ($row:expr, $row_above:expr, $row_self:expr, $row_below:expr) => {{
            let ghost = row_ghost_bits_from_windows(west_window, east_window);

            let next_row = advance_row_scalar($row_above, $row_self, $row_below, ghost);

            next[$row] = next_row;
            changed |= next_row != $row_self;
            has_live |= next_row != 0;
            border_west |= (next_row & 1) << $row;
            border_east |= ((next_row >> 63) & 1) << $row;
        }};
    }

    process_row!(0, current[1], current[0], ghost.south);
    west_window >>= 1;
    east_window >>= 1;
    for row in 1..(TILE_SIZE - 1) {
        process_row!(row, current[row + 1], current[row], current[row - 1]);
        west_window >>= 1;
        east_window >>= 1;
    }
    process_row!(
        TILE_SIZE - 1,
        ghost.north,
        current[TILE_SIZE - 1],
        current[TILE_SIZE - 2]
    );

    let north_row = next[TILE_SIZE - 1];
    let south_row = next[0];
    let corners = corners_from_rows(north_row, south_row);

    let border = BorderData::from_parts(north_row, south_row, border_west, border_east, corners);

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
        __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi256_pd, _mm256_extract_epi64,
        _mm256_loadu_si256, _mm256_movemask_pd, _mm256_or_si256, _mm256_setzero_si256,
        _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };

    let mut diff_acc = _mm256_setzero_si256();
    let mut live_acc = _mm256_setzero_si256();
    let mut border_west = 0u64;
    let mut border_east = 0u64;
    let mut border_north = 0u64;
    let mut border_south = 0u64;
    let current_ptr = current.as_ptr();

    let mut west_self_bits = ghost.west;
    let mut east_self_bits = ghost.east;
    let mut west_above_bits = (ghost.west >> 1) | ((ghost.nw as u64) << 63);
    let mut east_above_bits = (ghost.east >> 1) | ((ghost.ne as u64) << 63);
    let mut west_below_bits = (ghost.west << 1) | (ghost.sw as u64);
    let mut east_below_bits = (ghost.east << 1) | (ghost.se as u64);

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

        let ghost_w_self = west_self_bits & 0xF;
        let ghost_e_self = east_self_bits & 0xF;
        let ghost_w_above = west_above_bits & 0xF;
        let ghost_e_above = east_above_bits & 0xF;
        let ghost_w_below = west_below_bits & 0xF;
        let ghost_e_below = east_below_bits & 0xF;

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
        diff_acc = _mm256_or_si256(diff_acc, diff);
        live_acc = _mm256_or_si256(live_acc, next_rows);

        // Store to output and extract border bits directly from register
        // to avoid store-to-load forwarding stalls.
        unsafe {
            _mm256_storeu_si256(next.as_mut_ptr().add(row_base) as *mut __m256i, next_rows);
        }
        let east_mask = _mm256_movemask_pd(_mm256_castsi256_pd(next_rows)) as u64;
        let west_rows = _mm256_slli_epi64(next_rows, 63);
        let west_mask = _mm256_movemask_pd(_mm256_castsi256_pd(west_rows)) as u64;
        border_west |= west_mask << row_base;
        border_east |= east_mask << row_base;
        // Capture south (row 0) and north (row 63) from registers.
        if row_base == 0 {
            border_south = _mm256_extract_epi64(next_rows, 0) as u64;
        }
        if row_base == TILE_SIZE - 4 {
            border_north = _mm256_extract_epi64(next_rows, 3) as u64;
        }

        west_self_bits >>= 4;
        east_self_bits >>= 4;
        west_above_bits >>= 4;
        east_above_bits >>= 4;
        west_below_bits >>= 4;
        east_below_bits >>= 4;
    }

    let corners = corners_from_rows(border_north, border_south);

    let border = BorderData::from_parts(
        border_north,
        border_south,
        border_west,
        border_east,
        corners,
    );

    let changed = _mm256_testz_si256(diff_acc, diff_acc) == 0;
    let has_live = _mm256_testz_si256(live_acc, live_acc) == 0;

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

#[inline(always)]
unsafe fn advance_core_const<const USE_AVX2: bool>(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    if USE_AVX2 {
        #[cfg(target_arch = "x86_64")]
        {
            return unsafe { advance_core_avx2(current, next, ghost) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!("AVX2 backend selected on non-x86 target");
        }
    }
    advance_core_scalar(current, next, ghost)
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
unsafe fn advance_tile_fused_impl<const USE_AVX2: bool>(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    idx: usize,
) -> bool {
    // Inline ghost-zone gather (avoids function call + struct construction).
    let nb = unsafe { &*neighbors_ptr.add(idx) };
    let north_b = unsafe { &*borders_read_ptr.add(nb[0] as usize) };
    let south_b = unsafe { &*borders_read_ptr.add(nb[1] as usize) };
    let west_b = unsafe { &*borders_read_ptr.add(nb[2] as usize) };
    let east_b = unsafe { &*borders_read_ptr.add(nb[3] as usize) };
    let nw_b = unsafe { &*borders_read_ptr.add(nb[4] as usize) };
    let ne_b = unsafe { &*borders_read_ptr.add(nb[5] as usize) };
    let sw_b = unsafe { &*borders_read_ptr.add(nb[6] as usize) };
    let se_b = unsafe { &*borders_read_ptr.add(nb[7] as usize) };

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
    if ghost_is_empty(&ghost) && tile_is_empty(current) {
        unsafe {
            std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
            *next_borders_ptr.add(idx) = BorderData::default();
        }
        meta.set_changed(false);
        meta.set_has_live(false);
        meta.population = 0;
        return false;
    }

    let (changed, border, has_live) =
        unsafe { advance_core_const::<USE_AVX2>(current, next, &ghost) };

    unsafe {
        *next_borders_ptr.add(idx) = border;
    }

    meta.set_changed(changed);
    meta.set_has_live(has_live);
    if changed {
        meta.population = POPULATION_UNKNOWN;
    } else if !has_live {
        meta.population = 0;
    }

    changed
}

#[inline(always)]
pub unsafe fn advance_tile_fused_scalar(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    idx: usize,
) -> bool {
    unsafe {
        advance_tile_fused_impl::<false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_ptr,
            borders_read_ptr,
            neighbors_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn advance_tile_fused_avx2(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_ptr: *mut BorderData,
    borders_read_ptr: *const BorderData,
    neighbors_ptr: *const [u32; 8],
    idx: usize,
) -> bool {
    unsafe {
        advance_tile_fused_impl::<true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_ptr,
            borders_read_ptr,
            neighbors_ptr,
            idx,
        )
    }
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
    meta.set_has_live(has_live);
    if changed {
        meta.population = POPULATION_UNKNOWN;
    } else if !has_live {
        meta.population = 0;
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::{
        BorderData, GhostZone, TILE_SIZE, advance_core_scalar, corners_from_rows, ghost_bit,
        ghost_is_empty, tile_is_empty,
    };

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
        assert_eq!(border.live_mask, 0);
    }

    #[test]
    fn corners_from_rows_matches_corner_flags() {
        let corners = corners_from_rows(1 | (1u64 << 63), 1 | (1u64 << 63));
        assert_eq!(
            corners,
            BorderData::CORNER_NW
                | BorderData::CORNER_NE
                | BorderData::CORNER_SW
                | BorderData::CORNER_SE
        );
        assert_eq!(corners_from_rows(0, 0), 0);
    }

    #[test]
    fn ghost_is_empty_accounts_for_edges_and_corners() {
        let mut ghost = GhostZone::default();
        assert!(ghost_is_empty(&ghost));

        ghost.west = 1;
        assert!(!ghost_is_empty(&ghost));

        ghost.west = 0;
        ghost.ne = true;
        assert!(!ghost_is_empty(&ghost));
    }

    #[test]
    fn tile_is_empty_detects_live_rows() {
        let mut tile = [0u64; TILE_SIZE];
        assert!(tile_is_empty(&tile));

        tile[37] = 1 << 11;
        assert!(!tile_is_empty(&tile));
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
                assert_eq!(scalar.1.live_mask, avx.1.live_mask);
            }
        }
    }
}
