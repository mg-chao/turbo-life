//! Scalar bit-parallel kernel for TurboLife.
//!
//! Computes the next generation for a single tile using a full-adder chain.
//! Border extraction is fused into the main loop.
//! Works with split cell buffers (separate current/next vecs).

use super::tile::{
    BorderData, CellBuf, GhostZone, MISSING_ALL_NEIGHBORS, NeighborIdx, TILE_SIZE, TileMeta,
};
const _: [(); 1] = [(); (TILE_SIZE == 64) as usize];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBackend {
    Scalar,
    Avx2,
    Neon,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileAdvanceResult {
    pub changed: bool,
    pub has_live: bool,
    pub missing_mask: u8,
    pub live_mask: u8,
    pub neighbor_influence_mask: u8,
    pub prune_ready: bool,
}

impl TileAdvanceResult {
    #[inline(always)]
    pub const fn new(
        changed: bool,
        has_live: bool,
        missing_mask: u8,
        live_mask: u8,
        neighbor_influence_mask: u8,
        prune_ready: bool,
    ) -> Self {
        Self {
            changed,
            has_live,
            missing_mask,
            live_mask,
            neighbor_influence_mask,
            prune_ready,
        }
    }
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

const LIVE_N: u8 = 1 << 0;
const LIVE_S: u8 = 1 << 1;
const LIVE_W: u8 = 1 << 2;
const LIVE_E: u8 = 1 << 3;
const LIVE_NW: u8 = 1 << 4;
const LIVE_NE: u8 = 1 << 5;
const LIVE_SW: u8 = 1 << 6;
const LIVE_SE: u8 = 1 << 7;

#[inline(always)]
#[cfg(test)]
fn ghost_is_empty_from_neighbor_masks(neighbor_live_masks: [u8; 8]) -> bool {
    let [north, south, west, east, nw, ne, sw, se] = neighbor_live_masks;
    let ghost_activity = (north & LIVE_S)
        | (south & LIVE_N)
        | (west & LIVE_E)
        | (east & LIVE_W)
        | (nw & LIVE_SE)
        | (ne & LIVE_SW)
        | (sw & LIVE_NE)
        | (se & LIVE_NW);
    ghost_activity == 0
}

#[inline(always)]
#[cfg(test)]
pub(crate) fn ghost_is_empty(ghost: &GhostZone) -> bool {
    let corners = (ghost.nw as u64)
        | ((ghost.ne as u64) << 1)
        | ((ghost.sw as u64) << 2)
        | ((ghost.se as u64) << 3);
    (ghost.north | ghost.south | ghost.west | ghost.east | corners) == 0
}

#[inline(always)]
#[cfg(test)]
pub(crate) fn ghost_is_empty_from_live_masks(neighbor_live_masks: [u8; 8]) -> bool {
    ghost_is_empty_from_neighbor_masks(neighbor_live_masks)
}

#[inline(always)]
pub(crate) unsafe fn ghost_is_empty_from_live_masks_ptr(
    live_masks_ptr: *const u8,
    neighbors: &[NeighborIdx; 8],
) -> bool {
    // SAFETY: callers guarantee `live_masks_ptr` points to a live-mask array
    // indexed by every value in `neighbors` (including sentinel slot 0).
    unsafe {
        let north = *live_masks_ptr.add(neighbors[0] as usize);
        let south = *live_masks_ptr.add(neighbors[1] as usize);
        let west = *live_masks_ptr.add(neighbors[2] as usize);
        let east = *live_masks_ptr.add(neighbors[3] as usize);
        let nw = *live_masks_ptr.add(neighbors[4] as usize);
        let ne = *live_masks_ptr.add(neighbors[5] as usize);
        let sw = *live_masks_ptr.add(neighbors[6] as usize);
        let se = *live_masks_ptr.add(neighbors[7] as usize);
        ((north & LIVE_S)
            | (south & LIVE_N)
            | (west & LIVE_E)
            | (east & LIVE_W)
            | (nw & LIVE_SE)
            | (ne & LIVE_SW)
            | (sw & LIVE_NE)
            | (se & LIVE_NW))
            == 0
    }
}

#[inline(always)]
fn neighbor_influence_mask_from_borders(
    prev_north: u64,
    prev_south: u64,
    prev_west: u64,
    prev_east: u64,
    next_border: &BorderData,
) -> u8 {
    BorderData::compute_live_mask(
        next_border.north ^ prev_north,
        next_border.south ^ prev_south,
        next_border.west ^ prev_west,
        next_border.east ^ prev_east,
    )
}

#[inline(always)]
fn no_track_hint(changed: bool, prune_ready: bool) -> u8 {
    if changed {
        u8::MAX
    } else if prune_ready {
        1
    } else {
        0
    }
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

#[inline(always)]
pub(crate) fn clear_tile_if_needed(next: &mut [u64; TILE_SIZE]) {
    if !tile_is_empty(next) {
        unsafe {
            std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
        }
    }
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
    alive_mask & (t0 | row_self)
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

    let mut row = 1usize;
    while row < TILE_SIZE - 2 {
        process_row!(row, current[row + 1], current[row], current[row - 1]);
        west_window >>= 1;
        east_window >>= 1;

        let row2 = row + 1;
        process_row!(row2, current[row2 + 1], current[row2], current[row2 - 1]);
        west_window >>= 1;
        east_window >>= 1;
        row += 2;
    }
    process_row!(
        TILE_SIZE - 1,
        ghost.north,
        current[TILE_SIZE - 1],
        current[TILE_SIZE - 2]
    );

    let north_row = next[TILE_SIZE - 1];
    let south_row = next[0];
    let border = BorderData::from_edges(north_row, south_row, border_west, border_east);

    (changed, border, has_live)
}

const fn build_corner_birth_table() -> [u8; 32] {
    let mut table = [0u8; 32];
    let mut bits = 0usize;
    while bits < 32 {
        let mut count = 0u32;
        count += (bits & 1) as u32;
        count += ((bits >> 1) & 1) as u32;
        count += ((bits >> 2) & 1) as u32;
        count += ((bits >> 3) & 1) as u32;
        count += ((bits >> 4) & 1) as u32;
        table[bits] = (count == 3) as u8;
        bits += 1;
    }
    table
}

const CORNER_BIRTH_TABLE: [u8; 32] = build_corner_birth_table();

#[inline(always)]
fn birth_from_count5(bits: u64) -> u64 {
    debug_assert!(bits < 32);
    CORNER_BIRTH_TABLE[(bits as usize) & 0b1_1111] as u64
}

/// Specialized core for tiles whose current buffer is entirely dead.
///
/// Only boundary cells can be born from halo activity, so this bypasses
/// the full 64-row adder pipeline and writes a sparse edge-only result.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn advance_core_empty(
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    advance_core_empty_with_clear(next, ghost, true)
}

#[inline(always)]
pub(crate) fn advance_core_empty_with_clear(
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
    clear_next: bool,
) -> (bool, BorderData, bool) {
    let south_trip = (ghost.south << 1) & ghost.south & (ghost.south >> 1);
    let north_trip = (ghost.north << 1) & ghost.north & (ghost.north >> 1);
    let west_trip = (ghost.west << 1) & ghost.west & (ghost.west >> 1);
    let east_trip = (ghost.east << 1) & ghost.east & (ghost.east >> 1);

    let sw_bits = (ghost.south & 1)
        | (((ghost.south >> 1) & 1) << 1)
        | ((ghost.west & 1) << 2)
        | (((ghost.west >> 1) & 1) << 3)
        | ((ghost.sw as u64) << 4);
    let se_bits = ((ghost.south >> 63) & 1)
        | (((ghost.south >> 62) & 1) << 1)
        | ((ghost.east & 1) << 2)
        | (((ghost.east >> 1) & 1) << 3)
        | ((ghost.se as u64) << 4);
    let nw_bits = (ghost.north & 1)
        | (((ghost.north >> 1) & 1) << 1)
        | (((ghost.west >> 63) & 1) << 2)
        | (((ghost.west >> 62) & 1) << 3)
        | ((ghost.nw as u64) << 4);
    let ne_bits = ((ghost.north >> 63) & 1)
        | (((ghost.north >> 62) & 1) << 1)
        | (((ghost.east >> 63) & 1) << 2)
        | (((ghost.east >> 62) & 1) << 3)
        | ((ghost.ne as u64) << 4);

    let sw_birth = birth_from_count5(sw_bits);
    let se_birth = birth_from_count5(se_bits);
    let nw_birth = birth_from_count5(nw_bits);
    let ne_birth = birth_from_count5(ne_bits);

    let south_row = south_trip | sw_birth | (se_birth << 63);
    let north_row = north_trip | nw_birth | (ne_birth << 63);
    let border_west = west_trip | sw_birth | (nw_birth << 63);
    let border_east = east_trip | se_birth | (ne_birth << 63);

    if clear_next {
        clear_tile_if_needed(next);
    } else {
        debug_assert!(tile_is_empty(next));
    }
    next[0] = south_row;
    next[TILE_SIZE - 1] = north_row;

    let mut edge_rows = west_trip | east_trip;
    while edge_rows != 0 {
        let row = edge_rows.trailing_zeros() as usize;
        let west_bit = (west_trip >> row) & 1;
        let east_bit = ((east_trip >> row) & 1) << 63;
        next[row] |= west_bit | east_bit;
        edge_rows &= edge_rows - 1;
    }

    let has_live = (south_row | north_row | border_west | border_east) != 0;
    let border = BorderData::from_edges(north_row, south_row, border_west, border_east);

    (has_live, border, has_live)
}

// ── AVX2 kernel ─────────────────────────────────────────────────────────

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
unsafe fn avx2_set_u64x4_lane_order(
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::_mm256_set_epi64x;
    unsafe { _mm256_set_epi64x(a3 as i64, a2 as i64, a1 as i64, a0 as i64) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_set_u64x2_lane_order(a0: u64, a1: u64) -> std::arch::aarch64::uint64x2_t {
    unsafe { std::mem::transmute([a0, a1]) }
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

#[cfg(target_arch = "aarch64")]
const fn neon_build_carry_table(shift: u32) -> [[u64; 2]; 4] {
    let mut table = [[0u64; 2]; 4];
    let mut bits = 0usize;
    while bits < 4 {
        let b = bits as u64;
        table[bits] = [(b & 1) << shift, ((b >> 1) & 1) << shift];
        bits += 1;
    }
    table
}

#[cfg(target_arch = "aarch64")]
const NEON_CARRY_LO_TABLE: [[u64; 2]; 4] = neon_build_carry_table(0);

#[cfg(target_arch = "aarch64")]
const NEON_CARRY_HI_TABLE: [[u64; 2]; 4] = neon_build_carry_table(63);

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_carry_mask_lo(bits2: u64) -> std::arch::aarch64::uint64x2_t {
    use std::arch::aarch64::vld1q_u64;
    let row = unsafe { NEON_CARRY_LO_TABLE.get_unchecked(bits2 as usize) };
    unsafe { vld1q_u64(row.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_carry_mask_hi(bits2: u64) -> std::arch::aarch64::uint64x2_t {
    use std::arch::aarch64::vld1q_u64;
    let row = unsafe { NEON_CARRY_HI_TABLE.get_unchecked(bits2 as usize) };
    unsafe { vld1q_u64(row.as_ptr()) }
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
    let current_ptr = current.as_ptr();
    let next_ptr = next.as_mut_ptr();

    let mut west_self_bits = ghost.west;
    let mut east_self_bits = ghost.east;
    let mut west_above_bits = (ghost.west >> 1) | ((ghost.nw as u64) << 63);
    let mut east_above_bits = (ghost.east >> 1) | ((ghost.ne as u64) << 63);
    let mut west_below_bits = (ghost.west << 1) | (ghost.sw as u64);
    let mut east_below_bits = (ghost.east << 1) | (ghost.se as u64);

    macro_rules! process_chunk {
        ($row_base:expr, $row_above:expr, $row_self:expr, $row_below:expr) => {{
            let ghost_w_self = west_self_bits & 0xF;
            let ghost_e_self = east_self_bits & 0xF;
            let ghost_w_above = west_above_bits & 0xF;
            let ghost_e_above = east_above_bits & 0xF;
            let ghost_w_below = west_below_bits & 0xF;
            let ghost_e_below = east_below_bits & 0xF;

            let nw = _mm256_or_si256(_mm256_slli_epi64($row_above, 1), unsafe {
                avx2_carry_mask_lo(ghost_w_above)
            });
            let n = $row_above;
            let ne = _mm256_or_si256(_mm256_srli_epi64($row_above, 1), unsafe {
                avx2_carry_mask_hi(ghost_e_above)
            });
            let w = _mm256_or_si256(_mm256_slli_epi64($row_self, 1), unsafe {
                avx2_carry_mask_lo(ghost_w_self)
            });
            let e = _mm256_or_si256(_mm256_srli_epi64($row_self, 1), unsafe {
                avx2_carry_mask_hi(ghost_e_self)
            });
            let sw = _mm256_or_si256(_mm256_slli_epi64($row_below, 1), unsafe {
                avx2_carry_mask_lo(ghost_w_below)
            });
            let s = $row_below;
            let se = _mm256_or_si256(_mm256_srli_epi64($row_below, 1), unsafe {
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
            let next_rows = _mm256_and_si256(alive_mask, _mm256_or_si256(t0, $row_self));

            let diff = _mm256_xor_si256(next_rows, $row_self);
            diff_acc = _mm256_or_si256(diff_acc, diff);
            live_acc = _mm256_or_si256(live_acc, next_rows);

            unsafe {
                _mm256_storeu_si256(next_ptr.add($row_base) as *mut __m256i, next_rows);
            }
            let east_mask = _mm256_movemask_pd(_mm256_castsi256_pd(next_rows)) as u64;
            let west_rows = _mm256_slli_epi64(next_rows, 63);
            let west_mask = _mm256_movemask_pd(_mm256_castsi256_pd(west_rows)) as u64;
            border_west |= west_mask << $row_base;
            border_east |= east_mask << $row_base;

            next_rows
        }};
    }

    let row_self_0 = unsafe { _mm256_loadu_si256(current_ptr as *const __m256i) };
    let row_above_0 = unsafe { _mm256_loadu_si256(current_ptr.add(1) as *const __m256i) };
    let row_below_0 =
        unsafe { avx2_set_u64x4_lane_order(ghost.south, current[0], current[1], current[2]) };
    let next_rows_0 = process_chunk!(0, row_above_0, row_self_0, row_below_0);
    let border_south = _mm256_extract_epi64(next_rows_0, 0) as u64;
    west_self_bits >>= 4;
    east_self_bits >>= 4;
    west_above_bits >>= 4;
    east_above_bits >>= 4;
    west_below_bits >>= 4;
    east_below_bits >>= 4;

    let mut row_base = 4usize;
    while row_base < TILE_SIZE - 4 {
        let row_self = unsafe { _mm256_loadu_si256(current_ptr.add(row_base) as *const __m256i) };
        let row_above =
            unsafe { _mm256_loadu_si256(current_ptr.add(row_base + 1) as *const __m256i) };
        let row_below =
            unsafe { _mm256_loadu_si256(current_ptr.add(row_base - 1) as *const __m256i) };
        let _ = process_chunk!(row_base, row_above, row_self, row_below);
        west_self_bits >>= 4;
        east_self_bits >>= 4;
        west_above_bits >>= 4;
        east_above_bits >>= 4;
        west_below_bits >>= 4;
        east_below_bits >>= 4;
        row_base += 4;
    }

    let last_base = TILE_SIZE - 4;
    let row_self_last = unsafe { _mm256_loadu_si256(current_ptr.add(last_base) as *const __m256i) };
    let row_above_last =
        unsafe { avx2_set_u64x4_lane_order(current[61], current[62], current[63], ghost.north) };
    let row_below_last =
        unsafe { _mm256_loadu_si256(current_ptr.add(last_base - 1) as *const __m256i) };
    let next_rows_last = process_chunk!(last_base, row_above_last, row_self_last, row_below_last);
    let border_north = _mm256_extract_epi64(next_rows_last, 3) as u64;
    let border = BorderData::from_edges(border_north, border_south, border_west, border_east);

    let changed = _mm256_testz_si256(diff_acc, diff_acc) == 0;
    let has_live = _mm256_testz_si256(live_acc, live_acc) == 0;

    (changed, border, has_live)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_full_add(
    a: std::arch::aarch64::uint64x2_t,
    b: std::arch::aarch64::uint64x2_t,
    c: std::arch::aarch64::uint64x2_t,
) -> (
    std::arch::aarch64::uint64x2_t,
    std::arch::aarch64::uint64x2_t,
) {
    use std::arch::aarch64::{vandq_u64, veorq_u64, vorrq_u64};
    unsafe {
        let sum = veorq_u64(veorq_u64(a, b), c);
        let carry = vorrq_u64(vorrq_u64(vandq_u64(a, b), vandq_u64(b, c)), vandq_u64(a, c));
        (sum, carry)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_half_add(
    a: std::arch::aarch64::uint64x2_t,
    b: std::arch::aarch64::uint64x2_t,
) -> (
    std::arch::aarch64::uint64x2_t,
    std::arch::aarch64::uint64x2_t,
) {
    use std::arch::aarch64::{vandq_u64, veorq_u64};
    unsafe { (veorq_u64(a, b), vandq_u64(a, b)) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn advance_core_neon_impl_raw<const TRACK_DIFF: bool, const FORCE_STORE: bool>(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost_north: u64,
    ghost_south: u64,
    ghost_west: u64,
    ghost_east: u64,
    ghost_nw: u64,
    ghost_ne: u64,
    ghost_sw: u64,
    ghost_se: u64,
) -> (bool, BorderData, bool) {
    use std::arch::aarch64::{
        vandq_u64, vbicq_u64, vdupq_n_u64, veorq_u64, vget_high_u64, vget_lane_u64, vget_low_u64,
        vgetq_lane_u64, vld1q_u64, vorr_u64, vorrq_u64, vshlq_n_u64, vshrq_n_u64, vst1q_u64,
    };

    let mut changed = !TRACK_DIFF;
    let mut diff_acc = vdupq_n_u64(0);
    let mut live_acc_scalar = 0u64;
    let mut border_west = 0u64;
    let mut border_east = 0u64;
    let current_ptr = current.as_ptr();
    let next_ptr = next.as_mut_ptr();

    debug_assert!(ghost_nw <= 1);
    debug_assert!(ghost_ne <= 1);
    debug_assert!(ghost_sw <= 1);
    debug_assert!(ghost_se <= 1);

    let mut west_self_bits = ghost_west;
    let mut east_self_bits = ghost_east;
    let mut west_above_bits = (ghost_west >> 1) | (ghost_nw << 63);
    let mut east_above_bits = (ghost_east >> 1) | (ghost_ne << 63);
    let mut west_below_bits = (ghost_west << 1) | ghost_sw;
    let mut east_below_bits = (ghost_east << 1) | ghost_se;

    macro_rules! process_pair {
        (
            $row_base:expr,
            $row_above:expr,
            $row_self:expr,
            $row_below:expr,
            $ghost_w_above:expr,
            $ghost_e_above:expr,
            $ghost_w_self:expr,
            $ghost_e_self:expr,
            $ghost_w_below:expr,
            $ghost_e_below:expr
        ) => {{
            let nw = vorrq_u64(vshlq_n_u64($row_above, 1), unsafe {
                neon_carry_mask_lo($ghost_w_above)
            });
            let n = $row_above;
            let ne = vorrq_u64(vshrq_n_u64($row_above, 1), unsafe {
                neon_carry_mask_hi($ghost_e_above)
            });
            let w = vorrq_u64(vshlq_n_u64($row_self, 1), unsafe {
                neon_carry_mask_lo($ghost_w_self)
            });
            let e = vorrq_u64(vshrq_n_u64($row_self, 1), unsafe {
                neon_carry_mask_hi($ghost_e_self)
            });
            let sw = vorrq_u64(vshlq_n_u64($row_below, 1), unsafe {
                neon_carry_mask_lo($ghost_w_below)
            });
            let s = $row_below;
            let se = vorrq_u64(vshrq_n_u64($row_below, 1), unsafe {
                neon_carry_mask_hi($ghost_e_below)
            });

            let (a0, a1) = unsafe { neon_full_add(nw, n, ne) };
            let (s0, s1) = unsafe { neon_half_add(w, e) };
            let (b0, b1) = unsafe { neon_full_add(sw, s, se) };
            let (t0, t0c) = unsafe { neon_full_add(a0, s0, b0) };
            let (u0, u0c) = unsafe { neon_full_add(a1, s1, b1) };
            let (t1, t1c) = unsafe { neon_half_add(u0, t0c) };
            let (t2, _) = unsafe { neon_half_add(u0c, t1c) };

            let alive_mask = vbicq_u64(t1, t2);
            let next_rows = vandq_u64(alive_mask, vorrq_u64(t0, $row_self));

            if TRACK_DIFF {
                let diff = veorq_u64(next_rows, $row_self);
                if FORCE_STORE {
                    diff_acc = vorrq_u64(diff_acc, diff);
                    unsafe {
                        vst1q_u64(next_ptr.add($row_base), next_rows);
                    }
                } else {
                    // Horizontal OR via 64-bit lane vectors avoids extracting both
                    // lanes as scalars before combining.
                    let pair_changed =
                        vget_lane_u64(vorr_u64(vget_low_u64(diff), vget_high_u64(diff)), 0) != 0;
                    changed |= pair_changed;
                    if pair_changed {
                        unsafe {
                            vst1q_u64(next_ptr.add($row_base), next_rows);
                        }
                    }
                }
            } else {
                unsafe {
                    vst1q_u64(next_ptr.add($row_base), next_rows);
                }
            }
            let row0 = vgetq_lane_u64(next_rows, 0);
            let row1 = vgetq_lane_u64(next_rows, 1);
            live_acc_scalar |= row0 | row1;
            let west_bits = (row0 & 1) | ((row1 & 1) << 1);
            let east_bits = ((row0 >> 63) & 1) | (((row1 >> 63) & 1) << 1);
            border_west |= west_bits << $row_base;
            border_east |= east_bits << $row_base;
            (row0, row1)
        }};
    }

    let row_self_0 = unsafe { vld1q_u64(current_ptr) };
    let row_above_0 = unsafe { vld1q_u64(current_ptr.add(1)) };
    let row_below_0 = unsafe { neon_set_u64x2_lane_order(ghost_south, current[0]) };
    let (border_south, _) = process_pair!(
        0,
        row_above_0,
        row_self_0,
        row_below_0,
        west_above_bits & 0b11,
        east_above_bits & 0b11,
        west_self_bits & 0b11,
        east_self_bits & 0b11,
        west_below_bits & 0b11,
        east_below_bits & 0b11
    );
    west_self_bits >>= 2;
    east_self_bits >>= 2;
    west_above_bits >>= 2;
    east_above_bits >>= 2;
    west_below_bits >>= 2;
    east_below_bits >>= 2;

    let mut prev_above = row_above_0;
    let mut row_base = 2usize;
    // Process 8 rows (4 pairs) per iteration to reduce loop-control overhead.
    while row_base < TILE_SIZE - 8 {
        let byte_w_self = west_self_bits & 0xFF;
        let byte_e_self = east_self_bits & 0xFF;
        let byte_w_above = west_above_bits & 0xFF;
        let byte_e_above = east_above_bits & 0xFF;
        let byte_w_below = west_below_bits & 0xFF;
        let byte_e_below = east_below_bits & 0xFF;

        let row_below_0 = prev_above;
        let row_self_0 = unsafe { vld1q_u64(current_ptr.add(row_base)) };
        let row_above_0 = unsafe { vld1q_u64(current_ptr.add(row_base + 1)) };
        let _ = process_pair!(
            row_base,
            row_above_0,
            row_self_0,
            row_below_0,
            byte_w_above & 0b11,
            byte_e_above & 0b11,
            byte_w_self & 0b11,
            byte_e_self & 0b11,
            byte_w_below & 0b11,
            byte_e_below & 0b11
        );

        let row_below_1 = row_above_0;
        let row_self_1 = unsafe { vld1q_u64(current_ptr.add(row_base + 2)) };
        let row_above_1 = unsafe { vld1q_u64(current_ptr.add(row_base + 3)) };
        let _ = process_pair!(
            row_base + 2,
            row_above_1,
            row_self_1,
            row_below_1,
            (byte_w_above >> 2) & 0b11,
            (byte_e_above >> 2) & 0b11,
            (byte_w_self >> 2) & 0b11,
            (byte_e_self >> 2) & 0b11,
            (byte_w_below >> 2) & 0b11,
            (byte_e_below >> 2) & 0b11
        );

        let row_below_2 = row_above_1;
        let row_self_2 = unsafe { vld1q_u64(current_ptr.add(row_base + 4)) };
        let row_above_2 = unsafe { vld1q_u64(current_ptr.add(row_base + 5)) };
        let _ = process_pair!(
            row_base + 4,
            row_above_2,
            row_self_2,
            row_below_2,
            (byte_w_above >> 4) & 0b11,
            (byte_e_above >> 4) & 0b11,
            (byte_w_self >> 4) & 0b11,
            (byte_e_self >> 4) & 0b11,
            (byte_w_below >> 4) & 0b11,
            (byte_e_below >> 4) & 0b11
        );

        let row_below_3 = row_above_2;
        let row_self_3 = unsafe { vld1q_u64(current_ptr.add(row_base + 6)) };
        let row_above_3 = unsafe { vld1q_u64(current_ptr.add(row_base + 7)) };
        let _ = process_pair!(
            row_base + 6,
            row_above_3,
            row_self_3,
            row_below_3,
            (byte_w_above >> 6) & 0b11,
            (byte_e_above >> 6) & 0b11,
            (byte_w_self >> 6) & 0b11,
            (byte_e_self >> 6) & 0b11,
            (byte_w_below >> 6) & 0b11,
            (byte_e_below >> 6) & 0b11
        );

        prev_above = row_above_3;
        west_self_bits >>= 8;
        east_self_bits >>= 8;
        west_above_bits >>= 8;
        east_above_bits >>= 8;
        west_below_bits >>= 8;
        east_below_bits >>= 8;
        row_base += 8;
    }

    while row_base < TILE_SIZE - 4 {
        let nib_w_self = west_self_bits & 0b1111;
        let nib_e_self = east_self_bits & 0b1111;
        let nib_w_above = west_above_bits & 0b1111;
        let nib_e_above = east_above_bits & 0b1111;
        let nib_w_below = west_below_bits & 0b1111;
        let nib_e_below = east_below_bits & 0b1111;

        let row_below_0 = prev_above;
        let row_self_0 = unsafe { vld1q_u64(current_ptr.add(row_base)) };
        let row_above_0 = unsafe { vld1q_u64(current_ptr.add(row_base + 1)) };
        let _ = process_pair!(
            row_base,
            row_above_0,
            row_self_0,
            row_below_0,
            nib_w_above & 0b11,
            nib_e_above & 0b11,
            nib_w_self & 0b11,
            nib_e_self & 0b11,
            nib_w_below & 0b11,
            nib_e_below & 0b11
        );

        let row_below_1 = row_above_0;
        let row_self_1 = unsafe { vld1q_u64(current_ptr.add(row_base + 2)) };
        let row_above_1 = unsafe { vld1q_u64(current_ptr.add(row_base + 3)) };
        let _ = process_pair!(
            row_base + 2,
            row_above_1,
            row_self_1,
            row_below_1,
            (nib_w_above >> 2) & 0b11,
            (nib_e_above >> 2) & 0b11,
            (nib_w_self >> 2) & 0b11,
            (nib_e_self >> 2) & 0b11,
            (nib_w_below >> 2) & 0b11,
            (nib_e_below >> 2) & 0b11
        );
        prev_above = row_above_1;
        west_self_bits >>= 4;
        east_self_bits >>= 4;
        west_above_bits >>= 4;
        east_above_bits >>= 4;
        west_below_bits >>= 4;
        east_below_bits >>= 4;
        row_base += 4;
    }

    if row_base < TILE_SIZE - 2 {
        let row_below = prev_above;
        let row_self = unsafe { vld1q_u64(current_ptr.add(row_base)) };
        let row_above = unsafe { vld1q_u64(current_ptr.add(row_base + 1)) };
        let _ = process_pair!(
            row_base,
            row_above,
            row_self,
            row_below,
            west_above_bits & 0b11,
            east_above_bits & 0b11,
            west_self_bits & 0b11,
            east_self_bits & 0b11,
            west_below_bits & 0b11,
            east_below_bits & 0b11
        );
        prev_above = row_above;
        west_self_bits >>= 2;
        east_self_bits >>= 2;
        west_above_bits >>= 2;
        east_above_bits >>= 2;
        west_below_bits >>= 2;
        east_below_bits >>= 2;
    }

    let last_base = TILE_SIZE - 2;
    let row_self_last = unsafe { vld1q_u64(current_ptr.add(last_base)) };
    let row_above_last = unsafe { neon_set_u64x2_lane_order(current[TILE_SIZE - 1], ghost_north) };
    let row_below_last = prev_above;
    let (_, border_north) = process_pair!(
        last_base,
        row_above_last,
        row_self_last,
        row_below_last,
        west_above_bits & 0b11,
        east_above_bits & 0b11,
        west_self_bits & 0b11,
        east_self_bits & 0b11,
        west_below_bits & 0b11,
        east_below_bits & 0b11
    );

    if TRACK_DIFF && FORCE_STORE {
        changed = vget_lane_u64(vorr_u64(vget_low_u64(diff_acc), vget_high_u64(diff_acc)), 0) != 0;
    }

    let has_live = live_acc_scalar != 0;
    let border = BorderData::from_edges(border_north, border_south, border_west, border_east);
    (changed, border, has_live)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn advance_core_neon_impl<const TRACK_DIFF: bool, const FORCE_STORE: bool>(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    unsafe {
        advance_core_neon_impl_raw::<TRACK_DIFF, FORCE_STORE>(
            current,
            next,
            ghost.north,
            ghost.south,
            ghost.west,
            ghost.east,
            ghost.nw as u64,
            ghost.ne as u64,
            ghost.sw as u64,
            ghost.se as u64,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn advance_core_neon(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
    force_store: bool,
) -> (bool, BorderData, bool) {
    if force_store {
        unsafe { advance_core_neon_impl::<true, true>(current, next, ghost) }
    } else {
        unsafe { advance_core_neon_impl::<true, false>(current, next, ghost) }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn advance_core_neon_assume_changed(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
) -> (bool, BorderData, bool) {
    unsafe { advance_core_neon_impl::<false, true>(current, next, ghost) }
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
        KernelBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    unsafe { advance_core_neon(current, next, ghost, true) }
                } else {
                    advance_core_scalar(current, next, ghost)
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                advance_core_scalar(current, next, ghost)
            }
        }
    }
}

const CORE_BACKEND_SCALAR: u8 = 0;
const CORE_BACKEND_AVX2: u8 = 1;
const CORE_BACKEND_NEON: u8 = 2;

#[inline(always)]
unsafe fn advance_core_const<const CORE_BACKEND: u8, const ASSUME_CHANGED: bool>(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &GhostZone,
    force_store: bool,
) -> (bool, BorderData, bool) {
    debug_assert!(
        !ASSUME_CHANGED || CORE_BACKEND == CORE_BACKEND_NEON,
        "ASSUME_CHANGED is only supported for the NEON backend"
    );
    #[cfg(not(target_arch = "aarch64"))]
    let _ = force_store;

    if CORE_BACKEND == CORE_BACKEND_AVX2 {
        #[cfg(target_arch = "x86_64")]
        {
            return unsafe { advance_core_avx2(current, next, ghost) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!("AVX2 backend selected on non-x86 target");
        }
    }
    if CORE_BACKEND == CORE_BACKEND_NEON {
        #[cfg(target_arch = "aarch64")]
        {
            if ASSUME_CHANGED {
                return unsafe { advance_core_neon_assume_changed(current, next, ghost) };
            }
            return unsafe { advance_core_neon(current, next, ghost, force_store) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            unreachable!("NEON backend selected on non-aarch64 target");
        }
    }
    advance_core_scalar(current, next, ghost)
}

// Keep the empty-tile branch outlined; forcing this into the fused path
// can exhaust debug worker stacks by ballooning the caller frame.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn advance_tile_fused_empty_tile<
    const TRACK_NEIGHBOR_INFLUENCE: bool,
    const ASSUME_CHANGED: bool,
>(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    meta_slot: *mut TileMeta,
    missing_mask: u8,
    force_store: bool,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
    north_i: usize,
    south_i: usize,
    west_i: usize,
    east_i: usize,
    nw_i: usize,
    ne_i: usize,
    sw_i: usize,
    se_i: usize,
) -> TileAdvanceResult {
    if missing_mask == MISSING_ALL_NEIGHBORS {
        debug_assert!(tile_is_empty(current));
        if force_store {
            unsafe {
                std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
            }
        }
        unsafe {
            *next_borders_north_ptr.add(idx) = 0;
            *next_borders_south_ptr.add(idx) = 0;
            *next_borders_west_ptr.add(idx) = 0;
            *next_borders_east_ptr.add(idx) = 0;
            *next_live_masks_ptr.add(idx) = 0;
            (*meta_slot).update_after_step(false, false);
        }
        debug_assert!(force_store || tile_is_empty(next));
        let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE || ASSUME_CHANGED {
            0
        } else {
            no_track_hint(false, true)
        };
        return TileAdvanceResult::new(
            false,
            false,
            missing_mask,
            0,
            neighbor_influence_mask,
            true,
        );
    }

    let north_live = unsafe { *live_masks_read_ptr.add(north_i) };
    let south_live = unsafe { *live_masks_read_ptr.add(south_i) };
    let west_live = unsafe { *live_masks_read_ptr.add(west_i) };
    let east_live = unsafe { *live_masks_read_ptr.add(east_i) };
    let nw_live = unsafe { *live_masks_read_ptr.add(nw_i) };
    let ne_live = unsafe { *live_masks_read_ptr.add(ne_i) };
    let sw_live = unsafe { *live_masks_read_ptr.add(sw_i) };
    let se_live = unsafe { *live_masks_read_ptr.add(se_i) };
    let ghost_empty = ((north_live & LIVE_S)
        | (south_live & LIVE_N)
        | (west_live & LIVE_E)
        | (east_live & LIVE_W)
        | (nw_live & LIVE_SE)
        | (ne_live & LIVE_SW)
        | (sw_live & LIVE_NE)
        | (se_live & LIVE_NW))
        == 0;

    if ghost_empty {
        debug_assert!(tile_is_empty(current));
        if force_store {
            unsafe {
                std::ptr::write_bytes(next.as_mut_ptr(), 0, TILE_SIZE);
            }
        }
        unsafe {
            *next_borders_north_ptr.add(idx) = 0;
            *next_borders_south_ptr.add(idx) = 0;
            *next_borders_west_ptr.add(idx) = 0;
            *next_borders_east_ptr.add(idx) = 0;
            *next_live_masks_ptr.add(idx) = 0;
            (*meta_slot).update_after_step(false, false);
        }
        debug_assert!(force_store || tile_is_empty(next));
        let prune_ready = false;
        let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE || ASSUME_CHANGED {
            0
        } else {
            no_track_hint(false, prune_ready)
        };
        return TileAdvanceResult::new(
            false,
            false,
            missing_mask,
            0,
            neighbor_influence_mask,
            prune_ready,
        );
    }

    let ghost = GhostZone {
        north: unsafe { *borders_south_read_ptr.add(north_i) },
        south: unsafe { *borders_north_read_ptr.add(south_i) },
        west: unsafe { *borders_east_read_ptr.add(west_i) },
        east: unsafe { *borders_west_read_ptr.add(east_i) },
        nw: (nw_live & LIVE_SE) != 0,
        ne: (ne_live & LIVE_SW) != 0,
        sw: (sw_live & LIVE_NE) != 0,
        se: (se_live & LIVE_NW) != 0,
    };
    debug_assert!(tile_is_empty(current));
    let (changed, border, has_live) = advance_core_empty_with_clear(next, &ghost, force_store);
    let live_mask = border.live_mask();

    unsafe {
        *next_borders_north_ptr.add(idx) = border.north;
        *next_borders_south_ptr.add(idx) = border.south;
        *next_borders_west_ptr.add(idx) = border.west;
        *next_borders_east_ptr.add(idx) = border.east;
        *next_live_masks_ptr.add(idx) = live_mask;
    }

    let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
        if changed {
            let prev_north = unsafe { *borders_north_read_ptr.add(idx) };
            let prev_south = unsafe { *borders_south_read_ptr.add(idx) };
            let prev_west = unsafe { *borders_west_read_ptr.add(idx) };
            let prev_east = unsafe { *borders_east_read_ptr.add(idx) };
            neighbor_influence_mask_from_borders(
                prev_north, prev_south, prev_west, prev_east, &border,
            )
        } else {
            0
        }
    } else if ASSUME_CHANGED {
        0
    } else {
        no_track_hint(changed, false)
    };

    unsafe {
        (*meta_slot).update_after_step(changed, has_live);
    }

    TileAdvanceResult::new(
        changed,
        has_live,
        missing_mask,
        live_mask,
        neighbor_influence_mask,
        false,
    )
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
/// `meta_ptr[idx]`, `next_borders_ptr[idx]`, and `next_live_masks_ptr[idx]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn advance_tile_fused_impl<
    const CORE_BACKEND: u8,
    const ASSUME_CHANGED: bool,
    const TRACK_NEIGHBOR_INFLUENCE: bool,
>(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    debug_assert!(
        !ASSUME_CHANGED || CORE_BACKEND == CORE_BACKEND_NEON,
        "ASSUME_CHANGED is only supported for the NEON backend"
    );
    debug_assert!(
        !ASSUME_CHANGED || !TRACK_NEIGHBOR_INFLUENCE,
        "ASSUME_CHANGED mode does not emit neighbor influence"
    );

    let nb = unsafe { *neighbors_ptr.add(idx) };
    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta_slot = unsafe { meta_ptr.add(idx) };
    let meta = unsafe { *meta_slot };
    let missing_mask = meta.missing_mask;
    let tile_has_live = meta.has_live();
    let force_store = meta.alt_phase_dirty();
    let north_i = nb[0] as usize;
    let south_i = nb[1] as usize;
    let west_i = nb[2] as usize;
    let east_i = nb[3] as usize;
    let nw_i = nb[4] as usize;
    let ne_i = nb[5] as usize;
    let sw_i = nb[6] as usize;
    let se_i = nb[7] as usize;

    if !tile_has_live {
        return unsafe {
            advance_tile_fused_empty_tile::<TRACK_NEIGHBOR_INFLUENCE, ASSUME_CHANGED>(
                current,
                next,
                meta_slot,
                missing_mask,
                force_store,
                next_borders_north_ptr,
                next_borders_south_ptr,
                next_borders_west_ptr,
                next_borders_east_ptr,
                borders_north_read_ptr,
                borders_south_read_ptr,
                borders_west_read_ptr,
                borders_east_read_ptr,
                live_masks_read_ptr,
                next_live_masks_ptr,
                idx,
                north_i,
                south_i,
                west_i,
                east_i,
                nw_i,
                ne_i,
                sw_i,
                se_i,
            )
        };
    }

    let (changed, border, has_live) = {
        let ghost_north = unsafe { *borders_south_read_ptr.add(north_i) };
        let ghost_south = unsafe { *borders_north_read_ptr.add(south_i) };
        let ghost_west = unsafe { *borders_east_read_ptr.add(west_i) };
        let ghost_east = unsafe { *borders_west_read_ptr.add(east_i) };
        let nw_live = unsafe { *live_masks_read_ptr.add(nw_i) };
        let ne_live = unsafe { *live_masks_read_ptr.add(ne_i) };
        let sw_live = unsafe { *live_masks_read_ptr.add(sw_i) };
        let se_live = unsafe { *live_masks_read_ptr.add(se_i) };
        let ghost_nw = (nw_live & LIVE_SE) != 0;
        let ghost_ne = (ne_live & LIVE_SW) != 0;
        let ghost_sw = (sw_live & LIVE_NE) != 0;
        let ghost_se = (se_live & LIVE_NW) != 0;

        if CORE_BACKEND == CORE_BACKEND_NEON {
            #[cfg(target_arch = "aarch64")]
            {
                let (changed, border, has_live) = if ASSUME_CHANGED {
                    unsafe {
                        advance_core_neon_impl_raw::<false, true>(
                            current,
                            next,
                            ghost_north,
                            ghost_south,
                            ghost_west,
                            ghost_east,
                            ghost_nw as u64,
                            ghost_ne as u64,
                            ghost_sw as u64,
                            ghost_se as u64,
                        )
                    }
                } else {
                    unsafe {
                        advance_core_neon_impl_raw::<true, true>(
                            current,
                            next,
                            ghost_north,
                            ghost_south,
                            ghost_west,
                            ghost_east,
                            ghost_nw as u64,
                            ghost_ne as u64,
                            ghost_sw as u64,
                            ghost_se as u64,
                        )
                    }
                };
                (changed, border, has_live)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                unreachable!("NEON backend selected on non-aarch64 target");
            }
        } else {
            let ghost = GhostZone {
                north: ghost_north,
                south: ghost_south,
                west: ghost_west,
                east: ghost_east,
                nw: ghost_nw,
                ne: ghost_ne,
                sw: ghost_sw,
                se: ghost_se,
            };
            let (changed, border, has_live) = unsafe {
                advance_core_const::<CORE_BACKEND, ASSUME_CHANGED>(
                    current,
                    next,
                    &ghost,
                    force_store,
                )
            };
            (changed, border, has_live)
        }
    };
    let prune_ready = !has_live && missing_mask == MISSING_ALL_NEIGHBORS;
    let live_mask = border.live_mask();

    unsafe {
        *next_borders_north_ptr.add(idx) = border.north;
        *next_borders_south_ptr.add(idx) = border.south;
        *next_borders_west_ptr.add(idx) = border.west;
        *next_borders_east_ptr.add(idx) = border.east;
        *next_live_masks_ptr.add(idx) = live_mask;
    }

    let neighbor_influence_mask = if TRACK_NEIGHBOR_INFLUENCE {
        if changed {
            let prev_north = unsafe { *borders_north_read_ptr.add(idx) };
            let prev_south = unsafe { *borders_south_read_ptr.add(idx) };
            let prev_west = unsafe { *borders_west_read_ptr.add(idx) };
            let prev_east = unsafe { *borders_east_read_ptr.add(idx) };
            neighbor_influence_mask_from_borders(
                prev_north, prev_south, prev_west, prev_east, &border,
            )
        } else {
            0
        }
    } else if ASSUME_CHANGED {
        0
    } else {
        no_track_hint(changed, prune_ready)
    };

    unsafe {
        (*meta_slot).update_after_step(changed, has_live);
    }

    TileAdvanceResult::new(
        changed,
        has_live,
        missing_mask,
        live_mask,
        neighbor_influence_mask,
        prune_ready,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_scalar_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_SCALAR }, false, true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_scalar_no_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_SCALAR }, false, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_avx2_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_AVX2 }, false, true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_avx2_no_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_AVX2 }, false, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_neon_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_NEON }, false, true>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_neon_no_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_NEON }, false, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_neon_assume_changed_no_track(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_NEON }, true, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_neon_no_track_fast(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_NEON }, false, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
            idx,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn advance_tile_fused_neon_assume_changed_no_track_fast(
    current_ptr: *const CellBuf,
    next_ptr: *mut CellBuf,
    meta_ptr: *mut TileMeta,
    next_borders_north_ptr: *mut u64,
    next_borders_south_ptr: *mut u64,
    next_borders_west_ptr: *mut u64,
    next_borders_east_ptr: *mut u64,
    borders_north_read_ptr: *const u64,
    borders_south_read_ptr: *const u64,
    borders_west_read_ptr: *const u64,
    borders_east_read_ptr: *const u64,
    neighbors_ptr: *const [NeighborIdx; 8],
    live_masks_read_ptr: *const u8,
    next_live_masks_ptr: *mut u8,
    idx: usize,
) -> TileAdvanceResult {
    unsafe {
        advance_tile_fused_impl::<{ CORE_BACKEND_NEON }, true, false>(
            current_ptr,
            next_ptr,
            meta_ptr,
            next_borders_north_ptr,
            next_borders_south_ptr,
            next_borders_west_ptr,
            next_borders_east_ptr,
            borders_north_read_ptr,
            borders_south_read_ptr,
            borders_west_read_ptr,
            borders_east_read_ptr,
            neighbors_ptr,
            live_masks_read_ptr,
            next_live_masks_ptr,
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

    meta.update_after_step(changed, has_live);

    changed
}

#[cfg(test)]
mod tests {
    use super::{
        BorderData, CellBuf, GhostZone, TILE_SIZE, TileMeta, advance_core_empty,
        advance_core_scalar, advance_tile_fused_scalar_no_track, ghost_bit, ghost_is_empty,
        ghost_is_empty_from_live_masks, ghost_is_empty_from_live_masks_ptr, tile_is_empty,
    };
    use crate::turbolife::tile::{MISSING_ALL_NEIGHBORS, NeighborIdx};

    #[cfg(target_arch = "x86_64")]
    use super::advance_core_avx2;
    #[cfg(target_arch = "aarch64")]
    use super::{
        CORE_BACKEND_NEON, advance_core_const, advance_core_neon, advance_core_neon_assume_changed,
        advance_tile_fused_neon_assume_changed_no_track,
        advance_tile_fused_neon_assume_changed_no_track_fast, advance_tile_fused_neon_no_track,
        advance_tile_fused_neon_no_track_fast,
    };

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

    #[cfg(target_arch = "aarch64")]
    fn stable_block_tile() -> [u64; TILE_SIZE] {
        let mut tile = [0u64; TILE_SIZE];
        let block = (1u64 << 10) | (1u64 << 11);
        tile[10] = block;
        tile[11] = block;
        tile
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
        assert_eq!(border.live_mask(), 0);
    }

    #[test]
    fn scalar_fused_no_track_empty_ghost_requires_prune_verification() {
        let current = [CellBuf::empty(), CellBuf::empty()];
        let mut next = [CellBuf::empty(), CellBuf::empty()];
        let mut meta = [TileMeta::empty(), TileMeta::empty()];
        meta[0].missing_mask = 0;
        meta[0].set_has_live(false);
        meta[0].set_alt_phase_dirty(false);
        meta[1].set_has_live(false);
        meta[1].set_alt_phase_dirty(false);

        let mut next_borders_north = [u64::MAX; 2];
        let mut next_borders_south = [u64::MAX; 2];
        let mut next_borders_west = [u64::MAX; 2];
        let mut next_borders_east = [u64::MAX; 2];
        let borders_north_read = [0u64; 2];
        let borders_south_read = [0u64; 2];
        let borders_west_read = [0u64; 2];
        let borders_east_read = [0u64; 2];
        let neighbors = [[0 as NeighborIdx; 8], [0 as NeighborIdx; 8]];
        let live_masks_read = [0u8; 2];
        let mut next_live_masks = [u8::MAX; 2];

        let result = unsafe {
            advance_tile_fused_scalar_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(!result.changed);
        assert!(!result.has_live);
        assert!(!result.prune_ready);
        assert_eq!(next_live_masks[0], 0);
    }

    #[test]
    fn scalar_fused_no_track_dying_tile_with_empty_ghost_requires_prune_verification() {
        let mut current = [CellBuf::empty(), CellBuf::empty()];
        current[0].0[7] = 1u64 << 13;
        let mut next = [CellBuf::empty(), CellBuf::empty()];
        let mut meta = [TileMeta::empty(), TileMeta::empty()];
        meta[0].missing_mask = 0;
        meta[0].set_has_live(true);
        meta[0].set_alt_phase_dirty(false);
        meta[1].set_has_live(false);
        meta[1].set_alt_phase_dirty(false);

        let mut next_borders_north = [u64::MAX; 2];
        let mut next_borders_south = [u64::MAX; 2];
        let mut next_borders_west = [u64::MAX; 2];
        let mut next_borders_east = [u64::MAX; 2];
        let borders_north_read = [0u64; 2];
        let borders_south_read = [0u64; 2];
        let borders_west_read = [0u64; 2];
        let borders_east_read = [0u64; 2];
        let neighbors = [[0 as NeighborIdx; 8], [0 as NeighborIdx; 8]];
        let live_masks_read = [0u8; 2];
        let mut next_live_masks = [u8::MAX; 2];

        let result = unsafe {
            advance_tile_fused_scalar_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(result.changed);
        assert!(!result.has_live);
        assert!(!result.prune_ready);
    }

    #[test]
    fn scalar_fused_no_track_isolated_dying_tile_sets_prune_ready() {
        let mut current = [CellBuf::empty(), CellBuf::empty()];
        current[0].0[7] = 1u64 << 13;
        let mut next = [CellBuf::empty(), CellBuf::empty()];
        let mut meta = [TileMeta::empty(), TileMeta::empty()];
        meta[0].missing_mask = MISSING_ALL_NEIGHBORS;
        meta[0].set_has_live(true);
        meta[0].set_alt_phase_dirty(false);
        meta[1].set_has_live(false);
        meta[1].set_alt_phase_dirty(false);

        let mut next_borders_north = [u64::MAX; 2];
        let mut next_borders_south = [u64::MAX; 2];
        let mut next_borders_west = [u64::MAX; 2];
        let mut next_borders_east = [u64::MAX; 2];
        let borders_north_read = [0u64; 2];
        let borders_south_read = [0u64; 2];
        let borders_west_read = [0u64; 2];
        let borders_east_read = [0u64; 2];
        let neighbors = [[0 as NeighborIdx; 8], [0 as NeighborIdx; 8]];
        let live_masks_read = [0u8; 2];
        let mut next_live_masks = [u8::MAX; 2];

        let result = unsafe {
            advance_tile_fused_scalar_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(result.changed);
        assert!(!result.has_live);
        assert!(result.prune_ready);
    }

    #[test]
    fn scalar_fused_no_track_empty_ghost_can_hide_future_neighbor_border_births() {
        let mut current = [CellBuf::empty(), CellBuf::empty(), CellBuf::empty()];
        // Horizontal triplet one row above the south edge. This births a live
        // cell on the south edge next generation without any current south-edge
        // activity.
        current[2].0[1] = (1u64 << 10) | (1u64 << 11) | (1u64 << 12);

        let mut next = [CellBuf::empty(), CellBuf::empty(), CellBuf::empty()];
        let mut meta = [TileMeta::empty(), TileMeta::empty(), TileMeta::empty()];
        meta[1].missing_mask = MISSING_ALL_NEIGHBORS & !(1 << 0);
        meta[1].set_has_live(false);
        meta[2].missing_mask = MISSING_ALL_NEIGHBORS & !(1 << 1);
        meta[2].set_has_live(true);
        for slot in &mut meta {
            slot.set_alt_phase_dirty(false);
        }

        let mut next_borders_north = [u64::MAX; 3];
        let mut next_borders_south = [u64::MAX; 3];
        let mut next_borders_west = [u64::MAX; 3];
        let mut next_borders_east = [u64::MAX; 3];
        let borders_north_read = [0u64; 3];
        let borders_south_read = [0u64; 3];
        let borders_west_read = [0u64; 3];
        let borders_east_read = [0u64; 3];
        let mut neighbors = [[0 as NeighborIdx; 8]; 3];
        neighbors[1][0] = 2;
        neighbors[2][1] = 1;
        let live_masks_read = [0u8; 3];
        let mut next_live_masks = [u8::MAX; 3];

        let target_result = unsafe {
            advance_tile_fused_scalar_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                1,
            )
        };
        let north_result = unsafe {
            advance_tile_fused_scalar_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                2,
            )
        };

        assert!(!target_result.changed);
        assert!(!target_result.has_live);
        assert!(!target_result.prune_ready);

        assert!(north_result.changed);
        assert!(north_result.has_live);
        assert_ne!(next_live_masks[2] & super::LIVE_S, 0);
        assert!(!unsafe {
            ghost_is_empty_from_live_masks_ptr(next_live_masks.as_ptr(), &neighbors[1])
        });
    }

    #[test]
    fn corner_birth_table_matches_three_neighbor_rule() {
        for bits in 0u64..32 {
            let expected = (bits.count_ones() == 3) as u64;
            assert_eq!(super::birth_from_count5(bits), expected);
        }
    }

    #[test]
    fn empty_core_matches_scalar_for_random_ghosts() {
        let current = [0u64; TILE_SIZE];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFACE_B00C_5566_7788);

        for _ in 0..4096 {
            let ghost = random_ghost(&mut rng);
            let mut next_scalar = [0u64; TILE_SIZE];
            let mut next_empty = [0u64; TILE_SIZE];

            let scalar = advance_core_scalar(&current, &mut next_scalar, &ghost);
            let empty = advance_core_empty(&mut next_empty, &ghost);

            assert_eq!(next_scalar, next_empty);
            assert_eq!(scalar.0, empty.0);
            assert_eq!(scalar.2, empty.2);
            assert_eq!(scalar.1.north, empty.1.north);
            assert_eq!(scalar.1.south, empty.1.south);
            assert_eq!(scalar.1.west, empty.1.west);
            assert_eq!(scalar.1.east, empty.1.east);
            assert_eq!(scalar.1.live_mask(), empty.1.live_mask());
        }
    }

    #[test]
    fn empty_core_clears_stale_next_buffer_rows() {
        let current = [0u64; TILE_SIZE];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0DDC_0FFE_EE12_34AA);

        for _ in 0..1024 {
            let ghost = random_ghost(&mut rng);
            let mut next_scalar = random_tile(&mut rng);
            let mut next_empty = random_tile(&mut rng);

            let scalar = advance_core_scalar(&current, &mut next_scalar, &ghost);
            let empty = advance_core_empty(&mut next_empty, &ghost);

            assert_eq!(next_scalar, next_empty);
            assert_eq!(scalar.0, empty.0);
            assert_eq!(scalar.2, empty.2);
            assert_eq!(scalar.1.live_mask(), empty.1.live_mask());
        }
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
    fn ghost_live_masks_empty_check_matches_ghost_bits() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE_F00D_1234_5678);
        for _ in 0..2048 {
            let ghost = random_ghost(&mut rng);

            let north = BorderData::from_edges(0, ghost.north, 0, 0);
            let south = BorderData::from_edges(ghost.south, 0, 0, 0);
            let west = BorderData::from_edges(0, 0, 0, ghost.west);
            let east = BorderData::from_edges(0, 0, ghost.east, 0);
            let nw = BorderData::from_edges(0, (ghost.nw as u64) << 63, 0, 0);
            let ne = BorderData::from_edges(0, ghost.ne as u64, 0, 0);
            let sw = BorderData::from_edges((ghost.sw as u64) << 63, 0, 0, 0);
            let se = BorderData::from_edges(ghost.se as u64, 0, 0, 0);

            let from_masks = ghost_is_empty_from_live_masks([
                north.live_mask(),
                south.live_mask(),
                west.live_mask(),
                east.live_mask(),
                nw.live_mask(),
                ne.live_mask(),
                sw.live_mask(),
                se.live_mask(),
            ]);

            assert_eq!(from_masks, ghost_is_empty(&ghost));
        }
    }

    #[test]
    fn ghost_live_masks_ptr_check_matches_array_variant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11C_E7A5_900D_1234);
        for _ in 0..2048 {
            let mut live_masks = [0u8; 97];
            for mask in live_masks.iter_mut() {
                *mask = (rng.next_u64() & 0xFF) as u8;
            }
            // Index 0 is the NO_NEIGHBOR sentinel and must stay empty.
            live_masks[0] = 0;

            let mut neighbors = [0 as NeighborIdx; 8];
            for ni in neighbors.iter_mut() {
                *ni = (rng.next_u64() as usize % live_masks.len()) as NeighborIdx;
            }

            let from_array = ghost_is_empty_from_live_masks([
                live_masks[neighbors[0] as usize],
                live_masks[neighbors[1] as usize],
                live_masks[neighbors[2] as usize],
                live_masks[neighbors[3] as usize],
                live_masks[neighbors[4] as usize],
                live_masks[neighbors[5] as usize],
                live_masks[neighbors[6] as usize],
                live_masks[neighbors[7] as usize],
            ]);
            let from_ptr =
                unsafe { ghost_is_empty_from_live_masks_ptr(live_masks.as_ptr(), &neighbors) };

            assert_eq!(from_ptr, from_array);
        }
    }

    #[test]
    fn tile_is_empty_detects_live_rows() {
        let mut tile = [0u64; TILE_SIZE];
        assert!(tile_is_empty(&tile));

        tile[37] = 1 << 11;
        assert!(!tile_is_empty(&tile));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_randomized() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD00D_F00D_A11C_0DE5);
        for _ in 0..2048 {
            let current = random_tile(&mut rng);
            let ghost = random_ghost(&mut rng);
            let mut next_scalar = [0u64; TILE_SIZE];
            let mut next_neon = [0u64; TILE_SIZE];

            let scalar = advance_core_scalar(&current, &mut next_scalar, &ghost);
            let neon = unsafe { advance_core_neon(&current, &mut next_neon, &ghost, true) };

            assert_eq!(next_scalar, next_neon);
            assert_eq!(scalar.0, neon.0);
            assert_eq!(scalar.2, neon.2);
            assert_eq!(scalar.1.north, neon.1.north);
            assert_eq!(scalar.1.south, neon.1.south);
            assert_eq!(scalar.1.west, neon.1.west);
            assert_eq!(scalar.1.east, neon.1.east);
            assert_eq!(scalar.1.live_mask(), neon.1.live_mask());
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_assume_changed_preserves_outputs() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFACE_0000_A11C_0DE5);
        for _ in 0..2048 {
            let current = random_tile(&mut rng);
            let ghost = random_ghost(&mut rng);
            let mut next_neon = [0u64; TILE_SIZE];
            let mut next_assume = [0u64; TILE_SIZE];

            let neon = unsafe { advance_core_neon(&current, &mut next_neon, &ghost, true) };
            let assume =
                unsafe { advance_core_neon_assume_changed(&current, &mut next_assume, &ghost) };

            assert_eq!(next_neon, next_assume);
            assert!(assume.0);
            assert_eq!(neon.2, assume.2);
            assert_eq!(neon.1.north, assume.1.north);
            assert_eq!(neon.1.south, assume.1.south);
            assert_eq!(neon.1.west, assume.1.west);
            assert_eq!(neon.1.east, assume.1.east);
            assert_eq!(neon.1.live_mask(), assume.1.live_mask());
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_const_backend_respects_force_store_flag() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let current = [0u64; TILE_SIZE];
        let ghost = GhostZone::default();
        let mut next = [u64::MAX; TILE_SIZE];

        let (changed, border, has_live) = unsafe {
            advance_core_const::<{ CORE_BACKEND_NEON }, false>(&current, &mut next, &ghost, false)
        };

        assert!(!changed);
        assert!(!has_live);
        assert_eq!(border.north, 0);
        assert_eq!(border.south, 0);
        assert_eq!(border.west, 0);
        assert_eq!(border.east, 0);
        assert_eq!(next, [u64::MAX; TILE_SIZE]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_no_track_writes_when_alt_phase_is_clean() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut current = [CellBuf::empty()];
        current[0].0 = stable_block_tile();
        let mut next = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta = [TileMeta::empty()];
        meta[0].set_has_live(true);
        meta[0].set_alt_phase_dirty(false);

        let mut next_borders_north = [u64::MAX];
        let mut next_borders_south = [u64::MAX];
        let mut next_borders_west = [u64::MAX];
        let mut next_borders_east = [u64::MAX];
        let borders_north_read = [0u64];
        let borders_south_read = [0u64];
        let borders_west_read = [0u64];
        let borders_east_read = [0u64];
        let neighbors = [[0 as NeighborIdx; 8]];
        let live_masks_read = [0u8];
        let mut next_live_masks = [u8::MAX];

        let result = unsafe {
            advance_tile_fused_neon_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(!result.changed);
        assert!(result.has_live);
        assert_eq!(result.live_mask, 0);
        assert_eq!(next_borders_north[0], 0);
        assert_eq!(next_borders_south[0], 0);
        assert_eq!(next_borders_west[0], 0);
        assert_eq!(next_borders_east[0], 0);
        assert_eq!(next_live_masks[0], 0);
        assert_eq!(next[0].0, current[0].0);
        assert!(!meta[0].alt_phase_dirty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_no_track_writes_when_alt_phase_is_dirty() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut current = [CellBuf::empty()];
        current[0].0 = stable_block_tile();
        let mut next = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta = [TileMeta::empty()];
        meta[0].set_has_live(true);
        meta[0].set_alt_phase_dirty(true);

        let mut next_borders_north = [u64::MAX];
        let mut next_borders_south = [u64::MAX];
        let mut next_borders_west = [u64::MAX];
        let mut next_borders_east = [u64::MAX];
        let borders_north_read = [0u64];
        let borders_south_read = [0u64];
        let borders_west_read = [0u64];
        let borders_east_read = [0u64];
        let neighbors = [[0 as NeighborIdx; 8]];
        let live_masks_read = [0u8];
        let mut next_live_masks = [u8::MAX];

        let result = unsafe {
            advance_tile_fused_neon_no_track(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(!result.changed);
        assert!(result.has_live);
        assert_eq!(result.live_mask, 0);
        assert_eq!(next_borders_north[0], 0);
        assert_eq!(next_borders_south[0], 0);
        assert_eq!(next_borders_west[0], 0);
        assert_eq!(next_borders_east[0], 0);
        assert_eq!(next_live_masks[0], 0);
        assert_eq!(next[0].0, current[0].0);
        assert!(!meta[0].alt_phase_dirty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_no_track_fast_writes_border_cache_when_alt_phase_is_clean() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut current = [CellBuf::empty()];
        current[0].0 = stable_block_tile();
        let mut next = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta = [TileMeta::empty()];
        meta[0].set_has_live(true);
        meta[0].set_alt_phase_dirty(false);

        let mut next_borders_north = [u64::MAX];
        let mut next_borders_south = [u64::MAX];
        let mut next_borders_west = [u64::MAX];
        let mut next_borders_east = [u64::MAX];
        let borders_north_read = [0u64];
        let borders_south_read = [0u64];
        let borders_west_read = [0u64];
        let borders_east_read = [0u64];
        let neighbors = [[0 as NeighborIdx; 8]];
        let live_masks_read = [0u8];
        let mut next_live_masks = [u8::MAX];

        let result = unsafe {
            advance_tile_fused_neon_no_track_fast(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(!result.changed);
        assert!(result.has_live);
        assert_eq!(result.live_mask, 0);
        assert_eq!(next_borders_north[0], 0);
        assert_eq!(next_borders_south[0], 0);
        assert_eq!(next_borders_west[0], 0);
        assert_eq!(next_borders_east[0], 0);
        assert_eq!(next_live_masks[0], 0);
        assert_eq!(next[0].0, current[0].0);
        assert!(!meta[0].alt_phase_dirty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_no_track_fast_writes_when_alt_phase_is_dirty() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut current = [CellBuf::empty()];
        current[0].0 = stable_block_tile();
        let mut next = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta = [TileMeta::empty()];
        meta[0].set_has_live(true);
        meta[0].set_alt_phase_dirty(true);

        let mut next_borders_north = [u64::MAX];
        let mut next_borders_south = [u64::MAX];
        let mut next_borders_west = [u64::MAX];
        let mut next_borders_east = [u64::MAX];
        let borders_north_read = [0u64];
        let borders_south_read = [0u64];
        let borders_west_read = [0u64];
        let borders_east_read = [0u64];
        let neighbors = [[0 as NeighborIdx; 8]];
        let live_masks_read = [0u8];
        let mut next_live_masks = [u8::MAX];

        let result = unsafe {
            advance_tile_fused_neon_no_track_fast(
                current.as_ptr(),
                next.as_mut_ptr(),
                meta.as_mut_ptr(),
                next_borders_north.as_mut_ptr(),
                next_borders_south.as_mut_ptr(),
                next_borders_west.as_mut_ptr(),
                next_borders_east.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks.as_mut_ptr(),
                0,
            )
        };

        assert!(!result.changed);
        assert!(result.has_live);
        assert_eq!(result.live_mask, 0);
        assert_eq!(next_borders_north[0], 0);
        assert_eq!(next_borders_south[0], 0);
        assert_eq!(next_borders_west[0], 0);
        assert_eq!(next_borders_east[0], 0);
        assert_eq!(next_live_masks[0], 0);
        assert_eq!(next[0].0, current[0].0);
        assert!(!meta[0].alt_phase_dirty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_no_track_fast_matches_reference_with_live_ghost_when_dirty() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut current_ref = [CellBuf::empty()];
        current_ref[0].0 = stable_block_tile();
        let mut next_ref = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta_ref = [TileMeta::empty()];
        meta_ref[0].set_has_live(true);
        meta_ref[0].set_alt_phase_dirty(true);
        let mut next_borders_north_ref = [u64::MAX];
        let mut next_borders_south_ref = [u64::MAX];
        let mut next_borders_west_ref = [u64::MAX];
        let mut next_borders_east_ref = [u64::MAX];
        let borders_north_read = [0u64];
        let borders_south_read = [1u64 << 40];
        let borders_west_read = [0u64];
        let borders_east_read = [0u64];
        let neighbors = [[0 as NeighborIdx; 8]];
        let live_masks_read = [0u8];
        let mut next_live_masks_ref = [u8::MAX];

        let reference = unsafe {
            advance_tile_fused_neon_no_track(
                current_ref.as_ptr(),
                next_ref.as_mut_ptr(),
                meta_ref.as_mut_ptr(),
                next_borders_north_ref.as_mut_ptr(),
                next_borders_south_ref.as_mut_ptr(),
                next_borders_west_ref.as_mut_ptr(),
                next_borders_east_ref.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks_ref.as_mut_ptr(),
                0,
            )
        };

        let mut current_fast = [CellBuf::empty()];
        current_fast[0].0 = stable_block_tile();
        let mut next_fast = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta_fast = [TileMeta::empty()];
        meta_fast[0].set_has_live(true);
        meta_fast[0].set_alt_phase_dirty(true);
        let mut next_borders_north_fast = [u64::MAX];
        let mut next_borders_south_fast = [u64::MAX];
        let mut next_borders_west_fast = [u64::MAX];
        let mut next_borders_east_fast = [u64::MAX];
        let mut next_live_masks_fast = [u8::MAX];

        let fast = unsafe {
            advance_tile_fused_neon_no_track_fast(
                current_fast.as_ptr(),
                next_fast.as_mut_ptr(),
                meta_fast.as_mut_ptr(),
                next_borders_north_fast.as_mut_ptr(),
                next_borders_south_fast.as_mut_ptr(),
                next_borders_west_fast.as_mut_ptr(),
                next_borders_east_fast.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks_fast.as_mut_ptr(),
                0,
            )
        };

        assert!(!reference.changed);
        assert_eq!(fast.changed, reference.changed);
        assert_eq!(fast.has_live, reference.has_live);
        assert_eq!(fast.live_mask, reference.live_mask);
        assert_eq!(
            fast.neighbor_influence_mask,
            reference.neighbor_influence_mask
        );
        assert_eq!(next_fast[0].0, next_ref[0].0);
        assert_eq!(next_borders_north_fast[0], next_borders_north_ref[0]);
        assert_eq!(next_borders_south_fast[0], next_borders_south_ref[0]);
        assert_eq!(next_borders_west_fast[0], next_borders_west_ref[0]);
        assert_eq!(next_borders_east_fast[0], next_borders_east_ref[0]);
        assert_eq!(next_live_masks_fast[0], next_live_masks_ref[0]);
        assert_eq!(
            meta_fast[0].alt_phase_dirty(),
            meta_ref[0].alt_phase_dirty()
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_assume_changed_fast_matches_reference() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut current_ref = [CellBuf::empty()];
        current_ref[0].0 = stable_block_tile();
        let mut next_ref = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta_ref = [TileMeta::empty()];
        meta_ref[0].set_has_live(true);
        meta_ref[0].set_alt_phase_dirty(false);
        let mut next_borders_north_ref = [u64::MAX];
        let mut next_borders_south_ref = [u64::MAX];
        let mut next_borders_west_ref = [u64::MAX];
        let mut next_borders_east_ref = [u64::MAX];
        let borders_north_read = [0u64];
        let borders_south_read = [0u64];
        let borders_west_read = [0u64];
        let borders_east_read = [0u64];
        let neighbors = [[0 as NeighborIdx; 8]];
        let live_masks_read = [0u8];
        let mut next_live_masks_ref = [u8::MAX];

        let reference = unsafe {
            advance_tile_fused_neon_assume_changed_no_track(
                current_ref.as_ptr(),
                next_ref.as_mut_ptr(),
                meta_ref.as_mut_ptr(),
                next_borders_north_ref.as_mut_ptr(),
                next_borders_south_ref.as_mut_ptr(),
                next_borders_west_ref.as_mut_ptr(),
                next_borders_east_ref.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks_ref.as_mut_ptr(),
                0,
            )
        };

        let mut current_fast = [CellBuf::empty()];
        current_fast[0].0 = stable_block_tile();
        let mut next_fast = [CellBuf([u64::MAX; TILE_SIZE])];
        let mut meta_fast = [TileMeta::empty()];
        meta_fast[0].set_has_live(true);
        meta_fast[0].set_alt_phase_dirty(false);
        let mut next_borders_north_fast = [u64::MAX];
        let mut next_borders_south_fast = [u64::MAX];
        let mut next_borders_west_fast = [u64::MAX];
        let mut next_borders_east_fast = [u64::MAX];
        let mut next_live_masks_fast = [u8::MAX];

        let fast = unsafe {
            advance_tile_fused_neon_assume_changed_no_track_fast(
                current_fast.as_ptr(),
                next_fast.as_mut_ptr(),
                meta_fast.as_mut_ptr(),
                next_borders_north_fast.as_mut_ptr(),
                next_borders_south_fast.as_mut_ptr(),
                next_borders_west_fast.as_mut_ptr(),
                next_borders_east_fast.as_mut_ptr(),
                borders_north_read.as_ptr(),
                borders_south_read.as_ptr(),
                borders_west_read.as_ptr(),
                borders_east_read.as_ptr(),
                neighbors.as_ptr(),
                live_masks_read.as_ptr(),
                next_live_masks_fast.as_mut_ptr(),
                0,
            )
        };

        assert_eq!(fast.changed, reference.changed);
        assert_eq!(fast.has_live, reference.has_live);
        assert_eq!(fast.live_mask, reference.live_mask);
        assert_eq!(
            fast.neighbor_influence_mask,
            reference.neighbor_influence_mask
        );
        assert_eq!(next_fast[0].0, next_ref[0].0);
        assert_eq!(next_borders_north_fast[0], next_borders_north_ref[0]);
        assert_eq!(next_borders_south_fast[0], next_borders_south_ref[0]);
        assert_eq!(next_borders_west_fast[0], next_borders_west_ref[0]);
        assert_eq!(next_borders_east_fast[0], next_borders_east_ref[0]);
        assert_eq!(next_live_masks_fast[0], next_live_masks_ref[0]);
        assert_eq!(
            meta_fast[0].alt_phase_dirty(),
            meta_ref[0].alt_phase_dirty()
        );
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
                assert_eq!(scalar.1.live_mask(), avx.1.live_mask());
            }
        }
    }
}
