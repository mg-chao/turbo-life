//! Scalar bit-parallel kernel for TurboLife.
//!
//! Computes the next generation for a single tile using a full-adder chain.
//! Border extraction is fused into the main loop.
//! Works with split cell buffers (separate current/next vecs).

use super::tile::{BorderData, CellBuf, TileMeta, TILE_SIZE, POPULATION_UNKNOWN};

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
pub fn advance_core(
    current: &[u64; TILE_SIZE],
    next: &mut [u64; TILE_SIZE],
    ghost: &super::tile::GhostZone,
) -> (bool, BorderData, bool) {
    let mut changed = false;
    let mut has_live = false;
    let mut border_west = 0u64;
    let mut border_east = 0u64;

    for row in 0..TILE_SIZE {
        let row_above = if row == TILE_SIZE - 1 { ghost.north } else { current[row + 1] };
        let row_self = current[row];
        let row_below = if row == 0 { ghost.south } else { current[row - 1] };

        let ghost_w_above = if row == TILE_SIZE - 1 { ghost.nw } else { ghost_bit(ghost.west, row + 1) };
        let ghost_e_above = if row == TILE_SIZE - 1 { ghost.ne } else { ghost_bit(ghost.east, row + 1) };
        let ghost_w_self = ghost_bit(ghost.west, row);
        let ghost_e_self = ghost_bit(ghost.east, row);
        let ghost_w_below = if row == 0 { ghost.sw } else { ghost_bit(ghost.west, row - 1) };
        let ghost_e_below = if row == 0 { ghost.se } else { ghost_bit(ghost.east, row - 1) };

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
    if (next[63] & 1) != 0 { corners |= BorderData::CORNER_NW; }
    if ((next[63] >> 63) & 1) != 0 { corners |= BorderData::CORNER_NE; }
    if (next[0] & 1) != 0 { corners |= BorderData::CORNER_SW; }
    if ((next[0] >> 63) & 1) != 0 { corners |= BorderData::CORNER_SE; }

    let border = BorderData {
        north: next[63],
        south: next[0],
        west: border_west,
        east: border_east,
        corners,
    };

    (changed, border, has_live)
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
    ghost: &super::tile::GhostZone,
) -> bool {
    let current = unsafe { &(*current_ptr.add(idx)).0 };
    let next = unsafe { &mut (*next_ptr.add(idx)).0 };
    let meta = unsafe { &mut *meta_ptr.add(idx) };

    let (changed, border, has_live) = advance_core(current, next, ghost);

    unsafe { *next_borders_ptr.add(idx) = border; }

    meta.set_changed(changed);
    if changed {
        meta.population = POPULATION_UNKNOWN;
        meta.set_has_live(has_live);
    }

    changed
}
