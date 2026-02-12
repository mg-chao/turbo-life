//! Ghost zone synchronization for TurboLife.
//!
//! Branchless sentinel-based gather: NO_NEIGHBOR indices are remapped to
//! the sentinel slot (index 0, always zeroed), eliminating all 8 branches
//! per tile on the hottest path.

use super::arena::SENTINEL_IDX;
use super::tile::{BorderData, GhostZone, NO_NEIGHBOR, Neighbors, TileIdx};

/// Remap NO_NEIGHBOR to the sentinel index.
#[inline(always)]
fn sentinel_or(raw: u32) -> usize {
    if raw == NO_NEIGHBOR {
        SENTINEL_IDX
    } else {
        raw as usize
    }
}

/// Gather the ghost zone for a single tile using raw pointers.
///
/// # Safety
/// `borders_ptr` and `neighbors_ptr` must be valid for reads at `idx`
/// and all mapped neighbor indices (or sentinel index 0).
#[inline(always)]
pub unsafe fn gather_ghost_zone_raw(
    idx: usize,
    borders_ptr: *const BorderData,
    neighbors_ptr: *const Neighbors,
) -> GhostZone {
    debug_assert!(!borders_ptr.is_null());
    debug_assert!(!neighbors_ptr.is_null());

    unsafe {
        let nb = &*neighbors_ptr.add(idx);
        let north = &*borders_ptr.add(sentinel_or(nb[0]));
        let south = &*borders_ptr.add(sentinel_or(nb[1]));
        let west = &*borders_ptr.add(sentinel_or(nb[2]));
        let east = &*borders_ptr.add(sentinel_or(nb[3]));
        let nw = &*borders_ptr.add(sentinel_or(nb[4]));
        let ne = &*borders_ptr.add(sentinel_or(nb[5]));
        let sw = &*borders_ptr.add(sentinel_or(nb[6]));
        let se = &*borders_ptr.add(sentinel_or(nb[7]));

        GhostZone {
            north: north.south,
            south: south.north,
            west: west.east,
            east: east.west,
            nw: nw.se(),
            ne: ne.sw(),
            sw: sw.ne(),
            se: se.nw(),
        }
    }
}

/// Gather the ghost zone for a single tile — branchless via sentinel slot.
#[inline(always)]
#[allow(dead_code)]
pub fn gather_ghost_zone(
    idx: TileIdx,
    borders: &[BorderData],
    neighbors: &[Neighbors],
) -> GhostZone {
    unsafe { gather_ghost_zone_raw(idx.index(), borders.as_ptr(), neighbors.as_ptr()) }
}
