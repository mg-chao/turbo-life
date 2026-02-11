//! Ghost zone synchronization for TurboLife.
//!
//! Branchless sentinel-based gather: NO_NEIGHBOR is encoded directly as the
//! sentinel slot index (0, always zeroed), eliminating all 8 branches per tile
//! on the hottest path.

use super::tile::{BorderData, GhostZone, Neighbors, TileIdx};

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
        let north = &*borders_ptr.add(nb[0] as usize);
        let south = &*borders_ptr.add(nb[1] as usize);
        let west = &*borders_ptr.add(nb[2] as usize);
        let east = &*borders_ptr.add(nb[3] as usize);
        let nw = &*borders_ptr.add(nb[4] as usize);
        let ne = &*borders_ptr.add(nb[5] as usize);
        let sw = &*borders_ptr.add(nb[6] as usize);
        let se = &*borders_ptr.add(nb[7] as usize);

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
