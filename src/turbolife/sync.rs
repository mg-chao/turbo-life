//! Ghost zone synchronization for TurboLife.
//!
//! Branchless sentinel-based gather: NO_NEIGHBOR indices are remapped to
//! the sentinel slot (index 0, always zeroed), eliminating all 8 branches
//! per tile on the hottest path.

use super::arena::SENTINEL_IDX;
use super::tile::{BorderData, GhostZone, Neighbors, TileIdx, NO_NEIGHBOR};

/// Remap NO_NEIGHBOR to the sentinel index.
#[inline(always)]
fn sentinel_or(raw: u32) -> usize {
    if raw == NO_NEIGHBOR { SENTINEL_IDX } else { raw as usize }
}

/// Gather the ghost zone for a single tile — branchless via sentinel slot.
#[inline(always)]
pub fn gather_ghost_zone(
    idx: TileIdx,
    borders: &[BorderData],
    neighbors: &[Neighbors],
) -> GhostZone {
    let nb = &neighbors[idx.index()];
    unsafe {
        let b = borders.as_ptr();
        let north = &*b.add(sentinel_or(nb[0]));
        let south = &*b.add(sentinel_or(nb[1]));
        let west  = &*b.add(sentinel_or(nb[2]));
        let east  = &*b.add(sentinel_or(nb[3]));
        let nw    = &*b.add(sentinel_or(nb[4]));
        let ne    = &*b.add(sentinel_or(nb[5]));
        let sw    = &*b.add(sentinel_or(nb[6]));
        let se    = &*b.add(sentinel_or(nb[7]));

        GhostZone {
            north: north.south,
            south: south.north,
            west:  west.east,
            east:  east.west,
            nw: nw.se(),
            ne: ne.sw(),
            sw: sw.ne(),
            se: se.nw(),
        }
    }
}
