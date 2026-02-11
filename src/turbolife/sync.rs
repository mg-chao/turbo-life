//! Ghost zone synchronization for TurboLife.
//!
//! Gathers border data from 8 neighbors into a `GhostZone` struct.
//! Reads from the compact `borders` array rather than full `TileCells`,
//! keeping the working set small.

use super::tile::{BorderData, Direction, GhostZone, Neighbors, TileIdx, NO_NEIGHBOR};

/// Gather the ghost zone for a single tile from its neighbors' borders.
#[inline]
pub fn gather_ghost_zone(
    idx: TileIdx,
    borders: &[BorderData],
    neighbors: &[Neighbors],
) -> GhostZone {
    let nb = &neighbors[idx.index()];
    let north_i = nb[Direction::North.index()];
    let south_i = nb[Direction::South.index()];
    let west_i = nb[Direction::West.index()];
    let east_i = nb[Direction::East.index()];
    let nw_i = nb[Direction::NW.index()];
    let ne_i = nb[Direction::NE.index()];
    let sw_i = nb[Direction::SW.index()];
    let se_i = nb[Direction::SE.index()];

    GhostZone {
        north: if north_i == NO_NEIGHBOR { 0 } else { borders[north_i as usize].south },
        south: if south_i == NO_NEIGHBOR { 0 } else { borders[south_i as usize].north },
        west: if west_i == NO_NEIGHBOR { 0 } else { borders[west_i as usize].east },
        east: if east_i == NO_NEIGHBOR { 0 } else { borders[east_i as usize].west },
        nw: if nw_i == NO_NEIGHBOR { false } else { borders[nw_i as usize].se },
        ne: if ne_i == NO_NEIGHBOR { false } else { borders[ne_i as usize].sw },
        sw: if sw_i == NO_NEIGHBOR { false } else { borders[sw_i as usize].ne },
        se: if se_i == NO_NEIGHBOR { false } else { borders[se_i as usize].nw },
    }
}
