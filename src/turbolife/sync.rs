//! Ghost zone synchronization for TurboLife.
//!
//! Gathers border data from 8 neighbors into a `GhostZone` struct.
//! Reads from the compact `borders` array rather than full `TileCells`,
//! keeping the working set small.

use super::tile::{BorderData, Direction, GhostZone, Neighbors, TileIdx};

/// Gather the ghost zone for a single tile from its neighbors' borders.
#[inline]
pub fn gather_ghost_zone(
    idx: TileIdx,
    borders: &[BorderData],
    neighbors: &[Neighbors],
) -> GhostZone {
    let nb = &neighbors[idx.index()];

    #[inline(always)]
    fn read_u64(nb: &[Option<TileIdx>; 8], borders: &[BorderData], dir: usize, field: fn(&BorderData) -> u64) -> u64 {
        match nb[dir] {
            Some(ni) => field(&borders[ni.index()]),
            None => 0,
        }
    }

    #[inline(always)]
    fn read_bool(nb: &[Option<TileIdx>; 8], borders: &[BorderData], dir: usize, field: fn(&BorderData) -> bool) -> bool {
        match nb[dir] {
            Some(ni) => field(&borders[ni.index()]),
            None => false,
        }
    }

    GhostZone {
        north: read_u64(nb, borders, Direction::North.index(), |b| b.south),
        south: read_u64(nb, borders, Direction::South.index(), |b| b.north),
        west:  read_u64(nb, borders, Direction::West.index(),  |b| b.east),
        east:  read_u64(nb, borders, Direction::East.index(),  |b| b.west),
        nw:    read_bool(nb, borders, Direction::NW.index(),   |b| b.se),
        ne:    read_bool(nb, borders, Direction::NE.index(),   |b| b.sw),
        sw:    read_bool(nb, borders, Direction::SW.index(),   |b| b.ne),
        se:    read_bool(nb, borders, Direction::SE.index(),   |b| b.nw),
    }
}
