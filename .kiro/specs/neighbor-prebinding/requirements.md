# Requirements Document

## Introduction

The TurboLife engine uses a tiled Game of Life implementation with 64×64 tiles stored in a dense arena. Each generation, tiles must read border data from their 8 neighbors to fill ghost zones before computing the next state. Currently, neighbor lookup happens via `FxHashMap<(i64, i64), TileIdx>` — 8 hash lookups per active tile per generation. The neighbor pre-binding feature eliminates these hash lookups from the per-generation hot path by caching neighbor indices directly on each tile's metadata, maintaining them during allocation and release.

## Glossary

- **TileArena**: The arena allocator that manages tile storage, allocation, recycling, and coordinate-to-index mapping.
- **TileMeta**: Mutable metadata associated with each tile slot, including ghost zones, activity flags, and population.
- **TileCells**: Immutable (during sync/compute) cell buffer storage including coordinates and pre-cached border data.
- **TileIdx**: A newtype wrapper around `u32` used as an index into the arena's parallel arrays.
- **BorderData**: Pre-cached border row/column/corner data extracted from a tile's current buffer after each generation.
- **GhostZone**: A 1-cell halo around a tile, filled from 8 neighbors' `BorderData` before the compute phase.
- **Neighbor_Array**: A fixed-size array of 8 `Option<TileIdx>` values representing a tile's N, S, W, E, NW, NE, SW, SE neighbors.
- **Direction**: An enumeration of the 8 cardinal and intercardinal directions (N, S, W, E, NW, NE, SW, SE) with a consistent index mapping.
- **Reverse_Direction**: The opposite direction for a given direction (e.g., North ↔ South, NE ↔ SW), used for bidirectional linking.
- **Hot_Path**: The per-generation execution path consisting of ghost zone synchronization and tile computation (step → sync → compute).

## Requirements

### Requirement 1: Neighbor Array Storage

**User Story:** As a developer, I want each tile to store direct references to its 8 neighbors, so that ghost zone filling can use array indexing instead of hash lookups.

#### Acceptance Criteria

1. THE Neighbor_Array SHALL store 8 `Option<TileIdx>` values representing the N, S, W, E, NW, NE, SW, SE neighbors of a tile.
2. THE Direction SHALL use a consistent index mapping: 0=North (tx, ty+1), 1=South (tx, ty-1), 2=West (tx-1, ty), 3=East (tx+1, ty), 4=NW (tx-1, ty+1), 5=NE (tx+1, ty+1), 6=SW (tx-1, ty-1), 7=SE (tx+1, ty-1).
3. WHEN a tile slot is unoccupied, THE Neighbor_Array SHALL contain `None` in all 8 positions.
4. THE Neighbor_Array SHALL be stored alongside existing tile metadata in TileMeta or in a new parallel array in TileArena.

### Requirement 2: Bidirectional Linking on Allocation

**User Story:** As a developer, I want neighbor links to be established automatically when a tile is allocated, so that the neighbor array is always consistent with the arena's coordinate map.

#### Acceptance Criteria

1. WHEN TileArena allocates a tile at coordinate (tx, ty), THE TileArena SHALL look up each of the 8 neighbor coordinates in `coord_to_idx` and store found neighbor indices in the new tile's Neighbor_Array.
2. WHEN TileArena allocates a tile at coordinate (tx, ty) and a neighbor tile exists at a neighboring coordinate, THE TileArena SHALL update that neighbor tile's Neighbor_Array to point back to the newly allocated tile in the corresponding Reverse_Direction slot.
3. WHEN TileArena allocates a tile at a coordinate that already has an allocated tile, THE TileArena SHALL return the existing TileIdx without modifying any Neighbor_Array entries.

### Requirement 3: Unlinking on Release

**User Story:** As a developer, I want neighbor links to be cleaned up automatically when a tile is released, so that no stale indices remain in any tile's neighbor array.

#### Acceptance Criteria

1. WHEN TileArena releases a tile, THE TileArena SHALL iterate over the released tile's Neighbor_Array and for each non-None neighbor, clear the Reverse_Direction slot in that neighbor's Neighbor_Array.
2. WHEN TileArena releases a tile, THE TileArena SHALL set all 8 entries in the released tile's Neighbor_Array to `None`.
3. WHEN TileArena releases a tile that is already unoccupied, THE TileArena SHALL perform no modifications to any Neighbor_Array.

### Requirement 4: Ghost Zone Fill via Direct Indexing

**User Story:** As a developer, I want ghost zone synchronization to use pre-bound neighbor indices for direct array access, so that the hot path avoids all hash map lookups.

#### Acceptance Criteria

1. WHEN filling a tile's ghost zone, THE sync module SHALL read neighbor `BorderData` using the tile's Neighbor_Array indices as direct offsets into the `cell_data` array, instead of looking up coordinates in `coord_to_idx`.
2. WHEN a tile's Neighbor_Array entry is `None` for a given direction, THE sync module SHALL use default (all-zero) `BorderData` for that direction.
3. THE sync module's `fill_ghost_zone_direct` function SHALL accept the Neighbor_Array (or a reference to the neighbor storage) instead of a reference to `coord_to_idx`.

### Requirement 5: Serial Path Ghost Zone Fill

**User Story:** As a developer, I want the serial execution path in the engine to also use pre-bound neighbor indices, so that both serial and parallel paths benefit from the optimization.

#### Acceptance Criteria

1. WHEN the engine uses the serial path (fewer than 32 active tiles), THE engine SHALL fill ghost zones using the tile's Neighbor_Array for direct array access instead of calling `arena.idx_at()` with neighbor coordinates.
2. THE serial path ghost zone fill SHALL produce identical GhostZone values as the previous HashMap-based implementation for any given tile configuration.

### Requirement 6: HashMap Retention for Non-Hot-Path Operations

**User Story:** As a developer, I want `coord_to_idx` to remain available for coordinate-based lookups outside the hot path, so that `set_cell`, `get_cell`, and allocation-time neighbor binding continue to work.

#### Acceptance Criteria

1. THE TileArena SHALL retain the `coord_to_idx` HashMap for use by `set_cell`, `get_cell`, `allocate` (neighbor binding), and activity set operations.
2. WHEN the engine executes the Hot_Path (ghost zone fill and tile computation), THE engine SHALL perform zero lookups against `coord_to_idx`.

### Requirement 7: Behavioral Preservation

**User Story:** As a developer, I want the optimization to be purely internal, so that all existing tests pass without modification and the engine produces identical results.

#### Acceptance Criteria

1. WHEN the engine runs any Game of Life pattern, THE engine SHALL produce identical cell states generation-by-generation as the pre-optimization implementation.
2. THE existing parity tests (tests/parity.rs) SHALL continue to pass, confirming TurboLife produces identical results to QuickLife.
3. THE existing unit tests (tests/turbolife.rs) SHALL continue to pass without modification.

### Requirement 8: Neighbor Array Consistency Invariant

**User Story:** As a developer, I want the neighbor array to always be consistent with the arena's coordinate map, so that I can trust the cached indices during the hot path.

#### Acceptance Criteria

1. FOR ALL occupied tiles A and B in the TileArena, IF tile A's Neighbor_Array contains tile B's index at direction D, THEN tile B's Neighbor_Array SHALL contain tile A's index at Reverse_Direction of D.
2. FOR ALL occupied tiles in the TileArena, IF a tile's Neighbor_Array contains `Some(idx)` at direction D, THEN the tile at `idx` SHALL be occupied and located at the expected coordinate offset for direction D.
3. FOR ALL occupied tiles in the TileArena, IF a tile's Neighbor_Array contains `None` at direction D, THEN no occupied tile SHALL exist at the expected coordinate offset for direction D in `coord_to_idx`.
