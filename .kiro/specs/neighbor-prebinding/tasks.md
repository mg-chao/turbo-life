# Implementation Plan: Neighbor Pre-Binding

## Overview

Incrementally add neighbor pre-binding to the TurboLife engine. Each task builds on the previous one, starting with data structures, then allocation/release logic, then hot-path rewiring, and finally testing. The `coord_to_idx` HashMap remains for cold-path operations; only the per-generation ghost zone fill is changed to use direct array indexing.

## Tasks

- [x] 1. Add Direction enum, Neighbors type, and neighbor storage to TileArena
  - [x] 1.1 Add `Direction` enum to `src/turbolife/tile.rs`
    - Define `#[repr(u8)]` enum with 8 variants: North=0, South=1, West=2, East=3, NW=4, NE=5, SW=6, SE=7
    - Implement `ALL` constant array, `offset()`, `reverse()`, and `index()` methods
    - Add `pub type Neighbors = [Option<TileIdx>; 8]` and `pub const EMPTY_NEIGHBORS: Neighbors = [None; 8]`
    - _Requirements: 1.1, 1.2_
  - [x] 1.2 Add `neighbors: Vec<Neighbors>` parallel array to `TileArena` in `src/turbolife/arena.rs`
    - Add the field to the struct and initialize it in `TileArena::new()`
    - Add `#[inline] pub fn neighbors(&self, idx: TileIdx) -> &Neighbors` accessor
    - _Requirements: 1.4_

- [x] 2. Implement bidirectional linking in allocate() and unlinking in release()
  - [x] 2.1 Update `TileArena::allocate()` in `src/turbolife/arena.rs`
    - After inserting into `coord_to_idx`, iterate `Direction::ALL`
    - For each direction, look up the neighbor coord in `coord_to_idx`
    - If found, set `neighbors[idx][dir] = Some(neighbor_idx)` and `neighbors[neighbor_idx][reverse(dir)] = Some(idx)`
    - For recycled slots, reset `neighbors[recycled] = EMPTY_NEIGHBORS` before binding
    - For new slots, push `EMPTY_NEIGHBORS` to the neighbors vec
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 2.2 Update `TileArena::release()` in `src/turbolife/arena.rs`
    - Before clearing the slot, iterate the tile's neighbor array
    - For each `Some(neighbor_idx)` at direction `d`, set `neighbors[neighbor_idx][reverse(d)] = None`
    - Set `neighbors[idx] = EMPTY_NEIGHBORS`
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ]* 2.3 Write property test for neighbor consistency invariant
    - **Property 1: Neighbor Consistency Invariant**
    - Use `proptest` to generate random sequences of allocate/release operations on coordinates in range -3..3
    - After each operation, verify: bidirectional symmetry, coordinate consistency, completeness, unoccupied-clean
    - Add `proptest` as a dev-dependency in Cargo.toml
    - **Validates: Requirements 8.1, 8.2, 8.3, 1.3, 2.1, 2.2, 3.1, 3.2**
  - [ ]* 2.4 Write property test for allocation idempotence
    - **Property 2: Allocation Idempotence**
    - Generate random arena states, re-allocate at an existing coordinate, verify returned index and unchanged neighbors
    - **Validates: Requirements 2.3**

- [x] 3. Checkpoint — Verify arena linking
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Rewire ghost zone fill to use neighbor arrays
  - [x] 4.1 Rewrite `fill_ghost_zone_direct()` in `src/turbolife/sync.rs`
    - Change signature: replace `coord_to_idx: &FxHashMap<(i64, i64), TileIdx>` with `neighbors: &[Neighbors]`
    - Read `BorderData` via `neighbors[idx.index()][dir].map(|ni| cell_data[ni.index()].border).unwrap_or_default()`
    - Remove the `use rustc_hash::FxHashMap` import if no longer needed
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 4.2 Update the parallel path in `engine.rs` `step()` method
    - Pass `&self.arena.neighbors` instead of `&self.arena.coord_to_idx` to `fill_ghost_zone_direct`
    - _Requirements: 4.1, 6.2_
  - [x] 4.3 Update the serial path in `engine.rs` `step()` method
    - Replace the inline `border_of` closure that calls `self.arena.idx_at()` with neighbor-array-based reads using `self.arena.neighbors[idx.index()]`
    - _Requirements: 5.1, 5.2_
  - [ ]* 4.4 Write property test for ghost zone equivalence
    - **Property 3: Ghost Zone Equivalence**
    - Generate random arena states with random cell/border data
    - Compare ghost zone output from neighbor-array method vs HashMap method
    - **Validates: Requirements 5.2, 4.1, 4.2**

- [x] 5. Checkpoint — Verify hot-path rewiring
  - Ensure all existing tests pass (`cargo test`), ask the user if questions arise.

- [ ] 6. Behavioral preservation validation
  - [ ]* 6.1 Write property test for behavioral preservation
    - **Property 4: Behavioral Preservation (Simulation Round-Trip)**
    - Generate random initial patterns on small grids, run TurboLife for N steps, compare against naive reference implementation
    - **Validates: Requirements 7.1, 7.2, 7.3**
  - [ ]* 6.2 Write unit tests for Direction, single-tile allocation, adjacent linking, and release unlinking
    - Test Direction offset/reverse correctness (8 specific cases)
    - Test single tile has all-None neighbors
    - Test two adjacent tiles have bidirectional link
    - Test releasing middle tile clears outer tiles' links
    - _Requirements: 1.2, 2.1, 2.2, 3.1, 3.2_

- [x] 7. Final checkpoint — Full test suite
  - Ensure all tests pass (`cargo test`), ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The `proptest` crate must be added as a dev-dependency for property-based tests
- The `coord_to_idx` HashMap is intentionally retained — it's still needed for `set_cell`, `get_cell`, `allocate`, and activity management
- Existing tests in `tests/turbolife.rs` and `tests/parity.rs` serve as the primary behavioral preservation check
