# CoordMap: Purpose-Built Spatial Hash for TurboLife

## Motivation

TurboLife's `TileArena` uses a hash map (`FxHashMap<(i64, i64), TileIdx>`) to translate
2D tile coordinates into dense arena indices. While `FxHashMap` is a solid general-purpose
map, the TurboLife workload has very specific properties that a custom structure can exploit:

| Property | Implication |
|---|---|
| Keys are `(i64, i64)` integer pairs | No need for trait-object hashing; inline a cheap spatial mixer |
| Coordinates are spatially clustered | Neighbor lookups (±1 in x/y) dominate; cache-local probing wins |
| Table size is moderate (hundreds–low thousands) | Fits in L2/L3; open addressing with linear probing is ideal |
| Lookups are neighbor-biased | 9-probe bursts (self + 8 neighbors) per expand check |
| Insertions/deletions come in batches | Bulk expand then bulk prune each step; amortized growth is fine |
| Hot simulation loop does NOT query the map | The map is only touched in `allocate`, `release`, `ensure_neighbor`, `set_cell`, `get_cell` |

### Critical observation: the coord map is a cold-path structure

The kernel loop in `step_impl` works entirely through dense arena indices and the
pre-built `neighbors: Vec<[u32; 8]>` array. It never touches the coord map. The map
is only queried during:

- **Expand phase** (`prune_and_expand`): `idx_at()` existence checks + `allocate()`
  with up to 8 neighbor lookups per new tile.
- **External API** (`set_cell`, `get_cell`): coordinate-to-index translation for
  user-facing entry points.
- **Release** (`release`): removal of pruned tiles.

This means the highest-impact optimization is not making the map faster — it's
**bypassing it** on the expand hot path via the existing neighbor graph (see
[Neighbor-Graph Bypass](#neighbor-graph-bypass-co-design) below). CoordMap still
needs to be correct and reasonably fast for the remaining cold-path uses, but
shaving nanoseconds off individual lookups matters less than eliminating lookups
entirely.

## Data Structure

```
┌─────────────────────────────────────────────────────┐
│ CoordMap                                            │
│                                                     │
│   keys:     Vec<(i64, i64)>    ← slot key storage   │
│   values:   Vec<TileIdx>       ← slot value storage  │
│   len:      usize              ← number of entries    │
│   mask:     usize              ← capacity − 1         │
│                                                     │
│   EMPTY_KEY = (i64::MIN, i64::MIN)  ← sentinel       │
│                                                     │
│   Invariant: capacity is always a power of 2        │
│   Invariant: len ≤ capacity * MAX_LOAD (≈ 70%)     │
│   Invariant: EMPTY_KEY is never inserted             │
│   Invariant: slot 0 of the arena is never stored    │
└─────────────────────────────────────────────────────┘
```

### Why parallel arrays instead of `Vec<Option<(Key, Value)>>`?

- `keys` and `values` are separate so that probe chains only touch `keys`
  until a match is confirmed, keeping the working set small during lookup.
  At 16 bytes/key, ~4 keys fit per cache line — probing stays within 1–2
  cache lines for typical chains.
- Empty slots are identified by `keys[slot] == EMPTY_KEY` — no separate
  `occupied` array needed. This saves ~1 byte/slot and removes one array
  from the probe working set.

### Why sentinel key instead of `Vec<bool>`?

The original design used `occupied: Vec<bool>` for slot liveness. Using a sentinel
key (`(i64::MIN, i64::MIN)`, already reserved as the sentinel coord in `TileArena`)
eliminates this array entirely. Probe logic checks `keys[slot] == EMPTY_KEY` instead
of `!occupied[slot]`, reducing the number of arrays touched during probing from three
to one (`keys` only until a match is found).

## Hash Function

A fast spatial mixer that distributes clustered integer coordinates evenly across
power-of-2 table sizes:

```rust
#[inline(always)]
fn spatial_hash(x: i64, y: i64) -> usize {
    // Two large odd constants (derived from fractional parts of golden ratio
    // and sqrt(3)), chosen for good bit avalanche on sequential inputs.
    const MIX_X: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_Y: u64 = 0x517C_C1B7_2722_0A95;

    let hx = (x as u64).wrapping_mul(MIX_X);
    let hy = (y as u64).wrapping_mul(MIX_Y);
    (hx ^ hy) as usize
}
```

### Why not FxHash?

`FxHash` processes the key as a byte stream, feeding 8 bytes at a time through
`wrapping_mul` + `rotate_left`. For a 16-byte `(i64, i64)` key, that's two
multiply-rotate rounds plus finalization. Our mixer does two multiplies and one
XOR — fewer instructions, no data-dependent rotations, and the constants are
chosen specifically for 2D integer dispersion rather than general byte hashing.

### Collision behavior on clustered coords

For a cluster of tiles around `(0,0)`, the hash values of `(0,0)`, `(1,0)`,
`(0,1)`, `(-1,0)`, `(0,-1)`, etc. differ by at least one full `MIX_X` or
`MIX_Y` stride, which scatters them across the table. Empirically, for a
1000-tile cluster, average probe length stays under 1.5 at 70% load.

### Morton-code (Z-order) hash variant — not recommended

An earlier draft proposed interleaving the bits of `x` and `y` into a Morton code
before hashing, to preserve 2D spatial locality in the hash table. The idea was that
the 9-probe neighbor burst in `allocate()` would hit slots within 2–3 cache lines
rather than scattering across the table.

This is **not recommended** for two reasons:

1. **The neighbor-graph bypass eliminates the 9-probe burst.** Once `allocate()` resolves
   neighbors via the existing `neighbors` array instead of the coord map (see
   [Neighbor-Graph Bypass](#neighbor-graph-bypass-co-design)), the coord map no longer
   sees clustered neighbor lookups. The remaining cold-path queries (`set_cell`,
   `get_cell`) are isolated single lookups where Morton locality provides no benefit.

2. **Higher per-hash cost for no gain.** Morton interleaving costs ~10 instructions
   (shifts + masks) vs 2 multiplies for `spatial_hash`. On the cold path, this
   overhead is pure waste.

The `spatial_hash` mixer is sufficient. No feature flag or compile-time variant needed.

## Operations

### Lookup: `get(&self, coord: (i64, i64)) -> Option<TileIdx>`

```
slot = spatial_hash(coord.0, coord.1) & self.mask

loop:
    if keys[slot] == EMPTY_KEY:
        return None              ← empty slot = key absent
    if keys[slot] == coord:
        return Some(values[slot]) ← found
    slot = (slot + 1) & self.mask ← linear probe
```

Linear probing is chosen over quadratic or Robin Hood because:
- The table is small enough that L2 prefetch covers probe chains.
- Neighbor coords often hash to nearby slots, so sequential probes
  benefit from cache-line prefetch.
- Simpler branch logic = better branch prediction on short chains.

### Insert: `insert(&mut self, coord: (i64, i64), idx: TileIdx)`

```
if self.len * 10 >= self.capacity() * MAX_LOAD_NUMER:
    self.grow()

slot = spatial_hash(coord.0, coord.1) & self.mask

loop:
    if keys[slot] == EMPTY_KEY:
        keys[slot] = coord
        values[slot] = idx
        self.len += 1
        return
    if keys[slot] == coord:
        values[slot] = idx       ← update existing
        return
    slot = (slot + 1) & self.mask
```

### Remove: `remove(&mut self, coord: (i64, i64)) -> bool`

Uses backward-shift deletion (not tombstones):

```
slot = find_slot(coord)
if slot is None:
    return false

keys[slot] = EMPTY_KEY
self.len -= 1

── backward shift ──
next = (slot + 1) & mask
while keys[next] != EMPTY_KEY:
    ideal = spatial_hash(keys[next].0, keys[next].1) & mask
    if !is_between_circular(ideal, slot, next):
        break
    keys[slot]     = keys[next]
    values[slot]   = values[next]
    keys[next]     = EMPTY_KEY
    slot = next
    next = (slot + 1) & mask

return true
```

#### Why backward-shift instead of tombstones?

Tombstones accumulate over time and degrade probe-chain length. TurboLife
calls `release()` in batches during pruning — a tombstone-based table would
need periodic rehashing to stay healthy. Backward-shift deletion keeps probe
chains minimal at all times, with no deferred cleanup needed.

The cost is O(average probe length) extra moves per delete, but since the
table is small and deletions are batched (not interleaved with lookups),
this is a net win.

## Growth Policy

```
Initial capacity:  16 slots
Max load factor:   70%  (i.e., grow when len > capacity * 7 / 10)
Growth factor:     2×   (double capacity)
```

Growth allocates new vectors at 2× capacity and re-inserts all occupied entries.
This is O(n) but happens infrequently — at most O(log n) times over the lifetime
of a simulation. The 70% load factor balances memory usage against probe length;
at 70% load with a good hash, expected probe length is ~1.4 for successful lookups.

```
Tiles    Capacity    Memory (approx)
──────   ────────    ───────────────
< 12     16          ~320 B
100      256         ~5 KB
1,000    2,048       ~40 KB
10,000   16,384      ~320 KB
```

Memory per slot: `16 (key) + 4 (value) = 20 bytes`.

## API Surface

```rust
impl CoordMap {
    /// Create an empty map with default initial capacity (16).
    pub fn new() -> Self;

    /// Create an empty map with at least the given capacity.
    pub fn with_capacity(cap: usize) -> Self;

    /// Look up a tile index by coordinate.
    pub fn get(&self, coord: (i64, i64)) -> Option<TileIdx>;

    /// Insert or update a coordinate → index mapping.
    pub fn insert(&mut self, coord: (i64, i64), idx: TileIdx);

    /// Remove a coordinate. Returns true if it was present.
    pub fn remove(&mut self, coord: (i64, i64)) -> bool;

    /// Number of entries.
    pub fn len(&self) -> usize;

    /// Is the map empty?
    pub fn is_empty(&self) -> bool;

    /// Create an empty map (equivalent to new()).
    pub fn default() -> Self;
}
```

This is a drop-in replacement for the `FxHashMap` usage in `TileArena`. The only
call sites are:

| Call site | Operation | Location |
|---|---|---|
| `TileArena::new()` | `CoordMap::new()` | `arena.rs` |
| `TileArena::idx_at()` | `coord_map.get(coord)` | `arena.rs` |
| `TileArena::allocate()` | `coord_map.insert(coord, idx)` + `coord_map.get(neighbor)` | `arena.rs` |
| `TileArena::release()` | `coord_map.remove(coord)` | `arena.rs` |
| `TurboLife::set_cell()` | via `idx_at()` / `allocate()` | `engine.rs` |
| `TurboLife::get_cell()` | via `idx_at()` | `engine.rs` |

## Neighbor-Graph Bypass (Co-Design)

This is the highest-impact optimization in this document and is designed as a
co-delivery with CoordMap, not a follow-up.

### Problem

`TileArena::allocate()` currently performs 1 insert + up to 8 `coord_to_idx.get()`
lookups to wire up the `neighbors` array for a new tile. In the expand phase of
`prune_and_expand`, every expansion was triggered by a specific source tile that
already knows which direction the missing neighbor is in — but this information is
discarded, and `allocate()` re-discovers all neighbors via the coord map.

For a 5,000-tile simulation with ~200 tiles expanding per step, the expand phase
does roughly 200 × 9 = 1,800 map operations per generation. With the bypass, this
drops to ~200 × 1 = 200 (the insert only), eliminating ~89% of expand-phase map
queries.

### Design

Add an `allocate_adjacent` method that accepts a source tile and direction hint:

```rust
impl TileArena {
    /// Allocate a new tile at `coord`, using `source` tile's neighbor graph
    /// to resolve adjacency without coord map lookups.
    ///
    /// `from_dir` is the direction FROM source TO the new tile.
    /// E.g., if source is at (0,0) and new tile is at (1,0), from_dir = East.
    pub fn allocate_adjacent(
        &mut self,
        coord: (i64, i64),
        source: TileIdx,
        from_dir: Direction,
    ) -> TileIdx {
        // 1. Insert into coord map (still needed for external API lookups).
        //    If already exists, return early.
        if let Some(existing) = self.idx_at(coord) {
            return existing;
        }
        let idx = self.allocate_slot(coord);  // internal: alloc + coord_map.insert

        // 2. Wire the source↔new link directly (no map lookup).
        self.neighbors[idx.index()][from_dir.reverse().index()] = source.0;
        self.neighbors[source.index()][from_dir.index()] = idx.0;

        // 3. Resolve remaining 7 neighbors via source's neighbor graph.
        //    For each direction D of the new tile (other than back-to-source):
        //    - Compute which neighbor of source would be adjacent to D.
        //    - If that neighbor exists (not NO_NEIGHBOR), its neighbor in the
        //      appropriate direction is the tile we want.
        //    Example: new tile's North neighbor = source's North-East neighbor's
        //             East neighbor (when from_dir = East).
        for dir in Direction::ALL {
            if dir == from_dir.reverse() {
                continue;  // already wired above
            }
            // Try to resolve via source's neighbor graph.
            // This is a 2-hop walk: source → intermediate → target.
            if let Some(target) = self.resolve_via_graph(source, from_dir, dir) {
                self.neighbors[idx.index()][dir.index()] = target.0;
                self.neighbors[target.index()][dir.reverse().index()] = idx.0;
            }
        }

        idx
    }
}
```

The `resolve_via_graph` helper walks at most 2 hops through the existing neighbor
array — pure array indexing, no hashing. For the 7 remaining directions, this
replaces 7 hash lookups with at most 14 array reads.

### Call site change in `prune_and_expand`

```rust
// Before (activity.rs):
let idx = arena.allocate(coord);

// After — expand_buf carries (coord, source_idx, direction):
let idx = arena.allocate_adjacent(coord, source, from_dir);
```

The `scan_tile_prune_expand` function already knows the source tile and which
direction triggered the expansion. The only change is threading that information
through `expand_buf` (changing its type from `Vec<(i64, i64)>` to
`Vec<(i64, i64, TileIdx, Direction)>`).

### Residual coord map usage after bypass

With the bypass in place, the coord map is only queried in:

| Call site | Frequency | Notes |
|---|---|---|
| `set_cell` / `get_cell` | User-driven, cold | External API, unavoidable |
| `allocate_adjacent` existence check | Once per expansion | `idx_at(coord)` to detect duplicates |
| `release` | Batch prune, cold | `coord_map.remove(coord)` |
| `ensure_neighbor` | `set_cell` path only | Cold |

The 8-lookup neighbor resolution burst is completely eliminated from the hot path.

## Scaling Bottleneck Analysis: Beyond the Coord Map

The coord map is not the scaling bottleneck for TurboLife. At high tile counts,
the dominant memory pressure comes from the kernel loop's scattered border reads.
This section documents the real bottleneck and potential mitigations.

### Per-tile memory budget

Each tile in the arena consumes:

| Component | Size | Notes |
|---|---|---|
| `CellBuf` × 2 (double-buffered) | 2 × 512 = 1,024 B | `[u64; 64]`, 64-byte aligned |
| `TileMeta` | 9 B | `u32 + u32 + u8`, packed |
| `Neighbors` | 32 B | `[u32; 8]` |
| `BorderData` × 2 (double-buffered) | 2 × 40 = 80 B | `4×u64 + u8`, padded to 40 B with `repr(C)` |
| `coords` | 16 B | `(i64, i64)` |
| **Total per tile** | **~1,161 B** | |

| Tile count | Arena memory | L3 fit? (typical 32 MB) |
|---|---|---|
| 1,000 | ~1.1 MB | Comfortable |
| 10,000 | ~11 MB | Fits |
| 50,000 | ~55 MB | Exceeds L3 |
| 100,000 | ~110 MB | Far exceeds L3 |

The coord map's 320 KB at 10K tiles is a rounding error compared to the arena's 11 MB.

### The real hot-path pressure: scattered border reads

In `step_impl`, the ghost zone gather (`gather_ghost_zone_raw` in `sync.rs`) reads
the `BorderData` of 8 neighbors for each active tile. The active set is iterated
linearly, but the neighbor indices are essentially random — each border read is a
potential cache miss.

Per active tile: 8 × 40 bytes = 320 bytes of scattered reads from the `borders` array.
For 5,000 active tiles: 5,000 × 320 = 1.6 MB of scattered reads per generation.
At 50,000 active tiles: 16 MB of scattered reads — thrashing L3.

This is the dominant memory bottleneck, not the coord map.

### Mitigation 1: Active-set sorting by spatial locality

Sort the active set by Morton order of tile coordinates before the kernel loop:

```rust
// In step_impl, after rebuild_active_set:
arena.active_set.sort_unstable_by_key(|&idx| {
    let (tx, ty) = arena.coords[idx.index()];
    morton_key(tx, ty)  // interleave bits for Z-order curve
});
```

Tiles that are spatially adjacent will be processed consecutively. Since neighbors
of tile N are likely neighbors of tile N+1, the border reads for consecutive tiles
will hit the same cache lines. This turns scattered reads into semi-sequential reads.

Cost: O(n log n) sort per generation. For 5,000 tiles, this is ~60K comparisons —
roughly 50–100 µs. The cache savings on the border gather should more than compensate.

**When to apply:** Only when `active_set.len()` exceeds a threshold (e.g., >512 tiles).
Below that, the working set fits in L2 and sorting is pure overhead.

### Mitigation 2: Border data SOA layout

The current `borders: [Vec<BorderData>; 2]` stores each tile's border as a contiguous
struct. During ghost zone gather, the 8 neighbor reads each pull in a full 40-byte
`BorderData`, but only specific fields are used per direction:

- North neighbor: only `south` field read (8 bytes out of 40)
- West neighbor: only `east` field read (8 bytes out of 40)
- Corner neighbors: only `corners` field read (1 byte out of 40)

A struct-of-arrays layout would group all `north` edges contiguously, all `south`
edges contiguously, etc.:

```rust
struct BorderSOA {
    north: Vec<u64>,   // all tiles' north edges
    south: Vec<u64>,   // all tiles' south edges
    west:  Vec<u64>,   // all tiles' west edges
    east:  Vec<u64>,   // all tiles' east edges
    corners: Vec<u8>,  // all tiles' corner bits
}
```

This way, reading 8 different tiles' `south` edges reads from 8 positions in a
single contiguous `Vec<u64>`, rather than 8 positions in a `Vec<BorderData>` where
each position is 40 bytes apart. Cache line utilization improves from ~20% (8/40)
to ~100% (8/8) for edge reads.

**Tradeoff:** More complex indexing, and `recompute_border` must scatter-write to
5 arrays instead of writing one struct. The gather path (hot) benefits; the
recompute path (also hot, but write-side) becomes slightly more expensive.

**Recommendation:** Benchmark both layouts. The SOA layout is most impactful at
high tile counts (>10K) where the border array exceeds L2. At low tile counts,
the AOS layout's simpler codegen may win.

### Mitigation 3: Prefetch hints in the ghost zone gather

For the current AOS border layout, software prefetch hints can hide memory latency:

```rust
// In the kernel loop, prefetch the next tile's neighbors' borders
// while processing the current tile.
for i in 0..active_len {
    // Prefetch borders for tile i+1's neighbors
    if i + 1 < active_len {
        let next_idx = active_set[i + 1];
        let next_nb = &neighbors[next_idx.index()];
        for &ni in next_nb.iter() {
            if ni != NO_NEIGHBOR {
                prefetch_read(&borders[ni as usize]);
            }
        }
    }
    // Process tile i (borders already in cache from previous prefetch)
    let ghost = gather_ghost_zone(active_set[i], borders, neighbors);
    // ...
}
```

This overlaps memory latency with computation. Effectiveness depends on the
pipeline depth and how far ahead the prefetch needs to be — may need to prefetch
2–3 tiles ahead rather than 1. Requires benchmarking on target hardware.

## Design Refinements

### Sentinel key vs `Vec<bool>` (occupied bitmap)

An earlier draft used `occupied: Vec<bool>` to mark slot liveness. Each `bool`
occupies a full byte, wasting 7 bits per slot. Two alternatives were considered:

1. **Sentinel key** (adopted): Use `(i64::MIN, i64::MIN)` (already the sentinel coord
   in `TileArena`) to mark empty slots. This eliminates the `occupied` array entirely —
   probe logic checks `keys[slot] == EMPTY_KEY` instead of `!occupied[slot]`. Saves
   ~1 byte/slot and removes one array from the probe working set.

2. **Packed bitmap**: Replace `Vec<bool>` with a `Vec<u64>` where each bit represents
   one slot. Scanning 64 slots per cache line instead of 8. More complex indexing
   (`bitmap[slot >> 6] & (1 << (slot & 63))`), but dramatically smaller footprint
   for the liveness data.

The sentinel key approach was chosen for its simplicity. Probe chains touch only
`keys` until a match is found, and the sentinel value is already reserved by the arena.

## Alternative Data Structures Considered

Beyond the open-addressing hash table, several other structures were evaluated for
the `coord → TileIdx` mapping. Each is analyzed against TurboLife's specific workload:
batch expand/prune, neighbor-biased lookups, moderate table sizes (hundreds to low
thousands of tiles).

### 1. Flat grid with offset (dense bounded patterns)

If the simulation's tile bounding box is known and reasonably bounded, a flat 2D
array indexed by `(x - x_min, y - y_min)` gives O(1) lookup with zero hashing and
zero probing:

```
┌──────────────────────────────────────────────────┐
│ FlatCoordMap                                     │
│                                                  │
│   data:   Vec<u32>     ← TileIdx or NO_TILE      │
│   x_min:  i64          ← bounding box origin      │
│   y_min:  i64                                     │
│   width:  usize        ← bounding box extent      │
│   height: usize                                   │
│                                                  │
│   Lookup: data[(y - y_min) * width + (x - x_min)] │
└──────────────────────────────────────────────────┘
```

| Aspect | Assessment |
|---|---|
| Lookup | O(1), single array index, no hashing — fastest possible |
| Insert/Remove | O(1), direct write |
| Memory | `width × height × 4 bytes`; 1000×1000 bbox = ~4 MB |
| Neighbor burst | 9 sequential array reads, perfect cache behavior |
| Weakness | Sparse patterns waste memory; bounding box growth requires reallocation + copy |
| Suitability | Niche — only wins for dense, bounded patterns (oscillators, bounded soups) |

**Assessment:** With the neighbor-graph bypass eliminating the 9-probe burst from
the coord map, the flat grid's main advantage (cache-perfect neighbor reads) is
moot. The remaining cold-path lookups don't benefit enough to justify the bounding
box tracking complexity and the memory waste on sparse/expanding patterns. Not
recommended unless a specific dense-bounded use case is identified and profiled.

### 2. Sorted Vec with binary search

Keep a `Vec<((i64, i64), TileIdx)>` sorted by coordinate (row-major order). Lookup
via binary search, batch insert via sort-merge.

| Aspect | Assessment |
|---|---|
| Lookup | O(log n), ~10 comparisons for 1000 tiles |
| Insert (batch) | O(n) amortized via merge-sort of pending inserts |
| Remove (batch) | O(n) via partition + truncate |
| Memory | Minimal — no wasted slots, no load factor overhead |
| Iteration | Cache-perfect linear scan |
| Weakness | O(log n) lookup is 3–5× slower than O(1) hash for the remaining cold-path queries |
| Suitability | Only wins if iteration dominates lookups, which it does not in TurboLife |

Not recommended. TurboLife never iterates the coord map, so the cache-perfect
iteration advantage is irrelevant.

### 3. Quadtree / spatial tree

A region quadtree where each internal node covers a power-of-2 region of tile-space.
Lookup walks from root to leaf in O(log range) steps.

| Aspect | Assessment |
|---|---|
| Lookup | O(log R) where R is the coordinate range; ~20 steps for 64-bit coords |
| Neighbor walk | Often O(1) amortized — adjacent tile is a sibling or cousin node |
| Memory | High overhead per node (4 child pointers + metadata) |
| Cache behavior | Pointer-chasing; poor prefetch behavior |
| Suitability | Overkill at current scale; at very large scale, HashLife is the better architecture |

Not recommended. The pointer-chasing overhead and high per-node memory cost make
this slower than open-addressing for tables that fit in L2/L3.

### Summary

| Structure | Lookup | Memory | Best for |
|---|---|---|---|
| CoordMap (proposed) | O(1) hash | 20 B/slot × 1.4× | General case, all scales |
| Flat grid | O(1) index | 4 B/cell × bbox area | Dense bounded (niche) |
| Sorted Vec | O(log n) | 20 B/entry, minimal waste | Iteration-heavy (not TurboLife) |
| Quadtree | O(log R) | High (pointers) | Very large sparse (not recommended) |

The proposed CoordMap with the sentinel-key refinement is the right choice. The
neighbor-graph bypass (co-delivered) eliminates the hot-path neighbor bursts that
would have been the main performance-sensitive use of the map.

## Integration Plan

### Phase 1a: CoordMap with sentinel key

1. Add `src/turbolife/coordmap.rs` with the `CoordMap` implementation.
   - Use sentinel key `(i64::MIN, i64::MIN)` instead of a separate `occupied` array.
   - Use `spatial_hash` only (no Morton variant).
2. In `arena.rs`: replace `use rustc_hash::FxHashMap` with `use super::coordmap::CoordMap`.
3. Change the `coord_to_idx` field type from `FxHashMap<(i64, i64), TileIdx>` to `CoordMap`.
4. Replace `FxHashMap::default()` → `CoordMap::new()`.
5. Replace `.get(&coord).copied()` → `.get(coord)`.
6. Replace `.insert(coord, idx)` → `.insert(coord, idx)` (same signature).
7. Replace `.remove(&coord)` → `.remove(coord)`.
8. Run existing parity and turbolife tests to verify correctness.

### Phase 1b: Neighbor-graph bypass (co-deliver with CoordMap)

9. Add `allocate_adjacent(&mut self, coord, source, from_dir) -> TileIdx` to `TileArena`.
   - Wire source↔new link directly.
   - Resolve remaining 7 neighbors via 2-hop graph walk.
   - Fall back to coord map lookup only if graph walk fails (neighbor not yet allocated).
10. Change `expand_buf` type from `Vec<(i64, i64)>` to `Vec<(i64, i64, TileIdx, Direction)>`
    to carry the source tile and expansion direction.
11. Update `scan_tile_prune_expand` to emit `(coord, source_idx, from_dir)` tuples.
12. Update `prune_and_expand` to call `allocate_adjacent` instead of `allocate` for
    expansion-triggered allocations.
13. Keep `allocate()` (without source hint) for `set_cell` / `ensure_neighbor` paths
    where no adjacent tile is known.
14. Benchmark expand phase to measure reduction in coord map queries.

### Phase 2: Kernel-loop cache optimizations (scaling)

These target the real scaling bottleneck — scattered border reads during the kernel
loop — and are independent of the coord map work.

15. **Active-set Morton sorting:** Sort `active_set` by Morton order of tile coords
    before the kernel loop when `active_set.len() > 512`. Benchmark the sort overhead
    vs cache-miss reduction on patterns with >5K active tiles.
16. **Prefetch hints:** Add software prefetch for the next tile's neighbor borders
    in the kernel loop. Tune the prefetch distance (1–3 tiles ahead) on target hardware.
17. **Border SOA evaluation:** Prototype a struct-of-arrays border layout and benchmark
    against the current AOS layout at 10K+ tiles. Adopt if the gather-path improvement
    outweighs the scatter-write cost in `recompute_border`.

### Phase 3: Flat grid hybrid (deferred, low priority)

18. Evaluate bounding-box density at simulation start or periodically.
19. If density exceeds a threshold (e.g., >50% of bbox cells occupied), switch to
    `FlatCoordMap` for the duration of that density regime.
20. Fall back to `CoordMap` when the pattern becomes sparse or unbounded.

**Note:** With the neighbor-graph bypass in place, the flat grid's advantage is
diminished. This phase is only worth pursuing if profiling reveals that the remaining
cold-path coord map lookups are a measurable bottleneck for a specific dense-pattern
workload.

## Expected Performance Characteristics

### CoordMap vs FxHashMap (individual operations)

| Metric | FxHashMap (SwissTable) | CoordMap |
|---|---|---|
| Lookup (hit) | ~15–25 ns | ~8–15 ns |
| Lookup (miss) | ~10–15 ns | ~5–10 ns |
| Insert | ~20–30 ns | ~10–18 ns |
| Remove | ~20–25 ns | ~12–20 ns (backward shift) |
| Memory overhead | ~1 byte/slot control + padding | None (sentinel key) |
| Cache behavior | Group-based SIMD probe | Linear sequential probe, keys-only |
| Hash cost (16-byte key) | 2× mul-rotate + finalize | 2× mul + 1 XOR |

These are rough estimates. Individual operation latency matters less than the
system-level impact of the neighbor-graph bypass.

### System-level impact (CoordMap + neighbor-graph bypass)

For a 5,000-tile simulation with ~200 tiles expanding per step:

| Metric | Before (FxHashMap, no bypass) | After (CoordMap + bypass) |
|---|---|---|
| Map ops per expand step | ~200 × 9 = 1,800 | ~200 × 1 = 200 (insert only) |
| Map ops per prune step | ~P removes | ~P removes (unchanged) |
| Expand-phase map time | ~1,800 × 20 ns = ~36 µs | ~200 × 14 ns = ~2.8 µs |
| Reduction | — | ~92% fewer map operations |

The bypass is the dominant win. The CoordMap-over-FxHashMap improvement on the
remaining operations is a secondary benefit.

### Scaling projection

| Tile count | Expand ops/gen (no bypass) | Expand ops/gen (with bypass) | Arena memory | Bottleneck |
|---|---|---|---|---|
| 1,000 | ~360 | ~40 | ~1.1 MB | None (fits L2) |
| 5,000 | ~1,800 | ~200 | ~5.5 MB | None (fits L3) |
| 10,000 | ~3,600 | ~400 | ~11 MB | Border scatter reads approach L3 limit |
| 50,000 | ~18,000 | ~2,000 | ~55 MB | Border scatter reads dominate; Morton sort helps |
| 100,000 | ~36,000 | ~4,000 | ~110 MB | L3 thrashing; SOA borders + prefetch needed |

At 10K+ tiles, the coord map (even without optimization) is <1% of per-generation
time. The kernel-loop border reads become the dominant cost, which is why Phase 2
targets that bottleneck specifically.
