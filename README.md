# turbo-life

A high-performance Conway's Game of Life (B3/S23) engine in Rust.

## TurboLife

TurboLife is built to fully leverage the performance of modern computers. It uses multi-threading, and plans to incorporate SIMD and GPU acceleration to achieve ultimate calculation speed.

This architecture is designed from the ground up for acceleration:

- **Multi-threading** — already implemented via rayon with adaptive serial/parallel dispatch.
- **SIMD** — the row-independent full-adder kernel maps naturally to AVX2 (4 rows) and AVX-512 (8 rows).
- **GPU** — the branchless kernel and flat SoA data layout are ready to map onto compute shaders or CUDA kernels.

## Building

```
cargo build --release
```

## Running

```
cargo run --release
```

Seeds a 4096×4096 region at 42% density and runs 1000 generations, printing total and per-iteration timings.

## Threading behavior

- TurboLife defaults to a memory-bandwidth-aware cap derived from physical CPU cores for its internal rayon thread pool.
- You can override this with `TURBOLIFE_NUM_THREADS` (preferred) or `RAYON_NUM_THREADS`.
- TurboLife uses adaptive per-step parallelism to avoid over-aggressive threading on small/medium active sets.
- `TURBOLIFE_MAX_THREADS` can cap runtime parallel fan-out while keeping pool size unchanged.
- Example (PowerShell):

```
$env:TURBOLIFE_NUM_THREADS=12
$env:TURBOLIFE_MAX_THREADS=8
cargo run --release
```

## Kernel backend selection

- `TURBOLIFE_KERNEL=auto|scalar|avx2` controls kernel dispatch behavior.
- Default is `auto` (uses AVX2 on supported `x86_64` CPUs, otherwise scalar).
- `scalar` is useful for comparison and debugging.

## Tests

```
cargo test
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE) for details.
