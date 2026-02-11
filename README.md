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

## Tests

```
cargo test
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE) for details.
