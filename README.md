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

Seeds a 4096×4096 region at 42% density and runs 2000 generations (reported every 1000), printing total and per-iteration timings.

## Performance workflow

Main-harness benchmark helper (parses the `TurboLife` summary line and reports min/median/mean):

```
./scripts/bench_main.sh 10
```

Quiet mode prints median milliseconds only (useful for automation):

```
./scripts/bench_main.sh 10 target/release/turbo-life --quiet
```

PGO build helper (uses `src/main.rs` as the training workload, benchmarks before/after with the same harness, and rejects regressions):

```
./scripts/build_pgo.sh 3 9
```

Arguments:

- First positional: `train_runs` for profile collection (default `3`).
- Second positional: `bench_runs` for baseline vs PGO comparison (default `9`).

To pass CLI args through to both training and benchmarking harness runs, append them after run counts:

```
./scripts/build_pgo.sh 3 9 --threads 6 --kernel neon
```

`--pgo-train` is reserved for internal script use and should not be passed manually.

`build_pgo.sh` (and the PGO phase in `build_maxperf.sh`) now trains with `--pgo-train`, a TurboLife-only execution mode in `src/main.rs` that removes QuickLife/checkpoint I/O from profile collection so LLVM profile data stays focused on hot TurboLife kernels.

Max-performance auto-tuner (builds multiple compiler/feature variants, benchmarks each with `src/main.rs`, rejects regressions, then optionally tests PGO on top of the best non-PGO candidate):

```
./scripts/build_maxperf.sh 7 3
```

Arguments:

- First positional: benchmark runs per candidate (default `7`).
- Second positional: PGO training runs (default `3`).
- Remaining args are passed through to `src/main.rs` for both candidate benchmarking and PGO training.

When `pgo/turbo-life-main.profdata` is present, `build_maxperf.sh` now benchmarks a `repo-profdata` candidate before fresh training. Fresh PGO instrumentation still starts from the best non-PGO candidate, so the seeded profile can improve winner selection without changing the fresh-training baseline.

Output binary:

```
target/maxperf/best/turbo-life
```

Auto-tuned feature flags used by `build_maxperf.sh`:

- `aggressive-prefetch-aarch64` — enables AArch64 PRFM prefetch hints.

The flag is opt-in only and is rejected automatically when it regresses.

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

- `TURBOLIFE_KERNEL=auto|scalar|avx2|neon` controls kernel dispatch behavior.
- Default is `auto` (uses AVX2 on supported `x86_64`, NEON on supported `aarch64`, otherwise scalar).
- `scalar` is useful for comparison and debugging.

## Tests

```
cargo test
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE) for details.
