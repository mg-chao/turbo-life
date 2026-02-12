# TurboLife Performance Report Template

Date: <!-- YYYY-MM-DD -->
CPU: <!-- model -->
OS: <!-- distro/version -->
Rust: <!-- rustc -V -->

## Commands

```bash
./scripts/perf_linux.sh
```

Events:

- `task-clock`
- `cycles`
- `instructions`
- `cache-references`
- `cache-misses`
- `context-switches`
- `cpu-migrations`

## Baseline (before)

| Threads | Size | Density | Iters | avg_ms | task_clock_ms | cycles | instructions | cache_refs | cache_misses | ctx_switch | cpu_migr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |  |  |

## Optimized (after)

| Threads | Size | Density | Iters | avg_ms | task_clock_ms | cycles | instructions | cache_refs | cache_misses | ctx_switch | cpu_migr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |  |  |

## Derived Metrics

Use the following formulas per scenario:

- `time_per_step_ms = avg_ms`
- `core_utilization = task_clock_ms / (elapsed_ms * nproc)`
- `cache_hit_rate = 1 - cache_misses / cache_refs`
- `ipc = instructions / cycles`
- `speedup_vs_baseline = baseline_avg_ms / optimized_avg_ms`

## Acceptance Checks

- Dense geometric mean speedup (`4096`, `8192`, `16384`, density `0.42`) is `>= 1.25x`.
- Sparse guard (`2048`, density `0.05`) regression is `<= 10%`.
- No correctness regression in `cargo test --release`.

## Notes / Risk Review

- Unsafe scope limited to kernel fast path and existing pointer hot loops.
- AVX2 path validated against scalar path via randomized unit tests.
- If cache hit improves but speed regresses, inspect bandwidth saturation and thread cap tuning.
