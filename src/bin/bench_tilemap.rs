//! Micro-benchmark for TileMap operations: insert, get (hit/miss), remove, and resize.
//! Run with: cargo run --release --bin bench_tilemap

use std::time::Instant;

// We need to access the tilemap module. It's pub(crate), so we use a helper
// exposed from the library for benchmarking.
// Instead, we'll duplicate the minimal interface here via the public engine API
// and measure end-to-end tilemap-heavy workloads.

use rand::RngCore;
use rand::SeedableRng;
use turbo_life::turbolife::TurboLife;

const WARMUP_STEPS: u64 = 3;
const BENCH_STEPS: u64 = 30;
const RUNS: usize = 3;

fn bench_step_performance(label: &str, size: i64, density: f64, seed: u64) {
    let mut best_avg = f64::MAX;

    for run in 0..RUNS {
        let mut engine = TurboLife::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let threshold = (u64::MAX as f64 * density) as u64;

        for y in 0..size {
            for x in 0..size {
                if rng.next_u64() <= threshold {
                    engine.set_cell(x, y, true);
                }
            }
        }

        // Warmup
        engine.step_n(WARMUP_STEPS);

        let start = Instant::now();
        engine.step_n(BENCH_STEPS);
        let elapsed = start.elapsed();

        let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCH_STEPS as f64;
        let pop = engine.population();

        if avg_ms < best_avg {
            best_avg = avg_ms;
        }

        if run == RUNS - 1 {
            println!("{:<36} best={:.4} ms/step  pop={}", label, best_avg, pop);
        }
    }
}

/// Benchmark that stresses tilemap operations specifically:
/// many small patterns that create/destroy tiles frequently.
fn bench_tile_churn(label: &str, num_gliders: usize, steps: u64) {
    let mut best_total = f64::MAX;

    for run in 0..RUNS {
        let mut engine = TurboLife::new();
        // Place gliders spread across a large area to maximize tile creation
        for i in 0..num_gliders {
            let bx = (i % 100) as i64 * 200;
            let by = (i / 100) as i64 * 200;
            engine.set_cell(bx + 1, by, true);
            engine.set_cell(bx + 2, by - 1, true);
            engine.set_cell(bx, by - 2, true);
            engine.set_cell(bx + 1, by - 2, true);
            engine.set_cell(bx + 2, by - 2, true);
        }

        engine.step_n(WARMUP_STEPS);

        let start = Instant::now();
        engine.step_n(steps);
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        if total_ms < best_total {
            best_total = total_ms;
        }

        if run == RUNS - 1 {
            let avg_ms = best_total / steps as f64;
            println!(
                "{:<36} best={:.4} ms/step  ({:.1} ms total for {} steps)",
                label, avg_ms, best_total, steps
            );
        }
    }
}

/// Benchmark set_cell throughput (tilemap insert-heavy).
fn bench_set_cells(label: &str, count: usize, spread: i64, seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let coords: Vec<(i64, i64)> = (0..count)
        .map(|_| {
            let x = (rng.next_u64() % (spread as u64 * 2)) as i64 - spread;
            let y = (rng.next_u64() % (spread as u64 * 2)) as i64 - spread;
            (x, y)
        })
        .collect();

    let mut best_ms = f64::MAX;

    for run in 0..RUNS {
        let mut engine = TurboLife::new();
        let start = Instant::now();
        for &(x, y) in &coords {
            engine.set_cell(x, y, true);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        if elapsed_ms < best_ms {
            best_ms = elapsed_ms;
        }

        if run == RUNS - 1 {
            let per_cell_ns = best_ms * 1_000_000.0 / count as f64;
            println!(
                "{:<36} best={:.3} ms  ({:.1} ns/cell, {} cells)",
                label, best_ms, per_cell_ns, count
            );
        }
    }
}

fn main() {
    println!("=== TileMap & Tile Management Benchmark ===\n");

    println!("-- set_cell throughput (tilemap insert stress) --");
    bench_set_cells("10k cells, spread=500", 10_000, 500, 0xAABB);
    bench_set_cells("100k cells, spread=2000", 100_000, 2000, 0xCCDD);
    bench_set_cells("500k cells, spread=5000", 500_000, 5000, 0xEEFF);

    println!("\n-- tile churn (glider expansion/pruning) --");
    bench_tile_churn("100 gliders, 200 steps", 100, 200);
    bench_tile_churn("500 gliders, 200 steps", 500, 200);
    bench_tile_churn("2000 gliders, 100 steps", 2000, 100);

    println!("\n-- step performance (full engine, tilemap-heavy) --");
    bench_step_performance("512x512 d=0.30", 512, 0.30, 0x1111);
    bench_step_performance("1024x1024 d=0.30", 1024, 0.30, 0x2222);
    bench_step_performance("2048x2048 d=0.42", 2048, 0.42, 0x3333);
}
