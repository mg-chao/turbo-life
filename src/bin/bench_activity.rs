#[cfg(feature = "mimalloc-global")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rand::RngCore;
use rand::SeedableRng;
use std::time::Instant;
use turbo_life::turbolife::TurboLife;

fn bench(label: &str, size: i64, density: f64, iterations: u64) {
    let mut turbo = TurboLife::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF_CAFE);
    let threshold = (u64::MAX as f64 * density) as u64;

    for y in 0..size {
        for x in 0..size {
            if rng.next_u64() <= threshold {
                turbo.set_cell(x, y, true);
            }
        }
    }

    // Warm up: 2 steps to stabilize active set
    turbo.step_n(2);

    let start = Instant::now();
    turbo.step_n(iterations);
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_us = total_ms * 1000.0 / iterations as f64;
    let pop = turbo.population();
    println!(
        "{:<28} {:>6} iters  {:>10.1} ms total  {:>10.1} us/step  pop={}",
        label, iterations, total_ms, avg_us, pop
    );
}

fn bench_glider(steps: u64) {
    let mut turbo = TurboLife::new();
    // Single glider - very small active set
    turbo.set_cell(1, 0, true);
    turbo.set_cell(2, -1, true);
    turbo.set_cell(0, -2, true);
    turbo.set_cell(1, -2, true);
    turbo.set_cell(2, -2, true);

    turbo.step_n(2);

    let start = Instant::now();
    turbo.step_n(steps);
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_us = total_ms * 1000.0 / steps as f64;
    println!(
        "{:<28} {:>6} iters  {:>10.1} ms total  {:>10.1} us/step",
        "glider (tiny)", steps, total_ms, avg_us
    );
}

fn main() {
    println!("=== TurboLife Activity Benchmark ===\n");

    bench_glider(10000);
    bench("small 128x128 d=0.3", 128, 0.3, 500);
    bench("medium 512x512 d=0.3", 512, 0.3, 200);
    bench("medium 1024x1024 d=0.3", 1024, 0.3, 100);
    bench("large 2048x2048 d=0.3", 2048, 0.3, 50);
    bench("large 4096x4096 d=0.42", 4096, 0.42, 20);
    bench("sparse 2048x2048 d=0.05", 2048, 0.05, 50);
}
