use rand::RngCore;
use rand::SeedableRng;
use std::time::Instant;
use turbo_life::turbolife::TurboLife;

fn bench_turbo(size: i64, density: f64, iterations: u64) -> (f64, u64) {
    let mut turbo = TurboLife::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_1234_ABCD_EF01);
    let threshold = (u64::MAX as f64 * density) as u64;

    for y in 0..size {
        for x in 0..size {
            if rng.next_u64() <= threshold {
                turbo.set_cell(x, y, true);
            }
        }
    }

    let start = Instant::now();
    turbo.step_n(iterations);
    let duration = start.elapsed();

    let total_ms = duration.as_secs_f64() * 1000.0;
    let pop = turbo.population();
    (total_ms, pop)
}

fn main() {
    let scales: &[(i64, u64)] = &[
        (1024, 200), // ~256 tiles, below parallel threshold
        (2048, 200), // ~1024 tiles
        (4096, 100), // ~4096 tiles (current benchmark)
        (8192, 50),  // ~16384 tiles
        (16384, 20), // ~65536 tiles
    ];

    println!(
        "{:<10} {:>8} {:>12} {:>12} {:>10}",
        "Grid", "Tiles", "Iters", "Total(ms)", "Avg(ms)"
    );
    println!("{}", "-".repeat(58));

    for &(size, iters) in scales {
        let tiles = (size / 64) * (size / 64);
        let (total_ms, _pop) = bench_turbo(size, 0.42, iters);
        let avg_ms = total_ms / iters as f64;
        println!(
            "{:<10} {:>8} {:>12} {:>12.1} {:>10.4}",
            format!("{}x{}", size, size),
            tiles,
            iters,
            total_ms,
            avg_ms
        );
    }
}
