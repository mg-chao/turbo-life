//! Performance regression test for TurboLife optimizations.
//!
//! Runs multiple scenarios and reports timing. Use with `--release` for meaningful results.
//! Compare output across commits to detect regressions.

use rand::RngCore;
use rand::SeedableRng;
use std::time::Instant;
use turbo_life::turbolife::{TurboLife, TurboLifeConfig};

struct Scenario {
    name: &'static str,
    size: i64,
    density: f64,
    warmup: u64,
    iters: u64,
    seed: u64,
}

fn seed_board(engine: &mut TurboLife, size: i64, density: f64, seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let threshold = (u64::MAX as f64 * density) as u64;
    let mut cells = Vec::new();
    for y in 0..size {
        for x in 0..size {
            if rng.next_u64() <= threshold {
                cells.push((x, y));
            }
        }
    }
    engine.set_cells_alive(cells);
}

fn run_scenario(s: &Scenario, threads: Option<usize>) -> (f64, u64) {
    let config = if let Some(t) = threads {
        TurboLifeConfig::default().thread_count(t)
    } else {
        TurboLifeConfig::default()
    };
    let mut engine = TurboLife::with_config(config);
    seed_board(&mut engine, s.size, s.density, s.seed);

    if s.warmup > 0 {
        engine.step_n(s.warmup);
    }

    let start = Instant::now();
    engine.step_n(s.iters);
    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let pop = engine.population();
    (total_ms, pop)
}

fn main() {
    let scenarios = [
        Scenario {
            name: "small-sparse",
            size: 512,
            density: 0.10,
            warmup: 3,
            iters: 200,
            seed: 0xA1,
        },
        Scenario {
            name: "small-dense",
            size: 512,
            density: 0.42,
            warmup: 3,
            iters: 200,
            seed: 0xB2,
        },
        Scenario {
            name: "medium-sparse",
            size: 2048,
            density: 0.10,
            warmup: 3,
            iters: 50,
            seed: 0xC3,
        },
        Scenario {
            name: "medium-dense",
            size: 2048,
            density: 0.42,
            warmup: 3,
            iters: 50,
            seed: 0xD4,
        },
        Scenario {
            name: "large-sparse",
            size: 4096,
            density: 0.10,
            warmup: 3,
            iters: 30,
            seed: 0xE5,
        },
        Scenario {
            name: "large-dense",
            size: 4096,
            density: 0.42,
            warmup: 3,
            iters: 30,
            seed: 0xF6,
        },
        Scenario {
            name: "xlarge-dense",
            size: 8192,
            density: 0.42,
            warmup: 2,
            iters: 10,
            seed: 0x17,
        },
    ];

    println!(
        "{:<20} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "Scenario", "Threads", "Iters", "Total(ms)", "Avg(ms)", "Pop"
    );
    println!("{}", "-".repeat(78));

    // Run with auto thread count
    for s in &scenarios {
        let (total_ms, pop) = run_scenario(s, None);
        let avg_ms = total_ms / s.iters as f64;
        println!(
            "{:<20} {:>10} {:>10} {:>12.3} {:>12.6} {:>10}",
            s.name, "auto", s.iters, total_ms, avg_ms, pop
        );
    }

    println!();

    // Run single-threaded for comparison
    for s in &scenarios {
        let (total_ms, pop) = run_scenario(s, Some(1));
        let avg_ms = total_ms / s.iters as f64;
        println!(
            "{:<20} {:>10} {:>10} {:>12.3} {:>12.6} {:>10}",
            s.name, "1", s.iters, total_ms, avg_ms, pop
        );
    }
}
