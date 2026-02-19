#[cfg(feature = "mimalloc-global")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rand::RngCore;
use rand::SeedableRng;
use std::time::Instant;
use turbo_life::quicklife::QuickLife;
use turbo_life::turbolife::{KernelBackend, TurboLife, TurboLifeConfig};

const SEED_SIDE: i32 = 4096;
const LIVE_DENSITY: f64 = 0.42;
const TOTAL_ITERATIONS: u64 = 2000;
const CHECK_INTERVAL: u64 = 1000;

struct MainArgs {
    config: TurboLifeConfig,
    pgo_train: bool,
}

fn parse_args() -> MainArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut config = TurboLifeConfig::default();
    let mut pgo_train = false;
    let next_arg = |i: usize, flag: &str| -> &str {
        args.get(i)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("{flag} requires a value"))
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                i += 1;
                let n: usize = next_arg(i, "--threads")
                    .parse()
                    .expect("--threads requires a positive integer");
                config = config.thread_count(n);
            }
            "--max-threads" => {
                i += 1;
                let n: usize = next_arg(i, "--max-threads")
                    .parse()
                    .expect("--max-threads requires a positive integer");
                config = config.max_threads(n);
            }
            "--kernel" => {
                i += 1;
                let backend = match next_arg(i, "--kernel").to_ascii_lowercase().as_str() {
                    "scalar" => KernelBackend::Scalar,
                    "avx2" => KernelBackend::Avx2,
                    "neon" => KernelBackend::Neon,
                    other => {
                        panic!("unknown kernel backend: {other} (expected scalar, avx2, or neon)")
                    }
                };
                config = config.kernel(backend);
            }
            "--pgo-train" => {
                pgo_train = true;
            }
            other => panic!(
                "unknown argument: {other}\nusage: turbo-life [--threads N] [--max-threads N] [--kernel scalar|avx2|neon] [--pgo-train]"
            ),
        }
        i += 1;
    }
    MainArgs { config, pgo_train }
}

fn seed_random_world(turbo: &mut TurboLife, mut quick: Option<&mut QuickLife>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_1234_ABCD_EF01);
    let threshold = (u64::MAX as f64 * LIVE_DENSITY) as u64;

    for y in 0..SEED_SIDE {
        for x in 0..SEED_SIDE {
            if rng.next_u64() <= threshold {
                if let Some(quick) = quick.as_deref_mut() {
                    quick.set_cell(x, y, true);
                }
                turbo.set_cell_alive(x as i64, y as i64);
            }
        }
    }
}

fn run_checked(config: TurboLifeConfig) {
    let mut quick = QuickLife::new();
    let mut turbo = TurboLife::with_config(config);
    seed_random_world(&mut turbo, Some(&mut quick));

    let mut quick_total_duration = std::time::Duration::ZERO;
    let mut turbo_total_duration = std::time::Duration::ZERO;
    let mut quick_prev_duration = std::time::Duration::ZERO;
    let mut turbo_prev_duration = std::time::Duration::ZERO;

    for checkpoint in 1..=(TOTAL_ITERATIONS / CHECK_INTERVAL) {
        let iteration = checkpoint * CHECK_INTERVAL;

        let start = Instant::now();
        quick.step(CHECK_INTERVAL);
        quick_total_duration += start.elapsed();

        let start = Instant::now();
        turbo.step_n(CHECK_INTERVAL);
        turbo_total_duration += start.elapsed();

        let quick_phase = quick_total_duration - quick_prev_duration;
        let turbo_phase = turbo_total_duration - turbo_prev_duration;
        quick_prev_duration = quick_total_duration;
        turbo_prev_duration = turbo_total_duration;

        let quick_ms = quick_phase.as_secs_f64() * 1000.0;
        let turbo_ms = turbo_phase.as_secs_f64() * 1000.0;
        let quick_avg_ms = quick_ms / CHECK_INTERVAL as f64;
        let turbo_avg_ms = turbo_ms / CHECK_INTERVAL as f64;

        let quick_population = quick.population();
        let turbo_population = turbo.population();
        let match_status = if quick_population == turbo_population {
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!(
            "Iteration {iteration}: QuickLife pop = {quick_population}, TurboLife pop = {turbo_population} [{match_status}]"
        );
        println!(
            "  QuickLife: {quick_ms:.3} ms total, {quick_avg_ms:.6} ms/iter | TurboLife: {turbo_ms:.3} ms total, {turbo_avg_ms:.6} ms/iter"
        );
    }

    let quick_total_ms = quick_total_duration.as_secs_f64() * 1000.0;
    let turbo_total_ms = turbo_total_duration.as_secs_f64() * 1000.0;
    let quick_avg_ms = quick_total_ms / TOTAL_ITERATIONS as f64;
    let turbo_avg_ms = turbo_total_ms / TOTAL_ITERATIONS as f64;
    let speedup = quick_total_ms / turbo_total_ms;

    println!("\n--- Summary ({TOTAL_ITERATIONS} iterations) ---");
    println!("QuickLife: {quick_total_ms:.3} ms total, {quick_avg_ms:.6} ms/iter");
    println!("TurboLife: {turbo_total_ms:.3} ms total, {turbo_avg_ms:.6} ms/iter");
    println!("Speedup (QuickLife / TurboLife): {speedup:.2}x");
}

fn run_pgo_train(config: TurboLifeConfig) {
    let mut turbo = TurboLife::with_config(config);
    seed_random_world(&mut turbo, None);
    turbo.step_n(TOTAL_ITERATIONS);
    std::hint::black_box(turbo.population());
}

fn main() {
    let args = parse_args();
    if args.pgo_train {
        run_pgo_train(args.config);
    } else {
        run_checked(args.config);
    }
}
