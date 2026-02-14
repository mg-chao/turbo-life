use rand::RngCore;
use rand::SeedableRng;
use std::time::Instant;
use turbo_life::quicklife::QuickLife;
use turbo_life::turbolife::{KernelBackend, TurboLife, TurboLifeConfig};

fn parse_args() -> TurboLifeConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut config = TurboLifeConfig::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                i += 1;
                let n: usize = args[i]
                    .parse()
                    .expect("--threads requires a positive integer");
                config = config.thread_count(n);
            }
            "--max-threads" => {
                i += 1;
                let n: usize = args[i]
                    .parse()
                    .expect("--max-threads requires a positive integer");
                config = config.max_threads(n);
            }
            "--kernel" => {
                i += 1;
                let backend = match args[i].to_ascii_lowercase().as_str() {
                    "scalar" => KernelBackend::Scalar,
                    "avx2" => KernelBackend::Avx2,
                    other => panic!("unknown kernel backend: {other} (expected scalar or avx2)"),
                };
                config = config.kernel(backend);
            }
            other => panic!(
                "unknown argument: {other}\nusage: turbo-life [--threads N] [--max-threads N] [--kernel scalar|avx2]"
            ),
        }
        i += 1;
    }
    config
}

fn main() {
    let config = parse_args();

    let mut quick = QuickLife::new();
    let mut turbo = TurboLife::with_config(config);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_1234_ABCD_EF01);
    let threshold = (u64::MAX as f64 * 0.42) as u64;

    for y in 0..=4096 {
        for x in 0..=4096 {
            if rng.next_u64() <= threshold {
                quick.set_cell(x, y, true);
                turbo.set_cell(x as i64, y as i64, true);
            }
        }
    }

    let total_iterations = 2000u64;
    let check_interval = 1000u64;
    let mut quick_total_duration = std::time::Duration::ZERO;
    let mut turbo_total_duration = std::time::Duration::ZERO;
    let mut quick_prev_duration = std::time::Duration::ZERO;
    let mut turbo_prev_duration = std::time::Duration::ZERO;

    for checkpoint in 1..=(total_iterations / check_interval) {
        let iteration = checkpoint * check_interval;

        let start = Instant::now();
        quick.step(check_interval);
        quick_total_duration += start.elapsed();

        let start = Instant::now();
        turbo.step_n(check_interval);
        turbo_total_duration += start.elapsed();

        let quick_phase = quick_total_duration - quick_prev_duration;
        let turbo_phase = turbo_total_duration - turbo_prev_duration;
        quick_prev_duration = quick_total_duration;
        turbo_prev_duration = turbo_total_duration;

        let quick_ms = quick_phase.as_secs_f64() * 1000.0;
        let turbo_ms = turbo_phase.as_secs_f64() * 1000.0;
        let quick_avg_ms = quick_ms / check_interval as f64;
        let turbo_avg_ms = turbo_ms / check_interval as f64;

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
    let quick_avg_ms = quick_total_ms / total_iterations as f64;
    let turbo_avg_ms = turbo_total_ms / total_iterations as f64;
    let speedup = quick_total_ms / turbo_total_ms;

    println!("\n--- Summary ({total_iterations} iterations) ---");
    println!("QuickLife: {quick_total_ms:.3} ms total, {quick_avg_ms:.6} ms/iter");
    println!("TurboLife: {turbo_total_ms:.3} ms total, {turbo_avg_ms:.6} ms/iter");
    println!("Speedup (QuickLife / TurboLife): {speedup:.2}x");
}
