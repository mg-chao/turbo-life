use rand::RngCore;
use rand::SeedableRng;
use std::env;
use std::time::Instant;
use turbo_life::turbolife::{KernelBackend, TurboLife, TurboLifeConfig};

#[derive(Clone, Debug)]
struct BenchConfig {
    size: i64,
    density: f64,
    warmup: u64,
    iters: u64,
    seed: u64,
    threads: Option<usize>,
    json: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            size: 2048,
            density: 0.42,
            warmup: 3,
            iters: 30,
            seed: 0xA5A5_5EED_7788_1122,
            threads: None,
            json: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RunResult {
    total_ms: f64,
    avg_ms: f64,
    population: u64,
}

fn parse_args() -> BenchConfig {
    let mut cfg = BenchConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--size" => {
                if let Some(v) = args.next() {
                    cfg.size = v.parse().expect("--size expects i64");
                }
            }
            "--density" => {
                if let Some(v) = args.next() {
                    cfg.density = v.parse().expect("--density expects f64");
                }
            }
            "--warmup" => {
                if let Some(v) = args.next() {
                    cfg.warmup = v.parse().expect("--warmup expects u64");
                }
            }
            "--iters" => {
                if let Some(v) = args.next() {
                    cfg.iters = v.parse().expect("--iters expects u64");
                }
            }
            "--seed" => {
                if let Some(v) = args.next() {
                    cfg.seed = if let Some(hex) = v.strip_prefix("0x") {
                        u64::from_str_radix(hex, 16).expect("--seed hex parse failed")
                    } else {
                        v.parse().expect("--seed expects u64")
                    };
                }
            }
            "--threads" => {
                if let Some(v) = args.next() {
                    cfg.threads = Some(v.parse().expect("--threads expects usize"));
                }
            }
            "--json" => {
                cfg.json = true;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    cfg
}

fn seed_board_bulk(engine: &mut TurboLife, size: i64, density: f64, seed: u64) {
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

fn run_backend(cfg: &BenchConfig, backend: KernelBackend) -> RunResult {
    let mut config = TurboLifeConfig::default().kernel(backend);
    if let Some(t) = cfg.threads {
        config = config.thread_count(t);
    }
    let mut engine = TurboLife::with_config(config);
    seed_board_bulk(&mut engine, cfg.size, cfg.density, cfg.seed);

    if cfg.warmup > 0 {
        engine.step_n(cfg.warmup);
    }

    let start = Instant::now();
    engine.step_n(cfg.iters);
    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / cfg.iters as f64;
    let population = engine.population();

    RunResult {
        total_ms,
        avg_ms,
        population,
    }
}

fn main() {
    let cfg = parse_args();
    let scalar = run_backend(&cfg, KernelBackend::Scalar);

    let avx2_supported = {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    let avx2 = if avx2_supported {
        Some(run_backend(&cfg, KernelBackend::Avx2))
    } else {
        None
    };

    if cfg.json {
        match avx2 {
            Some(avx2_res) => {
                println!(
                    "{{\"size\":{},\"density\":{},\"warmup\":{},\"iters\":{},\"seed\":{},\"threads\":{},\"scalar\":{{\"total_ms\":{:.6},\"avg_ms\":{:.6},\"population\":{}}},\"avx2\":{{\"supported\":true,\"total_ms\":{:.6},\"avg_ms\":{:.6},\"population\":{}}},\"speedup\":{:.6}}}",
                    cfg.size,
                    cfg.density,
                    cfg.warmup,
                    cfg.iters,
                    cfg.seed,
                    cfg.threads.unwrap_or(0),
                    scalar.total_ms,
                    scalar.avg_ms,
                    scalar.population,
                    avx2_res.total_ms,
                    avx2_res.avg_ms,
                    avx2_res.population,
                    scalar.avg_ms / avx2_res.avg_ms,
                );
            }
            None => {
                println!(
                    "{{\"size\":{},\"density\":{},\"warmup\":{},\"iters\":{},\"seed\":{},\"threads\":{},\"scalar\":{{\"total_ms\":{:.6},\"avg_ms\":{:.6},\"population\":{}}},\"avx2\":{{\"supported\":false}}}}",
                    cfg.size,
                    cfg.density,
                    cfg.warmup,
                    cfg.iters,
                    cfg.seed,
                    cfg.threads.unwrap_or(0),
                    scalar.total_ms,
                    scalar.avg_ms,
                    scalar.population,
                );
            }
        }
    } else {
        println!(
            "scalar: total_ms={:.6}, avg_ms={:.6}, population={}",
            scalar.total_ms, scalar.avg_ms, scalar.population
        );
        match avx2 {
            Some(avx2_res) => {
                println!(
                    "avx2: total_ms={:.6}, avg_ms={:.6}, population={}, speedup={:.3}x",
                    avx2_res.total_ms,
                    avx2_res.avg_ms,
                    avx2_res.population,
                    scalar.avg_ms / avx2_res.avg_ms,
                );
            }
            None => {
                println!("avx2: unsupported");
            }
        }
    }
}
