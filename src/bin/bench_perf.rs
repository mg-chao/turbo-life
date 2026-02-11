#[cfg(feature = "mimalloc-global")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rand::RngCore;
use rand::SeedableRng;
use std::env;
use std::time::Instant;
use turbo_life::turbolife::TurboLife;

#[derive(Clone, Debug)]
struct BenchConfig {
    size: i64,
    density: f64,
    warmup: u64,
    iters: u64,
    seed: u64,
    json: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            size: 4096,
            density: 0.42,
            warmup: 3,
            iters: 50,
            seed: 0x5EED_1234_ABCD_EF01,
            json: false,
        }
    }
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
            "--json" => {
                cfg.json = true;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    cfg
}

fn seed_board(engine: &mut TurboLife, size: i64, density: f64, seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let threshold = (u64::MAX as f64 * density) as u64;
    for y in 0..size {
        for x in 0..size {
            if rng.next_u64() <= threshold {
                engine.set_cell(x, y, true);
            }
        }
    }
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

fn main() {
    let cfg = parse_args();

    let mut engine = TurboLife::new();
    if std::env::var("TURBOLIFE_BENCH_SEED_BULK").ok().as_deref() == Some("1") {
        seed_board_bulk(&mut engine, cfg.size, cfg.density, cfg.seed);
    } else {
        seed_board(&mut engine, cfg.size, cfg.density, cfg.seed);
    }

    if cfg.warmup > 0 {
        engine.step_n(cfg.warmup);
    }

    let start = Instant::now();
    engine.step_n(cfg.iters);
    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / cfg.iters as f64;
    let population = engine.population();

    if cfg.json {
        println!(
            "{{\"size\":{},\"density\":{},\"warmup\":{},\"iters\":{},\"seed\":{},\"total_ms\":{:.6},\"avg_ms\":{:.6},\"population\":{}}}",
            cfg.size, cfg.density, cfg.warmup, cfg.iters, cfg.seed, total_ms, avg_ms, population,
        );
    } else {
        println!(
            "size={},density={},warmup={},iters={},seed={},total_ms={:.6},avg_ms={:.6},population={}",
            cfg.size, cfg.density, cfg.warmup, cfg.iters, cfg.seed, total_ms, avg_ms, population,
        );
    }
}
