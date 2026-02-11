use std::collections::HashSet;

use quick_life::quicklife::QuickLife;
use quick_life::turbolife::TurboLife;
use rand::RngCore;
use rand::SeedableRng;

fn collect_quick(engine: &QuickLife) -> HashSet<(i32, i32)> {
    let mut out = HashSet::new();
    engine.for_each_live(|x, y| {
        out.insert((x, y));
    });
    out
}

fn collect_turbo(engine: &TurboLife) -> HashSet<(i32, i32)> {
    let mut out = HashSet::new();
    engine.for_each_live(|x, y| {
        out.insert((x as i32, y as i32));
    });
    out
}

fn run_parity_case(width: i32, height: i32, density: f64, steps: u64, seed: u64) {
    let mut quick = QuickLife::new();
    let mut turbo = TurboLife::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let threshold = (u64::MAX as f64 * density) as u64;

    for y in -(height / 2)..=(height / 2) {
        for x in -(width / 2)..=(width / 2) {
            if rng.next_u64() <= threshold {
                quick.set_cell(x, y, true);
                turbo.set_cell(x as i64, y as i64, true);
            }
        }
    }

    quick.step(steps);
    turbo.step_n(steps);

    let quick_population = quick.population();
    let turbo_population = turbo.population();
    assert_eq!(
        quick_population, turbo_population,
        "population mismatch for density {density} seed {seed}"
    );

    let quick_live = collect_quick(&quick);
    let turbo_live = collect_turbo(&turbo);
    assert_eq!(
        quick_live, turbo_live,
        "live-set mismatch for density {density} seed {seed}"
    );
}

#[test]
fn parity_sparse_mid_dense() {
    run_parity_case(96, 96, 0.10, 6, 0xA1);
    run_parity_case(96, 96, 0.42, 6, 0xB2);
    run_parity_case(96, 96, 0.83, 4, 0xC3);
}

#[test]
fn parity_multiple_seeds() {
    for seed in [11u64, 22, 33, 44] {
        run_parity_case(72, 72, 0.35, 7, seed);
    }
}
