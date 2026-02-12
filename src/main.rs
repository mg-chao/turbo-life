use rand::RngCore;
use rand::SeedableRng;
use std::time::Instant;
use turbo_life::quicklife::QuickLife;
use turbo_life::turbolife::TurboLife;

fn main() {
    let mut quick = QuickLife::new();
    let mut turbo = TurboLife::new();
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

    let iterations = 1000u64;

    let start = Instant::now();
    quick.step(iterations);
    let quick_duration = start.elapsed();

    let start = Instant::now();
    turbo.step_n(iterations);
    let turbo_duration = start.elapsed();

    let quick_total_ms = quick_duration.as_secs_f64() * 1000.0;
    let quick_avg_ms = quick_total_ms / iterations as f64;
    let turbo_total_ms = turbo_duration.as_secs_f64() * 1000.0;
    let turbo_avg_ms = turbo_total_ms / iterations as f64;

    println!("QuickLife total iteration time: {quick_total_ms:.3} ms");
    println!("QuickLife average time per iteration: {quick_avg_ms:.6} ms");
    println!("TurboLife total iteration time: {turbo_total_ms:.3} ms");
    println!("TurboLife average time per iteration: {turbo_avg_ms:.6} ms");

    let quick_population = quick.population();
    let turbo_population = turbo.population();
    println!("QuickLife population: {quick_population}");
    println!("TurboLife population: {turbo_population}");
}
