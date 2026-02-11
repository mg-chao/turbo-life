use std::collections::HashSet;

use turbo_life::turbolife::TurboLife;
use rand::Rng;
use rand::SeedableRng;

fn set_cells(engine: &mut TurboLife, cells: &[(i64, i64)]) {
    for &(x, y) in cells {
        engine.set_cell(x, y, true);
    }
}

fn collect_live(engine: &TurboLife) -> HashSet<(i64, i64)> {
    let mut out = HashSet::new();
    engine.for_each_live(|x, y| {
        out.insert((x, y));
    });
    out
}

fn assert_alive(engine: &TurboLife, cells: &[(i64, i64)]) {
    for &(x, y) in cells {
        assert!(engine.get_cell(x, y), "expected alive at ({x},{y})");
    }
}

fn assert_dead(engine: &TurboLife, cells: &[(i64, i64)]) {
    for &(x, y) in cells {
        assert!(!engine.get_cell(x, y), "expected dead at ({x},{y})");
    }
}

fn step_naive(cells: &HashSet<(i64, i64)>) -> HashSet<(i64, i64)> {
    let mut next = HashSet::new();
    let mut candidates = HashSet::new();
    for &(x, y) in cells {
        for dy in -1..=1 {
            for dx in -1..=1 {
                candidates.insert((x + dx, y + dy));
            }
        }
    }

    for (x, y) in candidates {
        let mut neighbors = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if cells.contains(&(x + dx, y + dy)) {
                    neighbors += 1;
                }
            }
        }
        let alive = cells.contains(&(x, y));
        let next_alive = if alive {
            neighbors == 2 || neighbors == 3
        } else {
            neighbors == 3
        };
        if next_alive {
            next.insert((x, y));
        }
    }

    next
}

#[test]
fn set_and_get_cell_round_trip() {
    let mut engine = TurboLife::new();
    engine.set_cell(3, -2, true);
    assert!(engine.get_cell(3, -2));
    engine.set_cell(3, -2, false);
    assert!(!engine.get_cell(3, -2));
}

#[test]
fn block_is_stable() {
    let mut engine = TurboLife::new();
    let block = [(0, 0), (1, 0), (0, 1), (1, 1)];
    set_cells(&mut engine, &block);

    engine.step();

    assert_alive(&engine, &block);
    assert_dead(&engine, &[(2, 0), (2, 1), (-1, 0), (-1, 1)]);
}

#[test]
fn blinker_oscillates() {
    let mut engine = TurboLife::new();
    set_cells(&mut engine, &[(0, 0), (1, 0), (2, 0)]);

    engine.step();
    assert_alive(&engine, &[(1, -1), (1, 0), (1, 1)]);
    assert_dead(&engine, &[(0, 0), (2, 0)]);

    engine.step();
    assert_alive(&engine, &[(0, 0), (1, 0), (2, 0)]);
    assert_dead(&engine, &[(1, -1), (1, 1)]);
}

#[test]
fn glider_moves_down_right_every_four_steps() {
    let mut engine = TurboLife::new();
    let glider = [(1, 0), (2, -1), (0, -2), (1, -2), (2, -2)];
    set_cells(&mut engine, &glider);

    engine.step_n(4);

    let shifted = [(2, -1), (3, -2), (1, -3), (2, -3), (3, -3)];
    assert_alive(&engine, &shifted);
    assert_dead(&engine, &[(1, 0), (0, -2), (1, -2), (2, -2)]);
}

#[test]
fn matches_naive_on_small_random_seed() {
    let mut engine = TurboLife::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBADC0FFEE);

    let mut naive = HashSet::new();
    for y in -8..=8 {
        for x in -8..=8 {
            if rng.random::<f64>() < 0.33 {
                engine.set_cell(x, y, true);
                naive.insert((x, y));
            }
        }
    }

    for _ in 0..8 {
        assert_eq!(collect_live(&engine), naive);
        engine.step();
        naive = step_naive(&naive);
    }
}

#[test]
fn tile_boundary_crossing_pattern_evolves() {
    let mut engine = TurboLife::new();
    set_cells(
        &mut engine,
        &[(63, 63), (64, 63), (63, 64), (64, 64), (62, 63), (65, 64)],
    );

    let before = collect_live(&engine);
    engine.step_n(3);
    let after = collect_live(&engine);

    assert!(!after.is_empty());
    assert_ne!(before, after);
}

#[test]
fn empty_universe_stays_empty() {
    let mut engine = TurboLife::new();
    engine.step_n(10);
    assert_eq!(engine.population(), 0);
    assert!(engine.is_empty());
    assert_eq!(engine.bounds(), None);
}

#[test]
fn mid_simulation_set_cell_mutation_works() {
    let mut engine = TurboLife::new();
    set_cells(&mut engine, &[(0, 0), (1, 0), (2, 0)]);

    engine.step();
    engine.set_cell(5, 5, true);
    assert!(engine.get_cell(5, 5));
    engine.step();
    assert!(!engine.is_empty());
}

#[test]
fn prune_behavior_indirectly_visible_after_extinction() {
    let mut engine = TurboLife::new();
    engine.set_cell(0, 0, true);
    engine.step_n(5);

    assert_eq!(engine.population(), 0);
    assert!(engine.is_empty());
    assert_eq!(engine.bounds(), None);
}

#[test]
fn deterministic_across_thread_counts() {
    let mut initial = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xD37E_A515);
    for y in -24..=24 {
        for x in -24..=24 {
            if rng.random::<f64>() < 0.3 {
                initial.push((x, y));
            }
        }
    }

    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build thread pool");

        pool.install(|| {
            let mut engine = TurboLife::new();
            for &(x, y) in &initial {
                engine.set_cell(x, y, true);
            }
            engine.step_n(12);
            let pop = engine.population();
            let live = collect_live(&engine);
            (pop, live)
        })
    };

    let (pop1, live1) = run(1);
    let (pop4, live4) = run(4);

    assert_eq!(pop1, pop4);
    assert_eq!(live1, live4);
}
