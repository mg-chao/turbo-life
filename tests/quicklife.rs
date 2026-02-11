use std::collections::HashSet;

use quick_life::quicklife::QuickLife;

fn set_cells(engine: &mut QuickLife, cells: &[(i32, i32)]) {
    for &(x, y) in cells {
        engine.set_cell(x, y, true);
    }
}

fn assert_alive(engine: &QuickLife, cells: &[(i32, i32)]) {
    for &(x, y) in cells {
        assert!(engine.get_cell(x, y), "expected alive at ({x},{y})");
    }
}

fn assert_dead(engine: &QuickLife, cells: &[(i32, i32)]) {
    for &(x, y) in cells {
        assert!(!engine.get_cell(x, y), "expected dead at ({x},{y})");
    }
}

fn collect_live(engine: &QuickLife) -> HashSet<(i32, i32)> {
    let mut out = HashSet::new();
    engine.for_each_live(|x, y| {
        out.insert((x, y));
    });
    out
}

fn step_naive(cells: &HashSet<(i32, i32)>) -> HashSet<(i32, i32)> {
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
    let mut engine = QuickLife::new();
    engine.set_cell(3, -2, true);
    assert!(engine.get_cell(3, -2));
    engine.set_cell(3, -2, false);
    assert!(!engine.get_cell(3, -2));
}

#[test]
fn block_is_stable() {
    let mut engine = QuickLife::new();
    let block = [(0, 0), (1, 0), (0, 1), (1, 1)];
    set_cells(&mut engine, &block);

    engine.step(1);

    assert_alive(&engine, &block);
    assert_dead(&engine, &[(2, 0), (2, 1), (-1, 0), (-1, 1)]);
}

#[test]
fn population_and_bounds() {
    let mut engine = QuickLife::new();
    assert_eq!(engine.population(), 0);
    assert!(engine.is_empty());

    let block = [(0, 0), (1, 0), (0, 1), (1, 1)];
    set_cells(&mut engine, &block);

    assert_eq!(engine.population(), 4);
    assert!(!engine.is_empty());
    assert_eq!(engine.bounds(), Some((0, 0, 1, 1)));
}

#[test]
fn blinker_oscillates() {
    let mut engine = QuickLife::new();
    set_cells(&mut engine, &[(0, 0), (1, 0), (2, 0)]);

    engine.step(1);

    assert_alive(&engine, &[(1, -1), (1, 0), (1, 1)]);
    assert_dead(&engine, &[(0, 0), (2, 0)]);

    engine.step(1);

    assert_alive(&engine, &[(0, 0), (1, 0), (2, 0)]);
    assert_dead(&engine, &[(1, -1), (1, 1)]);
}

#[test]
fn glider_moves_down_right_every_four_steps() {
    let mut engine = QuickLife::new();
    let glider = [(1, 0), (2, -1), (0, -2), (1, -2), (2, -2)];
    set_cells(&mut engine, &glider);

    engine.step(4);

    let shifted = [(2, -1), (3, -2), (1, -3), (2, -3), (3, -3)];
    assert_alive(&engine, &shifted);
    assert_dead(&engine, &[(1, 0), (0, -2), (1, -2), (2, -2)]);
}

#[test]
fn quicklife_matches_naive_for_small_pattern() {
    let mut engine = QuickLife::new();
    let seed = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 2), (2, 2), (3, 1)];
    set_cells(&mut engine, &seed);

    let mut naive: HashSet<(i32, i32)> = seed.into_iter().collect();

    for _ in 0..5 {
        let live = collect_live(&engine);
        assert_eq!(live, naive, "QuickLife diverged from naive stepper");
        engine.step(1);
        naive = step_naive(&naive);
    }
}
