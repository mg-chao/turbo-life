//! Rule table generation for B3/S23.

pub struct RuleTable {
    table: [u8; 65_536],
}

impl RuleTable {
    pub fn new() -> Self {
        let mut table = [0u8; 65_536];
        for pattern in 0u16..=u16::MAX {
            table[pattern as usize] = output_for(pattern);
        }
        Self { table }
    }

    #[inline(always)]
    pub fn lookup(&self, pattern: u16) -> u8 {
        self.table[pattern as usize]
    }
}

fn output_for(pattern: u16) -> u8 {
    let mut out = 0u8;
    for (dy, bit) in [(0usize, 5u8), (0usize, 4u8), (1usize, 1u8), (1usize, 0u8)] {
        let y = 1 + dy;
        let x = if bit == 5 || bit == 1 { 1 } else { 2 };
        let mut neighbors = 0u8;
        for ny in (y - 1)..=(y + 1) {
            for nx in (x - 1)..=(x + 1) {
                if nx == x && ny == y {
                    continue;
                }
                neighbors += cell_at(pattern, nx, ny);
            }
        }
        let alive = cell_at(pattern, x, y) == 1;
        let next_alive = if alive {
            neighbors == 2 || neighbors == 3
        } else {
            neighbors == 3
        };
        if next_alive {
            out |= 1 << bit;
        }
    }
    out
}

fn cell_at(pattern: u16, x: usize, y: usize) -> u8 {
    let bit = 15 - (y * 4 + x);
    ((pattern >> bit) & 1) as u8
}

#[cfg(test)]
mod tests {
    use super::RuleTable;

    fn cell_at(pattern: u16, x: usize, y: usize) -> u8 {
        let bit = 15 - (y * 4 + x);
        ((pattern >> bit) & 1) as u8
    }

    fn expected_output(pattern: u16) -> u8 {
        let mut out = 0u8;
        for (dy, bit) in [(0usize, 5u8), (0usize, 4u8), (1usize, 1u8), (1usize, 0u8)] {
            let y = 1 + dy;
            let x = if bit == 5 || bit == 1 { 1 } else { 2 };
            let mut neighbors = 0u8;
            for ny in (y - 1)..=(y + 1) {
                for nx in (x - 1)..=(x + 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    neighbors += cell_at(pattern, nx, ny);
                }
            }
            let alive = cell_at(pattern, x, y) == 1;
            let next_alive = if alive {
                neighbors == 2 || neighbors == 3
            } else {
                neighbors == 3
            };
            if next_alive {
                out |= 1 << bit;
            }
        }
        out
    }

    #[test]
    fn rule_table_matches_reference() {
        let table = RuleTable::new();
        for pattern in 0u16..=u16::MAX {
            let expected = expected_output(pattern);
            let got = table.lookup(pattern);
            assert_eq!(
                got, expected,
                "pattern {:04x} expected {:02x} got {:02x}",
                pattern, expected, got
            );
        }
    }
}
