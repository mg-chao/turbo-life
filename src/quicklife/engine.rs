//! QuickLife engine core.

use super::arena::{Arenas, MAX_LEVELS, NodeId, SuperId, TileId};
use super::rules::RuleTable;

pub struct QuickLife {
    arenas: Arenas,
    rule_table: RuleTable,
    root: NodeId,
    rootlev: usize,
    nullroot: NodeId,
    generation: u64,
    deltaforward: u32,
    cleandowncounter: i32,
    pop_valid: bool,
    population: u64,
    min: i32,
    max: i32,
    bmin: i32,
    bmax: i32,
    minlow32: i32,
}

impl Default for QuickLife {
    fn default() -> Self {
        Self::new()
    }
}

impl QuickLife {
    pub fn new() -> Self {
        let arenas = Arenas::new();
        let root = arenas.empty_tile.0;
        let mut engine = Self {
            arenas,
            rule_table: RuleTable::new(),
            root,
            rootlev: 0,
            nullroot: root,
            generation: 0,
            deltaforward: 0,
            cleandowncounter: 63,
            pop_valid: false,
            population: 0,
            min: 0,
            max: 31,
            bmin: 0,
            bmax: 31,
            minlow32: 0,
        };

        engine.uproot();
        engine
    }

    pub fn set_cell(&mut self, x: i32, y: i32, alive: bool) {
        let newstate = if alive { 1u32 } else { 0u32 };
        let mut x = x;
        let mut y = -y;
        let odd = (self.generation & 1) == 1;
        if odd {
            x -= 1;
            y -= 1;
        }

        while x < self.min || x > self.max || y < self.min || y > self.max {
            self.uproot();
        }

        let xdel = (x >> 5) - self.minlow32;
        let ydel = (y >> 5) - self.minlow32;
        let xc = x - (self.minlow32 << 5);
        let yc = y - (self.minlow32 << 5);

        if self.root == self.nullroot {
            let new_root = self.arenas.new_supertile(self.rootlev);
            self.root = new_root.0;
        }

        let mut node = self.root;
        let mut lev = self.rootlev;
        while lev > 0 {
            let mut d = 1u32;
            let i: usize;
            if (lev & 1) == 1 {
                let s = ((lev >> 1) + lev - 1) as i32;
                i = ((xdel >> s) & 7) as usize;
                let s_mask = (1i32 << (s + 5)) - 2;
                if (xc & s_mask) == if odd { s_mask } else { 0 } {
                    d += 2;
                }
                if (yc & s_mask) == if odd { s_mask } else { 0 } {
                    d += d << 9;
                }
            } else {
                let s = ((lev >> 1) + lev - 3) as i32;
                i = ((ydel >> s) & 7) as usize;
                let mut s_mask = (1i32 << (s + 5)) - 2;
                if (yc & s_mask) == if odd { s_mask } else { 0 } {
                    d += 2;
                }
                s_mask |= s_mask << 3;
                if (xc & s_mask) == if odd { s_mask } else { 0 } {
                    d += d << 9;
                }
            }

            let child = self.arenas.supers[node].d[i];
            let need_alloc = child == self.arenas.nullroots[lev - 1];
            let new_child = if need_alloc {
                if lev == 1 {
                    self.arenas.new_tile().0
                } else {
                    self.arenas.new_supertile(lev - 1).0
                }
            } else {
                child
            };
            {
                let supertile = &mut self.arenas.supers[node];
                if odd {
                    supertile.flags |= (d << i) | 0xf0000000;
                } else {
                    supertile.flags |= (d << (7 - i)) | 0xf0000000;
                }
                if need_alloc {
                    supertile.d[i] = new_child;
                }
            }
            node = new_child;

            lev -= 1;
        }

        let x = x & 31;
        let y = y & 31;
        let tile_id = TileId(node);
        let brick_index = ((y >> 3) & 0x3) as usize;
        let existing_brick = self.arenas.tiles[tile_id.0].b[brick_index];
        if existing_brick == self.arenas.empty_brick {
            let new_brick = self.arenas.new_brick();
            self.arenas.tiles[tile_id.0].b[brick_index] = new_brick;
        }
        let tile = &mut self.arenas.tiles[tile_id.0];

        let slice_index = ((x >> 2) & 0x7) as usize;
        let bit = 1u32 << (31 - (y & 7) * 4 - (x & 3));
        if odd {
            let mor = if (x & 2) != 0 { 3 } else { 1 } << ((x >> 2) & 0x7);
            tile.c[brick_index + 1] |= mor as u16;
            tile.flags = u32::MAX;
            if (y & 6) == 6 {
                tile.c[brick_index + 2] |= mor as u16;
            }
            let brick_id = tile.b[brick_index];
            if newstate == 1 {
                self.arenas.bricks[brick_id.0].d[8 + slice_index] |= bit;
            } else {
                self.arenas.bricks[brick_id.0].d[8 + slice_index] &= !bit;
            }
            tile.localdeltaforward |= bit;
        } else {
            let mor = if (x & 2) != 0 { 1 } else { 3 } << (7 - ((x >> 2) & 0x7));
            tile.c[brick_index + 1] |= mor as u16;
            tile.flags = u32::MAX;
            if (y & 6) == 0 {
                tile.c[brick_index] |= mor as u16;
            }
            let brick_id = tile.b[brick_index];
            if newstate == 1 {
                self.arenas.bricks[brick_id.0].d[slice_index] |= bit;
            } else {
                self.arenas.bricks[brick_id.0].d[slice_index] &= !bit;
            }
            tile.localdeltaforward |= bit;
        }
        self.pop_valid = false;
    }

    pub fn get_cell(&self, x: i32, y: i32) -> bool {
        let mut x = x;
        let mut y = -y;
        let odd = (self.generation & 1) == 1;
        if odd {
            x -= 1;
            y -= 1;
        }

        if x < self.min || x > self.max || y < self.min || y > self.max {
            return false;
        }

        let xdel = (x >> 5) - self.minlow32;
        let ydel = (y >> 5) - self.minlow32;

        if self.root == self.nullroot {
            return false;
        }

        let mut node = self.root;
        let mut lev = self.rootlev;
        while lev > 0 {
            let i: usize = if (lev & 1) == 1 {
                let s = ((lev >> 1) + lev - 1) as i32;
                ((xdel >> s) & 7) as usize
            } else {
                let s = ((lev >> 1) + lev - 3) as i32;
                ((ydel >> s) & 7) as usize
            };
            let supertile = &self.arenas.supers[node];
            let child = supertile.d[i];
            if child == self.arenas.nullroots[lev - 1] {
                return false;
            }
            node = child;
            lev -= 1;
        }

        let x = x & 31;
        let y = y & 31;
        let tile = &self.arenas.tiles[node];
        let brick_index = ((y >> 3) & 0x3) as usize;
        if tile.b[brick_index] == self.arenas.empty_brick {
            return false;
        }

        let slice_index = ((x >> 2) & 0x7) as usize;
        let bit = 1u32 << (31 - (y & 7) * 4 - (x & 3));
        let brick = &self.arenas.bricks[tile.b[brick_index].0];
        if odd {
            (brick.d[8 + slice_index] & bit) != 0
        } else {
            (brick.d[slice_index] & bit) != 0
        }
    }

    pub fn step(&mut self, generations: u64) {
        for _ in 0..generations {
            self.dogen();
        }
    }

    pub fn population(&mut self) -> u64 {
        if self.pop_valid {
            return self.population;
        }
        let count = self.popcount_node(self.root, self.rootlev);
        self.population = count;
        self.pop_valid = true;
        count
    }

    pub fn is_empty(&mut self) -> bool {
        self.population() == 0
    }

    pub fn bounds(&self) -> Option<(i32, i32, i32, i32)> {
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut seen = false;

        self.for_each_live(|x, y| {
            seen = true;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        });

        if seen {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    pub fn for_each_live<F>(&self, mut f: F)
    where
        F: FnMut(i32, i32),
    {
        let odd = (self.generation & 1) == 1;
        let mut spans = vec![(1i32, 1i32); self.rootlev + 1];
        for lev in 1..=self.rootlev {
            let (w, h) = spans[lev - 1];
            spans[lev] = if (lev & 1) == 1 {
                (w * 8, h)
            } else {
                (w, h * 8)
            };
        }
        self.visit_node(self.root, self.rootlev, 0, 0, odd, &spans, &mut f);
    }

    fn dogen(&mut self) {
        while self.uproot_needed() {
            self.uproot();
        }
        if (self.generation & 1) == 1 {
            self.doquad10(
                SuperId(self.root),
                SuperId(self.nullroot),
                SuperId(self.nullroot),
                SuperId(self.nullroot),
                self.rootlev,
            );
        } else {
            self.doquad01(
                SuperId(self.root),
                SuperId(self.nullroot),
                SuperId(self.nullroot),
                SuperId(self.nullroot),
                self.rootlev,
            );
        }
        self.deltaforward = 0;
        self.generation += 1;
        self.pop_valid = false;
        self.cleandowncounter -= 1;
        if self.cleandowncounter == 0 {
            self.cleandowncounter = 63;
            self.mdelete(self.root, self.rootlev);
        }
    }

    fn mdelete(&mut self, node: NodeId, lev: usize) -> NodeId {
        if lev == 0 {
            if node == self.arenas.empty_tile.0 {
                return node;
            }
            if (self.arenas.tiles[node].flags & 0xf) == 0 {
                return node;
            }
            let mut tile = self.arenas.tiles[node];
            if (tile.flags & 0xf) != 0 {
                let mut seen = 0;
                for i in 0..4 {
                    let brick_id = tile.b[i];
                    if brick_id == self.arenas.empty_brick {
                        continue;
                    }
                    if (tile.flags & (1 << i)) != 0 {
                        let brick = &self.arenas.bricks[brick_id.0];
                        let mut has_data = false;
                        for word in brick.d {
                            if word != 0 {
                                has_data = true;
                                break;
                            }
                        }
                        if has_data {
                            seen += 1;
                        } else {
                            self.arenas.release_brick(brick_id);
                            tile.b[i] = self.arenas.empty_brick;
                        }
                    } else {
                        seen += 1;
                    }
                }

                let c_any = ((tile.c[1] | tile.c[2] | tile.c[3] | tile.c[4]) & 0xff) != 0;
                let edge = if (self.generation & 1) == 1 {
                    tile.c[5]
                } else {
                    tile.c[0]
                };

                if seen != 0 || c_any || edge != 0 {
                    tile.flags &= 0xffff_fff0;
                } else {
                    self.arenas.release_tile(TileId(node));
                    return self.arenas.nullroots[lev];
                }
            }
            self.arenas.tiles[node] = tile;
            node
        } else {
            if node == self.arenas.nullroots[lev] {
                return node;
            }
            if (self.arenas.supers[node].flags & 0x1000_0000) == 0 {
                return node;
            }
            let mut supertile = self.arenas.supers[node];
            if (supertile.flags & 0x1000_0000) != 0 {
                let mut keep = 0;
                for i in 0..8 {
                    let child = supertile.d[i];
                    if child == self.arenas.nullroots[lev - 1] {
                        continue;
                    }
                    let new_child = self.mdelete(child, lev - 1);
                    supertile.d[i] = new_child;
                    if new_child != self.arenas.nullroots[lev - 1] {
                        keep += 1;
                    }
                }
                if keep != 0 || node == self.root || (supertile.flags & 0x3ffff) != 0 {
                    supertile.flags &= 0xefff_ffff;
                } else {
                    self.arenas.release_supertile(SuperId(node));
                    return self.arenas.nullroots[lev];
                }
            }
            self.arenas.supers[node] = supertile;
            node
        }
    }

    fn popcount_node(&self, node: NodeId, lev: usize) -> u64 {
        let odd = (self.generation & 1) == 1;
        if lev == 0 {
            if node == self.arenas.empty_tile.0 {
                return 0;
            }
            let tile = &self.arenas.tiles[node];
            let add = if odd { 8 } else { 0 };
            let mut count = 0u64;
            for brick_id in tile.b {
                if brick_id == self.arenas.empty_brick {
                    continue;
                }
                let brick = &self.arenas.bricks[brick_id.0];
                for slice in 0..8 {
                    count += brick.d[add + slice].count_ones() as u64;
                }
            }
            count
        } else {
            if node == self.arenas.nullroots[lev] {
                return 0;
            }
            let supertile = &self.arenas.supers[node];
            let null = self.arenas.nullroots[lev - 1];
            let mut count = 0u64;
            for child in supertile.d {
                if child != null {
                    count += self.popcount_node(child, lev - 1);
                }
            }
            count
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_node<F>(
        &self,
        node: NodeId,
        lev: usize,
        xdel: i32,
        ydel: i32,
        odd: bool,
        spans: &[(i32, i32)],
        f: &mut F,
    ) where
        F: FnMut(i32, i32),
    {
        if lev == 0 {
            self.visit_tile(TileId(node), xdel, ydel, odd, f);
            return;
        }
        if node == self.arenas.nullroots[lev] {
            return;
        }
        let supertile = &self.arenas.supers[node];
        let child_span = spans[lev - 1];
        for (i, child) in supertile.d.iter().enumerate() {
            if *child == self.arenas.nullroots[lev - 1] {
                continue;
            }
            let (cx, cy) = if (lev & 1) == 1 {
                (xdel + (i as i32) * child_span.0, ydel)
            } else {
                (xdel, ydel + (i as i32) * child_span.1)
            };
            self.visit_node(*child, lev - 1, cx, cy, odd, spans, f);
        }
    }

    fn visit_tile<F>(&self, tile_id: TileId, xdel: i32, ydel: i32, odd: bool, f: &mut F)
    where
        F: FnMut(i32, i32),
    {
        if tile_id == self.arenas.empty_tile {
            return;
        }
        let tile = &self.arenas.tiles[tile_id.0];
        let tile_x = (xdel + self.minlow32) << 5;
        let tile_y = (ydel + self.minlow32) << 5;
        let add = if odd { 8 } else { 0 };

        for (brick_index, brick_id) in tile.b.iter().enumerate() {
            if *brick_id == self.arenas.empty_brick {
                continue;
            }
            let brick = &self.arenas.bricks[brick_id.0];
            let base_y = tile_y + (brick_index as i32) * 8;
            for slice in 0..8 {
                let data = brick.d[add + slice];
                if data == 0 {
                    continue;
                }
                for bit_pos in 0..32 {
                    let mask = 1u32 << (31 - bit_pos);
                    if data & mask != 0 {
                        let row = bit_pos / 4;
                        let col = bit_pos % 4;
                        let ix = tile_x + (slice as i32) * 4 + col;
                        let iy = base_y + row;
                        let (ux, uy) = if odd { (ix + 1, -(iy + 1)) } else { (ix, -iy) };
                        f(ux, uy);
                    }
                }
            }
        }
    }

    fn uproot_needed(&self) -> bool {
        let root = &self.arenas.supers[self.root];
        if root.d[0] != self.arenas.nullroots[self.rootlev - 1]
            || root.d[7] != self.arenas.nullroots[self.rootlev - 1]
        {
            return true;
        }
        for i in 1..7 {
            let child = &self.arenas.supers[root.d[i]];
            if child.d[0] != self.arenas.nullroots[self.rootlev - 2]
                || child.d[7] != self.arenas.nullroots[self.rootlev - 2]
            {
                return true;
            }
        }
        false
    }

    fn uproot(&mut self) {
        if self.min < -100_000_000 {
            self.min = i32::MIN;
        } else {
            self.min = 8 * self.min - 128;
        }
        if self.max > 500_000_000 {
            self.max = i32::MAX;
        } else {
            self.max = 8 * self.max - 121;
        }
        self.bmin = (self.bmin << 3) - 128;
        self.bmax = (self.bmax << 3) - 121;
        self.minlow32 = 8 * self.minlow32 - 4;

        if self.rootlev >= 38 || self.rootlev >= MAX_LEVELS {
            panic!("quicklife: universe expanded too far");
        }

        for _ in 0..2 {
            let oroot = self.root;
            self.rootlev += 1;
            let new_root = self.arenas.new_supertile(self.rootlev);
            self.root = new_root.0;
            if self.rootlev > 1 {
                let oroot_flags = if self.rootlev == 1 {
                    0
                } else {
                    self.arenas.supers[oroot].flags
                };
                let shift = 3 + (self.generation & 1) as u32;
                self.arenas.supers[self.root].flags =
                    0xf0000000 | (upchanging(oroot_flags) << shift);
            }
            self.arenas.supers[self.root].d[4] = oroot;
            if oroot != self.nullroot {
                let nullroot = self.arenas.new_supertile(self.rootlev);
                self.nullroot = nullroot.0;
                self.arenas.nullroots[self.rootlev] = self.nullroot;
            } else {
                self.nullroot = self.root;
                self.arenas.nullroots[self.rootlev] = self.root;
            }
        }
        self.pop_valid = false;
    }

    fn doquad01(
        &mut self,
        zis_id: SuperId,
        edge_id: SuperId,
        par_id: SuperId,
        cor_id: SuperId,
        lev: usize,
    ) -> u32 {
        let edge = self.arenas.supers[edge_id.0];
        let par = self.arenas.supers[par_id.0];
        let cor = self.arenas.supers[cor_id.0];
        let zis_flags = self.arenas.supers[zis_id.0].flags;
        let mut changing =
            (zis_flags | (par.flags >> 19) | (((edge.flags >> 18) | (cor.flags >> 27)) & 1)) & 0xff;
        let mut nchanging = (zis_flags & 0x3ff00) << 10;

        let mut x: i32;
        let mut b: u32;
        let mut pf: NodeId;
        let mut pfu: NodeId;
        if (changing & 1) != 0 {
            x = 7;
            b = 1;
            pf = edge.d[0];
            pfu = cor.d[0];
        } else {
            b = lowbit(changing);
            x = 7 - ai_index(b) as i32;
            pf = self.arenas.supers[zis_id.0].d[(x + 1) as usize];
            pfu = par.d[(x + 1) as usize];
        }

        loop {
            if changing == 0 {
                break;
            }
            let idx = x as usize;
            let p = self.arenas.supers[zis_id.0].d[idx];
            let pu = par.d[idx];
            if (changing & b) != 0 {
                let mut child = p;
                if child == self.arenas.nullroots[lev - 1] {
                    child = if lev == 1 {
                        self.arenas.new_tile().0
                    } else {
                        self.arenas.new_supertile(lev - 1).0
                    };
                    self.arenas.supers[zis_id.0].d[idx] = child;
                }
                if lev == 1 {
                    nchanging |=
                        self.p01(TileId(child), TileId(pf), TileId(pu), TileId(pfu)) << idx;
                } else {
                    nchanging |= self.doquad01(
                        SuperId(child),
                        SuperId(pu),
                        SuperId(pf),
                        SuperId(pfu),
                        lev - 1,
                    ) << idx;
                }
                changing = changing.wrapping_sub(b);
            }
            b <<= 1;
            x -= 1;
            pfu = pu;
            pf = p;
        }

        self.arenas.supers[zis_id.0].flags = nchanging | 0xf0000000;
        upchanging(nchanging)
    }

    fn doquad10(
        &mut self,
        zis_id: SuperId,
        edge_id: SuperId,
        par_id: SuperId,
        cor_id: SuperId,
        lev: usize,
    ) -> u32 {
        let edge = self.arenas.supers[edge_id.0];
        let par = self.arenas.supers[par_id.0];
        let cor = self.arenas.supers[cor_id.0];
        let zis_flags = self.arenas.supers[zis_id.0].flags;
        let mut changing =
            (zis_flags | (par.flags >> 19) | (((edge.flags >> 18) | (cor.flags >> 27)) & 1)) & 0xff;
        let mut nchanging = (zis_flags & 0x3ff00) << 10;

        let mut x: i32;
        let mut b: u32;
        let mut pf: NodeId;
        let mut pfu: NodeId;
        if (changing & 1) != 0 {
            x = 0;
            b = 1;
            pf = edge.d[7];
            pfu = cor.d[7];
        } else {
            b = lowbit(changing);
            x = ai_index(b) as i32;
            pf = self.arenas.supers[zis_id.0].d[(x - 1) as usize];
            pfu = par.d[(x - 1) as usize];
        }

        loop {
            if changing == 0 {
                break;
            }
            let idx = x as usize;
            let p = self.arenas.supers[zis_id.0].d[idx];
            let pu = par.d[idx];
            if (changing & b) != 0 {
                let mut child = p;
                if child == self.arenas.nullroots[lev - 1] {
                    child = if lev == 1 {
                        self.arenas.new_tile().0
                    } else {
                        self.arenas.new_supertile(lev - 1).0
                    };
                    self.arenas.supers[zis_id.0].d[idx] = child;
                }
                if lev == 1 {
                    nchanging |=
                        self.p10(TileId(pfu), TileId(pu), TileId(pf), TileId(child)) << (7 - idx);
                } else {
                    nchanging |= self.doquad10(
                        SuperId(child),
                        SuperId(pu),
                        SuperId(pf),
                        SuperId(pfu),
                        lev - 1,
                    ) << (7 - idx);
                }
                changing = changing.wrapping_sub(b);
            }
            b <<= 1;
            x += 1;
            pfu = pu;
            pf = p;
        }

        self.arenas.supers[zis_id.0].flags = nchanging | 0xf0000000;
        upchanging(nchanging)
    }

    fn p01(&mut self, p_id: TileId, pr_id: TileId, pd_id: TileId, prd_id: TileId) -> u32 {
        let rule_table = &self.rule_table;
        let deltaforward = self.deltaforward;
        let arenas = &mut self.arenas;

        let pr = arenas.tiles[pr_id.0];
        let pd = arenas.tiles[pd_id.0];
        let prd = arenas.tiles[prd_id.0];
        let mut p_tile = arenas.tiles[p_id.0];

        let mut recomp = (p_tile.c[4] as u32
            | pd.c[0] as u32
            | ((pr.c[4] as u32) >> 9)
            | ((prd.c[0] as u32) >> 8))
            & 0xff;
        let mut db = pd.b[0];
        let mut rdb = prd.b[0];

        p_tile.c[5] = 0;
        p_tile.flags |= 0xfff0_0000;

        for i in (0..=3).rev() {
            let rb = pr.b[i];
            let mut b = p_tile.b[i];
            if recomp != 0 {
                if b == arenas.empty_brick {
                    let new_b = arenas.new_brick();
                    p_tile.b[i] = new_b;
                    b = new_b;
                }
                p_tile.flags |= 1 << i;

                let mut b_data = arenas.bricks[b.0].d;
                let db_data = arenas.bricks[db.0].d;
                let mut traildata: u32;
                let mut trailunderdata: u32;
                let mut cdelta: u32 = 0;
                let mut maskprev: u32 = 0;
                let mut j: i32;
                if (recomp & 1) != 0 {
                    j = 7;
                    traildata = arenas.bricks[rb.0].d[0];
                    trailunderdata = arenas.bricks[rdb.0].d[0];
                } else {
                    let shift = ai_index(lowbit(recomp));
                    j = 7 - shift as i32;
                    recomp >>= shift;
                    traildata = b_data[(j + 1) as usize];
                    trailunderdata = db_data[(j + 1) as usize];
                }
                trailunderdata = (traildata << 8) + (trailunderdata >> 24);
                loop {
                    if (recomp & 1) != 0 {
                        let zisdata = b_data[j as usize];
                        let underdata = (zisdata << 8) + (db_data[j as usize] >> 24);
                        let otherdata =
                            ((zisdata << 2) & 0xcccccccc) + ((traildata >> 2) & 0x33333333);
                        let otherunderdata =
                            ((underdata << 2) & 0xcccccccc) + ((trailunderdata >> 2) & 0x33333333);
                        let newv = (rule_table.lookup((zisdata >> 16) as u16) as u32) << 26
                            | (rule_table.lookup((underdata >> 16) as u16) as u32) << 18
                            | (rule_table.lookup((zisdata & 0xffff) as u16) as u32) << 10
                            | (rule_table.lookup((underdata & 0xffff) as u16) as u32) << 2
                            | (rule_table.lookup((otherdata >> 16) as u16) as u32) << 24
                            | (rule_table.lookup((otherunderdata >> 16) as u16) as u32) << 16
                            | (rule_table.lookup((otherdata & 0xffff) as u16) as u32) << 8
                            | (rule_table.lookup((otherunderdata & 0xffff) as u16) as u32);
                        let delta = (b_data[(j + 8) as usize] ^ newv)
                            | deltaforward
                            | p_tile.localdeltaforward;
                        b_data[(j + 8) as usize] = newv;
                        let maska = cdelta | (delta & 0x33333333);
                        let maskb = maska | maska.wrapping_neg();
                        maskprev = (maskprev << 1) | ((maskb >> 9) & 0x400000) | (maskb & 0x80);
                        cdelta = delta;
                        traildata = zisdata;
                        trailunderdata = underdata;
                    } else {
                        let maskb = cdelta | cdelta.wrapping_neg();
                        maskprev = (maskprev << 1) | ((maskb >> 9) & 0x400000) | (maskb & 0x80);
                        if recomp == 0 {
                            break;
                        }
                        cdelta = 0;
                        traildata = b_data[j as usize];
                        trailunderdata = (traildata << 8) + (db_data[j as usize] >> 24);
                    }
                    recomp >>= 1;
                    j -= 1;
                }
                arenas.bricks[b.0].d = b_data;
                let shift1 = (6 - j) as u32;
                let shift2 = (21 - j) as u32;
                let keep = (p_tile.c[i + 1] & 0x100) as u32;
                p_tile.c[i + 2] |= ((maskprev >> shift1) & 0x1ff) as u16;
                p_tile.c[i + 1] = ((keep << 1) | (maskprev >> shift2)) as u16;
            } else {
                p_tile.c[i + 1] = 0;
            }

            recomp = ((p_tile.c[i] as u32) | ((pr.c[i] as u32) >> 9)) & 0xff;
            db = b;
            rdb = rb;
        }

        p_tile.localdeltaforward = 0;
        arenas.tiles[p_id.0] = p_tile;
        let recomp = p_tile.c[5] as u32;
        let i_all = recomp
            | p_tile.c[0] as u32
            | p_tile.c[1] as u32
            | p_tile.c[2] as u32
            | p_tile.c[3] as u32
            | p_tile.c[4] as u32;
        if recomp != 0 {
            0x201 | ((recomp & 0x100) << 2) | ((i_all & 0x100) >> 7)
        } else if i_all != 0 {
            ((i_all & 0x100) >> 7) | 1
        } else {
            0
        }
    }

    fn p10(&mut self, plu_id: TileId, pu_id: TileId, pl_id: TileId, p_id: TileId) -> u32 {
        let rule_table = &self.rule_table;
        let deltaforward = self.deltaforward;
        let arenas = &mut self.arenas;

        let plu = arenas.tiles[plu_id.0];
        let pu = arenas.tiles[pu_id.0];
        let pl = arenas.tiles[pl_id.0];
        let mut p_tile = arenas.tiles[p_id.0];

        let mut recomp = (p_tile.c[1] as u32
            | pu.c[5] as u32
            | ((pl.c[1] as u32) >> 9)
            | ((plu.c[5] as u32) >> 8))
            & 0xff;
        let mut ub = pu.b[3];
        let mut lub = plu.b[3];

        p_tile.c[0] = 0;
        p_tile.flags |= 0x000f_ff00;

        for i in 0..=3 {
            let lb = pl.b[i];
            let mut b = p_tile.b[i];
            if recomp != 0 {
                if b == arenas.empty_brick {
                    let new_b = arenas.new_brick();
                    p_tile.b[i] = new_b;
                    b = new_b;
                }
                p_tile.flags |= 1 << i;

                let mut b_data = arenas.bricks[b.0].d;
                let ub_data = arenas.bricks[ub.0].d;
                let mut maskprev: u32 = 0;
                let mut cdelta: u32 = 0;
                let mut j: i32;
                let mut traildata: u32;
                let mut trailoverdata: u32;
                if (recomp & 1) != 0 {
                    j = 0;
                    traildata = arenas.bricks[lb.0].d[15];
                    trailoverdata = arenas.bricks[lub.0].d[15];
                } else {
                    let shift = ai_index(lowbit(recomp));
                    j = shift as i32;
                    traildata = b_data[(j + 7) as usize];
                    trailoverdata = ub_data[(j + 7) as usize];
                    recomp >>= shift;
                }
                trailoverdata = (traildata >> 8) + (trailoverdata << 24);
                loop {
                    if (recomp & 1) != 0 {
                        let zisdata = b_data[(j + 8) as usize];
                        let overdata = (zisdata >> 8) + (ub_data[(j + 8) as usize] << 24);
                        let otherdata =
                            ((zisdata >> 2) & 0x33333333) + ((traildata << 2) & 0xcccccccc);
                        let otheroverdata =
                            ((overdata >> 2) & 0x33333333) + ((trailoverdata << 2) & 0xcccccccc);
                        let newv = (rule_table.lookup((otheroverdata >> 16) as u16) as u32) << 26
                            | (rule_table.lookup((otherdata >> 16) as u16) as u32) << 18
                            | (rule_table.lookup((otheroverdata & 0xffff) as u16) as u32) << 10
                            | (rule_table.lookup((otherdata & 0xffff) as u16) as u32) << 2
                            | (rule_table.lookup((overdata >> 16) as u16) as u32) << 24
                            | (rule_table.lookup((zisdata >> 16) as u16) as u32) << 16
                            | (rule_table.lookup((overdata & 0xffff) as u16) as u32) << 8
                            | (rule_table.lookup((zisdata & 0xffff) as u16) as u32);
                        let delta =
                            (b_data[j as usize] ^ newv) | deltaforward | p_tile.localdeltaforward;
                        let maska = cdelta | (delta & 0xcccccccc);
                        maskprev = (maskprev << 1)
                            | (((maska | maska.wrapping_neg()) >> 9) & 0x400000)
                            | ((((maska >> 24) | 0x100) - 1) & 0x100);
                        b_data[j as usize] = newv;
                        cdelta = delta;
                        traildata = zisdata;
                        trailoverdata = overdata;
                    } else {
                        maskprev = (maskprev << 1)
                            | (((cdelta | cdelta.wrapping_neg()) >> 9) & 0x400000)
                            | ((((cdelta >> 24) | 0x100) - 1) & 0x100);
                        if recomp == 0 {
                            break;
                        }
                        cdelta = 0;
                        traildata = b_data[(j + 8) as usize];
                        trailoverdata = (traildata >> 8) + (ub_data[(j + 8) as usize] << 24);
                    }
                    recomp >>= 1;
                    j += 1;
                }
                arenas.bricks[b.0].d = b_data;
                let shift1 = (14 + j) as u32;
                let keep = (p_tile.c[i + 1] & 0x100) as u32;
                p_tile.c[i + 1] = ((keep << 1) | (maskprev >> shift1)) as u16;
                p_tile.c[i] |= ((maskprev >> j) & 0x1ff) as u16;
            } else {
                p_tile.c[i + 1] = 0;
            }

            recomp = ((p_tile.c[i + 2] as u32) | ((pl.c[i + 2] as u32) >> 9)) & 0xff;
            ub = b;
            lub = lb;
        }

        p_tile.localdeltaforward = 0;
        arenas.tiles[p_id.0] = p_tile;
        let recomp = p_tile.c[0] as u32;
        let i_all = recomp
            | p_tile.c[1] as u32
            | p_tile.c[2] as u32
            | p_tile.c[3] as u32
            | p_tile.c[4] as u32
            | p_tile.c[5] as u32;
        if recomp != 0 {
            0x201 | ((recomp & 0x100) << 2) | ((i_all & 0x100) >> 7)
        } else if i_all != 0 {
            ((i_all & 0x100) >> 7) | 1
        } else {
            0
        }
    }
}

fn lowbit(x: u32) -> u32 {
    x & x.wrapping_neg()
}

fn ai_index(x: u32) -> usize {
    if x == 0 {
        4
    } else {
        x.trailing_zeros() as usize
    }
}

fn upchanging(x: u32) -> u32 {
    let a = (x & 0x1feff) + 0x1feff;
    ((a >> 8) & 1) | ((a >> 16) & 2) | ((x << 1) & 0x200) | ((x >> 7) & 0x400)
}

#[cfg(test)]
mod tests {
    use super::{NodeId, QuickLife};

    fn count_non_empty_tiles(engine: &QuickLife) -> usize {
        fn walk(engine: &QuickLife, node: NodeId, lev: usize, count: &mut usize) {
            if lev == 0 {
                if node != engine.arenas.empty_tile.0 {
                    *count += 1;
                }
                return;
            }
            if node == engine.arenas.nullroots[lev] {
                return;
            }
            let supertile = engine.arenas.supers[node];
            for child in supertile.d {
                walk(engine, child, lev - 1, count);
            }
        }

        let mut count = 0;
        walk(engine, engine.root, engine.rootlev, &mut count);
        count
    }

    #[test]
    fn cleanup_removes_dead_tiles() {
        let mut engine = QuickLife::new();
        engine.set_cell(0, 0, true);
        engine.step(1);
        engine.step(63);

        let tile_count = count_non_empty_tiles(&engine);
        assert_eq!(
            tile_count, 0,
            "expected no tiles after cleanup, found {tile_count}"
        );
    }
}
