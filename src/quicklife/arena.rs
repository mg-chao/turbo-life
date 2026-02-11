//! Memory arenas and data structures for QuickLife.

pub const MAX_LEVELS: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperId(pub usize);

pub type NodeId = usize;

#[derive(Clone, Copy, Debug)]
pub struct Brick {
    pub d: [u32; 16],
}

impl Brick {
    pub fn empty() -> Self {
        Self { d: [0; 16] }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tile {
    pub b: [BrickId; 4],
    pub c: [u16; 6],
    pub flags: u32,
    pub localdeltaforward: u32,
}

impl Tile {
    pub fn empty(empty_brick: BrickId) -> Self {
        Self {
            b: [empty_brick; 4],
            c: [0; 6],
            flags: u32::MAX,
            localdeltaforward: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Supertile {
    pub d: [NodeId; 8],
    pub flags: u32,
}

impl Supertile {
    pub fn new(child: NodeId) -> Self {
        Self {
            d: [child; 8],
            flags: 0,
        }
    }
}

pub struct Arenas {
    pub bricks: Vec<Brick>,
    pub tiles: Vec<Tile>,
    pub supers: Vec<Supertile>,
    pub empty_brick: BrickId,
    pub empty_tile: TileId,
    pub nullroots: Vec<NodeId>,
    pub free_bricks: Vec<BrickId>,
    pub free_tiles: Vec<TileId>,
    pub free_supers: Vec<SuperId>,
}

impl Arenas {
    pub fn new() -> Self {
        let mut arenas = Self {
            bricks: Vec::with_capacity(16_384),
            tiles: Vec::with_capacity(4_096),
            supers: Vec::with_capacity(4_096),
            empty_brick: BrickId(0),
            empty_tile: TileId(0),
            nullroots: vec![0; MAX_LEVELS + 1],
            free_bricks: Vec::new(),
            free_tiles: Vec::new(),
            free_supers: Vec::new(),
        };

        let empty_brick = BrickId(arenas.bricks.len());
        arenas.bricks.push(Brick::empty());

        let empty_tile = TileId(arenas.tiles.len());
        arenas.tiles.push(Tile::empty(empty_brick));

        arenas.empty_brick = empty_brick;
        arenas.empty_tile = empty_tile;
        arenas.nullroots[0] = empty_tile.0;

        arenas
    }

    pub fn new_brick(&mut self) -> BrickId {
        if let Some(id) = self.free_bricks.pop() {
            self.bricks[id.0] = Brick::empty();
            id
        } else {
            let id = BrickId(self.bricks.len());
            self.bricks.push(Brick::empty());
            id
        }
    }

    pub fn new_tile(&mut self) -> TileId {
        if let Some(id) = self.free_tiles.pop() {
            self.tiles[id.0] = Tile::empty(self.empty_brick);
            id
        } else {
            let id = TileId(self.tiles.len());
            self.tiles.push(Tile::empty(self.empty_brick));
            id
        }
    }

    pub fn new_supertile(&mut self, level: usize) -> SuperId {
        let child = self.nullroots[level - 1];
        if let Some(id) = self.free_supers.pop() {
            self.supers[id.0] = Supertile::new(child);
            id
        } else {
            let id = SuperId(self.supers.len());
            self.supers.push(Supertile::new(child));
            id
        }
    }

    pub fn release_brick(&mut self, id: BrickId) {
        if id != self.empty_brick {
            self.free_bricks.push(id);
        }
    }

    pub fn release_tile(&mut self, id: TileId) {
        if id != self.empty_tile {
            self.free_tiles.push(id);
        }
    }

    pub fn release_supertile(&mut self, id: SuperId) {
        self.free_supers.push(id);
    }
}
