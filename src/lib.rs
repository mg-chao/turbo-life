//! QuickLife-based Conway's Game of Life engine (B3/S23).

pub mod quicklife;
pub mod turbolife;
pub use quicklife::QuickLife;
pub use turbolife::{KernelBackend, TurboLife, TurboLifeConfig};
