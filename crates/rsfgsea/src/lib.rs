pub mod algo;
pub mod core;
mod esruler_compat;
mod fastgsea_compat;
pub mod io;
pub mod prelude;
mod rng_compat;

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use rsfgsea_gpu::GpuEngine;
}

#[cfg(feature = "gpu")]
pub use gpu::GpuEngine;

pub use crate::algo::*;
pub use crate::core::*;
