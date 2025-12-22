pub mod algo;
pub mod core;
pub mod io;
pub mod prelude;

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use rsfgsea_gpu::GpuEngine;
}

#[cfg(feature = "gpu")]
pub use gpu::GpuEngine;

pub use crate::algo::*;
pub use crate::core::*;
