pub mod algo;
mod algo_support;
pub mod bindings;
pub mod core;
mod esruler_compat;
mod fastgsea_compat;
#[cfg(feature = "gpu")]
mod gpu_algo;
pub mod io;
mod multilevel;
pub mod prelude;
mod rng_compat;

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use rsfgsea_gpu::GpuEngine;
}

#[cfg(feature = "gpu")]
pub use gpu::GpuEngine;

pub use crate::algo::*;
pub use crate::bindings::*;
pub use crate::core::*;
