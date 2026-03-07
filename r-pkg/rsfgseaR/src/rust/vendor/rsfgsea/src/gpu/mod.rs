mod adapter;
mod compute;
mod workflow;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuResult {
    pub es: f32,
    pub peak_idx: u32,
}

pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

pub use workflow::{FgseaMultilevelResult, FgseaSimpleResult};
