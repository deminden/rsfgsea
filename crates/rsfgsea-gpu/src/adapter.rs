use crate::GpuEngine;
use anyhow::Result;

struct AdapterSelection {
    adapter: wgpu::Adapter,
    selected_score: i32,
}

fn gl_fallback_enabled() -> bool {
    let allow_gl_backend = std::env::var("RSFGSEA_GPU_ALLOW_GL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mesa_adapter_override = std::env::var("MESA_D3D12_DEFAULT_ADAPTER_NAME")
        .ok()
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    allow_gl_backend || mesa_adapter_override
}

fn requested_backends(gl_fallback_enabled: bool) -> wgpu::Backends {
    if let Ok(raw) = std::env::var("WGPU_BACKEND") {
        match raw.to_lowercase().as_str() {
            "vulkan" => wgpu::Backends::VULKAN,
            "dx12" => wgpu::Backends::DX12,
            "metal" => wgpu::Backends::METAL,
            "gl" => wgpu::Backends::GL,
            "all" => wgpu::Backends::all(),
            _ => wgpu::Backends::PRIMARY,
        }
    } else if gl_fallback_enabled {
        wgpu::Backends::all()
    } else {
        wgpu::Backends::PRIMARY
    }
}

fn select_adapter(
    instance: &wgpu::Instance,
    backends: wgpu::Backends,
    gl_fallback_enabled: bool,
) -> Option<AdapterSelection> {
    let adapters = instance.enumerate_adapters(backends);
    let mut selected_adapter = None;
    let mut selected_score = i32::MIN;

    log::debug!("Enumerating available GPUs");
    for adapter in adapters {
        let info = adapter.get_info();
        log::debug!(
            "GPU candidate: name={:?} type={:?} backend={:?} vendor=0x{:04x}",
            info.name,
            info.device_type,
            info.backend,
            info.vendor
        );

        if !gl_fallback_enabled && info.backend == wgpu::Backend::Gl {
            continue;
        }

        let mut score = 0i32;
        let name_lower = info.name.to_lowercase();

        if info.vendor == 0x10de || name_lower.contains("nvidia") {
            score += 100;
        }
        if matches!(
            info.backend,
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal
        ) {
            score += 40;
        }
        if matches!(info.device_type, wgpu::DeviceType::DiscreteGpu) {
            score += 30;
        }
        if matches!(info.device_type, wgpu::DeviceType::Cpu) || name_lower.contains("llvmpipe") {
            score -= 100;
        }

        if score > selected_score {
            selected_score = score;
            selected_adapter = Some(adapter);
        }
    }

    selected_adapter.map(|adapter| AdapterSelection {
        adapter,
        selected_score,
    })
}

impl GpuEngine {
    pub async fn new() -> Result<Self> {
        let gl_fallback_enabled = gl_fallback_enabled();
        let backends = requested_backends(gl_fallback_enabled);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter =
            if let Some(selection) = select_adapter(&instance, backends, gl_fallback_enabled) {
                log::debug!("Selected adapter score: {}", selection.selected_score);
                selection.adapter
            } else {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        ..Default::default()
                    })
                    .await
                    .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?
            };

        let selected_info = adapter.get_info();
        log::info!("Selected GPU: {:?}", selected_info.name);
        let selected_name = selected_info.name.to_lowercase();
        let unsuitable = matches!(selected_info.device_type, wgpu::DeviceType::Cpu)
            || selected_name.contains("llvmpipe")
            || (!gl_fallback_enabled && selected_info.backend == wgpu::Backend::Gl);
        if unsuitable {
            return Err(anyhow::anyhow!(
                "No suitable non-CPU GPU adapter found. Selected adapter was '{}', backend={:?}. \
Set WGPU_BACKEND=vulkan and, on WSL2, ensure NVIDIA Vulkan is visible (optionally set MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA). \
If you intentionally want GL translation, set RSFGSEA_GPU_ALLOW_GL=1 (or set MESA_D3D12_DEFAULT_ADAPTER_NAME).",
                selected_info.name,
                selected_info.backend
            ));
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rsfgsea-gpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gsea_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gsea_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }
}
