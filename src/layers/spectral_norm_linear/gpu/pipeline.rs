// src/layers/spectral_norm_linear/gpu/pipeline.rs
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::ComputePipeline;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
    DescriptorType,
};
use vulkano::pipeline::{
    compute::ComputePipelineCreateInfo,
    layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::spirv::ExecutionModel;

fn as_u32_slice(bytes: &[u8]) -> &[u32] {
    assert!(bytes.len() % 4 == 0, "SPIR‑V файл должен быть выровнен по 4 байта");
    let ptr = bytes.as_ptr() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, bytes.len() / 4) }
}

/// Пайплайны SpectrallyNormalizedLinear.
///
/// Power iteration (поэтапно):
///   matvec_v → normalize → matvec_u → normalize → sigma
///
/// Forward: y = (scale / sigma) · W · x + b.
///
/// Backward: единый шейдер с phase 0/1.
pub struct SpectrallyNormalizedLinearPipelines {
    pub matvec_v: Arc<ComputePipeline>,
    pub matvec_u: Arc<ComputePipeline>,
    pub normalize: Arc<ComputePipeline>,
    pub sigma: Arc<ComputePipeline>,
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>,
}

impl SpectrallyNormalizedLinearPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let matvec_v_bytes = include_bytes!("vulkan/shaders/spectral_norm_matvec_v.spv");
        let matvec_u_bytes = include_bytes!("vulkan/shaders/spectral_norm_matvec_u.spv");
        let normalize_bytes = include_bytes!("vulkan/shaders/spectral_norm_normalize.spv");
        let sigma_bytes = include_bytes!("vulkan/shaders/spectral_norm_sigma.spv");
        let fwd_bytes = include_bytes!("vulkan/shaders/spectral_norm_linear_fwd.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/spectral_norm_linear_bwd.spv");

        let matvec_v_spv = as_u32_slice(matvec_v_bytes);
        let matvec_u_spv = as_u32_slice(matvec_u_bytes);
        let normalize_spv = as_u32_slice(normalize_bytes);
        let sigma_spv = as_u32_slice(sigma_bytes);
        let fwd_spv = as_u32_slice(fwd_bytes);
        let bwd_spv = as_u32_slice(bwd_bytes);

        fn create_ds_layout(device: Arc<Device>, n: u32) -> Arc<DescriptorSetLayout> {
            let mut bindings = std::collections::BTreeMap::new();
            for binding in 0..n {
                bindings.insert(
                    binding,
                    DescriptorSetLayoutBinding {
                        binding_flags: Default::default(),
                        descriptor_type: DescriptorType::StorageBuffer,
                        descriptor_count: 1,
                        stages: ShaderStages::COMPUTE,
                        immutable_samplers: Vec::new(),
                        _ne: unsafe { std::mem::zeroed() },
                    },
                );
            }
            DescriptorSetLayout::new(
                device,
                DescriptorSetLayoutCreateInfo {
                    bindings,
                    ..Default::default()
                },
            )
            .expect("Failed to create descriptor set layout for SpectrallyNormalizedLinear")
        }

        fn build(
            device: Arc<Device>,
            spv: &[u32],
            ds_n: u32,
            push_size: u32,
            name: &str,
        ) -> Arc<ComputePipeline> {
            let layout = create_ds_layout(device.clone(), ds_n);
            let module = unsafe {
                ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(spv))
                    .unwrap_or_else(|_| panic!("Failed to create {} shader module", name))
            };
            let push = PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: push_size,
            };
            let pipeline_layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![layout],
                    push_constant_ranges: vec![push],
                    ..Default::default()
                },
            )
            .unwrap_or_else(|_| panic!("Failed to create {} pipeline layout", name));
            let entry = module
                .entry_point_with_execution("main", ExecutionModel::GLCompute)
                .unwrap_or_else(|| panic!("{} entry point not found", name));
            let stage = PipelineShaderStageCreateInfo::new(entry);
            ComputePipeline::new(
                device,
                None,
                ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout),
            )
            .unwrap_or_else(|_| panic!("Failed to create {} pipeline", name))
        }

        // matvec_v: 3 буфера, push = [in, out] (8 байт).
        let matvec_v = build(device.clone(), matvec_v_spv, 3, 8, "SN matvec_v");
        // matvec_u: 3 буфера, push = [in, out] (8 байт).
        let matvec_u = build(device.clone(), matvec_u_spv, 3, 8, "SN matvec_u");
        // normalize: 1 буфер, push = [len_bits, eps_bits] (8 байт).
        let normalize = build(device.clone(), normalize_spv, 1, 8, "SN normalize");
        // sigma: 4 буфера, push = [in, out] (8 байт).
        let sigma = build(device.clone(), sigma_spv, 4, 8, "SN sigma");
        // forward: 4 буфера, push = [batch, in, out, scale_bits, sigma_bits] (20 байт).
        let forward = build(device.clone(), fwd_spv, 4, 20, "SN forward");
        // backward: 7 буферов, push = [batch, in, out, scale_bits, sigma_bits, phase] (24 байта).
        let backward = build(device.clone(), bwd_spv, 7, 24, "SN backward");

        Self {
            matvec_v,
            matvec_u,
            normalize,
            sigma,
            forward,
            backward,
        }
    }
}