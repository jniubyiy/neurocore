// src/layers/combiner/gpu/pipeline.rs
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

/// Пайплайны Combiner.
///
/// Forward:  combiner_fwd        — один поток на (r, j), j ∈ [0, m).
/// Backward: combiner_bwd_dx     — один поток на (r, c), c ∈ [0, n);
///           combiner_bwd_params — один поток на параметр.
pub struct CombinerPipelines {
    pub forward: Arc<ComputePipeline>,
    pub backward_dx: Arc<ComputePipeline>,
    pub backward_params: Arc<ComputePipeline>,
}

impl CombinerPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_bytes        = include_bytes!("vulkan/shaders/combiner_fwd.spv");
        let bwd_dx_bytes     = include_bytes!("vulkan/shaders/combiner_bwd_dx.spv");
        let bwd_params_bytes = include_bytes!("vulkan/shaders/combiner_bwd_params.spv");

        let fwd_spv        = as_u32_slice(fwd_bytes);
        let bwd_dx_spv     = as_u32_slice(bwd_dx_bytes);
        let bwd_params_spv = as_u32_slice(bwd_params_bytes);

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
            .expect("Failed to create descriptor set layout for Combiner")
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

        // Все три пайплайна используют push = [batch, n, m] — 12 байт.
        const PUSH: u32 = 12;

        let forward         = build(device.clone(), fwd_spv,        7, PUSH, "Combiner forward");
        let backward_dx     = build(device.clone(), bwd_dx_spv,     6, PUSH, "Combiner bwd_dx");
        let backward_params = build(device.clone(), bwd_params_spv, 7, PUSH, "Combiner bwd_params");

        Self {
            forward,
            backward_dx,
            backward_params,
        }
    }
}