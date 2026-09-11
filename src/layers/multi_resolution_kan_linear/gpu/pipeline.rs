// src/layers/multi_resolution_kan_linear/gpu/pipeline.rs
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

/// Пайплайны MultiResolutionKANLinear.
///
/// Forward:  multi_resolution_kan_linear_fwd.
/// Backward: bwd_params (фаза 0) → bwd_input (фаза 1).
pub struct MultiResolutionKANLinearPipelines {
    pub forward: Arc<ComputePipeline>,
    pub bwd_params: Arc<ComputePipeline>,
    pub bwd_input: Arc<ComputePipeline>,
}

impl MultiResolutionKANLinearPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_bytes = include_bytes!("vulkan/shaders/multi_resolution_kan_linear_fwd.spv");
        let bwd_params_bytes =
            include_bytes!("vulkan/shaders/multi_resolution_kan_linear_bwd_params.spv");
        let bwd_input_bytes =
            include_bytes!("vulkan/shaders/multi_resolution_kan_linear_bwd_input.spv");

        let fwd_spv = as_u32_slice(fwd_bytes);
        let bwd_params_spv = as_u32_slice(bwd_params_bytes);
        let bwd_input_spv = as_u32_slice(bwd_input_bytes);

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
            .expect("Failed to create descriptor set layout for MultiResolutionKANLinear")
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

        // Все три шейдера используют push = [batch, in, out] (12 байт).
        let forward    = build(device.clone(), fwd_spv,        3, 12, "KAN forward");
        let bwd_params = build(device.clone(), bwd_params_spv, 4, 12, "KAN bwd_params");
        let bwd_input  = build(device.clone(), bwd_input_spv,  4, 12, "KAN bwd_input");

        Self {
            forward,
            bwd_params,
            bwd_input,
        }
    }
}