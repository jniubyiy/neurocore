// src/layers/batch_renorm/gpu/pipeline.rs
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

pub struct BatchRenormPipelines {
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>,
    pub col_stats: Arc<ComputePipeline>,
}

impl BatchRenormPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_bytes = include_bytes!("vulkan/shaders/batch_renorm_fwd.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/batch_renorm_bwd.spv");
        let stats_bytes = include_bytes!("vulkan/shaders/batch_renorm_col_stats.spv");

        let fwd_spv = as_u32_slice(fwd_bytes);
        let bwd_spv = as_u32_slice(bwd_bytes);
        let stats_spv = as_u32_slice(stats_bytes);

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
            .expect("Failed to create descriptor set layout for BatchRenorm")
        }

        // ==================== Forward ====================
        let fwd_layout = create_ds_layout(device.clone(), 5);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create BatchRenorm forward shader module")
        };
        let fwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 8, // batch, features
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![fwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create BatchRenorm forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("BatchRenorm forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create BatchRenorm forward pipeline");

        // ==================== Backward ====================
        let bwd_layout = create_ds_layout(device.clone(), 7);
        let bwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_spv))
                .expect("Failed to create BatchRenorm backward shader module")
        };
        let bwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 8, // batch, features
        };
        let bwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_layout],
                push_constant_ranges: vec![bwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create BatchRenorm backward pipeline layout");
        let bwd_entry = bwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("BatchRenorm backward entry point not found");
        let bwd_stage = PipelineShaderStageCreateInfo::new(bwd_entry);
        let backward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_stage, bwd_pipeline_layout),
        )
        .expect("Failed to create BatchRenorm backward pipeline");

        // ==================== Column Statistics ====================
        let stats_layout = create_ds_layout(device.clone(), 3);
        let stats_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(stats_spv))
                .expect("Failed to create BatchRenorm col stats shader module")
        };
        let stats_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 8, // batch, features
        };
        let stats_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![stats_layout],
                push_constant_ranges: vec![stats_push],
                ..Default::default()
            },
        )
        .expect("Failed to create BatchRenorm col stats pipeline layout");
        let stats_entry = stats_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("BatchRenorm col stats entry point not found");
        let stats_stage = PipelineShaderStageCreateInfo::new(stats_entry);
        let col_stats = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(stats_stage, stats_pipeline_layout),
        )
        .expect("Failed to create BatchRenorm col stats pipeline");

        Self { forward, backward, col_stats }
    }
}