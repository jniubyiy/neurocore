// src/layers/adaptive_normalization/gpu/pipeline.rs
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

pub struct AdaptiveNormalizationPipelines {
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>,
    pub row_stats: Arc<ComputePipeline>,
    pub col_stats: Arc<ComputePipeline>,
}

impl AdaptiveNormalizationPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_bytes = include_bytes!("vulkan/shaders/adaptive_norm_fwd.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/adaptive_norm_bwd.spv");
        let row_stats_bytes = include_bytes!("vulkan/shaders/adaptive_norm_row_stats.spv");
        let col_stats_bytes = include_bytes!("vulkan/shaders/adaptive_norm_col_stats.spv");

        let fwd_spv = as_u32_slice(fwd_bytes);
        let bwd_spv = as_u32_slice(bwd_bytes);
        let row_stats_spv = as_u32_slice(row_stats_bytes);
        let col_stats_spv = as_u32_slice(col_stats_bytes);

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
            .expect("Failed to create descriptor set layout for AdaptiveNormalization")
        }

        // ==================== Forward ====================
        let fwd_layout = create_ds_layout(device.clone(), 8);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create AdaptiveNormalization forward shader module")
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
        .expect("Failed to create AdaptiveNormalization forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("AdaptiveNormalization forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create AdaptiveNormalization forward pipeline");

        // ==================== Backward ====================
        let bwd_layout = create_ds_layout(device.clone(), 10);
        let bwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_spv))
                .expect("Failed to create AdaptiveNormalization backward shader module")
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
        .expect("Failed to create AdaptiveNormalization backward pipeline layout");
        let bwd_entry = bwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("AdaptiveNormalization backward entry point not found");
        let bwd_stage = PipelineShaderStageCreateInfo::new(bwd_entry);
        let backward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_stage, bwd_pipeline_layout),
        )
        .expect("Failed to create AdaptiveNormalization backward pipeline");

        // ==================== Row Statistics ====================
        let row_stats_layout = create_ds_layout(device.clone(), 4);
        let row_stats_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(row_stats_spv))
                .expect("Failed to create AdaptiveNormalization row stats shader module")
        };
        let row_stats_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 8, // batch, features
        };
        let row_stats_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![row_stats_layout],
                push_constant_ranges: vec![row_stats_push],
                ..Default::default()
            },
        )
        .expect("Failed to create AdaptiveNormalization row stats pipeline layout");
        let row_stats_entry = row_stats_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("AdaptiveNormalization row stats entry point not found");
        let row_stats_stage = PipelineShaderStageCreateInfo::new(row_stats_entry);
        let row_stats = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(row_stats_stage, row_stats_pipeline_layout),
        )
        .expect("Failed to create AdaptiveNormalization row stats pipeline");

        // ==================== Column Statistics ====================
        let col_stats_layout = create_ds_layout(device.clone(), 3);
        let col_stats_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(col_stats_spv))
                .expect("Failed to create AdaptiveNormalization col stats shader module")
        };
        let col_stats_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 8, // batch, features
        };
        let col_stats_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![col_stats_layout],
                push_constant_ranges: vec![col_stats_push],
                ..Default::default()
            },
        )
        .expect("Failed to create AdaptiveNormalization col stats pipeline layout");
        let col_stats_entry = col_stats_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("AdaptiveNormalization col stats entry point not found");
        let col_stats_stage = PipelineShaderStageCreateInfo::new(col_stats_entry);
        let col_stats = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(col_stats_stage, col_stats_pipeline_layout),
        )
        .expect("Failed to create AdaptiveNormalization col stats pipeline");

        Self { forward, backward, row_stats, col_stats }
    }
}