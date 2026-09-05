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

pub struct SpectrallyNormalizedLinearPipelines {
    pub power_iteration: Arc<ComputePipeline>,
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>,
}

impl SpectrallyNormalizedLinearPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let power_bytes = include_bytes!("vulkan/shaders/spectral_norm_linear_power_iteration.spv");
        let fwd_bytes = include_bytes!("vulkan/shaders/spectral_norm_linear_fwd.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/spectral_norm_linear_bwd.spv");

        let power_spv = as_u32_slice(power_bytes);
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

        // ==================== Power Iteration ====================
        let power_layout = create_ds_layout(device.clone(), 4);
        let power_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(power_spv))
                .expect("Failed to create power iteration shader module")
        };
        // push: in_features, out_features, eps (3 слова по 4 байта = 12 байт)
        let power_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12,
        };
        let power_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![power_layout],
                push_constant_ranges: vec![power_push],
                ..Default::default()
            },
        )
        .expect("Failed to create power iteration pipeline layout");
        let power_entry = power_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("power iteration entry point not found");
        let power_stage = PipelineShaderStageCreateInfo::new(power_entry);
        let power_iteration = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(power_stage, power_pipeline_layout),
        )
        .expect("Failed to create power iteration pipeline");

        // ==================== Forward ====================
        let fwd_layout = create_ds_layout(device.clone(), 4);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create forward shader module")
        };
        // push: batch, in_features, out_features, scale, sigma (3 u32 + 2 f32 = 20 байт)
        let fwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 20,
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![fwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create forward pipeline");

        // ==================== Backward ====================
        let bwd_layout = create_ds_layout(device.clone(), 7);
        let bwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_spv))
                .expect("Failed to create backward shader module")
        };
        // push: batch, in_features, out_features, scale, sigma, phase (3 u32 + 2 f32 + 1 u32 = 24 байта)
        let bwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 24,
        };
        let bwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_layout],
                push_constant_ranges: vec![bwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create backward pipeline layout");
        let bwd_entry = bwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("backward entry point not found");
        let bwd_stage = PipelineShaderStageCreateInfo::new(bwd_entry);
        let backward = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_stage, bwd_pipeline_layout),
        )
        .expect("Failed to create backward pipeline");

        Self {
            power_iteration,
            forward,
            backward,
        }
    }
}