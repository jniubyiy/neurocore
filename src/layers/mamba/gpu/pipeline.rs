// src/layers/mamba/gpu/pipeline.rs
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

pub struct MambaPipelines {
    pub discretize: Arc<ComputePipeline>,
    pub forward_step: Arc<ComputePipeline>,
    pub backward_step: Arc<ComputePipeline>,
    pub convert_grads: Arc<ComputePipeline>,
}

impl MambaPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let discretize_bytes = include_bytes!("vulkan/shaders/mamba_discretize.spv");
        let fwd_step_bytes = include_bytes!("vulkan/shaders/mamba_fwd_step.spv");
        let bwd_step_bytes = include_bytes!("vulkan/shaders/mamba_bwd_step.spv");
        let convert_grads_bytes = include_bytes!("vulkan/shaders/mamba_convert_grads.spv");

        let discretize_spv = as_u32_slice(discretize_bytes);
        let fwd_step_spv = as_u32_slice(fwd_step_bytes);
        let bwd_step_spv = as_u32_slice(bwd_step_bytes);
        let convert_grads_spv = as_u32_slice(convert_grads_bytes);

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
            .expect("Failed to create descriptor set layout for Mamba")
        }

        // ==================== Discretize ====================
        let discretize_layout = create_ds_layout(device.clone(), 4);
        let discretize_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(discretize_spv))
                .expect("Failed to create Mamba discretize shader module")
        };
        let discretize_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 16, // n, d, delta, padding
        };
        let discretize_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![discretize_layout],
                push_constant_ranges: vec![discretize_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Mamba discretize pipeline layout");
        let discretize_entry = discretize_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Mamba discretize entry point not found");
        let discretize_stage = PipelineShaderStageCreateInfo::new(discretize_entry);
        let discretize = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(discretize_stage, discretize_pipeline_layout),
        )
        .expect("Failed to create Mamba discretize pipeline");

        // ==================== Forward Step ====================
        let fwd_step_layout = create_ds_layout(device.clone(), 7);
        let fwd_step_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_step_spv))
                .expect("Failed to create Mamba forward step shader module")
        };
        let fwd_step_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 24, // batch, d, n, seq_len, t, phase
        };
        let fwd_step_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_step_layout],
                push_constant_ranges: vec![fwd_step_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Mamba forward step pipeline layout");
        let fwd_step_entry = fwd_step_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Mamba forward step entry point not found");
        let fwd_step_stage = PipelineShaderStageCreateInfo::new(fwd_step_entry);
        let forward_step = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_step_stage, fwd_step_pipeline_layout),
        )
        .expect("Failed to create Mamba forward step pipeline");

        // ==================== Backward Step ====================
        let bwd_step_layout = create_ds_layout(device.clone(), 14);
        let bwd_step_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_step_spv))
                .expect("Failed to create Mamba backward step shader module")
        };
        let bwd_step_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 20, // batch, d, n, seq_len, t
        };
        let bwd_step_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_step_layout],
                push_constant_ranges: vec![bwd_step_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Mamba backward step pipeline layout");
        let bwd_step_entry = bwd_step_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Mamba backward step entry point not found");
        let bwd_step_stage = PipelineShaderStageCreateInfo::new(bwd_step_entry);
        let backward_step = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_step_stage, bwd_step_pipeline_layout),
        )
        .expect("Failed to create Mamba backward step pipeline");

        // ==================== Convert Gradients ====================
        let convert_grads_layout = create_ds_layout(device.clone(), 7);
        let convert_grads_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(convert_grads_spv))
                .expect("Failed to create Mamba convert gradients shader module")
        };
        let convert_grads_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 16, // n, d, delta, padding
        };
        let convert_grads_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![convert_grads_layout],
                push_constant_ranges: vec![convert_grads_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Mamba convert gradients pipeline layout");
        let convert_grads_entry = convert_grads_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Mamba convert gradients entry point not found");
        let convert_grads_stage = PipelineShaderStageCreateInfo::new(convert_grads_entry);
        let convert_grads = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(convert_grads_stage, convert_grads_pipeline_layout),
        )
        .expect("Failed to create Mamba convert gradients pipeline");

        Self {
            discretize,
            forward_step,
            backward_step,
            convert_grads,
        }
    }
}