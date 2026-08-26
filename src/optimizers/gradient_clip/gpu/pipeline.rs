// src/optimizers/gradient_clip/gpu/pipeline.rs
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

pub struct GradientClipPipelines {
    pub forward: Arc<ComputePipeline>,
}

impl GradientClipPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_bytes = include_bytes!("shaders/grad_clip.spv");
        let fwd_spv = as_u32_slice(fwd_bytes);

        let fwd_layout = Self::create_ds_layout(device.clone(), 1);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create GradientClip shader module")
        };
        let fwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12, // min_val, max_val, total
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![fwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create GradientClip pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("GradientClip entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create GradientClip pipeline");

        Self { forward }
    }

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
        .expect("Failed to create descriptor set layout for GradientClip")
    }
}