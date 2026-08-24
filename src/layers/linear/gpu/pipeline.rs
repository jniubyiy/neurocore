// src/layers/linear/gpu/pipeline.rs
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

/// Преобразует включённые байты SPIR‑V в слайс u32.
fn as_u32_slice(bytes: &[u8]) -> &[u32] {
    assert!(bytes.len() % 4 == 0, "SPIR‑V файл должен быть выровнен по 4 байта");
    let ptr = bytes.as_ptr() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, bytes.len() / 4) }
}

/// Пайплайны для слоя Linear
pub struct LinearPipelines {
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>,
}

impl LinearPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        // Загружаем SPIR-V
        let fwd_bytes = include_bytes!("vulkan/shaders/linear_fwd.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/linear_bwd.spv");
        let fwd_spv = as_u32_slice(fwd_bytes);
        let bwd_spv = as_u32_slice(bwd_bytes);

        // Вспомогательная функция создания layout с N storage-буферами
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
            .expect("Failed to create descriptor set layout for Linear")
        }

        // Прямой пайплайн (4 буфера: input, weight, bias, output)
        let fwd_layout = create_ds_layout(device.clone(), 4);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create Linear forward shader module")
        };
        let fwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12, // batch, in_features, out_features
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![fwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Linear forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Linear forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create Linear forward pipeline");

        // Обратный пайплайн (6 буферов: input, weight, grad_output, grad_input, grad_weight, grad_bias)
        let bwd_layout = create_ds_layout(device.clone(), 6);
        let bwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_spv))
                .expect("Failed to create Linear backward shader module")
        };
        let bwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12, // batch, in_features, out_features
        };
        let bwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_layout],
                push_constant_ranges: vec![bwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Linear backward pipeline layout");
        let bwd_entry = bwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Linear backward entry point not found");
        let bwd_stage = PipelineShaderStageCreateInfo::new(bwd_entry);
        let backward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_stage, bwd_pipeline_layout),
        )
        .expect("Failed to create Linear backward pipeline");

        Self { forward, backward }
    }
}