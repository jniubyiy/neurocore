// src/layers/ind_rnn/gpu/pipeline.rs
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

/// Пайплайны для слоя IndRNN
pub struct IndRNNPipelines {
    pub forward_step: Arc<ComputePipeline>,
    pub backward_step: Arc<ComputePipeline>,
}

impl IndRNNPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        // Загружаем SPIR-V
        let fwd_bytes = include_bytes!("vulkan/shaders/ind_rnn_fwd_step.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/ind_rnn_bwd_step.spv");
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
            .expect("Failed to create descriptor set layout for IndRNN")
        }

        // Push-константы для обоих пайплайнов: batch, features, seq_len, t (16 байт)
        let push_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 16, // 4 * u32
        };

        // Прямой шаг (6 буферов)
        let fwd_layout = create_ds_layout(device.clone(), 6);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create IndRNN forward shader module")
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create IndRNN forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("IndRNN forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward_step = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create IndRNN forward pipeline");

        // Обратный шаг (11 буферов)
        let bwd_layout = create_ds_layout(device.clone(), 11);
        let bwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_spv))
                .expect("Failed to create IndRNN backward shader module")
        };
        let bwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create IndRNN backward pipeline layout");
        let bwd_entry = bwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("IndRNN backward entry point not found");
        let bwd_stage = PipelineShaderStageCreateInfo::new(bwd_entry);
        let backward_step = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_stage, bwd_pipeline_layout),
        )
        .expect("Failed to create IndRNN backward pipeline");

        Self { forward_step, backward_step }
    }
}