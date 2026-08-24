// src/layers/tanh/gpu/pipeline.rs
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

/// Пайплайны для слоя Tanh
pub struct TanhPipelines {
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>,
}

impl TanhPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        // Загружаем SPIR-V
        let fwd_bytes = include_bytes!("vulkan/shaders/tanh_fwd.spv");
        let bwd_bytes = include_bytes!("vulkan/shaders/tanh_bwd.spv");
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
            .expect("Failed to create descriptor set layout for Tanh")
        }

        // Прямой пайплайн (2 буфера)
        let fwd_layout = create_ds_layout(device.clone(), 2);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create Tanh forward shader module")
        };
        let fwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 4,
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![fwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Tanh forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Tanh forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create Tanh forward pipeline");

        // Обратный пайплайн (3 буфера)
        let bwd_layout = create_ds_layout(device.clone(), 3);
        let bwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_spv))
                .expect("Failed to create Tanh backward shader module")
        };
        let bwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 4,
        };
        let bwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_layout],
                push_constant_ranges: vec![bwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create Tanh backward pipeline layout");
        let bwd_entry = bwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("Tanh backward entry point not found");
        let bwd_stage = PipelineShaderStageCreateInfo::new(bwd_entry);
        let backward = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_stage, bwd_pipeline_layout),
        )
        .expect("Failed to create Tanh backward pipeline");

        Self { forward, backward }
    }
}