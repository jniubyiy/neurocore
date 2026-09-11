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

/// Пайплайны AdaptiveNormalization, разбитые на этапы.
///
/// Прямой проход:
///   row_stats → col_stats → forward
///
/// Обратный проход:
///   row_stats → col_stats → bwd_weights → bwd_row_sums → bwd_col_sums
///   → bwd_input , bwd_params
pub struct AdaptiveNormalizationPipelines {
    pub forward: Arc<ComputePipeline>,
    pub backward: Arc<ComputePipeline>, // оставлено для обратной совместимости полей
    pub row_stats: Arc<ComputePipeline>,
    pub col_stats: Arc<ComputePipeline>,
    pub bwd_weights: Arc<ComputePipeline>,
    pub bwd_row_sums: Arc<ComputePipeline>,
    pub bwd_col_sums: Arc<ComputePipeline>,
    pub bwd_input: Arc<ComputePipeline>,
    pub bwd_params: Arc<ComputePipeline>,
}

impl AdaptiveNormalizationPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_bytes = include_bytes!("vulkan/shaders/adaptive_norm_fwd.spv");
        let row_stats_bytes = include_bytes!("vulkan/shaders/adaptive_norm_row_stats.spv");
        let col_stats_bytes = include_bytes!("vulkan/shaders/adaptive_norm_col_stats.spv");
        let bwd_weights_bytes = include_bytes!("vulkan/shaders/adaptive_norm_bwd_weights.spv");
        let bwd_row_sums_bytes = include_bytes!("vulkan/shaders/adaptive_norm_bwd_row_sums.spv");
        let bwd_col_sums_bytes = include_bytes!("vulkan/shaders/adaptive_norm_bwd_col_sums.spv");
        let bwd_input_bytes = include_bytes!("vulkan/shaders/adaptive_norm_bwd_input.spv");
        let bwd_params_bytes = include_bytes!("vulkan/shaders/adaptive_norm_bwd_params.spv");

        let fwd_spv         = as_u32_slice(fwd_bytes);
        let row_stats_spv   = as_u32_slice(row_stats_bytes);
        let col_stats_spv   = as_u32_slice(col_stats_bytes);
        let bwd_weights_spv = as_u32_slice(bwd_weights_bytes);
        let bwd_row_sums_spv = as_u32_slice(bwd_row_sums_bytes);
        let bwd_col_sums_spv = as_u32_slice(bwd_col_sums_bytes);
        let bwd_input_spv   = as_u32_slice(bwd_input_bytes);
        let bwd_params_spv  = as_u32_slice(bwd_params_bytes);

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

        // row_stats: 4 буфера, push = [batch, features] (8 байт).
        let row_stats = build(device.clone(), row_stats_spv, 4, 8, "AdaptiveNorm row_stats");

        // col_stats: 3 буфера, push = [batch, features] (8 байт).
        let col_stats = build(device.clone(), col_stats_spv, 3, 8, "AdaptiveNorm col_stats");

        // forward: 8 буферов, push = [batch, features] (8 байт).
        let forward = build(device.clone(), fwd_spv, 8, 8, "AdaptiveNorm forward");

        // bwd_weights: 4 буфера, push = [features] (4 байта).
        let bwd_weights = build(device.clone(), bwd_weights_spv, 4, 4, "AdaptiveNorm bwd_weights");

        // bwd_row_sums: 8 буферов, push = [batch, features] (8 байт).
        let bwd_row_sums = build(device.clone(), bwd_row_sums_spv, 8, 8, "AdaptiveNorm bwd_row_sums");

        // bwd_col_sums: 6 буферов, push = [batch, features] (8 байт).
        let bwd_col_sums = build(device.clone(), bwd_col_sums_spv, 6, 8, "AdaptiveNorm bwd_col_sums");

        // bwd_input: 17 буферов, push = [batch, features] (8 байт).
        let bwd_input = build(device.clone(), bwd_input_spv, 17, 8, "AdaptiveNorm bwd_input");

        // bwd_params: 12 буферов, push = [batch, features] (8 байт).
        let bwd_params = build(device.clone(), bwd_params_spv, 12, 8, "AdaptiveNorm bwd_params");

        // Поле backward оставлено для совместимости полей. Дублируем указатель на bwd_input.
        let backward = bwd_input.clone();

        Self {
            forward,
            backward,
            row_stats,
            col_stats,
            bwd_weights,
            bwd_row_sums,
            bwd_col_sums,
            bwd_input,
            bwd_params,
        }
    }
}