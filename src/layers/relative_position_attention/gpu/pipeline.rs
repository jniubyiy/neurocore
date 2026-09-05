// src/layers/relative_position_attention/gpu/pipeline.rs
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

pub struct RelativePositionAttentionPipelines {
    pub prepare_qkv: Arc<ComputePipeline>,
    pub scores_softmax: Arc<ComputePipeline>,
    pub output: Arc<ComputePipeline>,
    pub backward_scores_softmax: Arc<ComputePipeline>,
    pub backward_qkv_params: Arc<ComputePipeline>,
    pub backward_output_params: Arc<ComputePipeline>,
    pub backward_values_weights: Arc<ComputePipeline>,
    pub backward_input_params: Arc<ComputePipeline>,
}

impl RelativePositionAttentionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let prepare_bytes = include_bytes!("vulkan/shaders/relative_position_attention_fwd_prepare_qkv.spv");
        let scores_bytes = include_bytes!("vulkan/shaders/relative_position_attention_fwd_scores_softmax.spv");
        let output_bytes = include_bytes!("vulkan/shaders/relative_position_attention_fwd_output.spv");
        let bwd_scores_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_scores_softmax.spv");
        let bwd_qkv_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_qkv_params.spv");
        let bwd_output_params_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_output_params.spv");
        let bwd_values_weights_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_values_weights.spv");
        let bwd_input_params_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_input_params.spv");

        let prepare_spv = as_u32_slice(prepare_bytes);
        let scores_spv = as_u32_slice(scores_bytes);
        let output_spv = as_u32_slice(output_bytes);
        let bwd_scores_spv = as_u32_slice(bwd_scores_bytes);
        let bwd_qkv_spv = as_u32_slice(bwd_qkv_bytes);
        let bwd_output_params_spv = as_u32_slice(bwd_output_params_bytes);
        let bwd_values_weights_spv = as_u32_slice(bwd_values_weights_bytes);
        let bwd_input_params_spv = as_u32_slice(bwd_input_params_bytes);

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
            .expect("Failed to create descriptor set layout for RelativePositionAttention")
        }

        // Push constant для всех пайплайнов: batch, seq_len, d_model (12 байт)
        let push_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12,
        };

        // ==================== Prepare QKV ====================
        let prepare_layout = create_ds_layout(device.clone(), 5);
        let prepare_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(prepare_spv))
                .expect("Failed to create prepare QKV shader module")
        };
        let prepare_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![prepare_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create prepare QKV pipeline layout");
        let prepare_entry = prepare_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("prepare QKV entry point not found");
        let prepare_stage = PipelineShaderStageCreateInfo::new(prepare_entry);
        let prepare_qkv = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(prepare_stage, prepare_pipeline_layout),
        )
        .expect("Failed to create prepare QKV pipeline");

        // ==================== Scores & Softmax ====================
        let scores_layout = create_ds_layout(device.clone(), 5);
        let scores_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(scores_spv))
                .expect("Failed to create scores/softmax shader module")
        };
        let scores_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![scores_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create scores/softmax pipeline layout");
        let scores_entry = scores_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("scores/softmax entry point not found");
        let scores_stage = PipelineShaderStageCreateInfo::new(scores_entry);
        let scores_softmax = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(scores_stage, scores_pipeline_layout),
        )
        .expect("Failed to create scores/softmax pipeline");

        // ==================== Output ====================
        let output_layout = create_ds_layout(device.clone(), 5);
        let output_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(output_spv))
                .expect("Failed to create output shader module")
        };
        let output_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![output_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create output pipeline layout");
        let output_entry = output_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("output entry point not found");
        let output_stage = PipelineShaderStageCreateInfo::new(output_entry);
        let output = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(output_stage, output_pipeline_layout),
        )
        .expect("Failed to create output pipeline");

        // ==================== Backward Scores & Softmax ====================
        let bwd_scores_layout = create_ds_layout(device.clone(), 3);
        let bwd_scores_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_scores_spv))
                .expect("Failed to create backward scores shader module")
        };
        let bwd_scores_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_scores_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create backward scores pipeline layout");
        let bwd_scores_entry = bwd_scores_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("backward scores entry point not found");
        let bwd_scores_stage = PipelineShaderStageCreateInfo::new(bwd_scores_entry);
        let backward_scores_softmax = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_scores_stage, bwd_scores_pipeline_layout),
        )
        .expect("Failed to create backward scores pipeline");

        // ==================== Backward QKV Params ====================
        let bwd_qkv_layout = create_ds_layout(device.clone(), 6);
        let bwd_qkv_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_qkv_spv))
                .expect("Failed to create backward QKV params shader module")
        };
        let bwd_qkv_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_qkv_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create backward QKV params pipeline layout");
        let bwd_qkv_entry = bwd_qkv_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("backward QKV params entry point not found");
        let bwd_qkv_stage = PipelineShaderStageCreateInfo::new(bwd_qkv_entry);
        let backward_qkv_params = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_qkv_stage, bwd_qkv_pipeline_layout),
        )
        .expect("Failed to create backward QKV params pipeline");

        // ==================== Backward Output Params (новый) ====================
        let bwd_output_params_layout = create_ds_layout(device.clone(), 6);
        let bwd_output_params_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_output_params_spv))
                .expect("Failed to create backward output params shader module")
        };
        let bwd_output_params_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_output_params_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create backward output params pipeline layout");
        let bwd_output_params_entry = bwd_output_params_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("backward output params entry point not found");
        let bwd_output_params_stage = PipelineShaderStageCreateInfo::new(bwd_output_params_entry);
        let backward_output_params = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_output_params_stage, bwd_output_params_pipeline_layout),
        )
        .expect("Failed to create backward output params pipeline");

        // ==================== Backward Values & Weights (новый) ====================
        let bwd_values_weights_layout = create_ds_layout(device.clone(), 5);
        let bwd_values_weights_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_values_weights_spv))
                .expect("Failed to create backward values/weights shader module")
        };
        let bwd_values_weights_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_values_weights_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create backward values/weights pipeline layout");
        let bwd_values_weights_entry = bwd_values_weights_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("backward values/weights entry point not found");
        let bwd_values_weights_stage = PipelineShaderStageCreateInfo::new(bwd_values_weights_entry);
        let backward_values_weights = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_values_weights_stage, bwd_values_weights_pipeline_layout),
        )
        .expect("Failed to create backward values/weights pipeline");

        // ==================== Backward Input Params (новый) ====================
        let bwd_input_params_layout = create_ds_layout(device.clone(), 7);
        let bwd_input_params_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_input_params_spv))
                .expect("Failed to create backward input params shader module")
        };
        let bwd_input_params_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_input_params_layout],
                push_constant_ranges: vec![push_range],
                ..Default::default()
            },
        )
        .expect("Failed to create backward input params pipeline layout");
        let bwd_input_params_entry = bwd_input_params_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("backward input params entry point not found");
        let bwd_input_params_stage = PipelineShaderStageCreateInfo::new(bwd_input_params_entry);
        let backward_input_params = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_input_params_stage, bwd_input_params_pipeline_layout),
        )
        .expect("Failed to create backward input params pipeline");

        Self {
            prepare_qkv,
            scores_softmax,
            output,
            backward_scores_softmax,
            backward_qkv_params,
            backward_output_params,
            backward_values_weights,
            backward_input_params,
        }
    }
}