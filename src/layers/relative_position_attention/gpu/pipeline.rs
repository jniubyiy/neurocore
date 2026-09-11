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

/// Пайплайны RelativePositionAttention, разбитые на этапы.
///
/// Forward:
///   prepare_qkv → scores_softmax → output
///
/// Backward:
///   backward_output_params → backward_dv, backward_dweights
///                          → backward_scores_softmax
///                          → backward_dq, backward_dk, backward_rel_bias
///                          → backward_input_params
pub struct RelativePositionAttentionPipelines {
    // Forward
    pub prepare_qkv: Arc<ComputePipeline>,
    pub scores_softmax: Arc<ComputePipeline>,
    pub output: Arc<ComputePipeline>,

    // Backward — выходной линейный слой
    pub backward_output_params: Arc<ComputePipeline>,

    // Backward — V и weights
    pub backward_dv: Arc<ComputePipeline>,
    pub backward_dweights: Arc<ComputePipeline>,

    // Backward — softmax скоров
    pub backward_scores_softmax: Arc<ComputePipeline>,

    // Backward — Q, K, rel_bias
    pub backward_dq: Arc<ComputePipeline>,
    pub backward_dk: Arc<ComputePipeline>,
    pub backward_rel_bias: Arc<ComputePipeline>,

    // Backward — вход и QKV-параметры
    pub backward_input_params: Arc<ComputePipeline>,
}

impl RelativePositionAttentionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let prepare_bytes      = include_bytes!("vulkan/shaders/relative_position_attention_fwd_prepare_qkv.spv");
        let scores_bytes       = include_bytes!("vulkan/shaders/relative_position_attention_fwd_scores_softmax.spv");
        let output_bytes       = include_bytes!("vulkan/shaders/relative_position_attention_fwd_output.spv");
        let bwd_scores_bytes   = include_bytes!("vulkan/shaders/relative_position_attention_bwd_scores_softmax.spv");
        let bwd_dq_bytes       = include_bytes!("vulkan/shaders/relative_position_attention_bwd_dq.spv");
        let bwd_dk_bytes       = include_bytes!("vulkan/shaders/relative_position_attention_bwd_dk.spv");
        let bwd_rel_bias_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_rel_bias.spv");
        let bwd_dv_bytes       = include_bytes!("vulkan/shaders/relative_position_attention_bwd_dv.spv");
        let bwd_dweights_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_dweights.spv");
        let bwd_output_params_bytes = include_bytes!("vulkan/shaders/relative_position_attention_bwd_output_params.spv");
        let bwd_input_params_bytes  = include_bytes!("vulkan/shaders/relative_position_attention_bwd_input_params.spv");

        let prepare_spv         = as_u32_slice(prepare_bytes);
        let scores_spv          = as_u32_slice(scores_bytes);
        let output_spv          = as_u32_slice(output_bytes);
        let bwd_scores_spv      = as_u32_slice(bwd_scores_bytes);
        let bwd_dq_spv          = as_u32_slice(bwd_dq_bytes);
        let bwd_dk_spv          = as_u32_slice(bwd_dk_bytes);
        let bwd_rel_bias_spv    = as_u32_slice(bwd_rel_bias_bytes);
        let bwd_dv_spv          = as_u32_slice(bwd_dv_bytes);
        let bwd_dweights_spv    = as_u32_slice(bwd_dweights_bytes);
        let bwd_output_params_spv = as_u32_slice(bwd_output_params_bytes);
        let bwd_input_params_spv  = as_u32_slice(bwd_input_params_bytes);

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

        // Все шейдеры RelativePositionAttention используют push = [batch, seq_len, d_model] (12 байт).
        const PUSH: u32 = 12;

        // Forward
        let prepare_qkv =
            build(device.clone(), prepare_spv, 5, PUSH, "RPA prepare_qkv");
        let scores_softmax =
            build(device.clone(), scores_spv, 5, PUSH, "RPA scores_softmax");
        let output =
            build(device.clone(), output_spv, 5, PUSH, "RPA output");

        // Backward — выходной линейный слой
        let backward_output_params =
            build(device.clone(), bwd_output_params_spv, 6, PUSH, "RPA bwd_output_params");

        // Backward — V и weights
        let backward_dv =
            build(device.clone(), bwd_dv_spv, 3, PUSH, "RPA bwd_dv");
        let backward_dweights =
            build(device.clone(), bwd_dweights_spv, 3, PUSH, "RPA bwd_dweights");

        // Backward — softmax скоров
        let backward_scores_softmax =
            build(device.clone(), bwd_scores_spv, 3, PUSH, "RPA bwd_scores_softmax");

        // Backward — Q, K, rel_bias
        let backward_dq =
            build(device.clone(), bwd_dq_spv, 3, PUSH, "RPA bwd_dq");
        let backward_dk =
            build(device.clone(), bwd_dk_spv, 3, PUSH, "RPA bwd_dk");
        let backward_rel_bias =
            build(device.clone(), bwd_rel_bias_spv, 2, PUSH, "RPA bwd_rel_bias");

        // Backward — вход и QKV-параметры
        let backward_input_params =
            build(device.clone(), bwd_input_params_spv, 7, PUSH, "RPA bwd_input_params");

        Self {
            prepare_qkv,
            scores_softmax,
            output,
            backward_output_params,
            backward_dv,
            backward_dweights,
            backward_scores_softmax,
            backward_dq,
            backward_dk,
            backward_rel_bias,
            backward_input_params,
        }
    }
}