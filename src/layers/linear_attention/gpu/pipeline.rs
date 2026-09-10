// src/layers/linear_attention/gpu/pipeline.rs
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

pub struct LinearAttentionPipelines {
    pub compute_kvz: Arc<ComputePipeline>,
    pub forward: Arc<ComputePipeline>,
    pub backward_main: Arc<ComputePipeline>,
    pub backward_params: Arc<ComputePipeline>,
    pub phi: Arc<ComputePipeline>,
    pub linear_per_token: Arc<ComputePipeline>,
}

impl LinearAttentionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        // Загрузка SPIR‑V
        let kvz_bytes = include_bytes!("vulkan/shaders/linear_attention_compute_kvz.spv");
        let fwd_bytes = include_bytes!("vulkan/shaders/linear_attention_fwd.spv");
        let bwd_main_bytes = include_bytes!("vulkan/shaders/linear_attention_bwd_main.spv");
        let bwd_params_bytes = include_bytes!("vulkan/shaders/linear_attention_bwd_params.spv");
        let phi_bytes = include_bytes!("vulkan/shaders/linear_attention_phi.spv");
        let per_token_bytes = include_bytes!("vulkan/shaders/linear_attention_linear_per_token.spv");

        let kvz_spv = as_u32_slice(kvz_bytes);
        let fwd_spv = as_u32_slice(fwd_bytes);
        let bwd_main_spv = as_u32_slice(bwd_main_bytes);
        let bwd_params_spv = as_u32_slice(bwd_params_bytes);
        let phi_spv = as_u32_slice(phi_bytes);
        let per_token_spv = as_u32_slice(per_token_bytes);

        // Вспомогательная функция создания layout с N storage-буферами.
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
            .expect("Failed to create descriptor set layout for LinearAttention")
        }

        // ==================== Compute KVZ ====================
        let kvz_layout = create_ds_layout(device.clone(), 4);
        let kvz_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(kvz_spv))
                .expect("Failed to create LinearAttention compute_kvz shader module")
        };
        let kvz_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12, // batch, seq_len, d_model
        };
        let kvz_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![kvz_layout],
                push_constant_ranges: vec![kvz_push],
                ..Default::default()
            },
        )
        .expect("Failed to create LinearAttention compute_kvz pipeline layout");
        let kvz_entry = kvz_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("LinearAttention compute_kvz entry point not found");
        let kvz_stage = PipelineShaderStageCreateInfo::new(kvz_entry);
        let compute_kvz = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(kvz_stage, kvz_pipeline_layout),
        )
        .expect("Failed to create LinearAttention compute_kvz pipeline");

        // ==================== Forward ====================
        // 6 bindings: q_phi, kv, z, w_o, b_o, y
        let fwd_layout = create_ds_layout(device.clone(), 6);
        let fwd_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(fwd_spv))
                .expect("Failed to create LinearAttention forward shader module")
        };
        let fwd_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12, // batch, seq_len, d_model
        };
        let fwd_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![fwd_layout],
                push_constant_ranges: vec![fwd_push],
                ..Default::default()
            },
        )
        .expect("Failed to create LinearAttention forward pipeline layout");
        let fwd_entry = fwd_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("LinearAttention forward entry point not found");
        let fwd_stage = PipelineShaderStageCreateInfo::new(fwd_entry);
        let forward = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(fwd_stage, fwd_pipeline_layout),
        )
        .expect("Failed to create LinearAttention forward pipeline");

        // ==================== Backward Main ====================
        // 9 bindings: go, w_o, q_phi, kv, z, d_attn_out, d_q_phi, d_kv, d_z
        let bwd_main_layout = create_ds_layout(device.clone(), 9);
        let bwd_main_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_main_spv))
                .expect("Failed to create LinearAttention backward main shader module")
        };
        let bwd_main_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12,
        };
        let bwd_main_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_main_layout],
                push_constant_ranges: vec![bwd_main_push],
                ..Default::default()
            },
        )
        .expect("Failed to create LinearAttention backward main pipeline layout");
        let bwd_main_entry = bwd_main_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("LinearAttention backward main entry point not found");
        let bwd_main_stage = PipelineShaderStageCreateInfo::new(bwd_main_entry);
        let backward_main = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_main_stage, bwd_main_pipeline_layout),
        )
        .expect("Failed to create LinearAttention backward main pipeline");

        // ==================== Backward Params ====================
        // 15 bindings: x, go, params, q_phi, k_phi, v, q_raw, k_raw,
        //              d_q_phi, d_kv, d_z, gi, grad_params, kv, z
        let bwd_params_layout = create_ds_layout(device.clone(), 15);
        let bwd_params_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(bwd_params_spv))
                .expect("Failed to create LinearAttention backward params shader module")
        };
        let bwd_params_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12,
        };
        let bwd_params_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![bwd_params_layout],
                push_constant_ranges: vec![bwd_params_push],
                ..Default::default()
            },
        )
        .expect("Failed to create LinearAttention backward params pipeline layout");
        let bwd_params_entry = bwd_params_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("LinearAttention backward params entry point not found");
        let bwd_params_stage = PipelineShaderStageCreateInfo::new(bwd_params_entry);
        let backward_params = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(bwd_params_stage, bwd_params_pipeline_layout),
        )
        .expect("Failed to create LinearAttention backward params pipeline");

        // ==================== Phi ====================
        // 2 bindings: x, y
        let phi_layout = create_ds_layout(device.clone(), 2);
        let phi_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(phi_spv))
                .expect("Failed to create LinearAttention phi shader module")
        };
        let phi_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 4, // total
        };
        let phi_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![phi_layout],
                push_constant_ranges: vec![phi_push],
                ..Default::default()
            },
        )
        .expect("Failed to create LinearAttention phi pipeline layout");
        let phi_entry = phi_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("LinearAttention phi entry point not found");
        let phi_stage = PipelineShaderStageCreateInfo::new(phi_entry);
        let phi = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(phi_stage, phi_pipeline_layout),
        )
        .expect("Failed to create LinearAttention phi pipeline");

        // ==================== Per-Token Linear ====================
        // 4 bindings: x, w, b, y
        let per_token_layout = create_ds_layout(device.clone(), 4);
        let per_token_module = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(per_token_spv))
                .expect("Failed to create LinearAttention per-token linear shader module")
        };
        let per_token_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 16, // batch, seq_len, in_features, out_features
        };
        let per_token_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![per_token_layout],
                push_constant_ranges: vec![per_token_push],
                ..Default::default()
            },
        )
        .expect("Failed to create LinearAttention per-token linear pipeline layout");
        let per_token_entry = per_token_module
            .entry_point_with_execution("main", ExecutionModel::GLCompute)
            .expect("LinearAttention per-token linear entry point not found");
        let per_token_stage = PipelineShaderStageCreateInfo::new(per_token_entry);
        let linear_per_token = ComputePipeline::new(
            device,
            None,
            ComputePipelineCreateInfo::stage_layout(per_token_stage, per_token_pipeline_layout),
        )
        .expect("Failed to create LinearAttention per-token linear pipeline");

        Self {
            compute_kvz,
            forward,
            backward_main,
            backward_params,
            phi,
            linear_per_token,
        }
    }
}

