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

/// Пайплайны LinearAttention, разбитые на этапы.
///
/// Forward:
///   linear_per_token (×3)  →  phi (×2)  →  compute_kvz  →  forward
///
/// Backward:
///   bwd_prepare → bwd_dq_phi_dkv → bwd_dk_phi → bwd_dv
///               → bwd_grad_wo  → bwd_gi_params
pub struct LinearAttentionPipelines {
    // Forward
    pub compute_kvz: Arc<ComputePipeline>,
    pub forward: Arc<ComputePipeline>,
    pub phi: Arc<ComputePipeline>,
    pub linear_per_token: Arc<ComputePipeline>,

    // Backward
    pub bwd_prepare: Arc<ComputePipeline>,
    pub bwd_dq_phi_dkv: Arc<ComputePipeline>,
    pub bwd_dk_phi: Arc<ComputePipeline>,
    pub bwd_dv: Arc<ComputePipeline>,
    pub bwd_grad_wo: Arc<ComputePipeline>,
    pub bwd_gi_params: Arc<ComputePipeline>,
}

impl LinearAttentionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let kvz_bytes = include_bytes!("vulkan/shaders/linear_attention_compute_kvz.spv");
        let fwd_bytes = include_bytes!("vulkan/shaders/linear_attention_fwd.spv");
        let phi_bytes = include_bytes!("vulkan/shaders/linear_attention_phi.spv");
        let per_token_bytes =
            include_bytes!("vulkan/shaders/linear_attention_linear_per_token.spv");
        let bwd_prepare_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_prepare.spv");
        let bwd_dq_phi_dkv_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_dq_phi_dkv.spv");
        let bwd_dk_phi_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_dk_phi.spv");
        let bwd_dv_bytes = include_bytes!("vulkan/shaders/linear_attention_bwd_dv.spv");
        let bwd_grad_wo_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_grad_wo.spv");
        let bwd_gi_params_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_gi_params.spv");

        let kvz_spv = as_u32_slice(kvz_bytes);
        let fwd_spv = as_u32_slice(fwd_bytes);
        let phi_spv = as_u32_slice(phi_bytes);
        let per_token_spv = as_u32_slice(per_token_bytes);
        let bwd_prepare_spv = as_u32_slice(bwd_prepare_bytes);
        let bwd_dq_phi_dkv_spv = as_u32_slice(bwd_dq_phi_dkv_bytes);
        let bwd_dk_phi_spv = as_u32_slice(bwd_dk_phi_bytes);
        let bwd_dv_spv = as_u32_slice(bwd_dv_bytes);
        let bwd_grad_wo_spv = as_u32_slice(bwd_grad_wo_bytes);
        let bwd_gi_params_spv = as_u32_slice(bwd_gi_params_bytes);

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

        // Forward
        let compute_kvz = build(device.clone(), kvz_spv, 4, 12, "LinearAttention compute_kvz");
        let forward = build(device.clone(), fwd_spv, 6, 12, "LinearAttention forward");
        let phi = build(device.clone(), phi_spv, 2, 4, "LinearAttention phi");
        let linear_per_token =
            build(device.clone(), per_token_spv, 4, 16, "LinearAttention per_token");

        // Backward
        let bwd_prepare =
            build(device.clone(), bwd_prepare_spv, 9, 12, "LinearAttention bwd_prepare");
        let bwd_dq_phi_dkv = build(
            device.clone(),
            bwd_dq_phi_dkv_spv,
            10,
            12,
            "LinearAttention bwd_dq_phi_dkv",
        );
        let bwd_dk_phi =
            build(device.clone(), bwd_dk_phi_spv, 4, 12, "LinearAttention bwd_dk_phi");
        let bwd_dv =
            build(device.clone(), bwd_dv_spv, 3, 12, "LinearAttention bwd_dv");
        let bwd_grad_wo =
            build(device.clone(), bwd_grad_wo_spv, 3, 12, "LinearAttention bwd_grad_wo");
        let bwd_gi_params = build(
            device.clone(),
            bwd_gi_params_spv,
            9,
            12,
            "LinearAttention bwd_gi_params",
        );

        Self {
            compute_kvz,
            forward,
            phi,
            linear_per_token,
            bwd_prepare,
            bwd_dq_phi_dkv,
            bwd_dk_phi,
            bwd_dv,
            bwd_grad_wo,
            bwd_gi_params,
        }
    }
}

