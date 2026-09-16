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

/// Пайплайны LinearAttention (multi-head, d_head = d_model / max_heads).
///
/// Forward:
///   phi → linear_per_token → compute_kvz → denom → forward
///
/// Backward (в порядке выполнения для одной головы):
///   bwd_dattn → bwd_grad_wo → bwd_grad_bo
///            → bwd_dnum_denom
///            → bwd_dq_phi → bwd_dz → bwd_dkv → bwd_dk_phi → bwd_dv
///            → bwd_dqk_raw
///            → bwd_grad_qkv_weights → bwd_grad_qkv_bias
///            → bwd_self_bias
///            → bwd_dx
pub struct LinearAttentionPipelines {
    // Forward
    pub phi: Arc<ComputePipeline>,
    pub linear_per_token: Arc<ComputePipeline>,
    pub compute_kvz: Arc<ComputePipeline>,
    pub denom: Arc<ComputePipeline>,
    pub forward: Arc<ComputePipeline>,

    // Backward
    pub bwd_dattn: Arc<ComputePipeline>,
    pub bwd_grad_wo: Arc<ComputePipeline>,
    pub bwd_grad_bo: Arc<ComputePipeline>,
    pub bwd_dnum_denom: Arc<ComputePipeline>,
    pub bwd_dq_phi: Arc<ComputePipeline>,
    pub bwd_dz: Arc<ComputePipeline>,
    pub bwd_dkv: Arc<ComputePipeline>,
    pub bwd_dk_phi: Arc<ComputePipeline>,
    pub bwd_dv: Arc<ComputePipeline>,
    pub bwd_dqk_raw: Arc<ComputePipeline>,
    pub bwd_grad_qkv_weights: Arc<ComputePipeline>,
    pub bwd_grad_qkv_bias: Arc<ComputePipeline>,
    pub bwd_self_bias: Arc<ComputePipeline>,
    pub bwd_dx: Arc<ComputePipeline>,
}

impl LinearAttentionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let phi_bytes            = include_bytes!("vulkan/shaders/linear_attention_phi.spv");
        let per_token_bytes      = include_bytes!("vulkan/shaders/linear_attention_linear_per_token.spv");
        let kvz_bytes            = include_bytes!("vulkan/shaders/linear_attention_compute_kvz.spv");
        let denom_bytes          = include_bytes!("vulkan/shaders/linear_attention_denom.spv");
        let fwd_bytes            = include_bytes!("vulkan/shaders/linear_attention_fwd.spv");

        let bwd_dattn_bytes      = include_bytes!("vulkan/shaders/linear_attention_bwd_dattn.spv");
        let bwd_grad_wo_bytes    = include_bytes!("vulkan/shaders/linear_attention_bwd_grad_wo.spv");
        let bwd_grad_bo_bytes    = include_bytes!("vulkan/shaders/linear_attention_bwd_grad_bo.spv");
        let bwd_dnum_denom_bytes = include_bytes!("vulkan/shaders/linear_attention_bwd_dnum_denom.spv");
        let bwd_dq_phi_bytes     = include_bytes!("vulkan/shaders/linear_attention_bwd_dq_phi.spv");
        let bwd_dz_bytes         = include_bytes!("vulkan/shaders/linear_attention_bwd_dz.spv");
        let bwd_dkv_bytes        = include_bytes!("vulkan/shaders/linear_attention_bwd_dkv.spv");
        let bwd_dk_phi_bytes     = include_bytes!("vulkan/shaders/linear_attention_bwd_dk_phi.spv");
        let bwd_dv_bytes         = include_bytes!("vulkan/shaders/linear_attention_bwd_dv.spv");
        let bwd_dqk_raw_bytes    = include_bytes!("vulkan/shaders/linear_attention_bwd_dqk_raw.spv");
        let bwd_grad_qkv_weights_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_grad_qkv_weights.spv");
        let bwd_grad_qkv_bias_bytes =
            include_bytes!("vulkan/shaders/linear_attention_bwd_grad_qkv_bias.spv");
        let bwd_self_bias_bytes  = include_bytes!("vulkan/shaders/linear_attention_bwd_self_bias.spv");
        let bwd_dx_bytes         = include_bytes!("vulkan/shaders/linear_attention_bwd_dx.spv");

        let phi_spv            = as_u32_slice(phi_bytes);
        let per_token_spv      = as_u32_slice(per_token_bytes);
        let kvz_spv            = as_u32_slice(kvz_bytes);
        let denom_spv          = as_u32_slice(denom_bytes);
        let fwd_spv            = as_u32_slice(fwd_bytes);

        let bwd_dattn_spv      = as_u32_slice(bwd_dattn_bytes);
        let bwd_grad_wo_spv    = as_u32_slice(bwd_grad_wo_bytes);
        let bwd_grad_bo_spv    = as_u32_slice(bwd_grad_bo_bytes);
        let bwd_dnum_denom_spv = as_u32_slice(bwd_dnum_denom_bytes);
        let bwd_dq_phi_spv     = as_u32_slice(bwd_dq_phi_bytes);
        let bwd_dz_spv         = as_u32_slice(bwd_dz_bytes);
        let bwd_dkv_spv        = as_u32_slice(bwd_dkv_bytes);
        let bwd_dk_phi_spv     = as_u32_slice(bwd_dk_phi_bytes);
        let bwd_dv_spv         = as_u32_slice(bwd_dv_bytes);
        let bwd_dqk_raw_spv    = as_u32_slice(bwd_dqk_raw_bytes);
        let bwd_grad_qkv_weights_spv = as_u32_slice(bwd_grad_qkv_weights_bytes);
        let bwd_grad_qkv_bias_spv    = as_u32_slice(bwd_grad_qkv_bias_bytes);
        let bwd_self_bias_spv  = as_u32_slice(bwd_self_bias_bytes);
        let bwd_dx_spv         = as_u32_slice(bwd_dx_bytes);

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

        // ===== Forward =====
        let phi = build(device.clone(), phi_spv, 2, 4, "LinearAttention phi");
        let linear_per_token = build(
            device.clone(), per_token_spv, 4, 16, "LinearAttention per_token",
        );
        let compute_kvz = build(device.clone(), kvz_spv, 4, 12, "LinearAttention compute_kvz");
        let denom = build(device.clone(), denom_spv, 3, 16, "LinearAttention denom");
        let forward = build(device.clone(), fwd_spv, 5, 16, "LinearAttention forward");

        // ===== Backward =====
        let bwd_dattn = build(
            device.clone(), bwd_dattn_spv, 3, 16, "LinearAttention bwd_dattn",
        );
        let bwd_grad_wo = build(
            device.clone(), bwd_grad_wo_spv, 3, 16, "LinearAttention bwd_grad_wo",
        );
        let bwd_grad_bo = build(
            device.clone(), bwd_grad_bo_spv, 2, 12, "LinearAttention bwd_grad_bo",
        );
        let bwd_dnum_denom = build(
            device.clone(), bwd_dnum_denom_spv, 5, 12, "LinearAttention bwd_dnum_denom",
        );
        let bwd_dq_phi = build(
            device.clone(), bwd_dq_phi_spv, 5, 12, "LinearAttention bwd_dq_phi",
        );
        let bwd_dz = build(
            device.clone(), bwd_dz_spv, 3, 12, "LinearAttention bwd_dz",
        );
        let bwd_dkv = build(
            device.clone(), bwd_dkv_spv, 3, 12, "LinearAttention bwd_dkv",
        );
        let bwd_dk_phi = build(
            device.clone(), bwd_dk_phi_spv, 4, 12, "LinearAttention bwd_dk_phi",
        );
        let bwd_dv = build(
            device.clone(), bwd_dv_spv, 4, 16, "LinearAttention bwd_dv",
        );
        let bwd_dqk_raw = build(
            device.clone(), bwd_dqk_raw_spv, 6, 12, "LinearAttention bwd_dqk_raw",
        );
        let bwd_grad_qkv_weights = build(
            device.clone(), bwd_grad_qkv_weights_spv, 7, 16,
            "LinearAttention bwd_grad_qkv_weights",
        );
        let bwd_grad_qkv_bias = build(
            device.clone(), bwd_grad_qkv_bias_spv, 6, 12,
            "LinearAttention bwd_grad_qkv_bias",
        );
        let bwd_self_bias = build(
            device.clone(), bwd_self_bias_spv, 4, 12, "LinearAttention bwd_self_bias",
        );
        let bwd_dx = build(
            device.clone(), bwd_dx_spv, 7, 16, "LinearAttention bwd_dx",
        );

        Self {
            phi,
            linear_per_token,
            compute_kvz,
            denom,
            forward,
            bwd_dattn,
            bwd_grad_wo,
            bwd_grad_bo,
            bwd_dnum_denom,
            bwd_dq_phi,
            bwd_dz,
            bwd_dkv,
            bwd_dk_phi,
            bwd_dv,
            bwd_dqk_raw,
            bwd_grad_qkv_weights,
            bwd_grad_qkv_bias,
            bwd_self_bias,
            bwd_dx,
        }
    }
}



