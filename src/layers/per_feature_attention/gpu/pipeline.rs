// src/layers/per_feature_attention/gpu/pipeline.rs
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

pub struct PerFeatureAttentionPipelines {
    // Forward
    pub fwd_qkv: Arc<ComputePipeline>,
    pub fwd_kvz: Arc<ComputePipeline>,
    pub fwd_out: Arc<ComputePipeline>,

    // Backward
    pub bwd_out: Arc<ComputePipeline>,
    pub bwd_grad_kvz: Arc<ComputePipeline>,
    pub bwd_grad_q_phi: Arc<ComputePipeline>,
    pub bwd_grad_v: Arc<ComputePipeline>,
    pub bwd_qk_raw: Arc<ComputePipeline>,
    pub bwd_qkv_params: Arc<ComputePipeline>,
    pub bwd_grad_x: Arc<ComputePipeline>,
}

impl PerFeatureAttentionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_qkv_bytes    = include_bytes!("vulkan/shaders/per_feature_attention_fwd_qkv.spv");
        let fwd_kvz_bytes    = include_bytes!("vulkan/shaders/per_feature_attention_fwd_kvz.spv");
        let fwd_out_bytes    = include_bytes!("vulkan/shaders/per_feature_attention_fwd_out.spv");
        let bwd_out_bytes    = include_bytes!("vulkan/shaders/per_feature_attention_bwd_out.spv");
        let bwd_gkvz_bytes   = include_bytes!("vulkan/shaders/per_feature_attention_bwd_grad_kvz.spv");
        let bwd_gqphi_bytes  = include_bytes!("vulkan/shaders/per_feature_attention_bwd_grad_q_phi.spv");
        let bwd_gv_bytes     = include_bytes!("vulkan/shaders/per_feature_attention_bwd_grad_v.spv");
        let bwd_qkraw_bytes  = include_bytes!("vulkan/shaders/per_feature_attention_bwd_qk_raw.spv");
        let bwd_qkvpar_bytes = include_bytes!("vulkan/shaders/per_feature_attention_bwd_qkv_params.spv");
        let bwd_gx_bytes     = include_bytes!("vulkan/shaders/per_feature_attention_bwd_grad_x.spv");

        let fwd_qkv_spv    = as_u32_slice(fwd_qkv_bytes);
        let fwd_kvz_spv    = as_u32_slice(fwd_kvz_bytes);
        let fwd_out_spv    = as_u32_slice(fwd_out_bytes);
        let bwd_out_spv    = as_u32_slice(bwd_out_bytes);
        let bwd_gkvz_spv   = as_u32_slice(bwd_gkvz_bytes);
        let bwd_gqphi_spv  = as_u32_slice(bwd_gqphi_bytes);
        let bwd_gv_spv     = as_u32_slice(bwd_gv_bytes);
        let bwd_qkraw_spv  = as_u32_slice(bwd_qkraw_bytes);
        let bwd_qkvpar_spv = as_u32_slice(bwd_qkvpar_bytes);
        let bwd_gx_spv     = as_u32_slice(bwd_gx_bytes);

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
            .expect("Failed to create descriptor set layout for PerFeatureAttention")
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

        const PUSH: u32 = 16; // [batch, seq_len, d_model, d_head]

        let fwd_qkv  = build(device.clone(), fwd_qkv_spv,    7, PUSH, "PFA fwd_qkv");
        let fwd_kvz  = build(device.clone(), fwd_kvz_spv,    4, PUSH, "PFA fwd_kvz");
        let fwd_out  = build(device.clone(), fwd_out_spv,    8, PUSH, "PFA fwd_out");
        let bwd_out  = build(device.clone(), bwd_out_spv,    9, PUSH, "PFA bwd_out");
        let bwd_grad_kvz = build(device.clone(), bwd_gkvz_spv, 5, PUSH, "PFA bwd_grad_kvz");
        let bwd_grad_q_phi = build(device.clone(), bwd_gqphi_spv, 5, PUSH, "PFA bwd_grad_q_phi");
        let bwd_grad_v = build(device.clone(), bwd_gv_spv,   5, PUSH, "PFA bwd_grad_v");
        let bwd_qk_raw = build(device.clone(), bwd_qkraw_spv, 8, PUSH, "PFA bwd_qk_raw");
        let bwd_qkv_params = build(device.clone(), bwd_qkvpar_spv, 5, PUSH, "PFA bwd_qkv_params");
        let bwd_grad_x = build(device.clone(), bwd_gx_spv,   5, PUSH, "PFA bwd_grad_x");

        Self {
            fwd_qkv,
            fwd_kvz,
            fwd_out,
            bwd_out,
            bwd_grad_kvz,
            bwd_grad_q_phi,
            bwd_grad_v,
            bwd_qk_raw,
            bwd_qkv_params,
            bwd_grad_x,
        }
    }
}

