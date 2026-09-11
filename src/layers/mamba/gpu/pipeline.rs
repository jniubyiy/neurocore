// src/layers/mamba/gpu/pipeline.rs
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

/// Пайплайны Mamba, разбитые на этапы.
///
/// Discretize (Rust-цикл из 11 шагов):
///   discretize_init → (discretize_accum → discretize_step) × 10
///                   → discretize_accum  → discretize_scale_b
///
/// Forward (Rust-цикл по t):
///   forward_step × seq_len  (две фазы: h_t, y_t)
///
/// Backward (Rust-цикл по t в обратную сторону):
///   bwd_dh_t → bwd_grad_A_bar (если t > 0) → bwd_grad_B_bar
///            → bwd_grad_C → bwd_grad_D → bwd_grad_input
///
/// Convert (один вызов):
///   convert_grads
pub struct MambaPipelines {
    // Discretize
    pub discretize_init: Arc<ComputePipeline>,
    pub discretize_accum: Arc<ComputePipeline>,
    pub discretize_step: Arc<ComputePipeline>,
    pub discretize_scale_b: Arc<ComputePipeline>,

    // Forward
    pub forward_step: Arc<ComputePipeline>,

    // Backward
    pub bwd_dh_t: Arc<ComputePipeline>,
    pub bwd_grad_A_bar: Arc<ComputePipeline>,
    pub bwd_grad_B_bar: Arc<ComputePipeline>,
    pub bwd_grad_C: Arc<ComputePipeline>,
    pub bwd_grad_D: Arc<ComputePipeline>,
    pub bwd_grad_input: Arc<ComputePipeline>,

    // Convert
    pub convert_grads: Arc<ComputePipeline>,
}

impl MambaPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let disc_init_bytes  = include_bytes!("vulkan/shaders/mamba_discretize_init.spv");
        let disc_accum_bytes = include_bytes!("vulkan/shaders/mamba_discretize_accum.spv");
        let disc_step_bytes  = include_bytes!("vulkan/shaders/mamba_discretize_step.spv");
        let disc_scale_bytes = include_bytes!("vulkan/shaders/mamba_discretize_scale_b.spv");
        let fwd_step_bytes   = include_bytes!("vulkan/shaders/mamba_fwd_step.spv");
        let bwd_dh_t_bytes   = include_bytes!("vulkan/shaders/mamba_bwd_dh_t.spv");
        let bwd_gA_bytes     = include_bytes!("vulkan/shaders/mamba_bwd_grad_A_bar.spv");
        let bwd_gB_bytes     = include_bytes!("vulkan/shaders/mamba_bwd_grad_B_bar.spv");
        let bwd_gC_bytes     = include_bytes!("vulkan/shaders/mamba_bwd_grad_C.spv");
        let bwd_gD_bytes     = include_bytes!("vulkan/shaders/mamba_bwd_grad_D.spv");
        let bwd_gi_bytes     = include_bytes!("vulkan/shaders/mamba_bwd_grad_input.spv");
        let conv_bytes       = include_bytes!("vulkan/shaders/mamba_convert_grads.spv");

        let disc_init_spv  = as_u32_slice(disc_init_bytes);
        let disc_accum_spv = as_u32_slice(disc_accum_bytes);
        let disc_step_spv  = as_u32_slice(disc_step_bytes);
        let disc_scale_spv = as_u32_slice(disc_scale_bytes);
        let fwd_step_spv   = as_u32_slice(fwd_step_bytes);
        let bwd_dh_t_spv   = as_u32_slice(bwd_dh_t_bytes);
        let bwd_gA_spv     = as_u32_slice(bwd_gA_bytes);
        let bwd_gB_spv     = as_u32_slice(bwd_gB_bytes);
        let bwd_gC_spv     = as_u32_slice(bwd_gC_bytes);
        let bwd_gD_spv     = as_u32_slice(bwd_gD_bytes);
        let bwd_gi_spv     = as_u32_slice(bwd_gi_bytes);
        let conv_spv       = as_u32_slice(conv_bytes);

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
            .expect("Failed to create descriptor set layout for Mamba")
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

        // Discretize
        // init:    2 буфера (T, A_bar),          push = [n]         (4 байта)
        // accum:   2 буфера (T, A_bar),          push = [n]         (4 байта)
        // step:    3 буфера (T, A, T_next),      push = [n,k,delta] (12 байт)
        // scale_b: 2 буфера (B, B_bar),          push = [n,d,delta] (12 байт)
        let discretize_init =
            build(device.clone(), disc_init_spv, 2, 4, "Mamba discretize_init");
        let discretize_accum =
            build(device.clone(), disc_accum_spv, 2, 4, "Mamba discretize_accum");
        let discretize_step =
            build(device.clone(), disc_step_spv, 3, 12, "Mamba discretize_step");
        let discretize_scale_b =
            build(device.clone(), disc_scale_spv, 2, 12, "Mamba discretize_scale_b");

        // Forward step: 7 буферов, push = [batch,d,n,seq,t,phase] (24 байта)
        let forward_step = build(device.clone(), fwd_step_spv, 7, 24, "Mamba forward_step");

        // Backward
        // push = [batch,d,n,seq,t] (20 байт)
        let bwd_dh_t =
            build(device.clone(), bwd_dh_t_spv, 5, 20, "Mamba bwd_dh_t");
        let bwd_grad_A_bar =
            build(device.clone(), bwd_gA_spv, 3, 20, "Mamba bwd_grad_A_bar");
        let bwd_grad_B_bar =
            build(device.clone(), bwd_gB_spv, 3, 20, "Mamba bwd_grad_B_bar");
        let bwd_grad_C =
            build(device.clone(), bwd_gC_spv, 3, 20, "Mamba bwd_grad_C");
        let bwd_grad_D =
            build(device.clone(), bwd_gD_spv, 3, 20, "Mamba bwd_grad_D");
        let bwd_grad_input =
            build(device.clone(), bwd_gi_spv, 5, 20, "Mamba bwd_grad_input");

        // Convert: 7 буферов, push = [n,d,delta,pad] (16 байт)
        let convert_grads =
            build(device.clone(), conv_spv, 7, 16, "Mamba convert_grads");

        Self {
            discretize_init,
            discretize_accum,
            discretize_step,
            discretize_scale_b,
            forward_step,
            bwd_dh_t,
            bwd_grad_A_bar,
            bwd_grad_B_bar,
            bwd_grad_C,
            bwd_grad_D,
            bwd_grad_input,
            convert_grads,
        }
    }
}