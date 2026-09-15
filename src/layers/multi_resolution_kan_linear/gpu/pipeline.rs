// src/layers/multi_resolution_kan_linear/gpu/pipeline.rs
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

/// Пайплайны MultiResolutionKANLinear (v2).
///
/// Forward:
///   fwd_edge    → edge_out[batch · out · in]   (3 буфера, push=12)
///   fwd_reduce  → y[batch · out]               (3 буфера, push=12)
///
/// Backward:
///   bwd_edge    → grad_params (атомарно)        (4 буфера, push=12)
///   bwd_gi      → grad_input                    (4 буфера, push=12)
pub struct MultiResolutionKANLinearPipelines {
    pub fwd_edge: Arc<ComputePipeline>,
    pub fwd_reduce: Arc<ComputePipeline>,
    pub bwd_edge: Arc<ComputePipeline>,
    pub bwd_gi: Arc<ComputePipeline>,
}

impl MultiResolutionKANLinearPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let fwd_edge_bytes = include_bytes!(
            "vulkan/shaders/multi_resolution_kan_linear_fwd_edge.spv"
        );
        let fwd_reduce_bytes = include_bytes!(
            "vulkan/shaders/multi_resolution_kan_linear_fwd_reduce.spv"
        );
        let bwd_edge_bytes = include_bytes!(
            "vulkan/shaders/multi_resolution_kan_linear_bwd_edge.spv"
        );
        let bwd_gi_bytes = include_bytes!(
            "vulkan/shaders/multi_resolution_kan_linear_bwd_gi.spv"
        );

        let fwd_edge_spv = as_u32_slice(fwd_edge_bytes);
        let fwd_reduce_spv = as_u32_slice(fwd_reduce_bytes);
        let bwd_edge_spv = as_u32_slice(bwd_edge_bytes);
        let bwd_gi_spv = as_u32_slice(bwd_gi_bytes);

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
            .expect("Failed to create descriptor set layout for KAN")
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

        // Push = [batch, in_features, out_features] — 12 байт.
        const PUSH: u32 = 12;

        let fwd_edge   = build(device.clone(), fwd_edge_spv,   3, PUSH, "KAN fwd_edge");
        let fwd_reduce = build(device.clone(), fwd_reduce_spv, 3, PUSH, "KAN fwd_reduce");
        let bwd_edge   = build(device.clone(), bwd_edge_spv,   4, PUSH, "KAN bwd_edge");
        let bwd_gi     = build(device.clone(), bwd_gi_spv,     4, PUSH, "KAN bwd_gi");

        Self {
            fwd_edge,
            fwd_reduce,
            bwd_edge,
            bwd_gi,
        }
    }
}
