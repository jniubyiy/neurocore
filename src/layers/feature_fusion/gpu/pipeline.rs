// src/layers/feature_fusion/gpu/pipeline.rs
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

/// Пайплайны FeatureFusion.
///
/// Конвейер прямого прохода:
///   forward:  softmax  →  output
///
/// Конвейер обратного прохода:
///   backward: softmax  →  grad_input
///                      →  grad_params (логиты + накопление dot_ldz)
///                      →  grad_T (финализация градиента по температуре)
///
/// `softmax` переиспользуется в forward и backward — softmax логитов
/// одинаков в обе стороны.
pub struct FeatureFusionPipelines {
    /// Softmax логитов с обучаемой температурой (общий для fwd и bwd).
    pub softmax: Arc<ComputePipeline>,
    /// Прямой проход: y = W_softmax · x  (без bias).
    pub output: Arc<ComputePipeline>,
    /// Обратный проход: gi = go · W_softmax.
    pub grad_input: Arc<ComputePipeline>,
    /// Обратный проход: grad_logits + накопление dot_ldz.
    pub grad_params: Arc<ComputePipeline>,
    /// Обратный проход: финализация grad_T_raw.
    pub grad_T: Arc<ComputePipeline>,
}

impl FeatureFusionPipelines {
    pub fn new(device: Arc<Device>) -> Self {
        let softmax_bytes      = include_bytes!("vulkan/shaders/feature_fusion_softmax.spv");
        let output_bytes       = include_bytes!("vulkan/shaders/feature_fusion_output.spv");
        let grad_in_bytes      = include_bytes!("vulkan/shaders/feature_fusion_grad_input.spv");
        let grad_params_bytes  = include_bytes!("vulkan/shaders/feature_fusion_grad_params.spv");
        let grad_T_bytes       = include_bytes!("vulkan/shaders/feature_fusion_grad_T.spv");

        let softmax_spv     = as_u32_slice(softmax_bytes);
        let output_spv      = as_u32_slice(output_bytes);
        let grad_in_spv     = as_u32_slice(grad_in_bytes);
        let grad_params_spv = as_u32_slice(grad_params_bytes);
        let grad_T_spv      = as_u32_slice(grad_T_bytes);

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
            .expect("Failed to create descriptor set layout for FeatureFusion")
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

        // softmax: 2 буфера (params, weights), push = [out, in] (8 байт).
        let softmax = build(device.clone(), softmax_spv, 2, 8, "FeatureFusion softmax");

        // output: 3 буфера (x, weights, y), push = [batch, in, out] (12 байт).
        let output = build(device.clone(), output_spv, 3, 12, "FeatureFusion output");

        // grad_input: 3 буфера (go, weights, gi), push = [batch, in, out] (12 байт).
        let grad_input = build(device.clone(), grad_in_spv, 3, 12, "FeatureFusion grad_input");

        // grad_params: 6 буферов (x, go, params, weights, grad_params, dot_ldz),
        //              push = [batch, in, out] (12 байт).
        let grad_params = build(device.clone(), grad_params_spv, 6, 12, "FeatureFusion grad_params");

        // grad_T: 3 буфера (params, dot_ldz, grad_params), push = [out, in] (8 байт).
        let grad_T = build(device.clone(), grad_T_spv, 3, 8, "FeatureFusion grad_T");

        Self {
            softmax,
            output,
            grad_input,
            grad_params,
            grad_T,
        }
    }
}

