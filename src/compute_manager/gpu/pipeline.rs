// src/compute_manager/gpu/pipeline.rs

use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::descriptor_set::layout::{
    DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::Device;
use vulkano::pipeline::{
    compute::ComputePipelineCreateInfo,
    layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
    ComputePipeline, PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages, spirv::ExecutionModel};

macro_rules! include_spv {
    ($file:expr) => {{
        const BYTES: &[u8] = include_bytes!($file);
        assert!(BYTES.len() % 4 == 0, "SPIR‑V файл должен быть выровнен по 4 байта");
        let len = BYTES.len() / 4;
        let ptr = BYTES.as_ptr() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }};
}

pub struct PipelineCache {
    device: Arc<Device>,
    mat_mul: Arc<ComputePipeline>,
    activation: Arc<ComputePipeline>,
    reduce: Arc<ComputePipeline>,
    unsqueeze: Arc<ComputePipeline>,
}

impl PipelineCache {
    pub fn new(device: Arc<Device>) -> Self {
        let mat_mul_spv    = include_spv!("shaders/mat_mul.spv");
        let activation_spv = include_spv!("shaders/activation.spv");
        let reduce_spv     = include_spv!("shaders/reduce.spv");
        let unsqueeze_spv  = include_spv!("shaders/unsqueeze.spv");

        let mat_mul_mod = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(mat_mul_spv))
                .expect("mat_mul shader")
        };
        let activation_mod = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(activation_spv))
                .expect("activation shader")
        };
        let reduce_mod = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(reduce_spv))
                .expect("reduce shader")
        };
        let unsqueeze_mod = unsafe {
            ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(unsqueeze_spv))
                .expect("unsqueeze shader")
        };

        fn storage_binding() -> DescriptorSetLayoutBinding {
            DescriptorSetLayoutBinding {
                binding_flags: DescriptorBindingFlags::empty(),
                descriptor_type: DescriptorType::StorageBuffer,
                descriptor_count: 1,
                stages: ShaderStages::COMPUTE,
                immutable_samplers: Vec::new(),
                _ne: unsafe { std::mem::zeroed() },
            }
        }

        let mut mat_mul_bindings = BTreeMap::new();
        mat_mul_bindings.insert(0, storage_binding());
        mat_mul_bindings.insert(1, storage_binding());
        mat_mul_bindings.insert(2, storage_binding());

        let mut activation_bindings = BTreeMap::new();
        activation_bindings.insert(0, storage_binding());
        activation_bindings.insert(1, storage_binding());

        let mat_mul_ds_layout = create_ds_layout(device.clone(), mat_mul_bindings);
        let activation_ds_layout = create_ds_layout(device.clone(), activation_bindings);
        let reduce_ds_layout = activation_ds_layout.clone();
        let unsqueeze_ds_layout = activation_ds_layout.clone();

        let mat_mul_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12,
        };
        let activation_push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 12,
        };

        let mat_mul    = build_pipeline(device.clone(), mat_mul_mod.clone(),    mat_mul_ds_layout, Some(mat_mul_push));
        let activation = build_pipeline(device.clone(), activation_mod.clone(), activation_ds_layout, Some(activation_push));
        let reduce     = build_pipeline(device.clone(), reduce_mod.clone(),     reduce_ds_layout,  None);
        let unsqueeze  = build_pipeline(device.clone(), unsqueeze_mod.clone(),  unsqueeze_ds_layout, None);

        Self {
            device,
            mat_mul,
            activation,
            reduce,
            unsqueeze,
        }
    }

    pub fn mat_mul_pipeline(&self) -> Arc<ComputePipeline> { self.mat_mul.clone() }
    pub fn activation_pipeline(&self) -> Arc<ComputePipeline> { self.activation.clone() }
    pub fn reduce_pipeline(&self) -> Arc<ComputePipeline> { self.reduce.clone() }
    pub fn unsqueeze_pipeline(&self) -> Arc<ComputePipeline> { self.unsqueeze.clone() }
    pub fn device(&self) -> Arc<Device> { self.device.clone() }
}

fn create_ds_layout(
    device: Arc<Device>,
    bindings: BTreeMap<u32, DescriptorSetLayoutBinding>,
) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings,
            ..Default::default()
        },
    )
    .expect("Failed to create descriptor set layout")
}

fn build_pipeline(
    device: Arc<Device>,
    shader: Arc<ShaderModule>,
    ds_layout: Arc<DescriptorSetLayout>,
    push_constants: Option<PushConstantRange>,
) -> Arc<ComputePipeline> {
    let ranges = push_constants.into_iter().collect();
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![ds_layout],
            push_constant_ranges: ranges,
            ..Default::default()
        },
    )
    .expect("Failed to create pipeline layout");

    let entry_point = shader
        .entry_point_with_execution("main", ExecutionModel::GLCompute)
        .expect("Shader entry point not found");

    let stage = PipelineShaderStageCreateInfo::new(entry_point);

    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("Failed to create compute pipeline")
}