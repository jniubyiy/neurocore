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

    // Общие пайплайны
    pub mat_mul: Arc<ComputePipeline>,
    pub reduce: Arc<ComputePipeline>,
    pub unsqueeze: Arc<ComputePipeline>,
}

fn create_ds_layout_n(device: Arc<Device>, n: u32) -> Arc<DescriptorSetLayout> {
    let mut bindings = BTreeMap::new();
    for binding in 0..n {
        bindings.insert(
            binding,
            DescriptorSetLayoutBinding {
                binding_flags: DescriptorBindingFlags::empty(),
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

impl PipelineCache {
    pub fn new(device: Arc<Device>) -> Self {
        // ==================== Загрузка SPIR‑V ====================
        let mat_mul_spv   = include_spv!("shaders/common/mat_mul.spv");
        let reduce_spv    = include_spv!("shaders/common/reduce.spv");
        let unsqueeze_spv = include_spv!("shaders/common/unsqueeze.spv");

        // ==================== Шейдерные модули ====================
        let mat_mul_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(mat_mul_spv)).expect("mat_mul") };
        let reduce_mod  = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(reduce_spv)).expect("reduce") };
        let unsqueeze_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(unsqueeze_spv)).expect("unsqueeze") };

        // ==================== Layout'ы ====================
        let mat_mul_ds   = create_ds_layout_n(device.clone(), 3);
        let reduce_ds    = create_ds_layout_n(device.clone(), 2);
        let unsqueeze_ds = create_ds_layout_n(device.clone(), 2);

        // ==================== Push-константы ====================
        let push_mat_mul = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 12 }; // M, N, K
        let push_reduce  = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 4 };  // rows

        // ==================== Сборка пайплайнов ====================
        let mat_mul   = build_pipeline(device.clone(), mat_mul_mod, mat_mul_ds, Some(push_mat_mul));
        let reduce    = build_pipeline(device.clone(), reduce_mod, reduce_ds, Some(push_reduce));
        let unsqueeze = build_pipeline(device.clone(), unsqueeze_mod, unsqueeze_ds, None);

        Self {
            device,
            mat_mul,
            reduce,
            unsqueeze,
        }
    }

    pub fn mat_mul_pipeline(&self) -> Arc<ComputePipeline> {
        self.mat_mul.clone()
    }

    pub fn reduce_pipeline(&self) -> Arc<ComputePipeline> {
        self.reduce.clone()
    }

    pub fn unsqueeze_pipeline(&self) -> Arc<ComputePipeline> {
        self.unsqueeze.clone()
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
}
 