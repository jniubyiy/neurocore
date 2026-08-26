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

    // Loss-пайплайны
    pub sub_fwd: Arc<ComputePipeline>,
    pub sub_bwd: Arc<ComputePipeline>,
    pub square_fwd: Arc<ComputePipeline>,
    pub square_bwd: Arc<ComputePipeline>,
    pub abs_fwd: Arc<ComputePipeline>,
    pub abs_bwd: Arc<ComputePipeline>,
    pub log1p_fwd: Arc<ComputePipeline>,
    pub log1p_bwd: Arc<ComputePipeline>,
    pub absdiff_fwd: Arc<ComputePipeline>,
    pub absdiff_bwd: Arc<ComputePipeline>,
    pub log_fwd: Arc<ComputePipeline>,
    pub log_bwd: Arc<ComputePipeline>,
    pub neg_fwd: Arc<ComputePipeline>,
    pub neg_bwd: Arc<ComputePipeline>,
    pub mul_fwd: Arc<ComputePipeline>,
    pub mul_bwd: Arc<ComputePipeline>,
    pub addscalar_fwd: Arc<ComputePipeline>,
    pub addscalar_bwd: Arc<ComputePipeline>,
    pub cross_entropy_fwd: Arc<ComputePipeline>,
    pub cross_entropy_bwd: Arc<ComputePipeline>,
    // Новые пайплайны для SumColumns
    pub sum_columns_fwd: Arc<ComputePipeline>,
    pub sum_columns_bwd: Arc<ComputePipeline>,

    // Optimizer-пайплайны
    pub scale_grad: Arc<ComputePipeline>,
    pub weight_decay: Arc<ComputePipeline>,
    pub grad_clip: Arc<ComputePipeline>,
    pub momentum: Arc<ComputePipeline>,
    pub nesterov_momentum: Arc<ComputePipeline>,
    pub adam: Arc<ComputePipeline>,
    pub apply_update: Arc<ComputePipeline>,
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
        let mat_mul_spv         = include_spv!("shaders/common/mat_mul.spv");
        let reduce_spv          = include_spv!("shaders/common/reduce.spv");
        let unsqueeze_spv       = include_spv!("shaders/common/unsqueeze.spv");

        // Новые шейдеры для SumColumns
        let sum_columns_fwd_spv = include_spv!("../../losses/sum_columns/gpu/shaders/sum_columns_fwd.spv");
        let sum_columns_bwd_spv = include_spv!("../../losses/sum_columns/gpu/shaders/sum_columns_bwd.spv");

        // Loss-шейдеры
        let sub_fwd_spv         = include_spv!("../../losses/sub/gpu/shaders/sub_fwd.spv");
        let sub_bwd_spv         = include_spv!("../../losses/sub/gpu/shaders/sub_bwd.spv");
        let square_fwd_spv      = include_spv!("../../losses/square/gpu/shaders/square_fwd.spv");
        let square_bwd_spv      = include_spv!("../../losses/square/gpu/shaders/square_bwd.spv");
        let abs_fwd_spv         = include_spv!("../../losses/abs/gpu/shaders/abs_fwd.spv");
        let abs_bwd_spv         = include_spv!("../../losses/abs/gpu/shaders/abs_bwd.spv");
        let log1p_fwd_spv       = include_spv!("../../losses/log1p/gpu/shaders/log1p_fwd.spv");
        let log1p_bwd_spv       = include_spv!("../../losses/log1p/gpu/shaders/log1p_bwd.spv");
        let absdiff_fwd_spv     = include_spv!("../../losses/abs_diff/gpu/shaders/absdiff_fwd.spv");
        let absdiff_bwd_spv     = include_spv!("../../losses/abs_diff/gpu/shaders/absdiff_bwd.spv");
        let log_fwd_spv         = include_spv!("../../losses/log/gpu/shaders/log_fwd.spv");
        let log_bwd_spv         = include_spv!("../../losses/log/gpu/shaders/log_bwd.spv");
        let neg_fwd_spv         = include_spv!("../../losses/neg/gpu/shaders/neg_fwd.spv");
        let neg_bwd_spv         = include_spv!("../../losses/neg/gpu/shaders/neg_bwd.spv");
        let mul_fwd_spv         = include_spv!("../../losses/mul/gpu/shaders/mul_fwd.spv");
        let mul_bwd_spv         = include_spv!("../../losses/mul/gpu/shaders/mul_bwd.spv");
        let addscalar_fwd_spv   = include_spv!("../../losses/add_scalar/gpu/shaders/addscalar_fwd.spv");
        let addscalar_bwd_spv   = include_spv!("../../losses/add_scalar/gpu/shaders/addscalar_bwd.spv");
        let cross_entropy_fwd_spv = include_spv!("../../losses/cross_entropy/gpu/shaders/cross_entropy_fwd.spv");
        let cross_entropy_bwd_spv = include_spv!("../../losses/cross_entropy/gpu/shaders/cross_entropy_bwd.spv");

        // Optimizer-шейдеры
        let scale_grad_spv      = include_spv!("shaders/optim/scale_grad.spv");
        let weight_decay_spv    = include_spv!("shaders/optim/weight_decay.spv");
        let grad_clip_spv       = include_spv!("shaders/optim/grad_clip.spv");
        let momentum_spv        = include_spv!("shaders/optim/momentum.spv");
        let nesterov_momentum_spv = include_spv!("shaders/optim/nesterov_momentum.spv");
        let adam_spv            = include_spv!("shaders/optim/adam.spv");
        let apply_update_spv    = include_spv!("shaders/optim/apply_update.spv");

        // ==================== Шейдерные модули ====================
        let mat_mul_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(mat_mul_spv)).expect("mat_mul") };
        let reduce_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(reduce_spv)).expect("reduce") };
        let unsqueeze_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(unsqueeze_spv)).expect("unsqueeze") };

        let sum_columns_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(sum_columns_fwd_spv)).expect("sum_columns_fwd") };
        let sum_columns_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(sum_columns_bwd_spv)).expect("sum_columns_bwd") };

        let sub_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(sub_fwd_spv)).expect("sub_fwd") };
        let sub_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(sub_bwd_spv)).expect("sub_bwd") };
        let square_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(square_fwd_spv)).expect("square_fwd") };
        let square_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(square_bwd_spv)).expect("square_bwd") };
        let abs_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(abs_fwd_spv)).expect("abs_fwd") };
        let abs_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(abs_bwd_spv)).expect("abs_bwd") };
        let log1p_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(log1p_fwd_spv)).expect("log1p_fwd") };
        let log1p_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(log1p_bwd_spv)).expect("log1p_bwd") };
        let absdiff_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(absdiff_fwd_spv)).expect("absdiff_fwd") };
        let absdiff_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(absdiff_bwd_spv)).expect("absdiff_bwd") };
        let log_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(log_fwd_spv)).expect("log_fwd") };
        let log_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(log_bwd_spv)).expect("log_bwd") };
        let neg_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(neg_fwd_spv)).expect("neg_fwd") };
        let neg_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(neg_bwd_spv)).expect("neg_bwd") };
        let mul_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(mul_fwd_spv)).expect("mul_fwd") };
        let mul_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(mul_bwd_spv)).expect("mul_bwd") };
        let addscalar_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(addscalar_fwd_spv)).expect("addscalar_fwd") };
        let addscalar_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(addscalar_bwd_spv)).expect("addscalar_bwd") };
        let ce_fwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(cross_entropy_fwd_spv)).expect("cross_entropy_fwd") };
        let ce_bwd_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(cross_entropy_bwd_spv)).expect("cross_entropy_bwd") };

        let scale_grad_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(scale_grad_spv)).expect("scale_grad") };
        let weight_decay_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(weight_decay_spv)).expect("weight_decay") };
        let grad_clip_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(grad_clip_spv)).expect("grad_clip") };
        let momentum_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(momentum_spv)).expect("momentum") };
        let nesterov_momentum_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(nesterov_momentum_spv)).expect("nesterov_momentum") };
        let adam_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(adam_spv)).expect("adam") };
        let apply_update_mod = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(apply_update_spv)).expect("apply_update") };

        // ==================== Layout'ы ====================
        let mat_mul_ds = create_ds_layout_n(device.clone(), 3);
        let reduce_ds = create_ds_layout_n(device.clone(), 2);
        let unsqueeze_ds = create_ds_layout_n(device.clone(), 2);

        let sum_columns_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let sum_columns_bwd_ds = create_ds_layout_n(device.clone(), 2);

        let sub_fwd_ds = create_ds_layout_n(device.clone(), 3);
        let sub_bwd_ds = create_ds_layout_n(device.clone(), 3);
        let square_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let square_bwd_ds = create_ds_layout_n(device.clone(), 3);
        let abs_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let abs_bwd_ds = create_ds_layout_n(device.clone(), 3);
        let log1p_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let log1p_bwd_ds = create_ds_layout_n(device.clone(), 3);
        let absdiff_fwd_ds = create_ds_layout_n(device.clone(), 3);
        let absdiff_bwd_ds = create_ds_layout_n(device.clone(), 5);
        let log_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let log_bwd_ds = create_ds_layout_n(device.clone(), 3);
        let neg_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let neg_bwd_ds = create_ds_layout_n(device.clone(), 2);
        let mul_fwd_ds = create_ds_layout_n(device.clone(), 3);
        let mul_bwd_ds = create_ds_layout_n(device.clone(), 5);
        let addscalar_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let addscalar_bwd_ds = create_ds_layout_n(device.clone(), 2);
        let ce_fwd_ds = create_ds_layout_n(device.clone(), 2);
        let ce_bwd_ds = create_ds_layout_n(device.clone(), 3);

        let scale_grad_ds = create_ds_layout_n(device.clone(), 1);
        let weight_decay_ds = create_ds_layout_n(device.clone(), 2);
        let grad_clip_ds = create_ds_layout_n(device.clone(), 1);
        let momentum_ds = create_ds_layout_n(device.clone(), 2);
        let adam_ds = create_ds_layout_n(device.clone(), 2);
        let apply_update_ds = create_ds_layout_n(device.clone(), 2);

        // ==================== Push-константы ====================
        let push_mat_mul = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 12 };
        let push_reduce = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 4 };
        let push_sum_columns = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 8 }; // rows, cols
        let push_total = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 4 };
        let push_total_scalar = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 8 };
        let push_ce = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 8 };
        let push_factor_total = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 8 };
        let push_decay_total = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 8 };
        let push_clip = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 12 };
        let push_beta = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 8 };
        let push_adam = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 24 };
        let push_optim_total = PushConstantRange { stages: ShaderStages::COMPUTE, offset: 0, size: 4 };

        // ==================== Сборка пайплайнов ====================
        let mat_mul = build_pipeline(device.clone(), mat_mul_mod, mat_mul_ds, Some(push_mat_mul));
        let reduce = build_pipeline(device.clone(), reduce_mod, reduce_ds, Some(push_reduce));
        let unsqueeze = build_pipeline(device.clone(), unsqueeze_mod, unsqueeze_ds, None);

        let sum_columns_fwd = build_pipeline(device.clone(), sum_columns_fwd_mod, sum_columns_fwd_ds, Some(push_sum_columns));
        let sum_columns_bwd = build_pipeline(device.clone(), sum_columns_bwd_mod, sum_columns_bwd_ds, Some(push_sum_columns));

        let sub_fwd = build_pipeline(device.clone(), sub_fwd_mod, sub_fwd_ds, Some(push_total));
        let sub_bwd = build_pipeline(device.clone(), sub_bwd_mod, sub_bwd_ds, Some(push_total));
        let square_fwd = build_pipeline(device.clone(), square_fwd_mod, square_fwd_ds, Some(push_total));
        let square_bwd = build_pipeline(device.clone(), square_bwd_mod, square_bwd_ds, Some(push_total));
        let abs_fwd = build_pipeline(device.clone(), abs_fwd_mod, abs_fwd_ds, Some(push_total));
        let abs_bwd = build_pipeline(device.clone(), abs_bwd_mod, abs_bwd_ds, Some(push_total));
        let log1p_fwd = build_pipeline(device.clone(), log1p_fwd_mod, log1p_fwd_ds, Some(push_total));
        let log1p_bwd = build_pipeline(device.clone(), log1p_bwd_mod, log1p_bwd_ds, Some(push_total));
        let absdiff_fwd = build_pipeline(device.clone(), absdiff_fwd_mod, absdiff_fwd_ds, Some(push_total));
        let absdiff_bwd = build_pipeline(device.clone(), absdiff_bwd_mod, absdiff_bwd_ds, Some(push_total));
        let log_fwd = build_pipeline(device.clone(), log_fwd_mod, log_fwd_ds, Some(push_total));
        let log_bwd = build_pipeline(device.clone(), log_bwd_mod, log_bwd_ds, Some(push_total));
        let neg_fwd = build_pipeline(device.clone(), neg_fwd_mod, neg_fwd_ds, Some(push_total));
        let neg_bwd = build_pipeline(device.clone(), neg_bwd_mod, neg_bwd_ds, Some(push_total));
        let mul_fwd = build_pipeline(device.clone(), mul_fwd_mod, mul_fwd_ds, Some(push_total));
        let mul_bwd = build_pipeline(device.clone(), mul_bwd_mod, mul_bwd_ds, Some(push_total));
        let addscalar_fwd = build_pipeline(device.clone(), addscalar_fwd_mod, addscalar_fwd_ds, Some(push_total_scalar));
        let addscalar_bwd = build_pipeline(device.clone(), addscalar_bwd_mod, addscalar_bwd_ds, Some(push_total));
        let cross_entropy_fwd = build_pipeline(device.clone(), ce_fwd_mod, ce_fwd_ds, Some(push_ce));
        let cross_entropy_bwd = build_pipeline(device.clone(), ce_bwd_mod, ce_bwd_ds, Some(push_ce));

        let scale_grad = build_pipeline(device.clone(), scale_grad_mod, scale_grad_ds, Some(push_factor_total));
        let weight_decay = build_pipeline(device.clone(), weight_decay_mod, weight_decay_ds, Some(push_decay_total));
        let grad_clip = build_pipeline(device.clone(), grad_clip_mod, grad_clip_ds, Some(push_clip));
        let momentum = build_pipeline(device.clone(), momentum_mod, momentum_ds.clone(), Some(push_beta));
        let nesterov_momentum = build_pipeline(device.clone(), nesterov_momentum_mod, momentum_ds, Some(push_beta));
        let adam = build_pipeline(device.clone(), adam_mod, adam_ds, Some(push_adam));
        let apply_update = build_pipeline(device.clone(), apply_update_mod, apply_update_ds, Some(push_optim_total));

        Self {
            device,
            mat_mul,
            reduce,
            unsqueeze,
            sub_fwd,
            sub_bwd,
            square_fwd,
            square_bwd,
            abs_fwd,
            abs_bwd,
            log1p_fwd,
            log1p_bwd,
            absdiff_fwd,
            absdiff_bwd,
            log_fwd,
            log_bwd,
            neg_fwd,
            neg_bwd,
            mul_fwd,
            mul_bwd,
            addscalar_fwd,
            addscalar_bwd,
            cross_entropy_fwd,
            cross_entropy_bwd,
            sum_columns_fwd,
            sum_columns_bwd,
            scale_grad,
            weight_decay,
            grad_clip,
            momentum,
            nesterov_momentum,
            adam,
            apply_update,
        }
    }

    pub fn mat_mul_pipeline(&self) -> Arc<ComputePipeline> { self.mat_mul.clone() }
    pub fn reduce_pipeline(&self) -> Arc<ComputePipeline> { self.reduce.clone() }
    pub fn unsqueeze_pipeline(&self) -> Arc<ComputePipeline> { self.unsqueeze.clone() }
    pub fn sum_columns_fwd_pipeline(&self) -> Arc<ComputePipeline> { self.sum_columns_fwd.clone() }
    pub fn sum_columns_bwd_pipeline(&self) -> Arc<ComputePipeline> { self.sum_columns_bwd.clone() }
    pub fn device(&self) -> Arc<Device> { self.device.clone() }
}
 