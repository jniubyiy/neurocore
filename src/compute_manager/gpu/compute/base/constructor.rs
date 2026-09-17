// src/compute_manager/gpu/compute/base/constructor.rs
//
//! Конструктор [`GpuCompute`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::MemoryExecutor;

use super::super::super::init::GpuContext;
use super::super::super::pipeline::PipelineCache;
use super::GpuCompute;

impl GpuCompute {
    pub fn new(
        context: Arc<GpuContext>,
        pipeline_cache: Arc<PipelineCache>,
        memory_executor: Arc<RwLock<MemoryExecutor>>,
        gpu_device_id: DeviceId,
    ) -> Self {
        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(context.device.clone(), Default::default()),
        );
        let command_buffer_allocator = Arc::new(
            StandardCommandBufferAllocator::new(context.device.clone(), Default::default()),
        );
        Self {
            context,
            pipeline_cache,
            descriptor_set_allocator,
            command_buffer_allocator,
            memory_executor,
            gpu_device_id,
            memory_states: Mutex::new(HashMap::new()),
            queue_lock: Mutex::new(()),

            relu_pipelines: OnceLock::new(),
            sigmoid_pipelines: OnceLock::new(),
            tanh_pipelines: OnceLock::new(),
            leaky_relu_pipelines: OnceLock::new(),
            linear_pipelines: OnceLock::new(),
            soft_sparse_gate_pipelines: OnceLock::new(),
            soft_keep_gate_pipelines: OnceLock::new(),
            dual_anchor_pipelines: OnceLock::new(),
            adaptive_activation_pipelines: OnceLock::new(),
            softmax_pipelines: OnceLock::new(),
            memory_pipelines: OnceLock::new(),
            splitter_pipelines: OnceLock::new(),
            combiner_pipelines: OnceLock::new(),

            dual_slope_relu_pipelines: OnceLock::new(),
            learnable_mish_pipelines: OnceLock::new(),
            learnable_softplus_pipelines: OnceLock::new(),
            rms_norm_learnable_eps_pipelines: OnceLock::new(),
            adaptive_dropout_pipelines: OnceLock::new(),
            feature_fusion_pipelines: OnceLock::new(),
            sparse_feature_selection_gate_pipelines: OnceLock::new(),
            multi_resolution_kan_linear_pipelines: OnceLock::new(),
            adaptive_normalization_pipelines: OnceLock::new(),
            batch_renorm_pipelines: OnceLock::new(),
            concrete_dropout_pipelines: OnceLock::new(),
            mamba_pipelines: OnceLock::new(),
            ind_rnn_pipelines: OnceLock::new(),
            spectral_norm_linear_pipelines: OnceLock::new(),

            linear_attention_pipelines: OnceLock::new(),
            relative_position_attention_pipelines: OnceLock::new(),

            per_feature_attention_pipelines: OnceLock::new(),

            scale_gradient_pipelines: OnceLock::new(),
            add_weight_decay_pipelines: OnceLock::new(),
            gradient_clip_pipelines: OnceLock::new(),
            momentum_pipelines: OnceLock::new(),
            nesterov_momentum_pipelines: OnceLock::new(),
            adam_pipelines: OnceLock::new(),
            apply_update_pipelines: OnceLock::new(),

            sub_pipelines: OnceLock::new(),
            square_pipelines: OnceLock::new(),
            abs_pipelines: OnceLock::new(),
            log1p_pipelines: OnceLock::new(),
            abs_diff_pipelines: OnceLock::new(),
            log_pipelines: OnceLock::new(),
            neg_pipelines: OnceLock::new(),
            mul_pipelines: OnceLock::new(),
            add_scalar_pipelines: OnceLock::new(),
            cross_entropy_pipelines: OnceLock::new(),
            sum_columns_pipelines: OnceLock::new(),
        }
    }
}