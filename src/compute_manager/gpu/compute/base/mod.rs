// src/compute_manager/gpu/compute/base/mod.rs
//
//! Определение структуры [`GpuCompute`] и её полей.
//!
//! Реализация `GpuCompute` разнесена по подмодулям:
//! * [`constructor`]   — конструктор `new`.
//! * [`pipelines`]     — ленивый доступ к пайплайнам слоёв, оптимизаторов и функций потерь.
//! * [`temp_buffers`]  — временные / staging-буферы и синхронное копирование между ними.
//! * [`dispatch`]      — запуск compute-шейдеров и низкоуровневая синхронизация очереди.
//! * [`handle_ops`]    — операции с управляемыми `MatrixBufferHandle` (аллокация,
//!                       загрузка/выгрузка данных, копирование регионов).
//!
//! Поля, используемые не только в `mod.rs`, помечены видимостью `pub(super)`:
//! они доступны в `base` и всех его потомках, но не за пределами `base`.

mod constructor;
mod dispatch;
mod handle_ops;
mod pipelines;
mod temp_buffers;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::{
    executor::RawBufferId,
    MemoryExecutor,
};

use super::super::init::GpuContext;
use super::super::pipeline::PipelineCache;

// Импорты пайплайнов слоёв
use crate::layers::relu::gpu::pipeline::ReLUPipelines;
use crate::layers::sigmoid::gpu::pipeline::SigmoidPipelines;
use crate::layers::tanh::gpu::pipeline::TanhPipelines;
use crate::layers::leaky_relu::gpu::pipeline::LeakyReLUPipelines;
use crate::layers::linear::gpu::pipeline::LinearPipelines;
use crate::layers::soft_sparse_gate::gpu::pipeline::SoftSparseGatePipelines;
use crate::layers::soft_keep_gate::gpu::pipeline::SoftKeepGatePipelines;
use crate::layers::dual_anchor::gpu::pipeline::DualAnchorPipelines;
use crate::layers::softmax::gpu::pipeline::SoftmaxPipelines;
use crate::layers::memory::gpu::pipeline::MemoryPipelines;
use crate::layers::splitter::gpu::pipeline::SplitterPipelines;
use crate::layers::combiner::gpu::pipeline::CombinerPipelines;
use crate::layers::adaptive_activation::gpu::pipeline::AdaptivePerFeatureActivationPipelines;

// Новые слои
use crate::layers::dual_slope_relu::gpu::pipeline::DualSlopeReLUPipelines;
use crate::layers::learnable_mish::gpu::pipeline::LearnableMishPipelines;
use crate::layers::learnable_softplus::gpu::pipeline::LearnableSoftplusPipelines;
use crate::layers::rms_norm_learnable_eps::gpu::pipeline::RMSNormWithLearnableEpsilonPipelines;
use crate::layers::adaptive_dropout::gpu::pipeline::AdaptiveDropoutPipelines;
use crate::layers::feature_fusion::gpu::pipeline::FeatureFusionPipelines;
use crate::layers::sparse_feature_selection_gate::gpu::pipeline::SparseFeatureSelectionGatePipelines;
use crate::layers::multi_resolution_kan_linear::gpu::pipeline::MultiResolutionKANLinearPipelines;
use crate::layers::adaptive_normalization::gpu::pipeline::AdaptiveNormalizationPipelines;
use crate::layers::batch_renorm::gpu::pipeline::BatchRenormPipelines;
use crate::layers::concrete_dropout::gpu::pipeline::ConcreteDropoutPipelines;
use crate::layers::mamba::gpu::pipeline::MambaPipelines;
use crate::layers::ind_rnn::gpu::pipeline::IndRNNPipelines;
use crate::layers::spectral_norm_linear::gpu::pipeline::SpectrallyNormalizedLinearPipelines;

// Импорты LinearAttention и RelativePositionAttention
use crate::layers::linear_attention::gpu::pipeline::LinearAttentionPipelines;
use crate::layers::relative_position_attention::gpu::pipeline::RelativePositionAttentionPipelines;

// Импорт PerFeatureAttention
use crate::layers::per_feature_attention::gpu::pipeline::PerFeatureAttentionPipelines;

// Пайплайны оптимизаторов
use crate::optimizers::scale_gradient::gpu::pipeline::ScaleGradientPipelines;
use crate::optimizers::add_weight_decay::gpu::pipeline::AddWeightDecayPipelines;
use crate::optimizers::gradient_clip::gpu::pipeline::GradientClipPipelines;
use crate::optimizers::momentum::gpu::pipeline::MomentumPipelines;
use crate::optimizers::nesterov_momentum::gpu::pipeline::NesterovMomentumPipelines;
use crate::optimizers::adam::gpu::pipeline::AdamPipelines;
use crate::optimizers::apply_update::gpu::pipeline::ApplyUpdatePipelines;

// Пайплайны функций потерь
use crate::losses::sub::gpu::pipeline::SubPipelines;
use crate::losses::square::gpu::pipeline::SquarePipelines;
use crate::losses::abs::gpu::pipeline::AbsPipelines;
use crate::losses::log1p::gpu::pipeline::Log1pPipelines;
use crate::losses::abs_diff::gpu::pipeline::AbsDiffPipelines;
use crate::losses::log::gpu::pipeline::LogPipelines;
use crate::losses::neg::gpu::pipeline::NegPipelines;
use crate::losses::mul::gpu::pipeline::MulPipelines;
use crate::losses::add_scalar::gpu::pipeline::AddScalarPipelines;
use crate::losses::cross_entropy::gpu::pipeline::CrossEntropyPipelines;
use crate::losses::sum_columns::gpu::pipeline::SumColumnsPipelines;

/// Основной держатель GPU-ресурсов и лениво инициализируемых пайплайнов.
///
/// Один экземпляр `GpuCompute` соответствует одному GPU-устройству.
/// Все пайплайны инициализируются лениво через `OnceLock` — в момент
/// первого обращения через методы вида `*_pipelines()`.
pub struct GpuCompute {
    pub context: Arc<GpuContext>,
    pub pipeline_cache: Arc<PipelineCache>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_executor: Arc<RwLock<MemoryExecutor>>,
    pub gpu_device_id: DeviceId,
    /// Хранилище состояний для каждого слоя Memory по индексу (memory_idx).
    pub memory_states: Mutex<HashMap<usize, (Subbuffer<[f32]>, RawBufferId)>>,

    pub(super) queue_lock: Mutex<()>,

    // Пайплайны слоёв (ленивая инициализация)
    pub(super) relu_pipelines: OnceLock<ReLUPipelines>,
    pub(super) sigmoid_pipelines: OnceLock<SigmoidPipelines>,
    pub(super) tanh_pipelines: OnceLock<TanhPipelines>,
    pub(super) leaky_relu_pipelines: OnceLock<LeakyReLUPipelines>,
    pub(super) linear_pipelines: OnceLock<LinearPipelines>,
    pub(super) soft_sparse_gate_pipelines: OnceLock<SoftSparseGatePipelines>,
    pub(super) soft_keep_gate_pipelines: OnceLock<SoftKeepGatePipelines>,
    pub(super) dual_anchor_pipelines: OnceLock<DualAnchorPipelines>,
    pub(super) adaptive_activation_pipelines: OnceLock<AdaptivePerFeatureActivationPipelines>,
    pub(super) softmax_pipelines: OnceLock<SoftmaxPipelines>,
    pub(super) memory_pipelines: OnceLock<MemoryPipelines>,
    pub(super) splitter_pipelines: OnceLock<SplitterPipelines>,
    pub(super) combiner_pipelines: OnceLock<CombinerPipelines>,

    // Новые слои
    pub(super) dual_slope_relu_pipelines: OnceLock<DualSlopeReLUPipelines>,
    pub(super) learnable_mish_pipelines: OnceLock<LearnableMishPipelines>,
    pub(super) learnable_softplus_pipelines: OnceLock<LearnableSoftplusPipelines>,
    pub(super) rms_norm_learnable_eps_pipelines: OnceLock<RMSNormWithLearnableEpsilonPipelines>,
    pub(super) adaptive_dropout_pipelines: OnceLock<AdaptiveDropoutPipelines>,
    pub(super) feature_fusion_pipelines: OnceLock<FeatureFusionPipelines>,
    pub(super) sparse_feature_selection_gate_pipelines: OnceLock<SparseFeatureSelectionGatePipelines>,
    pub(super) multi_resolution_kan_linear_pipelines: OnceLock<MultiResolutionKANLinearPipelines>,
    pub(super) adaptive_normalization_pipelines: OnceLock<AdaptiveNormalizationPipelines>,
    pub(super) batch_renorm_pipelines: OnceLock<BatchRenormPipelines>,
    pub(super) concrete_dropout_pipelines: OnceLock<ConcreteDropoutPipelines>,
    pub(super) mamba_pipelines: OnceLock<MambaPipelines>,
    pub(super) ind_rnn_pipelines: OnceLock<IndRNNPipelines>,
    pub(super) spectral_norm_linear_pipelines: OnceLock<SpectrallyNormalizedLinearPipelines>,

    // Пайплайны для внимания
    pub(super) linear_attention_pipelines: OnceLock<LinearAttentionPipelines>,
    pub(super) relative_position_attention_pipelines: OnceLock<RelativePositionAttentionPipelines>,

    // Пайплайны PerFeatureAttention
    pub(super) per_feature_attention_pipelines: OnceLock<PerFeatureAttentionPipelines>,

    // Пайплайны оптимизаторов
    pub(super) scale_gradient_pipelines: OnceLock<ScaleGradientPipelines>,
    pub(super) add_weight_decay_pipelines: OnceLock<AddWeightDecayPipelines>,
    pub(super) gradient_clip_pipelines: OnceLock<GradientClipPipelines>,
    pub(super) momentum_pipelines: OnceLock<MomentumPipelines>,
    pub(super) nesterov_momentum_pipelines: OnceLock<NesterovMomentumPipelines>,
    pub(super) adam_pipelines: OnceLock<AdamPipelines>,
    pub(super) apply_update_pipelines: OnceLock<ApplyUpdatePipelines>,

    // Пайплайны функций потерь
    pub(super) sub_pipelines: OnceLock<SubPipelines>,
    pub(super) square_pipelines: OnceLock<SquarePipelines>,
    pub(super) abs_pipelines: OnceLock<AbsPipelines>,
    pub(super) log1p_pipelines: OnceLock<Log1pPipelines>,
    pub(super) abs_diff_pipelines: OnceLock<AbsDiffPipelines>,
    pub(super) log_pipelines: OnceLock<LogPipelines>,
    pub(super) neg_pipelines: OnceLock<NegPipelines>,
    pub(super) mul_pipelines: OnceLock<MulPipelines>,
    pub(super) add_scalar_pipelines: OnceLock<AddScalarPipelines>,
    pub(super) cross_entropy_pipelines: OnceLock<CrossEntropyPipelines>,
    pub(super) sum_columns_pipelines: OnceLock<SumColumnsPipelines>,
}