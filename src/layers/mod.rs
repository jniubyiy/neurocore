// src/layers/mod.rs

pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;
pub mod memory;
pub mod splitter;
pub mod combiner;
pub mod splitter_connector;
pub mod combiner_connector;
pub mod leaky_relu;
pub mod identity;
pub mod soft_sparse_gate;
pub mod soft_keep_gate;
pub mod dual_anchor;
pub mod adaptive_activation;
pub mod adaptive_normalization;
pub mod batch_renorm;
pub mod concrete_dropout;
pub mod mamba;
pub mod linear_attention;
pub mod relative_position_attention;
pub mod ind_rnn;
pub mod spectral_norm_linear;

// Новые слои (полные внутренние реализации)
pub mod dual_slope_relu;
pub mod learnable_mish;
pub mod learnable_softplus;
pub mod rms_norm_learnable_eps;
pub mod adaptive_dropout;
pub mod feature_fusion;
pub mod sparse_feature_selection_gate;
pub mod multi_resolution_kan_linear;

pub mod layers_special;
pub mod buffered_context;

use crate::model_plan::param_store::ParamSlice;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

// ---------------------------------------------------------------------------
// Маркерный трейт UniversalLayer (для downcasting и общей информации)
// ---------------------------------------------------------------------------

pub trait UniversalLayer: Send + Sync + 'static {
    fn as_linear(&self) -> Option<&Linear> { None }
    fn as_relu(&self) -> Option<&ReLU> { None }
    fn as_sigmoid(&self) -> Option<&Sigmoid> { None }
    fn as_tanh(&self) -> Option<&Tanh> { None }
    fn as_leaky_relu(&self) -> Option<&LeakyReLU> { None }
    fn as_identity(&self) -> Option<&Identity> { None }
    fn as_softmax(&self) -> Option<&Softmax> { None }
    fn as_memory(&self) -> Option<&Memory> { None }
    fn as_soft_sparse_gate(&self) -> Option<&SoftSparseGate> { None }
    fn as_soft_keep_gate(&self) -> Option<&SoftKeepGate> { None }
    fn as_dual_anchor(&self) -> Option<&DualAnchor> { None }
    fn as_adaptive_activation(&self) -> Option<&AdaptivePerFeatureActivation> { None }
    fn as_adaptive_normalization(&self) -> Option<&AdaptiveNormalization> { None }
    fn as_batch_renorm(&self) -> Option<&BatchRenorm1d> { None }
    fn as_concrete_dropout(&self) -> Option<&ConcreteDropout> { None }
    fn as_mamba(&self) -> Option<&Mamba> { None }
    fn as_linear_attention(&self) -> Option<&LinearAttention> { None }
    fn as_relative_position_attention(&self) -> Option<&RelativePositionAttention> { None }
    fn as_ind_rnn(&self) -> Option<&IndRNN> { None }
    fn as_spectral_norm_linear(&self) -> Option<&SpectrallyNormalizedLinear> { None }
    fn as_reduce_mean(&self) -> Option<&ReduceMean> { None }
    fn as_unsqueeze(&self) -> Option<&Unsqueeze> { None }

    // Новые слои
    fn as_dual_slope_relu(&self) -> Option<&DualSlopeReLU> { None }
    fn as_learnable_mish(&self) -> Option<&LearnableMish> { None }
    fn as_learnable_softplus(&self) -> Option<&LearnableSoftplus> { None }
    fn as_rms_norm_learnable_eps(&self) -> Option<&RMSNormWithLearnableEpsilon> { None }
    fn as_adaptive_dropout(&self) -> Option<&AdaptiveDropout> { None }
    fn as_feature_fusion(&self) -> Option<&FeatureFusion> { None }
    fn as_sparse_feature_selection_gate(&self) -> Option<&SparseFeatureSelectionGate> { None }
    fn as_multi_resolution_kan_linear(&self) -> Option<&MultiResolutionKANLinear> { None }

    // Общая информация о слое, используемая планировщиком.
    // По умолчанию возвращает 0. Конкретные слои переопределяют.
    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }
}

// ---------------------------------------------------------------------------
// Новый трейт UniversalLayerBuffered – основа для работы с буферами
// ---------------------------------------------------------------------------

pub trait UniversalLayerBuffered: Send + Sync + 'static {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    );

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    );

    fn param_len(&self) -> usize;

    fn input_features(&self) -> usize;

    fn output_features(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Публичные реэкспорты
// ---------------------------------------------------------------------------

pub use linear::Linear;
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;
pub use memory::Memory;
pub use splitter::Splitter;
pub use combiner::Combiner;
pub use splitter_connector::SplitterConnector;
pub use combiner_connector::CombinerConnector;
pub use leaky_relu::LeakyReLU;
pub use identity::Identity;
pub use soft_sparse_gate::SoftSparseGate;
pub use soft_keep_gate::SoftKeepGate;
pub use dual_anchor::DualAnchor;
pub use adaptive_activation::AdaptivePerFeatureActivation;
pub use adaptive_normalization::AdaptiveNormalization;
pub use batch_renorm::BatchRenorm1d;
pub use concrete_dropout::ConcreteDropout;
pub use mamba::Mamba;
pub use linear_attention::LinearAttention;
pub use relative_position_attention::RelativePositionAttention;
pub use ind_rnn::IndRNN;
pub use spectral_norm_linear::SpectrallyNormalizedLinear;

// Новые реэкспорты
pub use dual_slope_relu::DualSlopeReLU;
pub use learnable_mish::LearnableMish;
pub use learnable_softplus::LearnableSoftplus;
pub use rms_norm_learnable_eps::RMSNormWithLearnableEpsilon;
pub use adaptive_dropout::AdaptiveDropout;
pub use feature_fusion::FeatureFusion;
pub use sparse_feature_selection_gate::SparseFeatureSelectionGate;
pub use multi_resolution_kan_linear::MultiResolutionKANLinear;

pub use layers_special::{DimReduce, DimExpand, ReduceMean, Unsqueeze};
pub use buffered_context::BufferedContext;