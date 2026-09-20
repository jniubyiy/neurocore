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

// PerFeatureAttention (CPU-only)
pub mod per_feature_attention;

pub mod layers_special;
pub mod buffered_context;

// Инфраструктура градиентных адаптеров (MIGRATION_PLAN.md §7, Фаза 3).
pub mod adapter;

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::compute_manager::operators_v2::memory_v2::buffer::TempMatrixPool;
use crate::model_plan::param_store::ParamSlice;

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

    // PerFeatureAttention
    fn as_per_feature_attention(&self) -> Option<&PerFeatureAttention> { None }

    // -----------------------------------------------------------------------
    // Градиентный адаптер (MIGRATION_PLAN.md §7, Фаза 3)
    // -----------------------------------------------------------------------
    //
    // Возвращает адаптер этого слоя, если он есть. Вызывается в
    // `GraphV2::adapter_pass` (Фаза 4) между `optimizer_modify_grads`
    // и `optimizer_apply_update`.
    //
    // Дефолт `None` означает «у слоя нет адаптера» — поведение полностью
    // совпадает с архитектурой без адаптеров. Все существующие слои
    // используют дефолт; переопределение появится в Фазе 5+.
    //
    // Инварианты:
    //   * I-4: адаптер живёт в папке слоя (`src/layers/<layer>/adapter/`);
    //   * I-5: устройство адаптера = устройство слоя;
    //   * I-9: каждый слой имеет право на свою формулу.
    #[inline]
    fn adapter(&self) -> Option<&dyn GradientAdapter> {
        None
    }

    // Общая информация о слое, используемая планировщиком.
    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }

    #[inline]
    fn output_features_for(&self, in_cols: usize) -> usize {
        let of = self.output_features();
        if of == 0 { in_cols } else { of }
    }

    #[inline]
    fn input_features_for(&self, fallback: usize) -> usize {
        let inf = self.input_features();
        if inf == 0 { fallback } else { inf }
    }
}

// ---------------------------------------------------------------------------
// UniversalLayerBuffered — единый контракт forward/backward
// ---------------------------------------------------------------------------

pub trait UniversalLayerBuffered: Send + Sync + 'static {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext;

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

pub use per_feature_attention::PerFeatureAttention;

pub use layers_special::{DimReduce, DimExpand, ReduceMean, Unsqueeze};
pub use buffered_context::BufferedContext;

// Инфраструктура адаптеров (Фаза 3).
//
// `GradientAdapter` здесь и вводит имя в текущий модуль (для сигнатуры
// `UniversalLayer::adapter()` выше), и реэкспортирует его наружу.
// Отдельный `use crate::layers::adapter::GradientAdapter;` избыточен
// и вызывает E0252.
pub use adapter::{AdapterContext, AdapterRegistry, AdapterTypeInfo, GradientAdapter};