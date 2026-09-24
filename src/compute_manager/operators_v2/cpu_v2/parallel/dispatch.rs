// src/compute_manager/cpu/parallel/dispatch.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
    DualSlopeReLU, LearnableMish, LearnableSoftplus, RMSNormWithLearnableEpsilon,
    AdaptiveDropout, FeatureFusion, SparseFeatureSelectionGate, MultiResolutionKANLinear,
    AdaptiveNormalization, BatchRenorm1d, ConcreteDropout, IndRNN, Mamba,
    SpectrallyNormalizedLinear, LinearAttention, RelativePositionAttention,
    PerFeatureAttention,
    AdaptiveSpaceCompress,
    BufferedContext,
};
use crate::model_plan::param_store::ParamSlice;

/// Единая точка вызова forward слоя (dense). Каждый слой сам строит свой
/// `BufferedContext`.
pub(super) fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    pool: &mut TempMatrixPool,
) -> BufferedContext {
    let l: &dyn UniversalLayer = layer.as_ref();

    macro_rules! fwd {
        ($ty:ty, $getter:ident) => {
            if let Some(x) = l.$getter() {
                return <$ty as UniversalLayerBuffered>::forward_buffered(
                    x, input, output, params, slice, pool,
                );
            }
        };
    }

    fwd!(Linear, as_linear);
    fwd!(ReLU, as_relu);
    fwd!(Sigmoid, as_sigmoid);
    fwd!(Tanh, as_tanh);
    fwd!(LeakyReLU, as_leaky_relu);
    fwd!(Identity, as_identity);
    fwd!(Softmax, as_softmax);
    fwd!(Memory, as_memory);
    fwd!(SoftSparseGate, as_soft_sparse_gate);
    fwd!(SoftKeepGate, as_soft_keep_gate);
    fwd!(DualAnchor, as_dual_anchor);
    fwd!(AdaptivePerFeatureActivation, as_adaptive_activation);
    fwd!(DualSlopeReLU, as_dual_slope_relu);
    fwd!(LearnableMish, as_learnable_mish);
    fwd!(LearnableSoftplus, as_learnable_softplus);
    fwd!(RMSNormWithLearnableEpsilon, as_rms_norm_learnable_eps);
    fwd!(AdaptiveDropout, as_adaptive_dropout);
    fwd!(FeatureFusion, as_feature_fusion);
    fwd!(SparseFeatureSelectionGate, as_sparse_feature_selection_gate);
    fwd!(MultiResolutionKANLinear, as_multi_resolution_kan_linear);
    fwd!(AdaptiveNormalization, as_adaptive_normalization);
    fwd!(BatchRenorm1d, as_batch_renorm);
    fwd!(ConcreteDropout, as_concrete_dropout);
    fwd!(IndRNN, as_ind_rnn);
    fwd!(Mamba, as_mamba);
    fwd!(SpectrallyNormalizedLinear, as_spectral_norm_linear);
    fwd!(LinearAttention, as_linear_attention);
    fwd!(RelativePositionAttention, as_relative_position_attention);
    fwd!(PerFeatureAttention, as_per_feature_attention);
    fwd!(AdaptiveSpaceCompress, as_adaptive_space_compress);

    unreachable!("Unsupported layer in parallel forward");
}

/// Ragged-версия: если задан `sample_lens`, предпочитаем
/// `forward_buffered_ragged` для слоёв, поддерживающих ragged-вход.
/// Дефолтная реализация `forward_buffered_ragged` внутри трейта
/// вызывает `forward_buffered` — так что для остальных слоёв поведение
/// не меняется.
pub(super) fn call_forward_buffered_ragged(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    sample_lens: Option<&[usize]>,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    pool: &mut TempMatrixPool,
) -> BufferedContext {
    let Some(lens) = sample_lens else {
        return call_forward_buffered(layer, input, output, params, slice, pool);
    };

    let l: &dyn UniversalLayer = layer.as_ref();

    macro_rules! fwd_ragged {
        ($ty:ty, $getter:ident) => {
            if let Some(x) = l.$getter() {
                return <$ty as UniversalLayerBuffered>::forward_buffered_ragged(
                    x, input, lens, output, params, slice, pool,
                );
            }
        };
    }

    fwd_ragged!(Linear, as_linear);
    fwd_ragged!(ReLU, as_relu);
    fwd_ragged!(Sigmoid, as_sigmoid);
    fwd_ragged!(Tanh, as_tanh);
    fwd_ragged!(LeakyReLU, as_leaky_relu);
    fwd_ragged!(Identity, as_identity);
    fwd_ragged!(Softmax, as_softmax);
    fwd_ragged!(Memory, as_memory);
    fwd_ragged!(SoftSparseGate, as_soft_sparse_gate);
    fwd_ragged!(SoftKeepGate, as_soft_keep_gate);
    fwd_ragged!(DualAnchor, as_dual_anchor);
    fwd_ragged!(AdaptivePerFeatureActivation, as_adaptive_activation);
    fwd_ragged!(DualSlopeReLU, as_dual_slope_relu);
    fwd_ragged!(LearnableMish, as_learnable_mish);
    fwd_ragged!(LearnableSoftplus, as_learnable_softplus);
    fwd_ragged!(RMSNormWithLearnableEpsilon, as_rms_norm_learnable_eps);
    fwd_ragged!(AdaptiveDropout, as_adaptive_dropout);
    fwd_ragged!(FeatureFusion, as_feature_fusion);
    fwd_ragged!(SparseFeatureSelectionGate, as_sparse_feature_selection_gate);
    fwd_ragged!(MultiResolutionKANLinear, as_multi_resolution_kan_linear);
    fwd_ragged!(AdaptiveNormalization, as_adaptive_normalization);
    fwd_ragged!(BatchRenorm1d, as_batch_renorm);
    fwd_ragged!(ConcreteDropout, as_concrete_dropout);
    fwd_ragged!(IndRNN, as_ind_rnn);
    fwd_ragged!(Mamba, as_mamba);
    fwd_ragged!(SpectrallyNormalizedLinear, as_spectral_norm_linear);
    fwd_ragged!(LinearAttention, as_linear_attention);
    fwd_ragged!(RelativePositionAttention, as_relative_position_attention);
    fwd_ragged!(PerFeatureAttention, as_per_feature_attention);
    fwd_ragged!(AdaptiveSpaceCompress, as_adaptive_space_compress);

    unreachable!("Unsupported layer in parallel ragged forward");
}

/// Единая точка вызова backward слоя.
pub(super) fn call_backward_buffered(
    layer: &Box<dyn UniversalLayer>,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params: &MatrixBufferHandle,
) {
    let l: &dyn UniversalLayer = layer.as_ref();

    macro_rules! bwd {
        ($ty:ty, $getter:ident) => {
            if let Some(x) = l.$getter() {
                <$ty as UniversalLayerBuffered>::backward_buffered(
                    x, ctx, grad_output, grad_input, params, slice, grad_params,
                );
                return;
            }
        };
    }

    bwd!(Linear, as_linear);
    bwd!(ReLU, as_relu);
    bwd!(Sigmoid, as_sigmoid);
    bwd!(Tanh, as_tanh);
    bwd!(LeakyReLU, as_leaky_relu);
    bwd!(Identity, as_identity);
    bwd!(Softmax, as_softmax);
    bwd!(Memory, as_memory);
    bwd!(SoftSparseGate, as_soft_sparse_gate);
    bwd!(SoftKeepGate, as_soft_keep_gate);
    bwd!(DualAnchor, as_dual_anchor);
    bwd!(AdaptivePerFeatureActivation, as_adaptive_activation);
    bwd!(DualSlopeReLU, as_dual_slope_relu);
    bwd!(LearnableMish, as_learnable_mish);
    bwd!(LearnableSoftplus, as_learnable_softplus);
    bwd!(RMSNormWithLearnableEpsilon, as_rms_norm_learnable_eps);
    bwd!(AdaptiveDropout, as_adaptive_dropout);
    bwd!(FeatureFusion, as_feature_fusion);
    bwd!(SparseFeatureSelectionGate, as_sparse_feature_selection_gate);
    bwd!(MultiResolutionKANLinear, as_multi_resolution_kan_linear);
    bwd!(AdaptiveNormalization, as_adaptive_normalization);
    bwd!(BatchRenorm1d, as_batch_renorm);
    bwd!(ConcreteDropout, as_concrete_dropout);
    bwd!(IndRNN, as_ind_rnn);
    bwd!(Mamba, as_mamba);
    bwd!(SpectrallyNormalizedLinear, as_spectral_norm_linear);
    bwd!(LinearAttention, as_linear_attention);
    bwd!(RelativePositionAttention, as_relative_position_attention);
    bwd!(PerFeatureAttention, as_per_feature_attention);
    bwd!(AdaptiveSpaceCompress, as_adaptive_space_compress);

    unreachable!("Unsupported layer in parallel backward");
}

/// `AdaptiveSpaceCompress` **не входит** в чёрный список: все его
/// операции идут per-example (`r`) независимо, никаких накоплений
/// по батчу нет. `p_soft` детерминирован параметрами слоя.
pub(crate) fn can_parallelize(layers: &[Box<dyn UniversalLayer>]) -> bool {
    !layers.iter().any(|l| {
        l.as_memory().is_some()
            || l.as_ind_rnn().is_some()
            || l.as_mamba().is_some()
            || l.as_concrete_dropout().is_some()
            || l.as_adaptive_dropout().is_some()
            || l.as_batch_renorm().is_some()
            || l.as_linear_attention().is_some()
            || l.as_relative_position_attention().is_some()
    })
}