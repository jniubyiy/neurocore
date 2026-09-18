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
    BufferedContext,
};
use crate::model_plan::param_store::ParamSlice;

/// Единая точка вызова forward слоя. Каждый слой сам строит свой
/// `BufferedContext` — включая per-chunk state, если он есть.
pub(super) fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    pool: &mut TempMatrixPool,
) -> BufferedContext {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_linear_attention() {
        <LinearAttention as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_relative_position_attention() {
        <RelativePositionAttention as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_per_feature_attention() {
        <PerFeatureAttention as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else {
        unreachable!("Unsupported layer in parallel forward");
    }
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
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_linear_attention() {
        <LinearAttention as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_relative_position_attention() {
        <RelativePositionAttention as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_per_feature_attention() {
        <PerFeatureAttention as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else {
        unreachable!("Unsupported layer in parallel backward");
    }
}

/// Определяет, можно ли распараллелить цепочку слоёв по чанкам батча.
///
/// `PerFeatureAttention` **не входит** в список несовместимых слоёв: все его
/// операции выполняются независимо по каждой строке батча `r` (в т.ч.
/// reductions kv/z идут по оси `t`, а не по `r`). State-буферы создаются
/// per-chunk в `forward_buffered` и передаются через `BufferedContext`.
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