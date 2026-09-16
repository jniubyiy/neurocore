// src/compute_manager/graph/backward/segments/processors.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
    DualSlopeReLU, LearnableMish, LearnableSoftplus, RMSNormWithLearnableEpsilon,
    AdaptiveDropout, FeatureFusion, SparseFeatureSelectionGate, MultiResolutionKANLinear,
    AdaptiveNormalization, BatchRenorm1d, ConcreteDropout, IndRNN, Mamba,
    SpectrallyNormalizedLinear, LinearAttention, RelativePositionAttention,
};
use crate::model_plan::param_store::ParamSlice;

impl crate::compute_manager::graph::model::MixedModel {
    /// Последовательный обратный проход через цепочку слоёв UniversalProcessor.
    /// Используется в CPU‑ветке, когда параллелизм не применяется.
    ///
    /// Контексты слоёв (`ctxs`) были созданы в forward и содержат всё
    /// per-chunk состояние, необходимое backward'у: у каждого слоя — свой
    /// `BufferedContext`, включая state-буферы (h_all, mask, arg, per-head
    /// буферы LinearAttention и т.д.). Слой не читает состояние из своих
    /// полей.
    pub(crate) fn backward_universal_batch_buffered_handle(
        &mut self,
        pool: &mut TempMatrixPool,
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        ctxs: &[&DynamicContext],
        grad_out: MatrixBufferHandle,
        params: &MatrixBufferHandle,
        grad_params_handle: &MatrixBufferHandle,
    ) -> MatrixBufferHandle {
        assert_eq!(
            layers.len(),
            slices.len(),
            "backward_universal_batch_buffered_handle: layers/slices count mismatch"
        );
        assert_eq!(
            layers.len(),
            ctxs.len(),
            "backward_universal_batch_buffered_handle: layers/contexts count mismatch"
        );

        let mut current_grad = grad_out;
        for i in (0..layers.len()).rev() {
            let layer = &layers[i];
            let slice = &slices[i];
            let ctx = ctxs[i];

            // Входная размерность слоя. Вся логика определения — в трейте:
            // слои с фиксированным input_features() возвращают его,
            // слои, сохраняющие размерность, — число столбцов текущего
            // буфера градиента.
            let in_features = layer.input_features_for(current_grad.cols());

            let batch = current_grad.rows();
            let mut grad_input = pool.acquire(batch, in_features);

            call_backward_buffered(
                layer,
                ctx,
                &current_grad,
                &mut grad_input,
                params,
                slice,
                grad_params_handle,
            );

            pool.release(current_grad);
            current_grad = grad_input;
        }
        current_grad
    }
}

/// Диспетчеризация обратного прохода для конкретного слоя.
///
/// В каждой ветке `as_*` передаётся та же сигнатура `backward_buffered`,
/// которую объявляет трейт `UniversalLayerBuffered`. Никакие сигнатуры
/// не менялись по сравнению с прежней версией — слои читают состояние
/// из `ctx` (вариант `DynamicContext::Buffered(BufferedContext::…)`),
/// а не из собственных полей.
fn call_backward_buffered(
    layer: &Box<dyn UniversalLayer>,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &mut MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params_handle: &MatrixBufferHandle,
) {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_linear_attention() {
        <LinearAttention as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(l) = layer.as_relative_position_attention() {
        <RelativePositionAttention as UniversalLayerBuffered>::backward_buffered(
            l, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else {
        unreachable!(
            "Layer {:?} does not have a buffered backward implementation",
            std::any::type_name_of_val(layer.as_ref())
        );
    }
}