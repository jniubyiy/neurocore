// src/compute_manager/graph/backward/segments/processors.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
};
use crate::model_plan::param_store::ParamSlice;

impl crate::compute_manager::graph::model::MixedModel {
    /// Последовательный обратный проход через цепочку слоёв UniversalProcessor.
    /// Используется в CPU‑ветке, когда параллелизм не применяется.
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
        let mut current_grad = grad_out;
        for i in (0..layers.len()).rev() {
            let layer = &layers[i];
            let slice = &slices[i];
            let ctx = ctxs[i];

            let in_features = if let Some(l) = layer.as_linear() {
                <dyn UniversalLayerBuffered>::input_features(l)
            } else if layer.as_relu().is_some()
                || layer.as_sigmoid().is_some()
                || layer.as_tanh().is_some()
                || layer.as_leaky_relu().is_some()
                || layer.as_identity().is_some()
                || layer.as_softmax().is_some()
                || layer.as_memory().is_some()
                || layer.as_soft_sparse_gate().is_some()
                || layer.as_soft_keep_gate().is_some()
                || layer.as_dual_anchor().is_some()
                || layer.as_adaptive_activation().is_some()   // <-- добавлено
            {
                current_grad.cols()
            } else {
                current_grad.cols()
            };

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
fn call_backward_buffered(
    layer: &Box<dyn UniversalLayer>,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &mut MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params_handle: &MatrixBufferHandle,
) {
    if let Some(linear) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::backward_buffered(
            linear, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(relu) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::backward_buffered(
            relu, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(sigmoid) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::backward_buffered(
            sigmoid, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(tanh) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::backward_buffered(
            tanh, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(leaky) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::backward_buffered(
            leaky, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(identity) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::backward_buffered(
            identity, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(softmax) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::backward_buffered(
            softmax, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(memory) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::backward_buffered(
            memory, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::backward_buffered(
            soft_sparse, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::backward_buffered(
            soft_keep, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(dual_anchor) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::backward_buffered(
            dual_anchor, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(adaptive) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::backward_buffered(
            adaptive, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else {
        unreachable!(
            "Layer {:?} does not have a buffered backward implementation",
            std::any::type_name_of_val(layer.as_ref())
        );
    }
}