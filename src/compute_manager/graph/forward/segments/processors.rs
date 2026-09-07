// src/compute_manager/graph/forward/segments/processors.rs

use std::sync::Arc;

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
    DualSlopeReLU, LearnableMish, LearnableSoftplus, RMSNormWithLearnableEpsilon,
    AdaptiveDropout, FeatureFusion, SparseFeatureSelectionGate, MultiResolutionKANLinear,
    AdaptiveNormalization, BatchRenorm1d, ConcreteDropout, IndRNN, Mamba,
    SpectrallyNormalizedLinear,
};
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    pub(crate) fn process_universal_processor_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        proc: &Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[ParamSlice],
        _model_index: usize,
        params: &MatrixBufferHandle,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        stream_indices: &Option<Vec<usize>>,
    ) -> Vec<DynamicContext> {
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..stream_buffers.len()).collect(),
        };

        let layers = proc.as_ref();
        let num_layers = layers.len();

        let mut new_stream: Vec<Option<MatrixBufferHandle>> = stream_buffers
            .iter()
            .map(|handle| Some(handle.clone()))
            .collect();

        let mut result_ctxs = Vec::new();

        for &stream_idx in &active_indices {
            let input_handle = stream_buffers[stream_idx].clone();
            let batch_size = input_handle.rows();
            let mut current_input = input_handle;
            let mut layer_ctxs: Vec<DynamicContext> = Vec::with_capacity(num_layers);

            for i in 0..num_layers {
                let layer = &layers[i];
                let slice = &slices[i];

                let out_features = get_buffered_output_features(layer, &current_input);
                let output_handle = pool.acquire(batch_size, out_features);

                call_forward_buffered(layer, &current_input, &output_handle, params, slice);

                let buffered_ctx = build_buffered_context(layer, &current_input, &output_handle, pool);
                layer_ctxs.push(DynamicContext::Buffered(buffered_ctx));

                current_input = output_handle;
            }

            new_stream[stream_idx] = Some(current_input);
            result_ctxs = layer_ctxs;
        }

        *stream_buffers = new_stream
            .into_iter()
            .map(|opt| opt.expect("Missing stream buffer after forward"))
            .collect();

        result_ctxs
    }
}

fn get_buffered_output_features(layer: &Box<dyn UniversalLayer>, input: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        return <Linear as UniversalLayerBuffered>::output_features(l);
    }
    if let Some(l) = layer.as_feature_fusion() {
        return <FeatureFusion as UniversalLayerBuffered>::output_features(l);
    }
    if let Some(l) = layer.as_multi_resolution_kan_linear() {
        return <MultiResolutionKANLinear as UniversalLayerBuffered>::output_features(l);
    }
    if let Some(l) = layer.as_spectral_norm_linear() {
        return <SpectrallyNormalizedLinear as UniversalLayerBuffered>::output_features(l);
    }

    if layer.as_relu().is_some()
        || layer.as_sigmoid().is_some()
        || layer.as_tanh().is_some()
        || layer.as_leaky_relu().is_some()
        || layer.as_identity().is_some()
        || layer.as_softmax().is_some()
        || layer.as_memory().is_some()
        || layer.as_soft_sparse_gate().is_some()
        || layer.as_soft_keep_gate().is_some()
        || layer.as_dual_anchor().is_some()
        || layer.as_adaptive_activation().is_some()
        || layer.as_dual_slope_relu().is_some()
        || layer.as_learnable_mish().is_some()
        || layer.as_learnable_softplus().is_some()
        || layer.as_rms_norm_learnable_eps().is_some()
        || layer.as_adaptive_dropout().is_some()
        || layer.as_sparse_feature_selection_gate().is_some()
        || layer.as_adaptive_normalization().is_some()
        || layer.as_batch_renorm().is_some()
        || layer.as_concrete_dropout().is_some()
        || layer.as_ind_rnn().is_some()
        || layer.as_mamba().is_some()
    {
        return input.cols();
    }

    input.cols()
}

fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
) {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else {
        unreachable!(
            "Layer {:?} does not implement UniversalLayerBuffered for CPU path",
            std::any::type_name_of_val(layer.as_ref())
        );
    }
}

fn build_buffered_context(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    pool: &mut TempMatrixPool,
) -> BufferedContext {
    if layer.as_linear().is_some() {
        BufferedContext::Linear { input: input.clone() }
    } else if layer.as_relu().is_some() {
        BufferedContext::ReLU { input: input.clone() }
    } else if layer.as_sigmoid().is_some() {
        BufferedContext::Sigmoid { output: output.clone() }
    } else if layer.as_tanh().is_some() {
        BufferedContext::Tanh { output: output.clone() }
    } else if layer.as_softmax().is_some() {
        BufferedContext::Softmax { output: output.clone() }
    } else if layer.as_leaky_relu().is_some() {
        BufferedContext::LeakyReLU { input: input.clone() }
    } else if layer.as_identity().is_some() {
        BufferedContext::Identity { input: input.clone() }
    } else if layer.as_memory().is_some() {
        BufferedContext::Memory { input: input.clone() }
    } else if layer.as_soft_sparse_gate().is_some() {
        BufferedContext::SoftSparseGate { input: input.clone() }
    } else if layer.as_soft_keep_gate().is_some() {
        BufferedContext::SoftKeepGate { input: input.clone() }
    } else if layer.as_dual_anchor().is_some() {
        BufferedContext::DualAnchor1D { input: input.clone() }
    } else if layer.as_adaptive_activation().is_some() {
        BufferedContext::AdaptiveActivation { input: input.clone() }
    } else if layer.as_dual_slope_relu().is_some() {
        BufferedContext::DualSlopeReLU { input: input.clone() }
    } else if layer.as_learnable_mish().is_some() {
        BufferedContext::LearnableMish { input: input.clone() }
    } else if layer.as_learnable_softplus().is_some() {
        BufferedContext::LearnableSoftplus { input: input.clone() }
    } else if layer.as_rms_norm_learnable_eps().is_some() {
        BufferedContext::RMSNormWithLearnableEpsilon { input: input.clone() }
    } else if layer.as_adaptive_dropout().is_some() {
        // Для CPU-ветки mask и arg не нужны, создаём пустые handle
        let empty_mask = pool.acquire(0, 0);
        let empty_arg = pool.acquire(0, 0);
        BufferedContext::AdaptiveDropout {
            input: input.clone(),
            mask: empty_mask,
            arg: empty_arg,
        }
    } else if layer.as_feature_fusion().is_some() {
        BufferedContext::FeatureFusion { input: input.clone() }
    } else if layer.as_sparse_feature_selection_gate().is_some() {
        BufferedContext::SparseFeatureSelectionGate { input: input.clone() }
    } else if layer.as_multi_resolution_kan_linear().is_some() {
        BufferedContext::MultiResolutionKANLinear { input: input.clone() }
    } else if layer.as_adaptive_normalization().is_some() {
        BufferedContext::AdaptiveNormalization { input: input.clone() }
    } else if layer.as_batch_renorm().is_some() {
        BufferedContext::BatchRenorm {
            input: input.clone(),
            mean: Vec::new(),
            var: Vec::new(),
            use_batch_stats: true, // заменить на слой если нужно
        }
    } else if layer.as_concrete_dropout().is_some() {
        let empty_arg = pool.acquire(0, 0);
        BufferedContext::ConcreteDropout {
            input: input.clone(),
            arg: empty_arg,
        }
    } else if layer.as_ind_rnn().is_some() {
        let empty_h = pool.acquire(0, 0);
        BufferedContext::IndRNN {
            input: input.clone(),
            h_all: empty_h,
        }
    } else if layer.as_mamba().is_some() {
        let empty_h = pool.acquire(0, 0);
        BufferedContext::Mamba {
            input: input.clone(),
            h_all: empty_h,
        }
    } else if layer.as_spectral_norm_linear().is_some() {
        BufferedContext::SpectralNormLinear { input: input.clone() }
    } else {
        BufferedContext::Identity { input: input.clone() }
    }
}