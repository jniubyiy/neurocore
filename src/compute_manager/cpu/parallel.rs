// src/compute_manager/cpu/parallel.rs

use std::sync::{Arc, Barrier, Mutex};

use crate::compute_manager::executor::Executor;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext};
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

/// Общая структура данных для одной задачи прямого прохода.
struct ForwardTaskShared {
    input: MatrixBufferHandle,
    output: MatrixBufferHandle,
    params: MatrixBufferHandle,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Arc<Vec<ParamSlice>>,
    pool: Arc<Mutex<TempMatrixPool>>,
}

/// Общая структура данных для одной задачи обратного прохода.
struct BackwardTaskShared {
    grad_output: MatrixBufferHandle,
    grad_input: MatrixBufferHandle,
    params: MatrixBufferHandle,
    grad_params: MatrixBufferHandle,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Arc<Vec<ParamSlice>>,
    contexts: ChunkedContexts,
    pool: Arc<Mutex<TempMatrixPool>>,
}

pub(crate) fn extract_chunk(
    input: &MatrixBufferHandle,
    start: usize,
    end: usize,
    pool: &mut TempMatrixPool,
) -> MatrixBufferHandle {
    let rows_total = input.rows();
    let cols = input.cols();
    assert!(end <= rows_total && start < end);

    let chunk_rows = end - start;
    let chunk = pool.acquire(chunk_rows, cols);

    let src_guard = input.read();
    let src = src_guard.as_slice().expect("CPU buffer");
    let mut dst_guard = chunk.write();
    let dst = dst_guard.as_slice_mut().expect("CPU buffer");

    for c in 0..cols {
        for r in 0..chunk_rows {
            dst[c * chunk_rows + r] = src[c * rows_total + start + r];
        }
    }
    chunk
}

pub(crate) fn write_chunk(
    output: &MatrixBufferHandle,
    chunk: &MatrixBufferHandle,
    start: usize,
) {
    let out_rows = output.rows();
    let cols = output.cols();
    let chunk_rows = chunk.rows();
    assert_eq!(cols, chunk.cols());
    assert!(start + chunk_rows <= out_rows);

    let mut out_guard = output.write();
    let out_slice = out_guard.as_slice_mut().expect("CPU buffer");
    let chunk_guard = chunk.read();
    let chunk_slice = chunk_guard.as_slice().expect("CPU buffer");

    for c in 0..cols {
        for r in 0..chunk_rows {
            out_slice[c * out_rows + start + r] = chunk_slice[c * chunk_rows + r];
        }
    }
}

fn get_output_features(layer: &Box<dyn UniversalLayer>, input: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::output_features(l)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::output_features(l)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::output_features(l)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::output_features(l)
    } else {
        // Все остальные слои сохраняют размерность
        input.cols()
    }
}

fn get_input_features(layer: &Box<dyn UniversalLayer>, grad_output: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::input_features(l)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::input_features(l)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::input_features(l)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::input_features(l)
    } else {
        grad_output.cols()
    }
}

fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
) {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else {
        unreachable!("Unsupported layer in parallel forward");
    }
}

fn call_backward_buffered(
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
    } else {
        unreachable!("Unsupported layer in parallel backward");
    }
}

fn build_buffered_context(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    pool_guard: &mut TempMatrixPool,
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
        // CPU-ветка: mask и arg не нужны в контексте, но требуются для типа.
        // Создаём пустые handle.
        let empty_mask = pool_guard.acquire(0, 0);
        let empty_arg = pool_guard.acquire(0, 0);
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
            use_batch_stats: true, // можно уточнить, но для CPU backward статистики берутся из слоя? В текущей реализации batch_renorm backward использует mean/var из ctx, но forward их не сохраняет. Временное решение: оставляем пустые. Лучше исправить сам слой, но пока так.
        }
    } else if layer.as_concrete_dropout().is_some() {
        let empty_arg = pool_guard.acquire(0, 0);
        BufferedContext::ConcreteDropout {
            input: input.clone(),
            arg: empty_arg,
        }
    } else if layer.as_ind_rnn().is_some() {
        let empty_h = pool_guard.acquire(0, 0);
        BufferedContext::IndRNN {
            input: input.clone(),
            h_all: empty_h,
        }
    } else if layer.as_mamba().is_some() {
        let empty_h = pool_guard.acquire(0, 0);
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

pub(crate) fn can_parallelize(layers: &[Box<dyn UniversalLayer>]) -> bool {
    // Запрещаем параллелизм для слоёв с внутренним состоянием,
    // которое общее для всех чанков батча.
    !layers.iter().any(|l| {
        l.as_memory().is_some()
            || l.as_ind_rnn().is_some()
            || l.as_mamba().is_some()
            || l.as_concrete_dropout().is_some()
            || l.as_adaptive_dropout().is_some()
            || l.as_batch_renorm().is_some()
    })
}

pub(crate) fn forward_universal_parallel(
    executor: &dyn Executor,
    pool: Arc<Mutex<TempMatrixPool>>,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Vec<ParamSlice>,
    params: MatrixBufferHandle,
    input: MatrixBufferHandle,
    output: MatrixBufferHandle,
) -> ChunkedContexts {
    let batch_size = input.rows();
    let chunks = executor.plan_chunks_assignment(batch_size);
    let all_chunks: Vec<(usize, usize, usize)> = chunks.into_iter().flatten().collect();
    let num_chunks = all_chunks.len();
    if num_chunks == 0 {
        return Vec::new();
    }

    let slices_arc = Arc::new(slices);
    let shared = Arc::new(ForwardTaskShared {
        input,
        output,
        params,
        layers,
        slices: slices_arc,
        pool,
    });

    let ctx_storage = Arc::new(Mutex::new(vec![Vec::new(); num_chunks]));
    let barrier = Arc::new(Barrier::new(num_chunks + 1));

    for (chunk_id, (start, _size, end)) in all_chunks.into_iter().enumerate() {
        let shared = shared.clone();
        let barrier = barrier.clone();
        let ctx_storage = ctx_storage.clone();

        executor.execute_dyn(Box::new(move || {
            {
                let mut pool_guard = shared.pool.lock().unwrap();
                let input_chunk = extract_chunk(&shared.input, start, end, &mut *pool_guard);
                let mut current = input_chunk;
                let mut chunk_ctxs = Vec::with_capacity(shared.layers.len());

                for (layer, slice) in shared.layers.iter().zip(shared.slices.iter()) {
                    let out_cols = get_output_features(layer, &current);
                    let out = pool_guard.acquire(current.rows(), out_cols);
                    let buffered_ctx = build_buffered_context(layer, &current, &out, &mut *pool_guard);
                    call_forward_buffered(layer, &current, &out, &shared.params, slice);
                    chunk_ctxs.push(DynamicContext::Buffered(buffered_ctx));
                    current = out;
                }

                write_chunk(&shared.output, &current, start);

                {
                    let mut storage = ctx_storage.lock().unwrap();
                    storage[chunk_id] = chunk_ctxs;
                }
            }
            barrier.wait();
        }));
    }

    barrier.wait();
    let storage = ctx_storage.lock().unwrap();
    storage.clone()
}

pub(crate) fn backward_universal_parallel(
    executor: &dyn Executor,
    pool: Arc<Mutex<TempMatrixPool>>,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Vec<ParamSlice>,
    contexts: ChunkedContexts,
    grad_output: MatrixBufferHandle,
    grad_input: MatrixBufferHandle,
    params: MatrixBufferHandle,
    grad_params: MatrixBufferHandle,
) {
    let batch_size = grad_output.rows();
    let chunks = executor.plan_chunks_assignment(batch_size);
    let all_chunks: Vec<(usize, usize, usize)> = chunks.into_iter().flatten().collect();
    let num_chunks = all_chunks.len();
    if num_chunks == 0 || num_chunks != contexts.len() {
        panic!("backward_universal_parallel: number of chunks does not match contexts");
    }

    let slices_arc = Arc::new(slices);
    let shared = Arc::new(BackwardTaskShared {
        grad_output,
        grad_input,
        params,
        grad_params,
        layers,
        slices: slices_arc,
        contexts,
        pool,
    });

    let param_len = shared.grad_params.rows();
    let mut temp_grads = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        temp_grads.push(
            shared
                .pool
                .lock()
                .unwrap()
                .acquire(param_len, 1),
        );
    }

    let barrier = Arc::new(Barrier::new(num_chunks + 1));

    for (chunk_id, (start, _size, end)) in all_chunks.into_iter().enumerate() {
        let shared = shared.clone();
        let barrier = barrier.clone();
        let temp_grad = temp_grads[chunk_id].clone();

        executor.execute_dyn(Box::new(move || {
            {
                let mut pool_guard = shared.pool.lock().unwrap();
                let grad_output_chunk = extract_chunk(&shared.grad_output, start, end, &mut *pool_guard);
                let mut current_grad = grad_output_chunk;

                let contexts_chunk = &shared.contexts[chunk_id];
                for i in (0..shared.layers.len()).rev() {
                    let layer = &shared.layers[i];
                    let slice = &shared.slices[i];
                    let ctx = &contexts_chunk[i];

                    let in_features = get_input_features(layer, &current_grad);
                    let grad_input_chunk = pool_guard.acquire(current_grad.rows(), in_features);

                    call_backward_buffered(
                        layer,
                        ctx,
                        &current_grad,
                        &grad_input_chunk,
                        &shared.params,
                        slice,
                        &temp_grad,
                    );

                    pool_guard.release(current_grad);
                    current_grad = grad_input_chunk;
                }

                write_chunk(&shared.grad_input, &current_grad, start);
                pool_guard.release(current_grad);
            }
            barrier.wait();
        }));
    }

    barrier.wait();

    let mut pool_guard = shared.pool.lock().unwrap();
    {
        let mut grad_guard = shared.grad_params.write();
        let grad_slice = grad_guard.as_slice_mut().expect("CPU buffer");
        for v in grad_slice.iter_mut() {
            *v = 0.0;
        }
    }
    for temp in temp_grads {
        let temp_guard = temp.read();
        let temp_slice = temp_guard.as_slice().expect("CPU buffer");
        let mut grad_guard = shared.grad_params.write();
        let grad_slice = grad_guard.as_slice_mut().expect("CPU buffer");
        for i in 0..grad_slice.len() {
            grad_slice[i] += temp_slice[i];
        }
        pool_guard.release(temp);
    }
}