// src/compute_manager/operators_v2/cpu_v2/chunk_dispatch_v2.rs
//
// Диспетчер forward/backward UniversalProcessor-сегмента на CPU + DimOp
// + ConnectorOp.
//
// Universal:
//   * Параллельный режим — через существующие
//     `forward_universal_parallel` / `backward_universal_parallel`.
//   * Последовательный режим — проход по слоям в вызывающем потоке.
//
// DimOp:
//   * вызов `dim_change::unsqueeze_mat_buffered_handle` /
//     `reduce_mat_buffered_handle` (CPU-only).
//
// Connector:
//   * Splitter / Combiner — CPU-реализация по формулам из старого пути.

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::parallel::{
    backward_universal_parallel, can_parallelize, forward_universal_parallel,
};
use crate::compute_manager::dim_change;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext};
use crate::compute_manager::jobs_v2::{
    BackwardSegmentJob, ConnectorDirection, ConnectorOpJob, ConnectorOpKind,
    DimOpJob, DimOpKind, ForwardContextsV2, ForwardSegmentJob, JobResult,
};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::{
    BufferedContext, UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
    DualSlopeReLU, LearnableMish, LearnableSoftplus, RMSNormWithLearnableEpsilon,
    AdaptiveDropout, FeatureFusion, SparseFeatureSelectionGate, MultiResolutionKANLinear,
    AdaptiveNormalization, BatchRenorm1d, ConcreteDropout, IndRNN, Mamba,
    SpectrallyNormalizedLinear, LinearAttention, RelativePositionAttention,
    PerFeatureAttention,
};
use crate::model_plan::param_store::ParamSlice;

// ============================================================================
// Forward Universal
// ============================================================================

pub fn execute_forward(
    job: ForwardSegmentJob,
    executor: &dyn Executor,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    let layers = Arc::clone(&job.layers);
    let slices = job.slices.clone();
    let params = job.params.clone();
    let input = job.input.clone();

    let batch = input.rows();

    let mut cols = input.cols();
    for layer in layers.iter() {
        cols = layer.output_features_for(cols);
    }
    let out_features = cols;

    let can_parallel = can_parallelize(layers.as_slice())
        && batch > 1
        && executor.num_workers() > 1;

    if can_parallel {
        let output = {
            let mut p = pool.lock().unwrap();
            p.acquire(batch, out_features)
        };

        let (chunk_ctxs, chunk_layout) = forward_universal_parallel(
            executor,
            pool,
            Arc::clone(&layers),
            slices,
            params,
            input,
            output.clone(),
        );

        JobResult::Forward {
            output,
            contexts: ForwardContextsV2::Chunked {
                contexts: chunk_ctxs,
                layout: chunk_layout,
            },
        }
    } else {
        let mut current_input = input;
        let mut ctxs: Vec<DynamicContext> = Vec::with_capacity(layers.len());

        for (layer, slice) in layers.iter().zip(slices.iter()) {
            let out_feat = layer.output_features_for(current_input.cols());
            let out = {
                let mut p = pool.lock().unwrap();
                p.acquire(batch, out_feat)
            };

            let ctx = dispatch_forward_one_layer(
                layer,
                &current_input,
                &out,
                &params,
                slice,
                &pool,
            );

            ctxs.push(DynamicContext::Buffered(ctx));
            current_input = out;
        }

        JobResult::Forward {
            output: current_input,
            contexts: ForwardContextsV2::Sequential(ctxs),
        }
    }
}

// ============================================================================
// Backward Universal
// ============================================================================

pub fn execute_backward(
    job: BackwardSegmentJob,
    executor: &dyn Executor,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    let layers = Arc::clone(&job.layers);
    let slices = job.slices.clone();
    let params = job.params.clone();
    let grad_params = job.grad_params.clone();
    let grad_output = job.grad_output.clone();

    let batch = grad_output.rows();

    let in_features = match layers.first() {
        Some(first) => first.input_features_for(grad_output.cols()),
        None => grad_output.cols(),
    };

    let can_parallel = can_parallelize(layers.as_slice())
        && batch > 1
        && executor.num_workers() > 1
        && matches!(job.contexts, ForwardContextsV2::Chunked { .. });

    if can_parallel {
        let grad_input = {
            let mut p = pool.lock().unwrap();
            p.acquire(batch, in_features)
        };

        let (ctxs, layout): (ChunkedContexts, Vec<(usize, usize, usize)>) =
            match job.contexts {
                ForwardContextsV2::Chunked { contexts, layout } => (contexts, layout),
                ForwardContextsV2::Sequential(v) => (vec![v], vec![(0, batch, batch)]),
            };

        backward_universal_parallel(
            executor,
            pool,
            layers,
            slices,
            ctxs,
            &layout,
            grad_output,
            grad_input.clone(),
            params,
            grad_params,
        );

        JobResult::Backward { grad_input }
    } else {
        let ctxs: Vec<DynamicContext> = job.contexts.first_chunk();
        if ctxs.len() != layers.len() {
            return JobResult::Failed(format!(
                "CpuOperatorV2::backward: contexts count ({}) != layers count ({})",
                ctxs.len(),
                layers.len()
            ));
        }

        let grad_input = {
            let mut p = pool.lock().unwrap();
            p.acquire(batch, in_features)
        };

        let mut current_grad = grad_output;
        for i in (0..layers.len()).rev() {
            let layer = &layers[i];
            let slice = &slices[i];
            let ctx = &ctxs[i];

            let layer_in_feat = layer.input_features_for(current_grad.cols());
            let next_grad = {
                let mut p = pool.lock().unwrap();
                p.acquire(current_grad.rows(), layer_in_feat)
            };

            dispatch_backward_one_layer(
                layer,
                ctx,
                &current_grad,
                &next_grad,
                &params,
                slice,
                &grad_params,
            );

            {
                let mut p = pool.lock().unwrap();
                p.release(current_grad);
            }
            current_grad = next_grad;
        }

        // Копируем финальный градиент в выделенный grad_input.
        {
            let src_guard = current_grad.read();
            let src = src_guard.as_slice().expect("CPU buffer");
            grad_input.write_range(0, src);
        }
        {
            let mut p = pool.lock().unwrap();
            p.release(current_grad);
        }

        JobResult::Backward { grad_input }
    }
}

// ============================================================================
// DimOp
// ============================================================================

/// Изменение формы буфера без потери элементов.
pub fn execute_dimop(job: DimOpJob, pool: Arc<Mutex<TempMatrixPool>>) -> JobResult {
    let mut p = pool.lock().unwrap();
    let result = match job.kind {
        DimOpKind::Unsqueeze(dims) => {
            dim_change::unsqueeze_mat_buffered_handle(&mut *p, job.input, &dims)
        }
        DimOpKind::ReduceMean(dims) => {
            dim_change::reduce_mat_buffered_handle(&mut *p, job.input, &dims)
        }
    };
    JobResult::Buffer(result)
}

// ============================================================================
// ConnectorOp
// ============================================================================

/// Диспетчер forward/backward коннекторов.
pub fn execute_connector(
    job: ConnectorOpJob,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    match job.direction {
        ConnectorDirection::Forward => execute_connector_forward(job, pool),
        ConnectorDirection::Backward => execute_connector_backward(job, pool),
    }
}

/// Forward-диспетчер коннекторов.
///
/// ВАЖНО: match по **клонированному** `kind`, а не по `&job.kind`.
/// Иначе borrow `job.kind` остаётся живым, пока `job` move-ится в одну
/// из вызываемых функций ниже (ошибка E0505).
fn execute_connector_forward(
    job: ConnectorOpJob,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    let kind = job.kind.clone();
    match kind {
        ConnectorOpKind::Splitter {
            input_dim,
            output_dims,
            slice,
        } => execute_splitter_forward(job, &input_dim, &output_dims, &slice, pool),
        ConnectorOpKind::Combiner {
            input_dim,
            output_dim,
            slice,
        } => execute_combiner_forward(job, &input_dim, &output_dim, &slice, pool),
    }
}

fn execute_splitter_forward(
    job: ConnectorOpJob,
    input_dim: &usize,
    output_dims: &[usize],
    slice: &ParamSlice,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    if job.inputs.len() != 1 {
        return JobResult::Failed(format!(
            "Splitter forward: expected 1 input, got {}",
            job.inputs.len()
        ));
    }
    let Some(params) = job.params.as_ref() else {
        return JobResult::Failed("Splitter forward: params required".into());
    };
    if output_dims.len() != 2 {
        return JobResult::Failed(format!(
            "Splitter forward: expected 2 output dims, got {}",
            output_dims.len()
        ));
    }

    let x = &job.inputs[0];
    let batch = x.rows();
    let n = *input_dim;
    let p_dim = output_dims[0];
    let q_dim = output_dims[1];

    if x.cols() != n {
        return JobResult::Failed(format!(
            "Splitter forward: input cols ({}) != input_dim ({})",
            x.cols(),
            n
        ));
    }

    let param_len = p_dim * n + q_dim * n + p_dim + q_dim;
    let params_data = params.read_range(slice.start, param_len);
    let x_data = x.read_range(0, batch * n);

    let wa_start = 0usize;
    let wb_start = wa_start + p_dim * n;
    let bias_a_start = wb_start + q_dim * n;
    let bias_b_start = bias_a_start + p_dim;

    let mut out_a_data = vec![0.0f32; batch * p_dim];
    let mut pre_a_data = vec![0.0f32; batch * p_dim];
    let mut out_b_data = vec![0.0f32; batch * q_dim];
    let mut pre_b_data = vec![0.0f32; batch * q_dim];

    for r in 0..batch {
        for j in 0..p_dim {
            let mut sum = params_data[bias_a_start + j];
            for k in 0..n {
                sum += x_data[k * batch + r] * params_data[wa_start + j * n + k];
            }
            pre_a_data[j * batch + r] = sum;
            out_a_data[j * batch + r] = sum.max(0.0);
        }
        for j in 0..q_dim {
            let mut sum = params_data[bias_b_start + j];
            for k in 0..n {
                sum += x_data[k * batch + r] * params_data[wb_start + j * n + k];
            }
            pre_b_data[j * batch + r] = sum;
            out_b_data[j * batch + r] = sum.max(0.0);
        }
    }

    let (out_a, pre_a, out_b, pre_b) = {
        let mut p = pool.lock().unwrap();
        let out_a = p.acquire(batch, p_dim);
        let pre_a = p.acquire(batch, p_dim);
        let out_b = p.acquire(batch, q_dim);
        let pre_b = p.acquire(batch, q_dim);
        (out_a, pre_a, out_b, pre_b)
    };

    out_a.write_range(0, &out_a_data);
    pre_a.write_range(0, &pre_a_data);
    out_b.write_range(0, &out_b_data);
    pre_b.write_range(0, &pre_b_data);

    JobResult::Buffers(vec![out_a, out_b, pre_a, pre_b])
}

fn execute_combiner_forward(
    job: ConnectorOpJob,
    input_dim: &usize,
    output_dim: &usize,
    slice: &ParamSlice,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    if job.inputs.len() != 2 {
        return JobResult::Failed(format!(
            "Combiner forward: expected 2 inputs, got {}",
            job.inputs.len()
        ));
    }
    let Some(params) = job.params.as_ref() else {
        return JobResult::Failed("Combiner forward: params required".into());
    };

    let a = &job.inputs[0];
    let b = &job.inputs[1];
    let batch = a.rows();
    let n = *input_dim;
    let m = *output_dim;

    if a.cols() != n || b.cols() != n {
        return JobResult::Failed(format!(
            "Combiner forward: input cols ({},{}) != input_dim ({})",
            a.cols(),
            b.cols(),
            n
        ));
    }

    let param_len = 2 * m * n + m;
    let params_data = params.read_range(slice.start, param_len);
    let a_data = a.read_range(0, batch * n);
    let b_data = b.read_range(0, batch * n);

    let wa_start = 0usize;
    let wb_start = wa_start + m * n;
    let bias_start = wb_start + m * n;

    let mut out_data = vec![0.0f32; batch * m];
    let mut pre_data = vec![0.0f32; batch * m];

    for r in 0..batch {
        for j in 0..m {
            let mut sum = params_data[bias_start + j];
            for k in 0..n {
                sum += a_data[k * batch + r] * params_data[wa_start + j * n + k];
                sum += b_data[k * batch + r] * params_data[wb_start + j * n + k];
            }
            pre_data[j * batch + r] = sum;
            out_data[j * batch + r] = sum.max(0.0);
        }
    }

    let (out, pre) = {
        let mut p = pool.lock().unwrap();
        let out = p.acquire(batch, m);
        let pre = p.acquire(batch, m);
        (out, pre)
    };

    out.write_range(0, &out_data);
    pre.write_range(0, &pre_data);

    JobResult::Buffers(vec![out, pre])
}

/// Backward-диспетчер коннекторов.
///
/// ВАЖНО: match по **клонированному** `kind` — см. комментарий к
/// `execute_connector_forward`.
fn execute_connector_backward(
    job: ConnectorOpJob,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    let kind = job.kind.clone();
    match kind {
        ConnectorOpKind::Splitter {
            input_dim,
            output_dims,
            slice,
        } => execute_splitter_backward(job, &input_dim, &output_dims, &slice, pool),
        ConnectorOpKind::Combiner {
            input_dim,
            output_dim,
            slice,
        } => execute_combiner_backward(job, &input_dim, &output_dim, &slice, pool),
    }
}

fn execute_splitter_backward(
    job: ConnectorOpJob,
    input_dim: &usize,
    output_dims: &[usize],
    slice: &ParamSlice,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    if job.inputs.len() != 2 || job.saved.len() != 3 {
        return JobResult::Failed(format!(
            "Splitter backward: expected inputs=2, saved=3, got inputs={}, saved={}",
            job.inputs.len(),
            job.saved.len()
        ));
    }
    let Some(params) = job.params.as_ref() else {
        return JobResult::Failed("Splitter backward: params required".into());
    };
    let Some(grad_params) = job.grad_params.as_ref() else {
        return JobResult::Failed("Splitter backward: grad_params required".into());
    };

    let delta_a = &job.inputs[0];
    let delta_b = &job.inputs[1];
    let x = &job.saved[0];
    let pre_a = &job.saved[1];
    let pre_b = &job.saved[2];

    let batch = delta_a.rows();
    let n = *input_dim;
    let p_dim = output_dims[0];
    let q_dim = output_dims[1];

    let param_len = p_dim * n + q_dim * n + p_dim + q_dim;

    let x_data = x.read_range(0, batch * n);
    let pre_a_data = pre_a.read_range(0, batch * p_dim);
    let pre_b_data = pre_b.read_range(0, batch * q_dim);
    let delta_a_data = delta_a.read_range(0, batch * p_dim);
    let delta_b_data = delta_b.read_range(0, batch * q_dim);
    let params_data = params.read_range(slice.start, param_len);

    let wa_start = 0usize;
    let wb_start = wa_start + p_dim * n;
    let bias_a_start = wb_start + q_dim * n;
    let bias_b_start = bias_a_start + p_dim;

    // dx[c, r] = Σ_j d_pre_a[j, r] * wa[j, c] + Σ_j d_pre_b[j, r] * wb[j, c]
    let mut dx_data = vec![0.0f32; batch * n];
    for c in 0..n {
        for r in 0..batch {
            let mut sum = 0.0f32;
            for j in 0..p_dim {
                let d_pre = if pre_a_data[j * batch + r] > 0.0 {
                    delta_a_data[j * batch + r]
                } else {
                    0.0
                };
                sum += d_pre * params_data[wa_start + j * n + c];
            }
            for j in 0..q_dim {
                let d_pre = if pre_b_data[j * batch + r] > 0.0 {
                    delta_b_data[j * batch + r]
                } else {
                    0.0
                };
                sum += d_pre * params_data[wb_start + j * n + c];
            }
            dx_data[c * batch + r] = sum;
        }
    }

    // grad_wa[j, k] = Σ_r d_pre_a[j, r] * x[k, r]
    let mut grad_wa = vec![0.0f32; p_dim * n];
    for j in 0..p_dim {
        for k in 0..n {
            let mut sum = 0.0f32;
            for r in 0..batch {
                let d_pre = if pre_a_data[j * batch + r] > 0.0 {
                    delta_a_data[j * batch + r]
                } else {
                    0.0
                };
                sum += d_pre * x_data[k * batch + r];
            }
            grad_wa[j * n + k] = sum;
        }
    }

    // grad_wb[j, k] = Σ_r d_pre_b[j, r] * x[k, r]
    let mut grad_wb = vec![0.0f32; q_dim * n];
    for j in 0..q_dim {
        for k in 0..n {
            let mut sum = 0.0f32;
            for r in 0..batch {
                let d_pre = if pre_b_data[j * batch + r] > 0.0 {
                    delta_b_data[j * batch + r]
                } else {
                    0.0
                };
                sum += d_pre * x_data[k * batch + r];
            }
            grad_wb[j * n + k] = sum;
        }
    }

    // grad_bias_a[j] = Σ_r d_pre_a[j, r]
    let mut grad_bias_a = vec![0.0f32; p_dim];
    for j in 0..p_dim {
        let mut sum = 0.0f32;
        for r in 0..batch {
            let d_pre = if pre_a_data[j * batch + r] > 0.0 {
                delta_a_data[j * batch + r]
            } else {
                0.0
            };
            sum += d_pre;
        }
        grad_bias_a[j] = sum;
    }

    // grad_bias_b[j] = Σ_r d_pre_b[j, r]
    let mut grad_bias_b = vec![0.0f32; q_dim];
    for j in 0..q_dim {
        let mut sum = 0.0f32;
        for r in 0..batch {
            let d_pre = if pre_b_data[j * batch + r] > 0.0 {
                delta_b_data[j * batch + r]
            } else {
                0.0
            };
            sum += d_pre;
        }
        grad_bias_b[j] = sum;
    }

    // Собираем все градиенты в один блок и пишем одним write_range.
    let mut grad_params_local = vec![0.0f32; param_len];
    grad_params_local[wa_start..wa_start + p_dim * n].copy_from_slice(&grad_wa);
    grad_params_local[wb_start..wb_start + q_dim * n].copy_from_slice(&grad_wb);
    grad_params_local[bias_a_start..bias_a_start + p_dim].copy_from_slice(&grad_bias_a);
    grad_params_local[bias_b_start..bias_b_start + q_dim].copy_from_slice(&grad_bias_b);

    grad_params.write_range(slice.start, &grad_params_local);

    let dx = {
        let mut p = pool.lock().unwrap();
        p.acquire(batch, n)
    };
    dx.write_range(0, &dx_data);

    JobResult::Buffers(vec![dx])
}

fn execute_combiner_backward(
    job: ConnectorOpJob,
    input_dim: &usize,
    output_dim: &usize,
    slice: &ParamSlice,
    pool: Arc<Mutex<TempMatrixPool>>,
) -> JobResult {
    if job.inputs.len() != 1 || job.saved.len() != 3 {
        return JobResult::Failed(format!(
            "Combiner backward: expected inputs=1, saved=3, got inputs={}, saved={}",
            job.inputs.len(),
            job.saved.len()
        ));
    }
    let Some(params) = job.params.as_ref() else {
        return JobResult::Failed("Combiner backward: params required".into());
    };
    let Some(grad_params) = job.grad_params.as_ref() else {
        return JobResult::Failed("Combiner backward: grad_params required".into());
    };

    let delta = &job.inputs[0];
    let a = &job.saved[0];
    let b = &job.saved[1];
    let pre = &job.saved[2];

    let batch = delta.rows();
    let n = *input_dim;
    let m = *output_dim;

    let param_len = 2 * m * n + m;

    let a_data = a.read_range(0, batch * n);
    let b_data = b.read_range(0, batch * n);
    let pre_data = pre.read_range(0, batch * m);
    let delta_data = delta.read_range(0, batch * m);
    let params_data = params.read_range(slice.start, param_len);

    let wa_start = 0usize;
    let wb_start = wa_start + m * n;
    let bias_start = wb_start + m * n;

    // d_pre[j, r] = (pre[j, r] > 0) ? delta[j, r] : 0
    // da[c, r] = Σ_j d_pre[j, r] * wa[j, c]
    // db[c, r] = Σ_j d_pre[j, r] * wb[j, c]
    let mut da_data = vec![0.0f32; batch * n];
    let mut db_data = vec![0.0f32; batch * n];
    for c in 0..n {
        for r in 0..batch {
            let mut sum_a = 0.0f32;
            let mut sum_b = 0.0f32;
            for j in 0..m {
                let d_pre = if pre_data[j * batch + r] > 0.0 {
                    delta_data[j * batch + r]
                } else {
                    0.0
                };
                sum_a += d_pre * params_data[wa_start + j * n + c];
                sum_b += d_pre * params_data[wb_start + j * n + c];
            }
            da_data[c * batch + r] = sum_a;
            db_data[c * batch + r] = sum_b;
        }
    }

    // grad_wa[j, k] = Σ_r d_pre[j, r] * a[k, r]
    let mut grad_wa = vec![0.0f32; m * n];
    for j in 0..m {
        for k in 0..n {
            let mut sum = 0.0f32;
            for r in 0..batch {
                let d_pre = if pre_data[j * batch + r] > 0.0 {
                    delta_data[j * batch + r]
                } else {
                    0.0
                };
                sum += d_pre * a_data[k * batch + r];
            }
            grad_wa[j * n + k] = sum;
        }
    }

    // grad_wb[j, k] = Σ_r d_pre[j, r] * b[k, r]
    let mut grad_wb = vec![0.0f32; m * n];
    for j in 0..m {
        for k in 0..n {
            let mut sum = 0.0f32;
            for r in 0..batch {
                let d_pre = if pre_data[j * batch + r] > 0.0 {
                    delta_data[j * batch + r]
                } else {
                    0.0
                };
                sum += d_pre * b_data[k * batch + r];
            }
            grad_wb[j * n + k] = sum;
        }
    }

    // grad_bias[j] = Σ_r d_pre[j, r]
    let mut grad_bias = vec![0.0f32; m];
    for j in 0..m {
        let mut sum = 0.0f32;
        for r in 0..batch {
            let d_pre = if pre_data[j * batch + r] > 0.0 {
                delta_data[j * batch + r]
            } else {
                0.0
            };
            sum += d_pre;
        }
        grad_bias[j] = sum;
    }

    let mut grad_params_local = vec![0.0f32; param_len];
    grad_params_local[wa_start..wa_start + m * n].copy_from_slice(&grad_wa);
    grad_params_local[wb_start..wb_start + m * n].copy_from_slice(&grad_wb);
    grad_params_local[bias_start..bias_start + m].copy_from_slice(&grad_bias);

    grad_params.write_range(slice.start, &grad_params_local);

    let da = {
        let mut p = pool.lock().unwrap();
        p.acquire(batch, n)
    };
    let db = {
        let mut p = pool.lock().unwrap();
        p.acquire(batch, n)
    };
    da.write_range(0, &da_data);
    db.write_range(0, &db_data);

    JobResult::Buffers(vec![da, db])
}

// ============================================================================
// Внутренние диспетчеры одного слоя
// ============================================================================

fn dispatch_forward_one_layer(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    pool: &Arc<Mutex<TempMatrixPool>>,
) -> BufferedContext {
    let mut pool_guard = pool.lock().unwrap();
    let l: &dyn UniversalLayer = layer.as_ref();

    macro_rules! fwd {
        ($ty:ty, $getter:ident) => {
            if let Some(x) = l.$getter() {
                return <$ty as UniversalLayerBuffered>::forward_buffered(
                    x, input, output, params, slice, &mut *pool_guard,
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

    unreachable!(
        "CpuOperatorV2::forward: layer {:?} has no buffered forward",
        std::any::type_name_of_val(l)
    );
}

fn dispatch_backward_one_layer(
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

    unreachable!(
        "CpuOperatorV2::backward: layer {:?} has no buffered backward",
        std::any::type_name_of_val(l)
    );
}