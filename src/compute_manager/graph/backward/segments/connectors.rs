// src/compute_manager/graph/backward/segments/connectors.rs

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    /// Обработка обратного прохода SplitterConnector.
    /// Возвращает два градиента, соответствующие входам разветвления.
    pub(crate) fn process_splitter_connector_backward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        stream_gradients: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        assert_eq!(stream_gradients.len(), 2);
        let delta_a = stream_gradients[0].clone();
        let delta_b = stream_gradients[1].clone();

        let in_a = {
            let handle = pool.acquire(delta_a.rows(), delta_a.cols());
            let mut mem = self.memory_executor.write().unwrap();
            mem.copy_cpu_buffer(delta_a.id(), handle.id());
            handle
        };
        let in_b = {
            let handle = pool.acquire(delta_b.rows(), delta_b.cols());
            let mut mem = self.memory_executor.write().unwrap();
            mem.copy_cpu_buffer(delta_b.id(), handle.id());
            handle
        };

        pool.release(delta_a);
        pool.release(delta_b);

        vec![in_a, in_b]
    }

    /// Обработка обратного прохода CombinerConnector.
    /// Ничего не делает, так как коннектор не выполняет вычислений.
    pub(crate) fn process_combiner_connector_backward_buffered(
        &mut self,
        _pool: &mut TempMatrixPool,
        _stream_gradients: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        // Пусто – коннектор не имеет параметров и не меняет градиенты.
        _stream_gradients
    }

    /// Обработка обратного прохода обучаемого Splitter.
    /// Принимает градиенты двух выходных ветвей, возвращает градиент по входу.
    pub(crate) fn process_splitter_backward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dim: usize,
        output_dims: &[usize],
        slice: ParamSlice,
        chunked_ctxs: &ChunkedContexts,
        params_handle: &MatrixBufferHandle,
        grad_params_handle: &MatrixBufferHandle,
        stream_gradients: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        let ctx = chunked_ctxs
            .first()
            .and_then(|chunk| chunk.first())
            .cloned()
            .expect("Missing Splitter context");
        let (x_handle, pre_a_handle, pre_b_handle) = match ctx {
            DynamicContext::Buffered(BufferedContext::Splitter {
                input,
                pre_a,
                pre_b,
            }) => (input, pre_a, pre_b),
            _ => panic!("Expected Splitter Buffered context"),
        };

        let da_handle = stream_gradients[0].clone();
        let db_handle = stream_gradients[1].clone();

        let batch = x_handle.rows();
        let n = input_dim;
        let p = output_dims[0];
        let q = output_dims[1];

        let dx_handle = pool.acquire(batch, n);

        let ids = [
            x_handle.id(),
            da_handle.id(),
            db_handle.id(),
            pre_a_handle.id(),
            pre_b_handle.id(),
            params_handle.id(),
            grad_params_handle.id(),
            dx_handle.id(),
        ];

        x_handle.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let da: &[f32] = &*second[0];
            let (third, rest) = rest.split_at_mut(1);
            let db: &[f32] = &*third[0];
            let (fourth, rest) = rest.split_at_mut(1);
            let pre_a: &[f32] = &*fourth[0];
            let (fifth, rest) = rest.split_at_mut(1);
            let pre_b: &[f32] = &*fifth[0];
            let (sixth, rest) = rest.split_at_mut(1);
            let params_ref: &[f32] = &*sixth[0];
            let (seventh, rest) = rest.split_at_mut(1);
            let gp: &mut [f32] = &mut *seventh[0];
            let dx_out: &mut [f32] = &mut *rest[0];

            let wa_start = slice.start;
            let wa_len = p * n;
            let wb_start = wa_start + wa_len;
            let wb_len = q * n;
            let bias_a_start = wb_start + wb_len;
            let bias_b_start = bias_a_start + p;

            for r in 0..batch {
                for c in 0..n {
                    let mut sum = 0.0;
                    for k in 0..p {
                        let d_pre_a_val = if pre_a[k * batch + r] > 0.0 { da[k * batch + r] } else { 0.0 };
                        sum += d_pre_a_val * params_ref[wa_start + k * n + c];
                    }
                    for k in 0..q {
                        let d_pre_b_val = if pre_b[k * batch + r] > 0.0 { db[k * batch + r] } else { 0.0 };
                        sum += d_pre_b_val * params_ref[wb_start + k * n + c];
                    }
                    dx_out[c * batch + r] = sum;
                }
            }

            for out_idx in 0..p {
                for in_idx in 0..n {
                    let mut sum = 0.0;
                    for r in 0..batch {
                        let d_pre_a_val = if pre_a[out_idx * batch + r] > 0.0 { da[out_idx * batch + r] } else { 0.0 };
                        sum += d_pre_a_val * x[in_idx * batch + r];
                    }
                    gp[wa_start + out_idx * n + in_idx] = sum;
                }
            }
            for out_idx in 0..q {
                for in_idx in 0..n {
                    let mut sum = 0.0;
                    for r in 0..batch {
                        let d_pre_b_val = if pre_b[out_idx * batch + r] > 0.0 { db[out_idx * batch + r] } else { 0.0 };
                        sum += d_pre_b_val * x[in_idx * batch + r];
                    }
                    gp[wb_start + out_idx * n + in_idx] = sum;
                }
            }

            for c in 0..p {
                let mut sum = 0.0;
                for r in 0..batch {
                    let d_pre_a_val = if pre_a[c * batch + r] > 0.0 { da[c * batch + r] } else { 0.0 };
                    sum += d_pre_a_val;
                }
                gp[bias_a_start + c] = sum;
            }
            for c in 0..q {
                let mut sum = 0.0;
                for r in 0..batch {
                    let d_pre_b_val = if pre_b[c * batch + r] > 0.0 { db[c * batch + r] } else { 0.0 };
                    sum += d_pre_b_val;
                }
                gp[bias_b_start + c] = sum;
            }
        });

        pool.release(da_handle);
        pool.release(db_handle);
        pool.release(x_handle);
        pool.release(pre_a_handle);
        pool.release(pre_b_handle);

        vec![dx_handle]
    }

    /// Обработка обратного прохода обучаемого Combiner.
    /// Принимает градиент по выходу, возвращает градиенты по двум входам.
    pub(crate) fn process_combiner_backward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dim: usize,
        output_dim: usize,
        slice: ParamSlice,
        chunked_ctxs: &ChunkedContexts,
        params_handle: &MatrixBufferHandle,
        grad_params_handle: &MatrixBufferHandle,
        stream_gradients: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        let ctx = chunked_ctxs
            .first()
            .and_then(|chunk| chunk.first())
            .cloned()
            .expect("Missing Combiner context");
        let (a_handle, b_handle, pre_handle) = match ctx {
            DynamicContext::Buffered(BufferedContext::Combiner {
                input_a,
                input_b,
                pre_act,
            }) => (input_a, input_b, pre_act),
            _ => panic!("Expected Combiner Buffered context"),
        };

        let dout_handle = stream_gradients[0].clone();

        let batch = a_handle.rows();
        let n = input_dim;
        let m = output_dim;

        let da_handle = pool.acquire(batch, n);
        let db_handle = pool.acquire(batch, n);

        let ids = [
            a_handle.id(),
            b_handle.id(),
            pre_handle.id(),
            dout_handle.id(),
            params_handle.id(),
            grad_params_handle.id(),
            da_handle.id(),
            db_handle.id(),
        ];

        a_handle.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let a: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let b: &[f32] = &*second[0];
            let (third, rest) = rest.split_at_mut(1);
            let pre: &[f32] = &*third[0];
            let (fourth, rest) = rest.split_at_mut(1);
            let dout: &[f32] = &*fourth[0];
            let (fifth, rest) = rest.split_at_mut(1);
            let params_ref: &[f32] = &*fifth[0];
            let (sixth, rest) = rest.split_at_mut(1);
            let gp: &mut [f32] = &mut *sixth[0];
            let (seventh, rest) = rest.split_at_mut(1);
            let da_out: &mut [f32] = &mut *seventh[0];
            let db_out: &mut [f32] = &mut *rest[0];

            let wa_start = slice.start;
            let wa_len = m * n;
            let wb_start = wa_start + wa_len;
            let wb_len = m * n;
            let bias_start = wb_start + wb_len;

            for r in 0..batch {
                for c in 0..n {
                    let mut sum_a = 0.0;
                    let mut sum_b = 0.0;
                    for k in 0..m {
                        let d_pre = if pre[k * batch + r] > 0.0 { dout[k * batch + r] } else { 0.0 };
                        sum_a += d_pre * params_ref[wa_start + k * n + c];
                        sum_b += d_pre * params_ref[wb_start + k * n + c];
                    }
                    da_out[c * batch + r] = sum_a;
                    db_out[c * batch + r] = sum_b;
                }
            }

            for out_idx in 0..m {
                for in_idx in 0..n {
                    let mut sum = 0.0;
                    for r in 0..batch {
                        let d_pre = if pre[out_idx * batch + r] > 0.0 { dout[out_idx * batch + r] } else { 0.0 };
                        sum += d_pre * a[in_idx * batch + r];
                    }
                    gp[wa_start + out_idx * n + in_idx] = sum;
                }
            }
            for out_idx in 0..m {
                for in_idx in 0..n {
                    let mut sum = 0.0;
                    for r in 0..batch {
                        let d_pre = if pre[out_idx * batch + r] > 0.0 { dout[out_idx * batch + r] } else { 0.0 };
                        sum += d_pre * b[in_idx * batch + r];
                    }
                    gp[wb_start + out_idx * n + in_idx] = sum;
                }
            }

            for c in 0..m {
                let mut sum = 0.0;
                for r in 0..batch {
                    let d_pre = if pre[c * batch + r] > 0.0 { dout[c * batch + r] } else { 0.0 };
                    sum += d_pre;
                }
                gp[bias_start + c] = sum;
            }
        });

        pool.release(dout_handle);
        pool.release(a_handle);
        pool.release(b_handle);
        pool.release(pre_handle);

        vec![da_handle, db_handle]
    }
}