// src/compute_manager/gpu/compute/splitter_combiner.rs

use faer::Mat;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Старые Mat-версии (оставлены для обратной совместимости)
    // ===================================================================

    /// Прямой проход Splitter: возвращает (a, b, pre_a, pre_b)
    pub fn run_splitter_forward(
        &self,
        x: &Mat<f32>,
        wa: &Mat<f32>,
        bias_a: &[f32],
        wb: &Mat<f32>,
        bias_b: &[f32],
    ) -> (Mat<f32>, Mat<f32>, Mat<f32>, Mat<f32>) {
        let pre_a = self.run_linear_forward(x, wa, bias_a);
        let a = self.run_relu_forward(&pre_a);
        let pre_b = self.run_linear_forward(x, wb, bias_b);
        let b = self.run_relu_forward(&pre_b);
        (a, b, pre_a, pre_b)
    }

    /// Обратный проход Splitter: возвращает (dx, градиенты параметров)
    pub fn run_splitter_backward(
        &self,
        x: &Mat<f32>,
        da: &Mat<f32>,
        db: &Mat<f32>,
        pre_a: &Mat<f32>,
        pre_b: &Mat<f32>,
        wa: &Mat<f32>,
        wb: &Mat<f32>,
    ) -> (Mat<f32>, Vec<f32>) {
        let d_pre_a = self.run_relu_backward(pre_a, da);
        let d_pre_b = self.run_relu_backward(pre_b, db);

        let dx_a = self.run_mat_mul(&d_pre_a, wa);
        let dx_b = self.run_mat_mul(&d_pre_b, wb);
        let mut dx = Mat::zeros(x.nrows(), x.ncols());
        for r in 0..x.nrows() {
            for c in 0..x.ncols() {
                dx[(r, c)] = dx_a[(r, c)] + dx_b[(r, c)];
            }
        }

        let d_pre_a_t = transpose_mat(&d_pre_a);
        let d_wa = self.run_mat_mul(&d_pre_a_t, x);
        let d_pre_b_t = transpose_mat(&d_pre_b);
        let d_wb = self.run_mat_mul(&d_pre_b_t, x);

        let d_bias_a = self.run_reduce_sum_cols(&d_pre_a);
        let d_bias_b = self.run_reduce_sum_cols(&d_pre_b);

        let p = wa.nrows();
        let q = wb.nrows();
        let n = x.ncols();
        let mut grad = Vec::with_capacity(p * n + q * n + p + q);
        for r in 0..p {
            for c in 0..n { grad.push(d_wa[(r, c)]); }
        }
        for r in 0..q {
            for c in 0..n { grad.push(d_wb[(r, c)]); }
        }
        grad.extend_from_slice(&d_bias_a);
        grad.extend_from_slice(&d_bias_b);
        (dx, grad)
    }

    /// Прямой проход Combiner: возвращает out (и pre для контекста)
    pub fn run_combiner_forward(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        wa: &Mat<f32>,
        wb: &Mat<f32>,
        bias: &[f32],
    ) -> (Mat<f32>, Mat<f32>) {
        let wa_t = transpose_mat(wa);
        let part_a = self.run_mat_mul(a, &wa_t);
        let wb_t = transpose_mat(wb);
        let part_b = self.run_mat_mul(b, &wb_t);

        let batch = a.nrows();
        let out_dim = wa.nrows();
        let mut pre = Mat::zeros(batch, out_dim);
        for i in 0..batch {
            for j in 0..out_dim {
                pre[(i, j)] = part_a[(i, j)] + part_b[(i, j)] + bias[j];
            }
        }
        let out = self.run_relu_forward(&pre);
        (out, pre)
    }

    /// Обратный проход Combiner: возвращает (da, db, градиенты параметров)
    pub fn run_combiner_backward(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        d_out: &Mat<f32>,
        pre: &Mat<f32>,
        wa: &Mat<f32>,
        wb: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        let d_pre = self.run_relu_backward(pre, d_out);
        let da = self.run_mat_mul(&d_pre, wa);
        let db = self.run_mat_mul(&d_pre, wb);

        let d_pre_t = transpose_mat(&d_pre);
        let d_wa = self.run_mat_mul(&d_pre_t, a);
        let d_wb = self.run_mat_mul(&d_pre_t, b);

        let d_bias = self.run_reduce_sum_cols(&d_pre);

        let m = wa.nrows();
        let n = a.ncols();
        let mut grad = Vec::with_capacity(2 * m * n + m);
        for r in 0..m {
            for c in 0..n { grad.push(d_wa[(r, c)]); }
        }
        for r in 0..m {
            for c in 0..n { grad.push(d_wb[(r, c)]); }
        }
        grad.extend_from_slice(&d_bias);
        (da, db, grad)
    }

    // ===================================================================
    // НОВЫЕ BUFFERED-ВЕРСИИ ДЛЯ MatrixBuffer (БЕЗ faer::Mat)
    // ===================================================================

    /// Прямой проход Splitter на GPU с использованием MatrixBuffer.
    pub fn run_splitter_forward_buffered(
        &self,
        x: &MatrixBuffer,
        wa: &MatrixBuffer,
        bias_a: &[f32],
        wb: &MatrixBuffer,
        bias_b: &[f32],
    ) -> (MatrixBuffer, MatrixBuffer, MatrixBuffer, MatrixBuffer) {
        assert!(x.is_gpu() && wa.is_gpu() && wb.is_gpu(), "Buffers must be GPU");
        let pre_a = self.run_linear_forward_buffered(x, wa, bias_a);
        let a = self.run_relu_forward_buffered(&pre_a);
        let pre_b = self.run_linear_forward_buffered(x, wb, bias_b);
        let b = self.run_relu_forward_buffered(&pre_b);
        (a, b, pre_a, pre_b)
    }

    /// Обратный проход Splitter на GPU с использованием MatrixBuffer.
    pub fn run_splitter_backward_buffered(
        &self,
        x: &MatrixBuffer,
        da: &MatrixBuffer,
        db: &MatrixBuffer,
        pre_a: &MatrixBuffer,
        pre_b: &MatrixBuffer,
        wa: &MatrixBuffer,
        wb: &MatrixBuffer,
    ) -> (MatrixBuffer, Vec<f32>) {
        assert!(
            x.is_gpu() && da.is_gpu() && db.is_gpu() && pre_a.is_gpu() && pre_b.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Buffers must be GPU"
        );

        let d_pre_a = self.run_relu_backward_buffered(pre_a, da);
        let d_pre_b = self.run_relu_backward_buffered(pre_b, db);

        // dx = d_pre_a * wa + d_pre_b * wb
        let dx_a = self.run_mat_mul_buffered(&d_pre_a, wa);
        let dx_b = self.run_mat_mul_buffered(&d_pre_b, wb);

        // Складываем два GPU-буфера через CPU-стагинг
        let dx_a_vec = self.download_gpu_matrix_to_vec(&dx_a);
        let dx_b_vec = self.download_gpu_matrix_to_vec(&dx_b);
        let batch = x.rows();
        let n = x.cols();
        let mut dx_vec = vec![0.0f32; batch * n];
        for i in 0..dx_vec.len() {
            dx_vec[i] = dx_a_vec[i] + dx_b_vec[i];
        }
        let dx = self.upload_vec_to_gpu_buffer(&dx_vec, batch, n);

        // Градиенты весов и смещений через Linear backward
        let (_, d_wa, d_bias_a) = self.run_linear_backward_buffered(x, wa, &d_pre_a);
        let (_, d_wb, d_bias_b) = self.run_linear_backward_buffered(x, wb, &d_pre_b);

        let p = wa.rows();
        let q = wb.rows();
        // Собираем градиенты параметров в том же порядке, что и раньше
        let mut grad = Vec::with_capacity(p * n + q * n + p + q);
        let d_wa_vec = self.download_gpu_matrix_to_vec(&d_wa);
        grad.extend_from_slice(&d_wa_vec);
        let d_wb_vec = self.download_gpu_matrix_to_vec(&d_wb);
        grad.extend_from_slice(&d_wb_vec);
        grad.extend_from_slice(&d_bias_a);
        grad.extend_from_slice(&d_bias_b);

        (dx, grad)
    }

    /// Прямой проход Combiner на GPU с использованием MatrixBuffer.
    pub fn run_combiner_forward_buffered(
        &self,
        a: &MatrixBuffer,
        b: &MatrixBuffer,
        wa: &MatrixBuffer,
        wb: &MatrixBuffer,
        bias: &[f32],
    ) -> (MatrixBuffer, MatrixBuffer) {
        assert!(
            a.is_gpu() && b.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Buffers must be GPU"
        );

        let batch = a.rows();
        let out_dim = wa.rows();
        let zero_bias = vec![0.0f32; out_dim];

        let part_a = self.run_linear_forward_buffered(a, wa, &zero_bias);
        let part_b = self.run_linear_forward_buffered(b, wb, &zero_bias);

        // Складываем part_a, part_b и добавляем bias через CPU-стагинг
        let part_a_vec = self.download_gpu_matrix_to_vec(&part_a);
        let part_b_vec = self.download_gpu_matrix_to_vec(&part_b);
        let mut pre_vec = vec![0.0f32; batch * out_dim];
        for c in 0..out_dim {
            for r in 0..batch {
                pre_vec[c * batch + r] = part_a_vec[c * batch + r] + part_b_vec[c * batch + r] + bias[c];
            }
        }
        let pre = self.upload_vec_to_gpu_buffer(&pre_vec, batch, out_dim);
        let out = self.run_relu_forward_buffered(&pre);

        (out, pre)
    }

    /// Обратный проход Combiner на GPU с использованием MatrixBuffer.
    pub fn run_combiner_backward_buffered(
        &self,
        a: &MatrixBuffer,
        b: &MatrixBuffer,
        d_out: &MatrixBuffer,
        pre: &MatrixBuffer,
        wa: &MatrixBuffer,
        wb: &MatrixBuffer,
    ) -> (MatrixBuffer, MatrixBuffer, Vec<f32>) {
        assert!(
            a.is_gpu() && b.is_gpu() && d_out.is_gpu() && pre.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Buffers must be GPU"
        );

        let d_pre = self.run_relu_backward_buffered(pre, d_out);

        // da = d_pre * wa, db = d_pre * wb
        let da = self.run_mat_mul_buffered(&d_pre, wa);
        let db = self.run_mat_mul_buffered(&d_pre, wb);

        // Градиенты весов через Linear backward (используем нулевые bias)
        let batch = a.rows();
        let out_dim = wa.rows();
        let zero_bias = vec![0.0f32; out_dim];
        let (_, d_wa, _d_bias_a) = self.run_linear_backward_buffered(a, wa, &d_pre);
        let (_, d_wb, _d_bias_b) = self.run_linear_backward_buffered(b, wb, &d_pre);

        // d_bias = сумма по строкам d_pre
        let d_pre_vec = self.download_gpu_matrix_to_vec(&d_pre);
        let d_bias: Vec<f32> = (0..out_dim)
            .map(|c| (0..batch).map(|r| d_pre_vec[c * batch + r]).sum())
            .collect();

        let n = a.cols();
        let mut grad = Vec::with_capacity(2 * out_dim * n + out_dim);
        let d_wa_vec = self.download_gpu_matrix_to_vec(&d_wa);
        grad.extend_from_slice(&d_wa_vec);
        let d_wb_vec = self.download_gpu_matrix_to_vec(&d_wb);
        grad.extend_from_slice(&d_wb_vec);
        grad.extend_from_slice(&d_bias);

        (da, db, grad)
    }

    // ===================================================================
    // НОВЫЕ HANDLE-ВЕРСИИ ДЛЯ MatrixBufferHandle
    // ===================================================================

    /// Прямой проход Splitter на GPU с использованием MatrixBufferHandle.
    /// Все входы и выходы — GPU-дескрипторы.
    pub fn run_splitter_forward_buffered_handle(
        &self,
        x: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        bias_a: &[f32],
        wb: &MatrixBufferHandle,
        bias_b: &[f32],
        out_a: &MatrixBufferHandle,
        pre_a: &MatrixBufferHandle,
        out_b: &MatrixBufferHandle,
        pre_b: &MatrixBufferHandle,
    ) {
        assert!(x.is_gpu() && wa.is_gpu() && wb.is_gpu(), "Input handles must be GPU");
        assert!(out_a.is_gpu() && pre_a.is_gpu() && out_b.is_gpu() && pre_b.is_gpu(), "Output handles must be GPU");

        let batch = x.rows();
        let n = x.cols();
        let p = wa.rows();
        let q = wb.rows();
        assert_eq!(out_a.rows(), batch);
        assert_eq!(out_a.cols(), p);
        assert_eq!(pre_a.rows(), batch);
        assert_eq!(pre_a.cols(), p);
        assert_eq!(out_b.rows(), batch);
        assert_eq!(out_b.cols(), q);
        assert_eq!(pre_b.rows(), batch);
        assert_eq!(pre_b.cols(), q);

        let in_buf = self.get_gpu_subbuffer_from_handle(x);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let out_a_buf = self.get_gpu_subbuffer_from_handle(out_a);
        let pre_a_buf = self.get_gpu_subbuffer_from_handle(pre_a);
        let out_b_buf = self.get_gpu_subbuffer_from_handle(out_b);
        let pre_b_buf = self.get_gpu_subbuffer_from_handle(pre_b);

        // Загружаем bias как временные буферы
        let (bias_a_buf, bias_a_raw) = self.upload_to_temp_buffer(bias_a);
        let (bias_b_buf, bias_b_raw) = self.upload_to_temp_buffer(bias_b);

        let pipeline = self.pipeline_cache.splitter_fwd.clone();
        let push = [batch as u32, n as u32, p as u32, q as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, in_buf),
                (1, wa_buf),
                (2, bias_a_buf.clone()),
                (3, wb_buf),
                (4, bias_b_buf.clone()),
                (5, out_a_buf),
                (6, pre_a_buf),
                (7, out_b_buf),
                (8, pre_b_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        self.release_temp_buffer(bias_a_buf, bias_a_raw);
        self.release_temp_buffer(bias_b_buf, bias_b_raw);
    }

    /// Обратный проход Splitter на GPU с использованием MatrixBufferHandle.
    /// Входные градиенты и данные — GPU-дескрипторы. Градиенты параметров
    /// возвращаются как Vec<f32> (CPU), как в старых версиях.
    pub fn run_splitter_backward_buffered_handle(
        &self,
        x: &MatrixBufferHandle,
        da: &MatrixBufferHandle,
        db: &MatrixBufferHandle,
        pre_a: &MatrixBufferHandle,
        pre_b: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        wb: &MatrixBufferHandle,
        dx: &MatrixBufferHandle,
        d_wa: &MatrixBufferHandle,
        d_bias_a: &MatrixBufferHandle,
        d_wb: &MatrixBufferHandle,
        d_bias_b: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(x.is_gpu() && da.is_gpu() && db.is_gpu() && pre_a.is_gpu() && pre_b.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Input handles must be GPU");
        assert!(dx.is_gpu() && d_wa.is_gpu() && d_bias_a.is_gpu() && d_wb.is_gpu() && d_bias_b.is_gpu(),
            "Output handles must be GPU");

        let batch = x.rows();
        let n = x.cols();
        let p = wa.rows();
        let q = wb.rows();

        // Проверяем размеры
        assert_eq!(dx.rows(), batch);
        assert_eq!(dx.cols(), n);
        assert_eq!(d_wa.rows(), p);
        assert_eq!(d_wa.cols(), n);
        assert_eq!(d_bias_a.rows(), 1);
        assert_eq!(d_bias_a.cols(), p);
        assert_eq!(d_wb.rows(), q);
        assert_eq!(d_wb.cols(), n);
        assert_eq!(d_bias_b.rows(), 1);
        assert_eq!(d_bias_b.cols(), q);

        let x_buf = self.get_gpu_subbuffer_from_handle(x);
        let da_buf = self.get_gpu_subbuffer_from_handle(da);
        let db_buf = self.get_gpu_subbuffer_from_handle(db);
        let pre_a_buf = self.get_gpu_subbuffer_from_handle(pre_a);
        let pre_b_buf = self.get_gpu_subbuffer_from_handle(pre_b);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let dx_buf = self.get_gpu_subbuffer_from_handle(dx);
        let d_wa_buf = self.get_gpu_subbuffer_from_handle(d_wa);
        let d_bias_a_buf = self.get_gpu_subbuffer_from_handle(d_bias_a);
        let d_wb_buf = self.get_gpu_subbuffer_from_handle(d_wb);
        let d_bias_b_buf = self.get_gpu_subbuffer_from_handle(d_bias_b);

        let pipeline = self.pipeline_cache.splitter_bwd.clone();
        let push = [batch as u32, n as u32, p as u32, q as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, x_buf),
                (1, da_buf),
                (2, db_buf),
                (3, pre_a_buf),
                (4, pre_b_buf),
                (5, wa_buf),
                (6, wb_buf),
                (7, dx_buf),
                (8, d_wa_buf),
                (9, d_bias_a_buf),
                (10, d_wb_buf),
                (11, d_bias_b_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        // Скачиваем градиенты параметров в CPU
        let mut grad = Vec::with_capacity(p * n + q * n + p + q);
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_wa));
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_wb));
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_bias_a));
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_bias_b));
        grad
    }

    /// Прямой проход Combiner на GPU с использованием MatrixBufferHandle.
    pub fn run_combiner_forward_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        wb: &MatrixBufferHandle,
        bias: &[f32],
        out: &MatrixBufferHandle,
        pre: &MatrixBufferHandle,
    ) {
        assert!(a.is_gpu() && b.is_gpu() && wa.is_gpu() && wb.is_gpu(), "Input handles must be GPU");
        assert!(out.is_gpu() && pre.is_gpu(), "Output handles must be GPU");

        let batch = a.rows();
        let n = a.cols();
        let m = wa.rows();
        assert_eq!(b.rows(), batch);
        assert_eq!(b.cols(), n);
        assert_eq!(out.rows(), batch);
        assert_eq!(out.cols(), m);
        assert_eq!(pre.rows(), batch);
        assert_eq!(pre.cols(), m);

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);
        let pre_buf = self.get_gpu_subbuffer_from_handle(pre);

        let (bias_buf, bias_raw) = self.upload_to_temp_buffer(bias);

        let pipeline = self.pipeline_cache.combiner_fwd.clone();
        let push = [batch as u32, n as u32, m as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, a_buf),
                (1, b_buf),
                (2, wa_buf),
                (3, wb_buf),
                (4, bias_buf.clone()),
                (5, out_buf),
                (6, pre_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        self.release_temp_buffer(bias_buf, bias_raw);
    }

    /// Обратный проход Combiner на GPU с использованием MatrixBufferHandle.
    /// Возвращает градиенты параметров как Vec<f32>.
    /// Входные/выходные градиенты — GPU-дескрипторы.
    pub fn run_combiner_backward_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        d_out: &MatrixBufferHandle,
        pre: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        wb: &MatrixBufferHandle,
        da: &MatrixBufferHandle,
        db: &MatrixBufferHandle,
        d_wa: &MatrixBufferHandle,
        d_wb: &MatrixBufferHandle,
        d_bias: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(a.is_gpu() && b.is_gpu() && d_out.is_gpu() && pre.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Input handles must be GPU");
        assert!(da.is_gpu() && db.is_gpu() && d_wa.is_gpu() && d_wb.is_gpu() && d_bias.is_gpu(),
            "Output handles must be GPU");

        let batch = a.rows();
        let n = a.cols();
        let m = wa.rows();
        assert_eq!(b.rows(), batch);
        assert_eq!(b.cols(), n);
        assert_eq!(d_out.rows(), batch);
        assert_eq!(d_out.cols(), m);
        assert_eq!(pre.rows(), batch);
        assert_eq!(pre.cols(), m);
        assert_eq!(da.rows(), batch);
        assert_eq!(da.cols(), n);
        assert_eq!(db.rows(), batch);
        assert_eq!(db.cols(), n);
        assert_eq!(d_wa.rows(), m);
        assert_eq!(d_wa.cols(), n);
        assert_eq!(d_wb.rows(), m);
        assert_eq!(d_wb.cols(), n);
        assert_eq!(d_bias.rows(), 1);
        assert_eq!(d_bias.cols(), m);

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let d_out_buf = self.get_gpu_subbuffer_from_handle(d_out);
        let pre_buf = self.get_gpu_subbuffer_from_handle(pre);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let da_buf = self.get_gpu_subbuffer_from_handle(da);
        let db_buf = self.get_gpu_subbuffer_from_handle(db);
        let d_wa_buf = self.get_gpu_subbuffer_from_handle(d_wa);
        let d_wb_buf = self.get_gpu_subbuffer_from_handle(d_wb);
        let d_bias_buf = self.get_gpu_subbuffer_from_handle(d_bias);

        // Шейдер combiner_bwd вычисляет только da и db; градиенты весов и смещений
        // вычисляем через отдельные операции на GPU с последующим копированием.
        // Для простоты используем CPU fallback: скачиваем необходимые данные и считаем на CPU,
        // затем загружаем результаты обратно. Это медленнее, но корректно и не требует
        // дополнительных шейдеров.
        let d_out_vec = self.download_gpu_handle_to_vec(d_out);
        let pre_vec = self.download_gpu_handle_to_vec(pre);
        let a_vec = self.download_gpu_handle_to_vec(a);
        let b_vec = self.download_gpu_handle_to_vec(b);
        let wa_vec = self.download_gpu_handle_to_vec(wa);
        let wb_vec = self.download_gpu_handle_to_vec(wb);

        // Вычисляем d_pre = d_out * relu'(pre)
        let mut d_pre_vec = vec![0.0f32; batch * m];
        for c in 0..m {
            for r in 0..batch {
                let pre_val = pre_vec[c * batch + r];
                let dpre = if pre_val > 0.0 { d_out_vec[c * batch + r] } else { 0.0 };
                d_pre_vec[c * batch + r] = dpre;
            }
        }

        // da = d_pre * wa^T  (batch x m) * (m x n) -> batch x n
        let mut da_vec = vec![0.0f32; batch * n];
        for r in 0..batch {
            for col in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    // d_pre[k, r] (column-major) = d_pre_vec[k * batch + r]
                    // wa[k, col] (column-major) = wa_vec[col * m + k]
                    sum += d_pre_vec[k * batch + r] * wa_vec[col * m + k];
                }
                da_vec[col * batch + r] = sum;
            }
        }

        // db = d_pre * wb^T
        let mut db_vec = vec![0.0f32; batch * n];
        for r in 0..batch {
            for col in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += d_pre_vec[k * batch + r] * wb_vec[col * m + k];
                }
                db_vec[col * batch + r] = sum;
            }
        }

        // d_wa = d_pre^T * a  (m x batch) * (batch x n) -> m x n
        let mut d_wa_vec = vec![0.0f32; m * n];
        for out_idx in 0..m {
            for in_idx in 0..n {
                let mut sum = 0.0;
                for r in 0..batch {
                    sum += d_pre_vec[out_idx * batch + r] * a_vec[in_idx * batch + r];
                }
                // column-major для d_wa: data[in_idx * m + out_idx]
                d_wa_vec[in_idx * m + out_idx] = sum;
            }
        }

        // d_wb = d_pre^T * b
        let mut d_wb_vec = vec![0.0f32; m * n];
        for out_idx in 0..m {
            for in_idx in 0..n {
                let mut sum = 0.0;
                for r in 0..batch {
                    sum += d_pre_vec[out_idx * batch + r] * b_vec[in_idx * batch + r];
                }
                d_wb_vec[in_idx * m + out_idx] = sum;
            }
        }

        // d_bias = сумма по r d_pre
        let d_bias_vec: Vec<f32> = (0..m)
            .map(|c| (0..batch).map(|r| d_pre_vec[c * batch + r]).sum())
            .collect();

        // Загружаем результаты обратно в GPU handles
        self.copy_slice_to_gpu_handle(da, &da_vec);
        self.copy_slice_to_gpu_handle(db, &db_vec);
        self.copy_slice_to_gpu_handle(d_wa, &d_wa_vec);
        self.copy_slice_to_gpu_handle(d_wb, &d_wb_vec);
        self.copy_slice_to_gpu_handle(d_bias, &d_bias_vec);

        // Формируем общий градиент параметров
        let mut grad = Vec::with_capacity(2 * m * n + m);
        grad.extend_from_slice(&d_wa_vec);
        grad.extend_from_slice(&d_wb_vec);
        grad.extend_from_slice(&d_bias_vec);
        grad
    }
}

// Вспомогательная функция для старых Mat-версий
fn transpose_mat(mat: &Mat<f32>) -> Mat<f32> {
    Mat::from_fn(mat.ncols(), mat.nrows(), |r, c| mat[(c, r)])
}