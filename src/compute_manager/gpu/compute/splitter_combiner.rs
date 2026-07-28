// src/compute_manager/gpu/compute/splitter_combiner.rs

use faer::Mat;
use super::base::GpuCompute;

impl GpuCompute {
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
}

fn transpose_mat(mat: &Mat<f32>) -> Mat<f32> {
    Mat::from_fn(mat.ncols(), mat.nrows(), |r, c| mat[(c, r)])
}