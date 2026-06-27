// src/layers/splitter/splitter.rs

use crate::model_plan::param_store::ParamSlice;
use faer::Mat;

pub struct Splitter {
    input_dim: usize,
    output_dims: Vec<usize>,
}

impl Splitter {
    pub fn new(input_dim: usize, output_dims: Vec<usize>) -> Self {
        assert_eq!(output_dims.len(), 2, "Splitter requires exactly two outputs");
        Self { input_dim, output_dims }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dims(&self) -> &[usize] { &self.output_dims }

    pub(crate) fn get_weights_and_biases(&self, params: &[f32], slice: &ParamSlice) -> (Mat<f32>, Mat<f32>, Vec<f32>, Vec<f32>) {
        let n = self.input_dim;
        let p = self.output_dims[0];
        let q = self.output_dims[1];
        let base = slice.start;

        let wa_len = p * n;
        let wa = Mat::from_fn(p, n, |r, c| params[base + r * n + c]);

        let wb_start = base + wa_len;
        let wb = Mat::from_fn(q, n, |r, c| params[wb_start + r * n + c]);

        let bias_a_start = wb_start + q * n;
        let bias_a = params[bias_a_start..bias_a_start + p].to_vec();

        let bias_b_start = bias_a_start + p;
        let bias_b = params[bias_b_start..bias_b_start + q].to_vec();

        (wa, wb, bias_a, bias_b)
    }

    pub fn forward_mat(
        &self,
        x: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Mat<f32>, Mat<f32>, Mat<f32>) {
        let (wa, wb, bias_a, bias_b) = self.get_weights_and_biases(params, slice);
        let batch = x.nrows();
        let p = self.output_dims[0];
        let q = self.output_dims[1];

        let mut pre_a = x * wa.transpose();
        pre_a += Mat::from_fn(batch, p, |_, j| bias_a[j]);
        let a = pre_a.map(|v| v.max(0.0));

        let mut pre_b = x * wb.transpose();
        pre_b += Mat::from_fn(batch, q, |_, j| bias_b[j]);
        let b = pre_b.map(|v| v.max(0.0));

        (a, b, pre_a, pre_b)
    }

    pub fn backward_mat(
        &self,
        x: &Mat<f32>,
        da: &Mat<f32>,
        db: &Mat<f32>,
        pre_a: &Mat<f32>,
        pre_b: &Mat<f32>,
        wa: &Mat<f32>,
        wb: &Mat<f32>,
    ) -> (Mat<f32>, Vec<f32>) {
        let n = self.input_dim;
        let p = self.output_dims[0];
        let q = self.output_dims[1];

        let d_pre_a = relu_backward_mat(pre_a, da);
        let d_pre_b = relu_backward_mat(pre_b, db);

        let dx = &d_pre_a * wa + &d_pre_b * wb;

        let d_wa = d_pre_a.transpose() * x;
        let d_wb = d_pre_b.transpose() * x;

        let d_bias_a = col_sum_vec(&d_pre_a);
        let d_bias_b = col_sum_vec(&d_pre_b);

        let mut grad = Vec::with_capacity(self.param_len());
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

    pub fn param_len(&self) -> usize {
        let p = self.output_dims[0];
        let q = self.output_dims[1];
        self.input_dim * p + self.input_dim * q + p + q
    }
}

fn relu_backward_mat(pre: &Mat<f32>, dout: &Mat<f32>) -> Mat<f32> {
    Mat::from_fn(pre.nrows(), pre.ncols(), |i, j| {
        if pre[(i, j)] > 0.0 { dout[(i, j)] } else { 0.0 }
    })
}

fn col_sum_vec(mat: &Mat<f32>) -> Vec<f32> {
    let ncols = mat.ncols();
    let nrows = mat.nrows();
    let mut sums = vec![0.0f32; ncols];
    for j in 0..ncols {
        let mut s = 0.0;
        for i in 0..nrows {
            s += mat[(i, j)];
        }
        sums[j] = s;
    }
    sums
}