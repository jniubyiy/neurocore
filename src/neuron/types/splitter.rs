// src/neuron/types/splitter.rs

use faer::Mat;
use crate::neuron::base::Neuron;
use crate::neuron::ReLU;
use crate::neuron::Linear as LinearNeuron;

/// Нейрон-разделитель: принимает матрицу x (batch × n) и выдаёт два выхода a (batch × p) и b (batch × q).
/// Использует p+q независимых линейных нейронов и ReLU.
pub struct Splitter {
    pub n: usize,
    pub p: usize,
    pub q: usize,
    pub wa: Vec<f32>,     // p * n (row-major)
    pub wb: Vec<f32>,     // q * n
    pub bias_a: Vec<f32>, // p
    pub bias_b: Vec<f32>, // q
}

impl Splitter {
    pub fn new(n: usize, p: usize, q: usize) -> Self {
        Self {
            n, p, q,
            wa: vec![0.0; p * n],
            wb: vec![0.0; q * n],
            bias_a: vec![0.0; p],
            bias_b: vec![0.0; q],
        }
    }

    pub fn param_count(&self) -> usize {
        self.p * self.n + self.q * self.n + self.p + self.q
    }

    pub fn get_params(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.wa);
        v.extend_from_slice(&self.wb);
        v.extend_from_slice(&self.bias_a);
        v.extend_from_slice(&self.bias_b);
        v
    }

    pub fn set_params(&mut self, values: &[f32]) {
        let wa_len = self.p * self.n;
        let wb_len = self.q * self.n;
        let ba_len = self.p;
        let bb_len = self.q;
        assert_eq!(values.len(), wa_len + wb_len + ba_len + bb_len);
        self.wa.copy_from_slice(&values[..wa_len]);
        self.wb.copy_from_slice(&values[wa_len..wa_len + wb_len]);
        self.bias_a.copy_from_slice(&values[wa_len + wb_len..wa_len + wb_len + ba_len]);
        self.bias_b.copy_from_slice(&values[wa_len + wb_len + ba_len..]);
    }

    /// Прямой проход: возвращает (a, b, pre_a, pre_b)
    pub fn forward_mat(&self, x: &Mat<f32>) -> (Mat<f32>, Mat<f32>, Mat<f32>, Mat<f32>) {
        let batch = x.nrows();
        assert_eq!(x.ncols(), self.n);
        let relu = ReLU;

        // Ветка a
        let mut pre_a = Mat::zeros(batch, self.p);
        for out_idx in 0..self.p {
            let w_start = out_idx * self.n;
            let weights: Vec<f32> = self.wa[w_start..w_start + self.n].to_vec();
            let neuron = LinearNeuron::new(weights, self.bias_a[out_idx]);
            let col = neuron.forward_mat(x); // (batch, 1)
            for i in 0..batch {
                pre_a[(i, out_idx)] = col[(i, 0)];
            }
        }
        let mut a = Mat::zeros(batch, self.p);
        for out_idx in 0..self.p {
            let col = pre_a.submatrix(0, out_idx, batch, 1).to_owned();
            let act = relu.forward_mat(&col);
            for i in 0..batch {
                a[(i, out_idx)] = act[(i, 0)];
            }
        }

        // Ветка b
        let mut pre_b = Mat::zeros(batch, self.q);
        for out_idx in 0..self.q {
            let w_start = out_idx * self.n;
            let weights: Vec<f32> = self.wb[w_start..w_start + self.n].to_vec();
            let neuron = LinearNeuron::new(weights, self.bias_b[out_idx]);
            let col = neuron.forward_mat(x);
            for i in 0..batch {
                pre_b[(i, out_idx)] = col[(i, 0)];
            }
        }
        let mut b = Mat::zeros(batch, self.q);
        for out_idx in 0..self.q {
            let col = pre_b.submatrix(0, out_idx, batch, 1).to_owned();
            let act = relu.forward_mat(&col);
            for i in 0..batch {
                b[(i, out_idx)] = act[(i, 0)];
            }
        }

        (a, b, pre_a, pre_b)
    }

    /// Обратный проход (матричный, для эффективности)
    pub fn backward_mat(
        &self,
        x: &Mat<f32>,
        da: &Mat<f32>,
        db: &Mat<f32>,
        pre_a: &Mat<f32>,
        pre_b: &Mat<f32>,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = x.nrows();
        assert_eq!(x.ncols(), self.n);

        let d_pre_a = relu_backward_mat(pre_a, da);
        let d_pre_b = relu_backward_mat(pre_b, db);

        let wa_mat = Mat::from_fn(self.p, self.n, |r, c| self.wa[r * self.n + c]);
        let wb_mat = Mat::from_fn(self.q, self.n, |r, c| self.wb[r * self.n + c]);

        let dx = &d_pre_a * &wa_mat + &d_pre_b * &wb_mat;
        let d_wa = &d_pre_a.transpose() * x;
        let d_wb = &d_pre_b.transpose() * x;

        let mut d_bias_a = vec![0.0; self.p];
        let mut d_bias_b = vec![0.0; self.q];
        for i in 0..batch {
            for j in 0..self.p { d_bias_a[j] += d_pre_a[(i, j)]; }
            for j in 0..self.q { d_bias_b[j] += d_pre_b[(i, j)]; }
        }

        let mut grad = Vec::with_capacity(self.param_count());
        for r in 0..self.p {
            for c in 0..self.n { grad.push(d_wa[(r, c)]); }
        }
        for r in 0..self.q {
            for c in 0..self.n { grad.push(d_wb[(r, c)]); }
        }
        grad.extend_from_slice(&d_bias_a);
        grad.extend_from_slice(&d_bias_b);

        (dx, grad)
    }
}

fn relu_backward_mat(pre: &Mat<f32>, dout: &Mat<f32>) -> Mat<f32> {
    let mut out = Mat::zeros(pre.nrows(), pre.ncols());
    for i in 0..pre.nrows() {
        for j in 0..pre.ncols() {
            if pre[(i, j)] > 0.0 {
                out[(i, j)] = dout[(i, j)];
            }
        }
    }
    out
}