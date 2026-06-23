// src/neuron/types/combiner.rs

use faer::Mat;
use crate::neuron::base::Neuron;
use crate::neuron::ReLU;

/// Одиночный нейрон-объединитель: принимает два входа (batch, in_features),
/// возвращает (batch, 1) = ReLU(wa·a + wb·b + bias).
pub struct Combiner {
    pub wa: Vec<f32>,   // длина in_features
    pub wb: Vec<f32>,   // длина in_features
    pub bias: f32,
}

impl Combiner {
    pub fn new(wa: Vec<f32>, wb: Vec<f32>, bias: f32) -> Self {
        assert_eq!(wa.len(), wb.len(), "Combiner neuron: wa and wb must have same length");
        Self { wa, wb, bias }
    }

    /// Прямой проход: принимает две матрицы a и b размера (batch, in_features),
    /// возвращает матрицу (batch, 1).
    pub fn forward_mat(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let batch = a.nrows();
        let in_features = a.ncols();
        assert_eq!(b.nrows(), batch, "Combiner: batch sizes must match");
        assert_eq!(b.ncols(), in_features, "Combiner: input dimensions must match");
        assert_eq!(self.wa.len(), in_features, "Combiner neuron: wa length must match input columns");
        assert_eq!(self.wb.len(), in_features, "Combiner neuron: wb length must match input columns");

        let mut pre = Mat::zeros(batch, 1);
        for i in 0..batch {
            let mut sum = self.bias;
            for j in 0..in_features {
                sum += a[(i, j)] * self.wa[j] + b[(i, j)] * self.wb[j];
            }
            pre[(i, 0)] = sum;
        }

        ReLU.forward_mat(&pre)
    }

    /// Обратный проход: получает градиент выхода d_out (batch, 1) и исходные матрицы a, b.
    /// Возвращает градиенты по входам da, db (оба batch, in_features) и градиенты параметров [d_wa, d_wb, d_bias].
    pub fn backward_mat(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        d_out: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        let batch = a.nrows();
        let in_features = a.ncols();
        assert_eq!(b.nrows(), batch);
        assert_eq!(b.ncols(), in_features);
        assert_eq!(d_out.nrows(), batch);
        assert_eq!(d_out.ncols(), 1);

        let mut da = Mat::zeros(batch, in_features);
        let mut db = Mat::zeros(batch, in_features);
        let mut d_wa = vec![0.0f32; in_features];
        let mut d_wb = vec![0.0f32; in_features];
        let mut d_bias = 0.0f32;

        for i in 0..batch {
            let mut pre = self.bias;
            for j in 0..in_features {
                pre += a[(i, j)] * self.wa[j] + b[(i, j)] * self.wb[j];
            }
            if pre > 0.0 {
                let grad = d_out[(i, 0)];
                for j in 0..in_features {
                    da[(i, j)] = grad * self.wa[j];
                    db[(i, j)] = grad * self.wb[j];
                    d_wa[j] += grad * a[(i, j)];
                    d_wb[j] += grad * b[(i, j)];
                }
                d_bias += grad;
            }
        }

        let mut param_grad = Vec::with_capacity(2 * in_features + 1);
        param_grad.extend_from_slice(&d_wa);
        param_grad.extend_from_slice(&d_wb);
        param_grad.push(d_bias);
        (da, db, param_grad)
    }
}