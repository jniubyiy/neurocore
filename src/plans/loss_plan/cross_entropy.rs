// src/plans/loss_plan/cross_entropy.rs

use std::any::Any;
use faer::Mat;
use super::cubes::ElemCube;
use super::cubes::BufferedElemCube;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

/// Кросс‑энтропия с логитами.
///
/// Принимает матрицу размера `(batch, num_classes + 1)`, где первые `num_classes`
/// столбцов — это логиты (предсказания модели), а последний столбец содержит
/// индекс правильного класса (как `f32`, который приводится к `usize`).
///
/// Возвращает матрицу `(batch, 1)` со значениями потерь для каждого сэмпла.
///
/// Этот кубик полностью совместим с новым векторным представлением:
/// `pred_features = num_classes`, `target_features = 1`.
#[derive(Debug)]
pub struct CrossEntropyWithLogits {
    pub num_classes: usize,
}

impl CrossEntropyWithLogits {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl ElemCube for CrossEntropyWithLogits {
    fn in_features(&self) -> usize {
        self.num_classes + 1   // логиты + индекс класса
    }

    fn out_features(&self) -> usize {
        1
    }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let nclass = self.num_classes;

        let loss = Mat::from_fn(batch, 1, |i, _| {
            let class_idx = input[(i, nclass)] as usize;

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..nclass {
                max_val = max_val.max(input[(i, c)]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..nclass {
                exp_sum += (input[(i, c)] - max_val).exp();
            }

            -input[(i, class_idx)] + max_val + exp_sum.ln()
        });

        loss
    }

    fn backward_batch(
        &self,
        input: &Mat<f32>,
        _output_cache: &Mat<f32>,
        grad_out: &Mat<f32>,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let nclass = self.num_classes;

        let mut grad = Mat::zeros(batch, nclass + 1);

        for i in 0..batch {
            let class_idx = input[(i, nclass)] as usize;
            let g = grad_out[(i, 0)];

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..nclass {
                max_val = max_val.max(input[(i, c)]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..nclass {
                exp_sum += (input[(i, c)] - max_val).exp();
            }

            for j in 0..nclass {
                let softmax_j = ((input[(i, j)] - max_val).exp()) / exp_sum;
                let indicator = if j == class_idx { 1.0 } else { 0.0 };
                grad[(i, j)] = g * (softmax_j - indicator);
            }
            grad[(i, nclass)] = 0.0;
        }

        grad
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BufferedElemCube for CrossEntropyWithLogits {
    fn in_features(&self) -> usize {
        self.num_classes + 1
    }

    fn out_features(&self) -> usize {
        1
    }

    fn forward_buffered(&self, input: &MatrixBuffer, output: &mut MatrixBuffer) {
        assert!(!input.is_gpu() && !output.is_gpu(),
            "BufferedElemCube for CrossEntropyWithLogits supports only CPU buffers");

        let batch = input.rows();
        let nclass = self.num_classes;
        let src = input.as_slice();
        let dst = output.as_slice_mut();

        // входная матрица имеет размер (batch, nclass+1), column-major
        for r in 0..batch {
            let class_idx = src[nclass * batch + r] as usize;

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..nclass {
                max_val = max_val.max(src[c * batch + r]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..nclass {
                exp_sum += (src[c * batch + r] - max_val).exp();
            }

            dst[r] = -src[class_idx * batch + r] + max_val + exp_sum.ln();
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBuffer,
        _output_cache: &MatrixBuffer,
        grad_out: &MatrixBuffer,
        grad_in: &mut MatrixBuffer,
    ) {
        assert!(!input.is_gpu() && !grad_out.is_gpu() && !grad_in.is_gpu(),
            "BufferedElemCube for CrossEntropyWithLogits supports only CPU buffers");

        let batch = input.rows();
        let nclass = self.num_classes;
        let src = input.as_slice();
        let go = grad_out.as_slice(); // размер (batch,1)
        let gi = grad_in.as_slice_mut(); // размер (batch, nclass+1)

        for r in 0..batch {
            let class_idx = src[nclass * batch + r] as usize;
            let g = go[r];

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..nclass {
                max_val = max_val.max(src[c * batch + r]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..nclass {
                exp_sum += (src[c * batch + r] - max_val).exp();
            }

            for c in 0..nclass {
                let softmax_c = (src[c * batch + r] - max_val).exp() / exp_sum;
                let indicator = if c == class_idx { 1.0 } else { 0.0 };
                gi[c * batch + r] = g * (softmax_c - indicator);
            }
            gi[nclass * batch + r] = 0.0;
        }
    }
}