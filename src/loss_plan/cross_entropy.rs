// src/loss_plan/cross_entropy.rs

use faer::Mat;
use super::cubes::ElemCube;

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
        self.num_classes + 1
    }

    fn out_features(&self) -> usize {
        1
    }

    fn forward_batch(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let nclass = self.num_classes;

        // Вычисляем потери: -log(softmax[class_idx])
        let loss = Mat::from_fn(batch, 1, |i, _| {
            let class_idx = input[(i, nclass)] as usize;

            // Устойчивый softmax: вычитаем max по строке
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

        // Градиенты по логитам: softmax - one_hot, по индексу класса — 0
        let mut grad = Mat::zeros(batch, nclass + 1);

        for i in 0..batch {
            let class_idx = input[(i, nclass)] as usize;
            let g = grad_out[(i, 0)];

            // Устойчивый softmax
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
            grad[(i, nclass)] = 0.0; // градиент по индексу класса отсутствует
        }

        grad
    }
}