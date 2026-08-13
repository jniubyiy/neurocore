// src/compute_manager/gpu/compute/linear.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

impl GpuCompute {
    // ===================================================================
    // Старые Mat-версии (оставлены для обратной совместимости)
    // ===================================================================

    pub fn run_linear_forward(
        &self,
        input: &Mat<f32>,
        weight: &Mat<f32>,
        bias: &[f32],
    ) -> Mat<f32> {
        let weight_t = Mat::from_fn(weight.ncols(), weight.nrows(), |r, c| weight[(c, r)]);
        let mut out = self.run_mat_mul(input, &weight_t);
        let batch = input.nrows();
        let out_features = weight.nrows();
        for r in 0..batch {
            for c in 0..out_features {
                out[(r, c)] += bias[c];
            }
        }
        out
    }

    pub fn run_linear_backward(
        &self,
        input: &Mat<f32>,
        weight: &Mat<f32>,
        grad_output: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        // grad_input = grad_output * weight
        let grad_input = self.run_mat_mul(grad_output, weight);

        // grad_weight = grad_output^T * input
        let grad_output_t = Mat::from_fn(grad_output.ncols(), grad_output.nrows(), |r, c| grad_output[(c, r)]);
        let grad_weight = self.run_mat_mul(&grad_output_t, input);

        // grad_bias = reduce_sum_cols (теперь GPU)
        let grad_bias = self.run_reduce_sum_cols(grad_output);

        (grad_input, grad_weight, grad_bias)
    }

    // ===================================================================
    // Новые buffered-версии для MatrixBuffer
    // ===================================================================

    /// Прямой проход Linear на GPU с использованием MatrixBuffer.
    ///
    /// Вход `input` имеет размер `(batch, in_features)`.
    /// Веса `weight` имеют размер `(out_features, in_features)`.
    /// Смещения `bias` — срез длиной `out_features`.
    /// Результат — `(batch, out_features)`.
    ///
    /// Внутри не создаются `faer::Mat`; используются только GPU-буферы и временные `Vec`.
    pub fn run_linear_forward_buffered(
        &self,
        input: &MatrixBuffer,
        weight: &MatrixBuffer,
        bias: &[f32],
    ) -> MatrixBuffer {
        assert!(input.is_gpu() && weight.is_gpu(), "Buffers must be GPU");
        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(bias.len(), out_features, "Bias length mismatch");

        // Транспонируем веса: (out_features, in_features) -> (in_features, out_features)
        let weight_t = self.transpose_gpu_matrix(weight);

        // Выполняем матричное умножение: (batch, in_features) * (in_features, out_features)
        let mut out = self.run_mat_mul_buffered(input, &weight_t);

        // Добавляем bias через CPU-стагинг (временный Vec)
        let mut out_vec = self.download_gpu_matrix_to_vec(&out);
        for c in 0..out_features {
            let bias_val = bias[c];
            for r in 0..batch {
                out_vec[c * batch + r] += bias_val;
            }
        }
        out = self.upload_vec_to_gpu_buffer(&out_vec, batch, out_features);

        out
    }

    /// Обратный проход Linear на GPU с использованием MatrixBuffer.
    ///
    /// Принимает вход `input` `(batch, in_features)`, веса `weight` `(out_features, in_features)`
    /// и градиент по выходу `grad_output` `(batch, out_features)`.
    /// Возвращает:
    /// * `grad_input` `(batch, in_features)`
    /// * `grad_weight` `(out_features, in_features)`
    /// * `grad_bias` — `Vec<f32>` длиной `out_features`
    ///
    /// Внутри не создаются `faer::Mat`.
    pub fn run_linear_backward_buffered(
        &self,
        input: &MatrixBuffer,
        weight: &MatrixBuffer,
        grad_output: &MatrixBuffer,
    ) -> (MatrixBuffer, MatrixBuffer, Vec<f32>) {
        assert!(
            input.is_gpu() && weight.is_gpu() && grad_output.is_gpu(),
            "Buffers must be GPU"
        );
        let batch = input.rows();
        let in_features = input.cols();
        let out_features = grad_output.cols();
        assert_eq!(weight.rows(), out_features, "Weight shape mismatch");
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(input.rows(), batch, "Batch mismatch");

        // grad_input = grad_output * weight
        let grad_input = self.run_mat_mul_buffered(grad_output, weight);

        // grad_weight = grad_output^T * input
        let grad_output_t = self.transpose_gpu_matrix(grad_output); // (out_features, batch)
        let grad_weight = self.run_mat_mul_buffered(&grad_output_t, input); // (out_features, in_features)

        // grad_bias = сумма по строкам grad_output
        let go_vec = self.download_gpu_matrix_to_vec(grad_output);
        let grad_bias: Vec<f32> = (0..out_features)
            .map(|c| {
                (0..batch)
                    .map(|r| go_vec[c * batch + r])
                    .sum()
            })
            .collect();

        (grad_input, grad_weight, grad_bias)
    }
}