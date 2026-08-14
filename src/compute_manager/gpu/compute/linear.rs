// src/compute_manager/gpu/compute/linear.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

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
        let grad_input = self.run_mat_mul(grad_output, weight);
        let grad_output_t = Mat::from_fn(grad_output.ncols(), grad_output.nrows(), |r, c| grad_output[(c, r)]);
        let grad_weight = self.run_mat_mul(&grad_output_t, input);
        let grad_bias = self.run_reduce_sum_cols(grad_output);
        (grad_input, grad_weight, grad_bias)
    }

    // ===================================================================
    // Буферизованные версии для MatrixBuffer
    // ===================================================================

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

        let weight_t = self.transpose_gpu_matrix(weight);
        let mut out = self.run_mat_mul_buffered(input, &weight_t);

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

        let grad_input = self.run_mat_mul_buffered(grad_output, weight);

        let grad_output_t = self.transpose_gpu_matrix(grad_output);
        let grad_weight = self.run_mat_mul_buffered(&grad_output_t, input);

        let go_vec = self.download_gpu_matrix_to_vec(grad_output);
        let grad_bias: Vec<f32> = (0..out_features)
            .map(|c| (0..batch).map(|r| go_vec[c * batch + r]).sum())
            .collect();

        (grad_input, grad_weight, grad_bias)
    }

    // ===================================================================
    // НОВЫЕ Handle-версии (MatrixBufferHandle)
    // ===================================================================

    /// Прямой проход Linear на GPU с использованием MatrixBufferHandle.
    /// `input`, `weight` и `output` должны быть GPU-буферами.
    /// Веса передаются в формате (out_features, in_features).
    pub fn run_linear_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferHandle,
        bias: &[f32],
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(weight.is_gpu(), "Weight handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(output.rows(), batch, "Output rows mismatch");
        assert_eq!(output.cols(), out_features, "Output cols mismatch");
        assert_eq!(bias.len(), out_features, "Bias length mismatch");

        // Получаем данные из GPU в CPU для простой корректной реализации.
        // В дальнейшем можно заменить на чисто GPU-операции.
        let input_vec = self.download_gpu_handle_to_vec(input);
        let weight_vec = self.download_gpu_handle_to_vec(weight);

        // Входные данные хранятся column-major: input_vec[c * batch + r]
        // Веса хранятся column-major? weight_vec[col * out_features + row] или row-major?
        // В MatrixBuffer column-major, поэтому weight_vec[in_idx * out_features + out_idx] для элемента (out_idx, in_idx).
        // Проверим: MatrixBuffer хранит column-major, т.е. data[col * rows + row].
        // Для weight размером (out_features, in_features) => data[in_idx * out_features + out_idx] = W[out_idx, in_idx].
        // Нам удобнее использовать row-major для вычислений на CPU, но мы можем напрямую индексировать.
        let mut out_vec = vec![0.0f32; batch * out_features];
        for r in 0..batch {
            for c in 0..out_features {
                let mut sum = bias[c];
                for k in 0..in_features {
                    // input[k, r] (column-major) => input_vec[k * batch + r]
                    // weight[c, k] (column-major) => weight_vec[k * out_features + c]
                    sum += input_vec[k * batch + r] * weight_vec[k * out_features + c];
                }
                out_vec[c * batch + r] = sum;
            }
        }

        // Загружаем результат обратно в GPU output handle.
        self.copy_slice_to_gpu_handle(output, &out_vec);
    }

    /// Обратный проход Linear на GPU с использованием MatrixBufferHandle.
    /// `input`, `weight`, `grad_output`, `grad_input`, `grad_weight` должны быть GPU-буферами.
    /// Возвращает градиент по bias как Vec<f32>.
    pub fn run_linear_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        grad_weight: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(weight.is_gpu(), "Weight handle must be GPU");
        assert!(grad_output.is_gpu(), "grad_output handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_weight.is_gpu(), "grad_weight handle must be GPU");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = grad_output.cols();
        assert_eq!(weight.rows(), out_features, "Weight shape mismatch");
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(grad_input.rows(), batch, "grad_input rows mismatch");
        assert_eq!(grad_input.cols(), in_features, "grad_input cols mismatch");
        assert_eq!(grad_weight.rows(), out_features, "grad_weight rows mismatch");
        assert_eq!(grad_weight.cols(), in_features, "grad_weight cols mismatch");
        assert_eq!(grad_output.rows(), batch, "grad_output rows mismatch");

        // Получаем данные из GPU в CPU.
        let input_vec = self.download_gpu_handle_to_vec(input);
        let weight_vec = self.download_gpu_handle_to_vec(weight);
        let go_vec = self.download_gpu_handle_to_vec(grad_output);

        // Вычисляем grad_input = grad_output * weight (batch x out_features) * (out_features x in_features) -> (batch x in_features)
        let mut gi_vec = vec![0.0f32; batch * in_features];
        for r in 0..batch {
            for c in 0..in_features {
                let mut sum = 0.0;
                for k in 0..out_features {
                    // grad_output[k, r] => go_vec[k * batch + r]
                    // weight[k, c] => weight_vec[c * out_features + k]
                    sum += go_vec[k * batch + r] * weight_vec[c * out_features + k];
                }
                gi_vec[c * batch + r] = sum;
            }
        }

        // Вычисляем grad_weight = grad_output^T * input (out_features x batch) * (batch x in_features) -> (out_features x in_features)
        let mut gw_vec = vec![0.0f32; out_features * in_features];
        for out_idx in 0..out_features {
            for in_idx in 0..in_features {
                let mut sum = 0.0;
                for r in 0..batch {
                    // grad_output[out_idx, r] => go_vec[out_idx * batch + r]
                    // input[in_idx, r] => input_vec[in_idx * batch + r]
                    sum += go_vec[out_idx * batch + r] * input_vec[in_idx * batch + r];
                }
                // Column-major: data[in_idx * out_features + out_idx]
                gw_vec[in_idx * out_features + out_idx] = sum;
            }
        }

        // Вычисляем grad_bias = сумма по батчу grad_output
        let grad_bias: Vec<f32> = (0..out_features)
            .map(|c| (0..batch).map(|r| go_vec[c * batch + r]).sum())
            .collect();

        // Загружаем результаты в GPU handles.
        self.copy_slice_to_gpu_handle(grad_input, &gi_vec);
        self.copy_slice_to_gpu_handle(grad_weight, &gw_vec);

        grad_bias
    }
}