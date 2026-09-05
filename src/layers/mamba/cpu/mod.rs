// src/layers/mamba/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::mamba::{Mamba, MambaForwardCache};

// Вспомогательные функции линейной алгебры

/// Умножение матрицы (n x m) на вектор (m) -> вектор (n).
/// Матрица хранится row-major.
fn mat_vec_mul(mat: &[f32], vec: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut res = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            sum += mat[i * m + j] * vec[j];
        }
        res[i] = sum;
    }
    res
}

/// Транспонированное умножение матрицы (n x m) на вектор (n) -> вектор (m).
fn mat_transpose_vec_mul(mat: &[f32], vec: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut res = vec![0.0f32; m];
    for j in 0..m {
        let mut sum = 0.0;
        for i in 0..n {
            sum += mat[i * m + j] * vec[i];
        }
        res[j] = sum;
    }
    res
}

/// Приближённая матричная экспонента через ряд Тейлора (10 членов).
fn expm_taylor(mat: &[f32], n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n * n];
    let mut term = vec![0.0f32; n * n];
    // term = I
    for i in 0..n {
        term[i * n + i] = 1.0;
    }
    let mut factorial = 1.0f32;
    for k in 1..=10 {
        factorial *= k as f32;
        // term = term * mat / k
        let mut next = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += term[i * n + l] * mat[l * n + j];
                }
                next[i * n + j] = sum / k as f32;
            }
        }
        term = next;
        for i in 0..n {
            for j in 0..n {
                result[i * n + j] += term[i * n + j];
            }
        }
    }
    result
}

impl UniversalLayerBuffered for Mamba {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let seq = self.seq_len;
        let d = self.input_dim;
        let n = self.state_dim;
        let total_tokens = seq * d;

        debug_assert_eq!(input.cols(), total_tokens);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), total_tokens);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "Mamba: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;

            // Смещения параметров
            let a_start = base;
            let b_start = a_start + n * n;
            let c_start = b_start + n * d;
            let d_idx = c_start + d * n;
            let delta_idx = d_idx + 1;

            // Извлекаем параметры
            let A = &p[a_start..a_start + n * n];
            let B = &p[b_start..b_start + n * d];
            let C = &p[c_start..c_start + d * n];
            let D = p[d_idx];
            let delta = p[delta_idx];

            // Дискретизация
            // A_bar = exp(delta * A)
            let mut delta_A = vec![0.0f32; n * n];
            for i in 0..n * n {
                delta_A[i] = delta * A[i];
            }
            let A_bar = expm_taylor(&delta_A, n);

            // B_bar = delta * B
            let B_bar: Vec<f32> = B.iter().map(|v| delta * v).collect();

            // Кэш для обратного прохода
            let mut h_all = vec![0.0f32; batch * seq * n];
            let input_copy = x.to_vec(); // сохраняем копию входа

            // Прямой проход
            for r in 0..batch {
                let mut h_prev = vec![0.0f32; n];
                for t in 0..seq {
                    // Извлекаем x_t для этого r и t (длина d)
                    let mut x_t = vec![0.0f32; d];
                    for j in 0..d {
                        x_t[j] = x[(t * d + j) * batch + r];
                    }

                    // h_t = A_bar @ h_prev + B_bar @ x_t
                    let ah = mat_vec_mul(&A_bar, &h_prev, n, n);
                    let bx = mat_vec_mul(&B_bar, &x_t, n, d);
                    let mut h_t = vec![0.0f32; n];
                    for i in 0..n {
                        h_t[i] = ah[i] + bx[i];
                    }

                    // y_t = C @ h_t + D * x_t
                    let ch = mat_vec_mul(C, &h_t, d, n);
                    let mut y_t = vec![0.0f32; d];
                    for j in 0..d {
                        y_t[j] = ch[j] + D * x_t[j];
                    }

                    // Сохраняем h_t и y_t
                    let h_offset = (r * seq + t) * n;
                    for i in 0..n {
                        h_all[h_offset + i] = h_t[i];
                    }
                    let y_offset = (t * d + j) * batch + r; // j от 0..d, но мы записываем в цикле ниже
                    // Правильнее: сохранить y_t в выходной буфер по column-major
                    // Выходной буфер y имеет ту же структуру, что и вход: признак (t*d + j) для батча r.
                    // Поэтому запишем каждый компонент j.
                    for j in 0..d {
                        let out_idx = (t * d + j) * batch + r;
                        y[out_idx] = y_t[j];
                    }

                    h_prev = h_t;
                }
            }

            // Сохраняем кэш
            self.store_cache(MambaForwardCache {
                input: input_copy,
                h_all,
                A_bar,
                B_bar,
            });
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let _input_handle = match bc {
            BufferedContext::Mamba { input } => input,
            _ => panic!("Expected Mamba context"),
        };

        let batch = grad_output.rows();
        let seq = self.seq_len;
        let d = self.input_dim;
        let n = self.state_dim;
        let total_tokens = seq * d;

        debug_assert_eq!(grad_output.cols(), total_tokens);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), total_tokens);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "Mamba backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "Mamba backward: grad parameter slice out of bounds"
        );

        // Извлекаем кэш
        let cache = self
            .take_cache()
            .expect("Mamba backward called without forward cache");

        let ids = [
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];

        grad_output
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let go: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *second[0];
                let (third, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*third[0];
                let gp: &mut [f32] = &mut *rest[0];

                let base = slice.start;

                let a_start = base;
                let b_start = a_start + n * n;
                let c_start = b_start + n * d;
                let d_idx = c_start + d * n;
                let delta_idx = d_idx + 1;

                let A = &p[a_start..a_start + n * n];
                let B = &p[b_start..b_start + n * d];
                let C = &p[c_start..c_start + d * n];
                let D = p[d_idx];
                let delta = p[delta_idx];

                let A_bar = &cache.A_bar;
                let B_bar = &cache.B_bar;
                let x = &cache.input;
                let h_all = &cache.h_all;

                // Инициализируем градиенты параметров нулями
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                // Локальные накопители
                let mut grad_A = vec![0.0f32; n * n];
                let mut grad_B = vec![0.0f32; n * d];
                let mut grad_C = vec![0.0f32; d * n];
                let mut grad_D = 0.0f32;
                let mut grad_delta = 0.0f32;

                // Инициализируем градиент по входу нулями
                for i in 0..(batch * total_tokens) {
                    gi[i] = 0.0;
                }

                // Для каждого r храним dh_next (градиент от будущих шагов)
                let mut dh_next = vec![0.0f32; batch * n];

                // Обратное распространение во времени
                for t in (0..seq).rev() {
                    for r in 0..batch {
                        // Извлекаем dy_t из grad_output
                        let mut dy_t = vec![0.0f32; d];
                        for j in 0..d {
                            dy_t[j] = go[(t * d + j) * batch + r];
                        }

                        // dh_t = C^T @ dy_t + A_bar^T @ dh_next[r]
                        let c_t_dy = mat_transpose_vec_mul(C, &dy_t, d, n);
                        let at_dh = mat_transpose_vec_mul(A_bar, &dh_next[r * n..(r + 1) * n], n, n);
                        let mut dh_t = vec![0.0f32; n];
                        for i in 0..n {
                            dh_t[i] = c_t_dy[i] + at_dh[i];
                        }

                        // Градиенты по C и D
                        for j in 0..d {
                            for i in 0..n {
                                grad_C[j * n + i] += dy_t[j] * h_all[(r * seq + t) * n + i];
                            }
                        }
                        for j in 0..d {
                            let x_t = x[(t * d + j) * batch + r];
                            grad_D += dy_t[j] * x_t;
                        }

                        // Градиент по x_t = B_bar^T @ dh_t + D * dy_t
                        let b_t_dh = mat_transpose_vec_mul(B_bar, &dh_t, n, d);
                        for j in 0..d {
                            let dx = b_t_dh[j] + D * dy_t[j];
                            gi[(t * d + j) * batch + r] = dx;
                        }

                        // Градиенты по A_bar и B_bar
                        let h_prev = if t > 0 {
                            &h_all[(r * seq + t - 1) * n..(r * seq + t - 1) * n + n]
                        } else {
                            // на первом шаге h_prev = 0, поэтому вклад нулевой
                            &vec![0.0f32; n]
                        };
                        let x_t: Vec<f32> = (0..d)
                            .map(|j| x[(t * d + j) * batch + r])
                            .collect();

                        // dA_bar += dh_t ⊗ h_prev
                        for i in 0..n {
                            for j in 0..n {
                                // Накопление во временные переменные не делаем здесь, а сразу в grad_A_bar (но у нас нет отдельного grad_A_bar, сразу пересчитываем в grad_A)
                                // Для точности мы можем накапливать grad_A_bar и grad_B_bar, а потом преобразовать.
                                // Но мы используем приближение: dA = delta * dA_bar, dB = delta * dB_bar, ddelta = sum(A*dA_bar + B*dB_bar)
                                // Поэтому сразу накапливаем в grad_A, grad_B, grad_delta.
                                // grad_A[i,j] += delta * dh_t[i] * h_prev[j]  (поскольку dA_bar = dh_t[i] * h_prev[j])
                                grad_A[i * n + j] += delta * dh_t[i] * h_prev[j];
                                grad_delta += A[i * n + j] * dh_t[i] * h_prev[j];
                            }
                        }
                        for i in 0..n {
                            for j in 0..d {
                                grad_B[i * d + j] += delta * dh_t[i] * x_t[j];
                                grad_delta += B[i * d + j] * dh_t[i] * x_t[j];
                            }
                        }

                        // Обновляем dh_next для предыдущего шага
                        dh_next[r * n..(r + 1) * n].copy_from_slice(&dh_t);
                    }
                }

                // Записываем градиенты параметров
                for i in 0..n * n {
                    gp[a_start + i] = grad_A[i];
                }
                for i in 0..n * d {
                    gp[b_start + i] = grad_B[i];
                }
                for i in 0..d * n {
                    gp[c_start + i] = grad_C[i];
                }
                gp[d_idx] = grad_D;
                gp[delta_idx] = grad_delta;
            });
    }

    fn param_len(&self) -> usize {
        let n = self.state_dim;
        let d = self.input_dim;
        n * n + n * d + d * n + 2
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}