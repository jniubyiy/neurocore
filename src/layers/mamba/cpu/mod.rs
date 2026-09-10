// src/layers/mamba/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::mamba::mamba::{Mamba, MambaForwardCache, MambaForwardState};

// =============== Вспомогательные функции линейной алгебры ===============
// Все матрицы хранятся в row-major, векторы — линейные.

/// Умножение матрицы `mat` (n × m, row-major) на вектор `vec` длиной `m`.
/// Результат — вектор длиной `n`.
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

/// Умножение транспонированной матрицы `mat` (n × m, row-major) на вектор `vec`
/// длиной `n`. Результат — вектор длиной `m`.
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

/// Приближённая матричная экспонента exp(M) через ряд Тейлора (10 членов).
/// Матрица `mat` имеет размер n × n, row-major.
fn expm_taylor(mat: &[f32], n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n * n];
    let mut term = vec![0.0f32; n * n];
    for i in 0..n {
        term[i * n + i] = 1.0;
    }
    for k in 1..=10 {
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
        for i in 0..n * n {
            result[i] += term[i];
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
        let (input_copy, h_all, a_bar, b_bar) = input
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let y: &mut [f32] = &mut *second[0];
                let p: &[f32] = &*rest[0];

                let base = slice.start;

                let a_start = base;
                let b_start = a_start + n * n;
                let c_start = b_start + n * d;
                let d_idx = c_start + d * n;
                let delta_idx = d_idx + 1;

                let a_slice = &p[a_start..a_start + n * n];
                let b_slice = &p[b_start..b_start + n * d];
                let c_slice = &p[c_start..c_start + d * n];
                let d_par = p[d_idx];
                let delta = p[delta_idx];

                // Дискретизация: A_bar = exp(Δ A), B_bar = Δ B.
                let mut delta_a = vec![0.0f32; n * n];
                for i in 0..n * n {
                    delta_a[i] = delta * a_slice[i];
                }
                let a_bar_local = expm_taylor(&delta_a, n);
                let b_bar_local: Vec<f32> = b_slice.iter().map(|v| delta * v).collect();

                let mut h_all_local = vec![0.0f32; batch * seq * n];
                let input_copy = x.to_vec();

                for r in 0..batch {
                    let mut h_prev = vec![0.0f32; n];
                    for t in 0..seq {
                        // x_t: d-вектор.
                        let mut x_t = vec![0.0f32; d];
                        for j in 0..d {
                            x_t[j] = x[(t * d + j) * batch + r];
                        }

                        // h_t = A_bar * h_prev + B_bar * x_t.
                        let ah = mat_vec_mul(&a_bar_local, &h_prev, n, n);
                        let bx = mat_vec_mul(&b_bar_local, &x_t, n, d);
                        let mut h_t = vec![0.0f32; n];
                        for i in 0..n {
                            h_t[i] = ah[i] + bx[i];
                        }

                        // y_t = C * h_t + D * x_t.
                        let ch = mat_vec_mul(c_slice, &h_t, d, n);
                        let mut y_t = vec![0.0f32; d];
                        for j in 0..d {
                            y_t[j] = ch[j] + d_par * x_t[j];
                        }

                        // Сохранение h_t.
                        let h_offset = (r * seq + t) * n;
                        for i in 0..n {
                            h_all_local[h_offset + i] = h_t[i];
                        }

                        // Запись y.
                        for j in 0..d {
                            let out_idx = (t * d + j) * batch + r;
                            y[out_idx] = y_t[j];
                        }

                        h_prev = h_t;
                    }
                }

                (input_copy, h_all_local, a_bar_local, b_bar_local)
            });

        self.store_cache(MambaForwardCache {
            input: input_copy,
            h_all,
            a_bar,
            b_bar,
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
            BufferedContext::Mamba { input, .. } => input,
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

        // Извлекаем кэш и инвалидируем состояние.
        // std::mem::replace избегает клонирования больших векторов.
        let cache = {
            let mut guard = self.state.write().unwrap();
            assert!(
                guard.valid,
                "Mamba backward called without forward cache"
            );
            guard.valid = false;
            std::mem::replace(
                &mut guard.cache,
                MambaForwardCache {
                    input: Vec::new(),
                    h_all: Vec::new(),
                    a_bar: Vec::new(),
                    b_bar: Vec::new(),
                },
            )
        };

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

                let a_slice = &p[a_start..a_start + n * n];
                let b_slice = &p[b_start..b_start + n * d];
                let c_slice = &p[c_start..c_start + d * n];
                let d_par = p[d_idx];
                let delta = p[delta_idx];

                let a_bar = &cache.a_bar;
                let b_bar = &cache.b_bar;
                let x = &cache.input;
                let h_all = &cache.h_all;

                // Обнуление градиентов параметров.
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                let mut grad_a = vec![0.0f32; n * n];
                let mut grad_b = vec![0.0f32; n * d];
                let mut grad_c = vec![0.0f32; d * n];
                let mut grad_d = 0.0f32;
                let mut grad_delta = 0.0f32;

                // Обнуление градиента по входу.
                for i in 0..(batch * total_tokens) {
                    gi[i] = 0.0;
                }

                // dh_next: (batch * n), row-major.
                let mut dh_next = vec![0.0f32; batch * n];

                for t in (0..seq).rev() {
                    for r in 0..batch {
                        // dy_t: d-вектор.
                        let mut dy_t = vec![0.0f32; d];
                        for j in 0..d {
                            dy_t[j] = go[(t * d + j) * batch + r];
                        }

                        // dh_t = Cᵀ · dy_t + A_barᵀ · dh_next.
                        let dh_next_slice = &dh_next[r * n..(r + 1) * n];
                        let c_t_dy = mat_transpose_vec_mul(c_slice, &dy_t, d, n);
                        let at_dh = mat_transpose_vec_mul(a_bar, dh_next_slice, n, n);
                        let mut dh_t = vec![0.0f32; n];
                        for i in 0..n {
                            dh_t[i] = c_t_dy[i] + at_dh[i];
                        }

                        // Градиенты по C и D.
                        for j in 0..d {
                            let h_off = (r * seq + t) * n;
                            for i in 0..n {
                                grad_c[j * n + i] += dy_t[j] * h_all[h_off + i];
                            }
                        }
                        for j in 0..d {
                            let x_t = x[(t * d + j) * batch + r];
                            grad_d += dy_t[j] * x_t;
                        }

                        // Градиент по входу: gi = B_barᵀ · dh_t + D · dy_t.
                        let b_t_dh = mat_transpose_vec_mul(b_bar, &dh_t, n, d);
                        for j in 0..d {
                            let dx = b_t_dh[j] + d_par * dy_t[j];
                            gi[(t * d + j) * batch + r] = dx;
                        }

                        // h_prev (или нули при t = 0).
                        let h_prev: &[f32];
                        let zeros;
                        if t > 0 {
                            let h_off = (r * seq + t - 1) * n;
                            h_prev = &h_all[h_off..h_off + n];
                        } else {
                            zeros = vec![0.0f32; n];
                            h_prev = &zeros;
                        }

                        // x_t (d-вектор).
                        let x_t: Vec<f32> = (0..d)
                            .map(|j| x[(t * d + j) * batch + r])
                            .collect();

                        // grad_A и grad_delta:
                        //   A_bar = exp(Δ A) ⇒ ∂A_bar/∂A = Δ · exp(Δ A) = Δ · A_bar (приближение).
                        //   В формуле grad_A[i,j] += Δ · dh_t[i] · h_prev[j]
                        //   (это упрощение, принятое в исходной реализации).
                        for i in 0..n {
                            for j in 0..n {
                                grad_a[i * n + j] += delta * dh_t[i] * h_prev[j];
                                grad_delta += a_slice[i * n + j] * dh_t[i] * h_prev[j];
                            }
                        }
                        // grad_B и grad_delta:
                        //   B_bar = Δ B ⇒ ∂B_bar/∂B = Δ, ∂B_bar/∂Δ = B.
                        for i in 0..n {
                            for j in 0..d {
                                grad_b[i * d + j] += delta * dh_t[i] * x_t[j];
                                grad_delta += b_slice[i * d + j] * dh_t[i] * x_t[j];
                            }
                        }

                        // Сохранение dh_t для следующего (более раннего) шага.
                        dh_next[r * n..(r + 1) * n].copy_from_slice(&dh_t);
                    }
                }

                // Запись градиентов параметров.
                for i in 0..n * n {
                    gp[a_start + i] = grad_a[i];
                }
                for i in 0..n * d {
                    gp[b_start + i] = grad_b[i];
                }
                for i in 0..d * n {
                    gp[c_start + i] = grad_c[i];
                }
                gp[d_idx] = grad_d;
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