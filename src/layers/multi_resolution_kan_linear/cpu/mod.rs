// src/layers/multi_resolution_kan_linear/cpu/mod.rs

use once_cell::sync::Lazy;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::multi_resolution_kan_linear::MultiResolutionKANLinear;

// ---------------------------------------------------------------------------
// Конфигурация spline-сеток
// ---------------------------------------------------------------------------

const G_COARSE: usize = 3;
const G_FINE: usize = 8;
const K: usize = 3;
const COARSE_NUM_COEFFS: usize = G_COARSE + K; // 6
const FINE_NUM_COEFFS: usize = G_FINE + K; // 11
const MIXTURE_BRANCHES: usize = 2;
const GRID_MIN: f32 = -1.0;
const GRID_MAX: f32 = 1.0;
const TEMP_MIN: f32 = 1e-3;

fn build_grid(g: usize, k: usize, a: f32, b: f32) -> Vec<f32> {
    let dt = (b - a) / g as f32;
    let n = g + 2 * k + 1;
    (0..n).map(|i| a + (i as f32 - k as f32) * dt).collect()
}

static COARSE_GRID: Lazy<Vec<f32>> =
    Lazy::new(|| build_grid(G_COARSE, K, GRID_MIN, GRID_MAX));
static FINE_GRID: Lazy<Vec<f32>> =
    Lazy::new(|| build_grid(G_FINE, K, GRID_MIN, GRID_MAX));

#[inline]
fn bspline_basis(grid: &[f32], i: usize, k: usize, x: f32) -> f32 {
    if x < grid[i] || x >= grid[i + k + 1] {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    let d1 = grid[i + k] - grid[i];
    let d2 = grid[i + k + 1] - grid[i + 1];
    let mut acc = 0.0f32;
    if d1.abs() > 1e-30 {
        acc += (x - grid[i]) / d1 * bspline_basis(grid, i, k - 1, x);
    }
    if d2.abs() > 1e-30 {
        acc += (grid[i + k + 1] - x) / d2 * bspline_basis(grid, i + 1, k - 1, x);
    }
    acc
}

#[inline]
fn bspline_deriv(grid: &[f32], i: usize, k: usize, x: f32) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let d1 = grid[i + k] - grid[i];
    let d2 = grid[i + k + 1] - grid[i + 1];
    let mut acc = 0.0f32;
    if d1.abs() > 1e-30 {
        acc += bspline_basis(grid, i, k - 1, x) / d1;
    }
    if d2.abs() > 1e-30 {
        acc -= bspline_basis(grid, i + 1, k - 1, x) / d2;
    }
    acc * (k as f32)
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_deriv(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

#[inline]
fn mix_weights(logit_c: f32, logit_f: f32, temp: f32) -> (f32, f32) {
    let m = logit_c.max(logit_f);
    let inv_t = 1.0 / temp;
    let e_c = ((logit_c - m) * inv_t).exp();
    let e_f = ((logit_f - m) * inv_t).exp();
    let denom = e_c + e_f;
    (e_c / denom, e_f / denom)
}

#[derive(Clone, Copy)]
struct Offsets {
    bias: usize,
    mix_logits: usize,
    mix_temp_raw: usize,
    spline_coarse: usize,
    spline_fine: usize,
    base_weight: usize,
}

impl Offsets {
    fn new(base: usize, in_feat: usize, out_feat: usize) -> Self {
        let bias = base;
        let mix_logits = bias + out_feat;
        let mix_temp_raw = mix_logits + in_feat * out_feat * MIXTURE_BRANCHES;
        let spline_coarse = mix_temp_raw + in_feat * out_feat;
        let spline_fine = spline_coarse + in_feat * out_feat * COARSE_NUM_COEFFS;
        let base_weight = spline_fine + in_feat * out_feat * FINE_NUM_COEFFS;
        Self {
            bias,
            mix_logits,
            mix_temp_raw,
            spline_coarse,
            spline_fine,
            base_weight,
        }
    }
}

#[inline]
fn spline_scale(in_features: usize) -> f32 {
    1.0 / (in_features as f32).sqrt()
}

impl UniversalLayerBuffered for MultiResolutionKANLinear {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let batch = input.rows();
        let in_feat = self.in_features;
        let out_feat = self.out_features;
        debug_assert_eq!(input.cols(), in_feat);
        debug_assert_eq!(output.cols(), out_feat);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let off = Offsets::new(slice.start, in_feat, out_feat);
        let coarse_grid: &[f32] = &COARSE_GRID;
        let fine_grid: &[f32] = &FINE_GRID;
        let scale = spline_scale(in_feat);

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            for r in 0..batch {
                for j in 0..out_feat {
                    let mut sum = p[off.bias + j];
                    for i in 0..in_feat {
                        let x_val = x[i * batch + r];
                        let mi = j * in_feat + i;

                        let temp_raw = p[off.mix_temp_raw + mi];
                        let temp = TEMP_MIN + temp_raw.exp();
                        let logit_c = p[off.mix_logits + mi * MIXTURE_BRANCHES];
                        let logit_f = p[off.mix_logits + mi * MIXTURE_BRANCHES + 1];
                        let (w_c, w_f) = mix_weights(logit_c, logit_f, temp);

                        let coarse_base = off.spline_coarse + mi * COARSE_NUM_COEFFS;
                        let mut s_c = 0.0f32;
                        for g in 0..COARSE_NUM_COEFFS {
                            s_c += p[coarse_base + g]
                                * bspline_basis(coarse_grid, g, K, x_val);
                        }

                        let fine_base = off.spline_fine + mi * FINE_NUM_COEFFS;
                        let mut s_f = 0.0f32;
                        for g in 0..FINE_NUM_COEFFS {
                            s_f += p[fine_base + g]
                                * bspline_basis(fine_grid, g, K, x_val);
                        }

                        let bw = p[off.base_weight + mi];
                        let base = bw * silu(x_val);

                        sum += scale * (w_c * s_c + w_f * s_f) + base;
                    }
                    y[j * batch + r] = sum;
                }
            }
        });

        BufferedContext::MultiResolutionKANLinear {
            input: input.clone(),
        }
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
        let input_handle = match bc {
            BufferedContext::MultiResolutionKANLinear { input } => input,
            _ => panic!("Expected MultiResolutionKANLinear context"),
        };

        let batch = grad_output.rows();
        let in_feat = self.in_features;
        let out_feat = self.out_features;

        let off = Offsets::new(slice.start, in_feat, out_feat);
        let coarse_grid: &[f32] = &COARSE_GRID;
        let fine_grid: &[f32] = &FINE_GRID;
        let param_len = self.param_len();
        let scale = spline_scale(in_feat);

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];
        input_handle
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let go: &[f32] = &*second[0];
                let (third, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*fourth[0];
                let gp: &mut [f32] = &mut *rest[0];

                for v in gp[slice.start..slice.start + param_len].iter_mut() {
                    *v = 0.0;
                }
                for v in gi.iter_mut() {
                    *v = 0.0;
                }

                for r in 0..batch {
                    for j in 0..out_feat {
                        let gout = go[j * batch + r];
                        gp[off.bias + j] += gout;

                        for i in 0..in_feat {
                            let x_val = x[i * batch + r];
                            let mi = j * in_feat + i;

                            let temp_raw = p[off.mix_temp_raw + mi];
                            let temp = TEMP_MIN + temp_raw.exp();
                            let logit_c = p[off.mix_logits + mi * MIXTURE_BRANCHES];
                            let logit_f = p[off.mix_logits + mi * MIXTURE_BRANCHES + 1];
                            let (w_c, w_f) = mix_weights(logit_c, logit_f, temp);

                            let coarse_base =
                                off.spline_coarse + mi * COARSE_NUM_COEFFS;
                            let fine_base = off.spline_fine + mi * FINE_NUM_COEFFS;

                            let mut s_c = 0.0f32;
                            let mut ds_c = 0.0f32;
                            let mut basis_c = [0.0f32; COARSE_NUM_COEFFS];
                            for g in 0..COARSE_NUM_COEFFS {
                                let b = bspline_basis(coarse_grid, g, K, x_val);
                                basis_c[g] = b;
                                let c = p[coarse_base + g];
                                s_c += c * b;
                                ds_c += c * bspline_deriv(coarse_grid, g, K, x_val);
                            }

                            let mut s_f = 0.0f32;
                            let mut ds_f = 0.0f32;
                            let mut basis_f = [0.0f32; FINE_NUM_COEFFS];
                            for g in 0..FINE_NUM_COEFFS {
                                let b = bspline_basis(fine_grid, g, K, x_val);
                                basis_f[g] = b;
                                let c = p[fine_base + g];
                                s_f += c * b;
                                ds_f += c * bspline_deriv(fine_grid, g, K, x_val);
                            }

                            for g in 0..COARSE_NUM_COEFFS {
                                gp[coarse_base + g] += scale * gout * w_c * basis_c[g];
                            }
                            for g in 0..FINE_NUM_COEFFS {
                                gp[fine_base + g] += scale * gout * w_f * basis_f[g];
                            }

                            let silu_val = silu(x_val);
                            let bw = p[off.base_weight + mi];
                            gp[off.base_weight + mi] += gout * silu_val;

                            let s_mixed = w_c * s_c + w_f * s_f;
                            let inv_t = 1.0 / temp;

                            gp[off.mix_logits + mi * MIXTURE_BRANCHES] +=
                                scale * gout * inv_t * w_c * (s_c - s_mixed);
                            gp[off.mix_logits + mi * MIXTURE_BRANCHES + 1] +=
                                scale * gout * inv_t * w_f * (s_f - s_mixed);

                            let l_w = logit_c * w_c + logit_f * w_f;
                            let s_w_l = s_c * w_c * logit_c + s_f * w_f * logit_f;
                            let dt_draw = temp - TEMP_MIN;
                            let inv_t2 = inv_t * inv_t;
                            gp[off.mix_temp_raw + mi] +=
                                scale * gout * dt_draw * inv_t2 * (l_w * s_mixed - s_w_l);

                            let s_prime = silu_deriv(x_val);
                            let dx =
                                scale * (w_c * ds_c + w_f * ds_f) + bw * s_prime;
                            gi[i * batch + r] += gout * dx;
                        }
                    }
                }
            });
    }

    fn param_len(&self) -> usize {
        self.out_features
            + self.in_features
                * self.out_features
                * (2 + 1 + (G_COARSE + K) + (G_FINE + K) + 1)
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}