// src/layers/adaptive_space_compress/cpu/mod.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::adaptive_space_compress::AdaptiveSpaceCompress;

const MIN_PLANES: usize = 1;
const P_SOFT_TAU: f32 = 0.5;
const HEAD_WEIGHT_K: f32 = 4.0;

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

#[inline]
fn compute_p_soft(p_raw: f32, p_max: usize) -> f32 {
    if MIN_PLANES >= p_max { return MIN_PLANES as f32; }
    let n_trans = p_max - MIN_PLANES;
    let mut p = MIN_PLANES as f32;
    for k in 1..=n_trans {
        let theta_k = (k as f32) - 1.0;
        p += sigmoid((p_raw - theta_k) / P_SOFT_TAU);
    }
    p
}

#[inline]
fn compute_p_soft_derivative(p_raw: f32, p_max: usize) -> f32 {
    if MIN_PLANES >= p_max { return 0.0; }
    let n_trans = p_max - MIN_PLANES;
    let mut d = 0.0f32;
    for k in 1..=n_trans {
        let theta_k = (k as f32) - 1.0;
        let s = sigmoid((p_raw - theta_k) / P_SOFT_TAU);
        d += s * (1.0 - s) / P_SOFT_TAU;
    }
    d
}

#[inline]
fn plane_weight(p_soft: f32, p: usize) -> f32 {
    let x = p_soft - p as f32;
    0.5 * (1.0 + ((x - 0.5) * HEAD_WEIGHT_K).tanh())
}

#[inline]
fn plane_weight_derivative(p_soft: f32, p: usize) -> f32 {
    let x = p_soft - p as f32;
    let t = ((x - 0.5) * HEAD_WEIGHT_K).tanh();
    0.5 * HEAD_WEIGHT_K * (1.0 - t * t)
}

#[inline]
fn compute_assign_one(
    len_r: usize,
    p_max: usize,
    center: &[f32],
    b_l: &[f32],
    assign_out: &mut [f32],
) {
    debug_assert!(len_r > 0, "len_r must be positive");
    debug_assert_eq!(assign_out.len(), p_max * len_r);

    let inv_len = 1.0 / (len_r as f32);
    for j in 0..len_r {
        let pos_j = (j as f32 + 0.5) * inv_len;
        let mut max_l = f32::NEG_INFINITY;
        for p in 0..p_max {
            let d = pos_j - center[p];
            let l = -d * d + b_l[p];
            if l > max_l { max_l = l; }
        }
        let mut sum_exp = 0.0f32;
        for p in 0..p_max {
            let d = pos_j - center[p];
            let l = -d * d + b_l[p];
            let e = (l - max_l).exp();
            assign_out[p * len_r + j] = e;
            sum_exp += e;
        }
        let inv = 1.0 / sum_exp;
        for p in 0..p_max {
            assign_out[p * len_r + j] *= inv;
        }
    }
}

fn forward_impl(
    layer: &AdaptiveSpaceCompress,
    input: &MatrixBufferHandle,
    sample_lens: &[usize],
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
) -> BufferedContext {
    let batch = input.rows();
    let in_f = input.cols();
    let out_f = layer.out_features;
    let p_max = layer.p_max;

    debug_assert!(in_f > 0);
    debug_assert_eq!(sample_lens.len(), batch);
    debug_assert_eq!(output.rows(), batch);
    debug_assert_eq!(output.cols(), out_f);
    debug_assert!(slice.start + layer.param_len() <= params.rows() * params.cols());

    let ids = [input.id(), output.id(), params.id()];
    let mut p_soft_local = 0.0f32;

    input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
        let (first, rest) = slices.split_at_mut(1);
        let x: &[f32] = &*first[0];
        let (second, rest) = rest.split_at_mut(1);
        let y: &mut [f32] = &mut *second[0];
        let (third, _) = rest.split_at_mut(1);
        let p: &[f32] = &*third[0];

        let base = slice.start;
        let center_off = base + layer.center_offset();
        let b_l_off    = base + layer.b_l_offset();
        let comp_off   = base + layer.compress_offset();
        let w_off      = base + layer.w_offset();
        let b_off      = base + layer.b_offset();
        let p_raw_off  = base + layer.p_raw_offset();

        let p_raw = p[p_raw_off];
        let p_soft = compute_p_soft(p_raw, p_max);
        p_soft_local = p_soft;

        let mut w = vec![0.0f32; p_max];
        for pi in 0..p_max { w[pi] = plane_weight(p_soft, pi); }

        let center = &p[center_off..center_off + p_max];
        let b_l    = &p[b_l_off..b_l_off + p_max];

        let mut assign = vec![0.0f32; p_max * in_f];
        for r in 0..batch {
            let len_r = sample_lens[r];
            if len_r == 0 || len_r > in_f {
                for k in 0..out_f { y[k * batch + r] = 0.0; }
                continue;
            }
            compute_assign_one(len_r, p_max, center, b_l, &mut assign);

            let mut value = vec![0.0f32; p_max];
            for pi in 0..p_max {
                let a_row = &assign[pi * len_r..(pi + 1) * len_r];
                let mut s = 0.0f32;
                for j in 0..len_r {
                    s += a_row[j] * x[j * batch + r];
                }
                value[pi] = s;
            }

            let mut compressed = vec![0.0f32; p_max];
            for pi in 0..p_max {
                compressed[pi] = value[pi] * p[comp_off + pi] * w[pi];
            }

            for k in 0..out_f {
                let mut s = p[b_off + k];
                for pi in 0..p_max {
                    s += compressed[pi] * p[w_off + pi * out_f + k];
                }
                y[k * batch + r] = s;
            }
        }
    });

    BufferedContext::AdaptiveSpaceCompress {
        input: input.clone(),
        p_soft: p_soft_local,
        sample_lens: sample_lens.to_vec(),
    }
}

fn backward_impl(
    layer: &AdaptiveSpaceCompress,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params: &MatrixBufferHandle,
) {
    let DynamicContext::Buffered(bc) = ctx;
    let (input_handle, p_soft, sample_lens) = match bc {
        BufferedContext::AdaptiveSpaceCompress { input, p_soft, sample_lens } => {
            (input, *p_soft, sample_lens.clone())
        }
        _ => panic!("Expected AdaptiveSpaceCompress context"),
    };

    let batch = grad_output.rows();
    let in_f = input_handle.cols();
    let out_f = layer.out_features;
    let p_max = layer.p_max;

    debug_assert_eq!(sample_lens.len(), batch);
    debug_assert_eq!(grad_output.cols(), out_f);
    debug_assert_eq!(grad_input.rows(), batch);
    debug_assert_eq!(grad_input.cols(), in_f);
    debug_assert_eq!(input_handle.rows(), batch);
    debug_assert!(slice.start + layer.param_len() <= params.rows() * params.cols());
    debug_assert!(slice.start + layer.param_len() <= grad_params.rows() * grad_params.cols());

    let ids = [
        input_handle.id(),
        grad_output.id(),
        grad_input.id(),
        params.id(),
        grad_params.id(),
    ];

    input_handle.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
        let (first, rest) = slices.split_at_mut(1);
        let x: &[f32] = &*first[0];
        let (second, rest) = rest.split_at_mut(1);
        let go: &[f32] = &*second[0];
        let (third, rest) = rest.split_at_mut(1);
        let gi: &mut [f32] = &mut *third[0];
        let (fourth, rest) = rest.split_at_mut(1);
        let p: &[f32] = &*fourth[0];
        let (fifth, _) = rest.split_at_mut(1);
        let gp: &mut [f32] = &mut *fifth[0];

        let base = slice.start;
        let center_off = base + layer.center_offset();
        let b_l_off    = base + layer.b_l_offset();
        let comp_off   = base + layer.compress_offset();
        let w_off      = base + layer.w_offset();
        let b_off      = base + layer.b_offset();
        let p_raw_off  = base + layer.p_raw_offset();

        let p_raw = p[p_raw_off];

        for i in 0..layer.param_len() { gp[base + i] = 0.0; }
        for i in 0..(batch * in_f) { gi[i] = 0.0; }

        let mut w = vec![0.0f32; p_max];
        for pi in 0..p_max { w[pi] = plane_weight(p_soft, pi); }

        let center = &p[center_off..center_off + p_max];
        let b_l    = &p[b_l_off..b_l_off + p_max];

        let mut gp_b_l = vec![0.0f32; p_max];
        let mut gp_center = vec![0.0f32; p_max];
        let mut gp_compress = vec![0.0f32; p_max];
        let mut d_w = vec![0.0f32; p_max];
        let mut gp_w = vec![0.0f32; p_max * out_f];

        for k in 0..out_f {
            let mut s = 0.0f32;
            for r in 0..batch { s += go[k * batch + r]; }
            gp[b_off + k] = s;
        }

        let mut assign = vec![0.0f32; p_max * in_f];
        for r in 0..batch {
            let len_r = sample_lens[r];
            if len_r == 0 || len_r > in_f { continue; }
            compute_assign_one(len_r, p_max, center, b_l, &mut assign);

            let mut value = vec![0.0f32; p_max];
            for pi in 0..p_max {
                let a_row = &assign[pi * len_r..(pi + 1) * len_r];
                let mut s = 0.0f32;
                for j in 0..len_r { s += a_row[j] * x[j * batch + r]; }
                value[pi] = s;
            }

            let mut compressed = vec![0.0f32; p_max];
            for pi in 0..p_max {
                compressed[pi] = value[pi] * p[comp_off + pi] * w[pi];
            }

            let mut d_compressed = vec![0.0f32; p_max];
            for pi in 0..p_max {
                let mut s = 0.0f32;
                for k in 0..out_f {
                    s += go[k * batch + r] * p[w_off + pi * out_f + k];
                }
                d_compressed[pi] = s;
            }

            for pi in 0..p_max {
                for k in 0..out_f {
                    gp_w[pi * out_f + k] += go[k * batch + r] * compressed[pi];
                }
            }

            for pi in 0..p_max {
                gp_compress[pi] += d_compressed[pi] * value[pi] * w[pi];
                d_w[pi] += d_compressed[pi] * value[pi] * p[comp_off + pi];
            }

            let mut d_value = vec![0.0f32; p_max];
            for pi in 0..p_max {
                d_value[pi] = d_compressed[pi] * p[comp_off + pi] * w[pi];
            }

            let mut d_assign = vec![0.0f32; p_max * len_r];
            for pi in 0..p_max {
                for j in 0..len_r {
                    d_assign[pi * len_r + j] = d_value[pi] * x[j * batch + r];
                }
            }

            let mut d_l = vec![0.0f32; p_max * len_r];
            for j in 0..len_r {
                let mut dot = 0.0f32;
                for pi in 0..p_max {
                    dot += assign[pi * len_r + j] * d_assign[pi * len_r + j];
                }
                for pi in 0..p_max {
                    let a = assign[pi * len_r + j];
                    d_l[pi * len_r + j] = a * (d_assign[pi * len_r + j] - dot);
                }
            }

            let inv_len = 1.0 / (len_r as f32);
            for pi in 0..p_max {
                for j in 0..len_r {
                    let dl = d_l[pi * len_r + j];
                    gp_b_l[pi] += dl;
                    let pos_j = (j as f32 + 0.5) * inv_len;
                    let diff = pos_j - center[pi];
                    gp_center[pi] += dl * 2.0 * diff;
                }
            }

            for j in 0..len_r {
                let mut s = 0.0f32;
                for pi in 0..p_max {
                    s += d_value[pi] * assign[pi * len_r + j];
                }
                gi[j * batch + r] = s;
            }
        }

        for pi in 0..p_max {
            gp[b_l_off + pi] = gp_b_l[pi];
            gp[center_off + pi] = gp_center[pi];
            gp[comp_off + pi] = gp_compress[pi];
        }
        for i in 0..(p_max * out_f) {
            gp[w_off + i] = gp_w[i];
        }

        let mut d_p_soft = 0.0f32;
        for pi in 0..p_max {
            d_p_soft += d_w[pi] * plane_weight_derivative(p_soft, pi);
        }
        let d_p_raw = d_p_soft * compute_p_soft_derivative(p_raw, p_max);
        gp[p_raw_off] = d_p_raw;
    });
}

impl UniversalLayerBuffered for AdaptiveSpaceCompress {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let batch = input.rows();
        let in_f = input.cols();
        let lens = vec![in_f; batch];
        forward_impl(self, input, &lens, output, params, slice)
    }

    fn forward_buffered_ragged(
        &self,
        input: &MatrixBufferHandle,
        sample_lens: &[usize],
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        forward_impl(self, input, sample_lens, output, params, slice)
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
        backward_impl(self, ctx, grad_output, grad_input, params, slice, grad_params)
    }

    fn backward_buffered_ragged(
        &self,
        ctx: &DynamicContext,
        _sample_lens: &[usize],
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        // sample_lens уже в контексте — параметр здесь не нужен.
        backward_impl(self, ctx, grad_output, grad_input, params, slice, grad_params)
    }

    /// Возвращает реальный размер входа, прочитанный из forward-контекста.
    ///
    /// Это то самое переопределение, которое позволяет оркестратору
    /// (CPU-оператору) остаться **нейтральным**: он не знает про
    /// `AdaptiveSpaceCompress`, а просто спрашивает слой.
    fn input_features_from_ctx(
        &self,
        ctx: &DynamicContext,
        fallback: usize,
    ) -> usize {
        if let DynamicContext::Buffered(
            BufferedContext::AdaptiveSpaceCompress { input, .. },
        ) = ctx
        {
            input.cols()
        } else {
            fallback
        }
    }

    fn param_len(&self) -> usize {
        self.p_max * (3 + self.out_features) + self.out_features + 1
    }

    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}

