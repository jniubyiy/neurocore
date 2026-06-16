// src/loss/ops/mod.rs
use std::any::Any;
use super::common::stable_softmax;

// ── Объявления трейтов ─────────────────────────────────────────
pub trait LossInput: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn to_flat(&self) -> Vec<f32>;
    fn zero_clone(&self) -> Box<dyn LossInput>;
    fn fill_from_flat(&mut self, data: &[f32]);
}

pub trait LossJacobian: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn params(&self) -> usize;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn to_flat(&self) -> Vec<f32>;
    fn zero_clone(&self) -> Box<dyn LossJacobian>;
    fn fill_from_flat(&mut self, data: &[f32]);
}

// ── Подключаем реализации для каждой размерности ──────────────
mod ops1d;
mod ops2d;
mod ops3d;
mod ops4d;
mod ops5d;

// ── Элементарные операции (работают через трейты, без привязки к типу) ──
pub fn sub(
    pred: &dyn LossInput,
    target: &dyn LossInput,
    j_pred: &dyn LossJacobian,
) -> (Box<dyn LossInput>, Box<dyn LossJacobian>) {
    let rows = pred.rows();
    let cols = pred.cols();
    let params = j_pred.params();
    let p_flat = pred.to_flat();
    let t_flat = target.to_flat();
    let j_flat = j_pred.to_flat();

    let n = rows * cols;
    let mut out_val = vec![0.0f32; n];
    let mut out_jac = vec![0.0f32; n * params];

    for i in 0..n {
        out_val[i] = p_flat[i] - t_flat[i];
        let base = i * params;
        for k in 0..params {
            out_jac[base + k] = j_flat[base + k];
        }
    }

    let mut result_val = pred.zero_clone();
    result_val.fill_from_flat(&out_val);
    let mut result_jac = j_pred.zero_clone();
    result_jac.fill_from_flat(&out_jac);
    (result_val, result_jac)
}

pub fn square(
    val: &dyn LossInput,
    j_val: &dyn LossJacobian,
) -> (Box<dyn LossInput>, Box<dyn LossJacobian>) {
    let rows = val.rows();
    let cols = val.cols();
    let params = j_val.params();
    let v = val.to_flat();
    let jv = j_val.to_flat();

    let n = rows * cols;
    let mut out = vec![0.0f32; n];
    let mut j_out = vec![0.0f32; n * params];
    for i in 0..n {
        let x = v[i];
        out[i] = x * x;
        let base = i * params;
        for k in 0..params {
            j_out[base + k] = 2.0 * x * jv[base + k];
        }
    }

    let mut res_val = val.zero_clone();
    res_val.fill_from_flat(&out);
    let mut res_jac = j_val.zero_clone();
    res_jac.fill_from_flat(&j_out);
    (res_val, res_jac)
}

pub fn abs(
    val: &dyn LossInput,
    j_val: &dyn LossJacobian,
) -> (Box<dyn LossInput>, Box<dyn LossJacobian>) {
    let rows = val.rows();
    let cols = val.cols();
    let params = j_val.params();
    let v = val.to_flat();
    let jv = j_val.to_flat();

    let n = rows * cols;
    let mut out = vec![0.0f32; n];
    let mut j_out = vec![0.0f32; n * params];
    for i in 0..n {
        let x = v[i];
        out[i] = x.abs();
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let base = i * params;
        for k in 0..params {
            j_out[base + k] = sign * jv[base + k];
        }
    }

    let mut res_val = val.zero_clone();
    res_val.fill_from_flat(&out);
    let mut res_jac = j_val.zero_clone();
    res_jac.fill_from_flat(&j_out);
    (res_val, res_jac)
}

pub fn mean(val: &dyn LossInput, j_val: &dyn LossJacobian) -> (f32, Vec<f32>) {
    let rows = val.rows();
    let cols = val.cols();
    let params = j_val.params();
    let n = (rows * cols) as f32;
    let v = val.to_flat();
    let jv = j_val.to_flat();

    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; params];
    for i in 0..(rows * cols) {
        loss += v[i];
        let base = i * params;
        for k in 0..params {
            grad[k] += jv[base + k];
        }
    }
    loss /= n;
    for k in 0..params { grad[k] /= n; }
    (loss, grad)
}

pub fn softmax(
    logits: &dyn LossInput,
    j_logits: &dyn LossJacobian,
) -> (Box<dyn LossInput>, Box<dyn LossJacobian>) {
    let rows = logits.rows();
    let cols = logits.cols();
    let params = j_logits.params();
    let v = logits.to_flat();
    let jv = j_logits.to_flat();

    let mut out = vec![0.0f32; rows * cols];
    let mut j_out = vec![0.0f32; rows * cols * params];

    for r in 0..rows {
        let start = r * cols;
        let row = &v[start..start + cols];
        let sm = stable_softmax(row);
        for c in 0..cols {
            out[start + c] = sm[c];
            let base_out = (start + c) * params;
            for p in 0..params {
                let mut deriv = 0.0;
                for k in 0..cols {
                    let delta = if c == k { 1.0 } else { 0.0 };
                    let dsoft = sm[c] * (delta - sm[k]);
                    deriv += dsoft * jv[(start + k) * params + p];
                }
                j_out[base_out + p] = deriv;
            }
        }
    }

    let mut res_val = logits.zero_clone();
    res_val.fill_from_flat(&out);
    let mut res_jac = j_logits.zero_clone();
    res_jac.fill_from_flat(&j_out);
    (res_val, res_jac)
}

pub fn log(
    val: &dyn LossInput,
    j_val: &dyn LossJacobian,
) -> (Box<dyn LossInput>, Box<dyn LossJacobian>) {
    let rows = val.rows();
    let cols = val.cols();
    let params = j_val.params();
    let v = val.to_flat();
    let jv = j_val.to_flat();

    let mut out = vec![0.0f32; rows * cols];
    let mut j_out = vec![0.0f32; rows * cols * params];
    for i in 0..(rows * cols) {
        let x = v[i];
        out[i] = x.ln();
        let deriv = 1.0 / x;
        let base = i * params;
        for k in 0..params {
            j_out[base + k] = deriv * jv[base + k];
        }
    }

    let mut res_val = val.zero_clone();
    res_val.fill_from_flat(&out);
    let mut res_jac = j_val.zero_clone();
    res_jac.fill_from_flat(&j_out);
    (res_val, res_jac)
}

pub fn gather_neg_mean(
    val: &dyn LossInput,
    j_val: &dyn LossJacobian,
    target: &dyn LossInput,
) -> (f32, Vec<f32>) {
    let rows = val.rows();
    let cols = val.cols();
    let params = j_val.params();
    let v = val.to_flat();
    let jv = j_val.to_flat();
    let t = target.to_flat();

    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; params];
    for r in 0..rows {
        let class_idx = t[r] as usize;
        assert!(class_idx < cols, "gather_neg_mean: class index {} out of bounds (cols={})", class_idx, cols);
        let idx = r * cols + class_idx;
        loss += v[idx];
        let base = idx * params;
        for p in 0..params {
            grad[p] += jv[base + p];
        }
    }
    let n = rows as f32;
    loss = -loss / n;
    for p in 0..params {
        grad[p] = -grad[p] / n;
    }
    (loss, grad)
}