// src/plans/optimizer_plan/cubes.rs

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use super::cube::OptimizerCube;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

// ----------------------------------------------------------------
// ScaleGradient
// ----------------------------------------------------------------

/// Умножает градиент на заданный коэффициент (обычно learning rate).
pub struct ScaleGradient {
    pub factor: f32,
}

impl ScaleGradient {
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl OptimizerCube for ScaleGradient {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu(), "ScaleGradient: grads must be CPU");
        let mut grad_guard = grads.write();
        let grad_slice = grad_guard.as_slice_mut().expect("ScaleGradient: expected CPU buffer");
        for g in grad_slice.iter_mut() {
            *g *= self.factor;
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

// ----------------------------------------------------------------
// AddWeightDecay
// ----------------------------------------------------------------

/// Добавляет L2‑регуляризацию к градиенту:
/// `grads[i] += decay * params[i]`
pub struct AddWeightDecay {
    pub decay: f32,
}

impl AddWeightDecay {
    pub fn new(decay: f32) -> Self {
        Self { decay }
    }
}

impl OptimizerCube for AddWeightDecay {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(!params.is_gpu() && !grads.is_gpu(), "AddWeightDecay: params and grads must be CPU");
        let param_guard = params.read();
        let p_slice = param_guard.as_slice().expect("AddWeightDecay: expected CPU buffer");
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("AddWeightDecay: expected CPU buffer");

        debug_assert_eq!(p_slice.len(), g_slice.len());

        for i in 0..g_slice.len() {
            g_slice[i] += self.decay * p_slice[i];
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

// ----------------------------------------------------------------
// GradientClip
// ----------------------------------------------------------------

/// Обрезает градиент по значениям `[min, max]`.
pub struct GradientClip {
    pub min: Option<f32>,
    pub max: Option<f32>,
}

impl GradientClip {
    pub fn new(min: Option<f32>, max: Option<f32>) -> Self {
        Self { min, max }
    }
}

impl OptimizerCube for GradientClip {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu(), "GradientClip: grads must be CPU");
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("GradientClip: expected CPU buffer");

        for g in g_slice.iter_mut() {
            if let Some(min_val) = self.min {
                *g = g.max(min_val);
            }
            if let Some(max_val) = self.max {
                *g = g.min(max_val);
            }
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

// ----------------------------------------------------------------
// Momentum
// ----------------------------------------------------------------

/// Классический момент (momentum).
/// Состояние: одно число на параметр — скорость `v`.
pub struct Momentum {
    pub beta: f32,
}

impl Momentum {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl OptimizerCube for Momentum {
    fn state_size_per_param(&self) -> usize {
        1
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu() && !state.is_gpu(), "Momentum: grads and state must be CPU");
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("Momentum: expected CPU buffer");
        let mut state_guard = state.write();
        let s_slice = state_guard.as_slice_mut().expect("Momentum: expected CPU buffer");

        debug_assert_eq!(g_slice.len(), s_slice.len());

        for i in 0..g_slice.len() {
            let v = self.beta * s_slice[i] + g_slice[i];
            s_slice[i] = v;
            g_slice[i] = v;
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

// ----------------------------------------------------------------
// NesterovMomentum
// ----------------------------------------------------------------

/// Момент Нестерова (NAG).
/// Состояние: одно число на параметр — скорость `v`.
pub struct NesterovMomentum {
    pub beta: f32,
}

impl NesterovMomentum {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl OptimizerCube for NesterovMomentum {
    fn state_size_per_param(&self) -> usize {
        1
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu() && !state.is_gpu(), "NesterovMomentum: grads and state must be CPU");
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("NesterovMomentum: expected CPU buffer");
        let mut state_guard = state.write();
        let s_slice = state_guard.as_slice_mut().expect("NesterovMomentum: expected CPU buffer");

        debug_assert_eq!(g_slice.len(), s_slice.len());

        for i in 0..g_slice.len() {
            let v_old = s_slice[i];
            let v_new = self.beta * v_old + g_slice[i];
            g_slice[i] += self.beta * v_new;
            s_slice[i] = v_new;
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

// ----------------------------------------------------------------
// AdamTransform
// ----------------------------------------------------------------

/// Выполняет полное преобразование градиента по алгоритму Adam.
/// Состояние: два числа на параметр — `m` и `v`.
/// Счётчик шага реализован на `AtomicUsize`, что безопасно для потоков.
pub struct AdamTransform {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    step_counter: AtomicUsize,
}

impl AdamTransform {
    pub fn new(beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            beta1,
            beta2,
            eps,
            step_counter: AtomicUsize::new(0),
        }
    }
}

impl OptimizerCube for AdamTransform {
    fn state_size_per_param(&self) -> usize {
        2
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu() && !state.is_gpu(), "AdamTransform: grads and state must be CPU");
        let n = grads.rows() * grads.cols();

        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("AdamTransform: expected CPU buffer");
        let mut state_guard = state.write();
        let s_slice = state_guard.as_slice_mut().expect("AdamTransform: expected CPU buffer");

        debug_assert_eq!(s_slice.len(), n * 2);

        let (m_slice, v_slice) = s_slice.split_at_mut(n);

        let t = self.step_counter.fetch_add(1, Ordering::SeqCst) + 1;

        let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

        for i in 0..n {
            m_slice[i] = self.beta1 * m_slice[i] + (1.0 - self.beta1) * g_slice[i];
            v_slice[i] = self.beta2 * v_slice[i] + (1.0 - self.beta2) * g_slice[i] * g_slice[i];

            let m_hat = m_slice[i] / bias_correction1;
            let v_hat = v_slice[i] / bias_correction2;

            g_slice[i] = m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

// ----------------------------------------------------------------
// ApplyUpdate
// ----------------------------------------------------------------

/// Применяет накопленный градиент к параметрам: `params[i] -= grads[i]`
/// Этот кубик всегда должен быть последним в цепочке.
pub struct ApplyUpdate;

impl ApplyUpdate {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizerCube for ApplyUpdate {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(!params.is_gpu() && !grads.is_gpu(), "ApplyUpdate: params and grads must be CPU");
        let mut param_guard = params.write();
        let p_slice = param_guard.as_slice_mut().expect("ApplyUpdate: expected CPU buffer");
        let grad_guard = grads.read();
        let g_slice = grad_guard.as_slice().expect("ApplyUpdate: expected CPU buffer");

        debug_assert_eq!(p_slice.len(), g_slice.len());

        for i in 0..p_slice.len() {
            p_slice[i] -= g_slice[i];
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}