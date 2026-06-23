// src/optimizer/cubes.rs

use super::cube::OptimizerCube;
use std::sync::atomic::{AtomicUsize, Ordering};

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

    fn apply(&self, _params: &mut [f32], grads: &mut [f32], _state: &mut [f32]) {
        for g in grads.iter_mut() {
            *g *= self.factor;
        }
    }
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

    fn apply(&self, params: &mut [f32], grads: &mut [f32], _state: &mut [f32]) {
        for (p, g) in params.iter().zip(grads.iter_mut()) {
            *g += self.decay * *p;
        }
    }
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

    fn apply(&self, _params: &mut [f32], grads: &mut [f32], _state: &mut [f32]) {
        for g in grads.iter_mut() {
            if let Some(min_val) = self.min {
                *g = g.max(min_val);
            }
            if let Some(max_val) = self.max {
                *g = g.min(max_val);
            }
        }
    }
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

    fn apply(&self, _params: &mut [f32], grads: &mut [f32], state: &mut [f32]) {
        let n = grads.len();
        for i in 0..n {
            let v = self.beta * state[i] + grads[i];
            state[i] = v;
            grads[i] = v;
        }
    }
}

// ----------------------------------------------------------------
// NesterovMomentum
// ----------------------------------------------------------------

/// Момент Нестерова.
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

    fn apply(&self, _params: &mut [f32], grads: &mut [f32], state: &mut [f32]) {
        let n = grads.len();
        for i in 0..n {
            let v_old = state[i];
            let v_new = self.beta * v_old + grads[i];
            grads[i] = self.beta * v_new + grads[i];
            state[i] = v_new;
        }
    }
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

    fn apply(&self, _params: &mut [f32], grads: &mut [f32], state: &mut [f32]) {
        let n = grads.len();
        let (m_slice, v_slice) = state.split_at_mut(n);

        // Атомарно увеличиваем счётчик шагов и получаем текущий номер шага
        let t = self.step_counter.fetch_add(1, Ordering::SeqCst) + 1;

        let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

        for i in 0..n {
            m_slice[i] = self.beta1 * m_slice[i] + (1.0 - self.beta1) * grads[i];
            v_slice[i] = self.beta2 * v_slice[i] + (1.0 - self.beta2) * grads[i] * grads[i];

            let m_hat = m_slice[i] / bias_correction1;
            let v_hat = v_slice[i] / bias_correction2;

            grads[i] = m_hat / (v_hat.sqrt() + self.eps);
        }
    }
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

    fn apply(&self, params: &mut [f32], grads: &mut [f32], _state: &mut [f32]) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p -= *g;
        }
    }
}