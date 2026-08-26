use std::sync::atomic::{AtomicUsize, Ordering};

/// Выполняет полное преобразование градиента по алгоритму Adam.
/// Состояние: два числа на параметр — `m` и `v`.
/// Счётчик шага реализован на `AtomicUsize`, что безопасно для потоков.
pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    step_counter: AtomicUsize,
}

impl Adam {
    pub fn new(beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            beta1,
            beta2,
            eps,
            step_counter: AtomicUsize::new(0),
        }
    }
}