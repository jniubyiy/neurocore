// src/plans/optimizer_plan/expr.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};

use super::chain::OptimizerChain;

/// Интерпретатор оптимизатора, объединяющий цепочку кубиков и их состояние.
///
/// Работает полностью на `MatrixBufferHandle` через буферизованный путь.
pub struct OptimizerExpr {
    chain: OptimizerChain,
    /// Состояния для каждого кубика в буферизованном пути.
    states: Vec<MatrixBufferHandle>,
    step_counter: usize,
}

impl OptimizerExpr {
    /// Создаёт оптимизатор, который работает полностью на `MatrixBufferHandle`.
    ///
    /// Для каждого кубика выделяется отдельный `MatrixBufferHandle` через
    /// `TempMatrixPool`. Состояния сохраняются между вызовами `step_buffered_handle`.
    ///
    /// # Аргументы
    /// * `memory_executor` – менеджер памяти (используется косвенно через `pool`).
    /// * `num_params` – количество оптимизируемых параметров.
    /// * `chain` – цепочка кубиков.
    /// * `pool` – пул временных матриц для выделения состояний.
    pub fn new_buffered_handle(
        _memory_executor: Arc<Mutex<MemoryExecutor>>,
        num_params: usize,
        chain: OptimizerChain,
        pool: &mut TempMatrixPool,
    ) -> Self {
        let mut states = Vec::with_capacity(chain.cubes().len());
        for cube in chain.cubes() {
            let state_size = cube.state_size_per_param();
            if state_size > 0 {
                // Храним состояние как вектор размером `num_params * state_size` в столбце.
                let handle = pool.acquire(num_params * state_size, 1);
                states.push(handle);
            } else {
                // Для кубиков без состояния используем пустой handle.
                let empty = pool.acquire(0, 0);
                states.push(empty);
            }
        }

        Self {
            chain,
            states,
            step_counter: 0,
        }
    }

    /// Выполняет один шаг оптимизации, работая полностью с `MatrixBufferHandle`.
    ///
    /// Параметры и градиенты изменяются in‑place через `write()`/`read()`.
    /// Никаких копирований в `Vec<f32>` не выполняется.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров (CPU).
    /// * `grads`  – дескриптор градиентов (CPU).
    ///
    /// # Паника
    /// Паникует, если `params` или `grads` являются GPU‑буферами, или если
    /// буферизованный путь не был инициализирован (состояния отсутствуют).
    pub fn step_buffered_handle(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
    ) {
        assert!(!params.is_gpu() && !grads.is_gpu(),
            "step_buffered_handle supports only CPU handles");
        assert_eq!(self.states.len(), self.chain.cubes().len(),
            "OptimizerExpr was not initialized with new_buffered_handle");

        self.chain.apply_all_buffered_handle(params, grads, &self.states);
        self.step_counter += 1;
    }

    /// Возвращает номер текущего шага (начиная с 1 после первого вызова шага).
    pub fn current_step(&self) -> usize {
        self.step_counter
    }
}