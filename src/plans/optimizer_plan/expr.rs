// src/optimizer/expr.rs

use super::chain::OptimizerChain;
use super::state::OptimizerState;

/// Интерпретатор оптимизатора, объединяющий цепочку кубиков и их состояние.
pub struct OptimizerExpr {
    chain: OptimizerChain,
    state: OptimizerState,
    step_counter: usize,
}

impl OptimizerExpr {
    /// Создаёт новый оптимизатор для заданного количества параметров и цепочки кубиков.
    pub fn new(num_params: usize, chain: OptimizerChain) -> Self {
        let total_state = chain.total_state_size_per_param();
        Self {
            chain,
            state: OptimizerState::new(num_params, total_state),
            step_counter: 0,
        }
    }

    /// Выполняет один шаг оптимизации, изменяя параметры in‑place.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        let mut grads_mut = grads.to_vec();
        self.chain.apply_all(params, &mut grads_mut, self.state.as_mut_slice());
        self.step_counter += 1;
    }

    /// Возвращает номер текущего шага (начиная с 1 после первого вызова `step`).
    pub fn current_step(&self) -> usize {
        self.step_counter
    }
}