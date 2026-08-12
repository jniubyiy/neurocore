// src/plans/optimizer_plan/expr.rs

use super::chain::OptimizerChain;
use super::state::OptimizerState;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

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
    ///
    /// # Аргументы
    /// * `params` – мутабельный срез всех параметров модели.
    /// * `grads`  – срез градиентов. Кубики могут изменять градиенты
    ///   во временном буфере, но исходный `grads` не изменяется.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        let mut grads_mut = grads.to_vec();
        self.chain.apply_all(params, &mut grads_mut, self.state.as_mut_slice());
        self.step_counter += 1;
    }

    /// Выполняет один шаг оптимизации, принимая параметры и градиенты
    /// в виде управляемых буферов `MatrixBuffer` (только CPU).
    ///
    /// Внутри извлекаются обычные слайсы, а градиенты копируются во временный
    /// вектор, поэтому исходный `grads` не изменяется. Этот метод удобен для
    /// интеграции с буферизованным графом, когда параметры и градиенты уже
    /// находятся под управлением `MemoryExecutor`.
    ///
    /// # Паника
    /// Паникует, если `params` или `grads` являются GPU‑буферами.
    pub fn step_buffered(&mut self, params: &mut MatrixBuffer, grads: &MatrixBuffer) {
        assert!(!params.is_gpu() && !grads.is_gpu(),
            "step_buffered supports only CPU buffers");

        let mut grads_mut = grads.as_slice().to_vec();
        self.chain.apply_all(
            params.as_slice_mut(),
            &mut grads_mut,
            self.state.as_mut_slice(),
        );
        self.step_counter += 1;
    }

    /// Возвращает номер текущего шага (начиная с 1 после первого вызова `step`).
    pub fn current_step(&self) -> usize {
        self.step_counter
    }
}