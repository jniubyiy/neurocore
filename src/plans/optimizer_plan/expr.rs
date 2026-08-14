// src/plans/optimizer_plan/expr.rs

use super::chain::OptimizerChain;
use super::state::OptimizerState;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

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

    /// Выполняет один шаг оптимизации, принимая параметры, градиенты и
    /// состояние в виде дескрипторов `MatrixBufferHandle` (все CPU).
    ///
    /// Метод копирует данные из дескрипторов во временные векторы, применяет
    /// цепочку кубиков и записывает результаты обратно. Это позволяет
    /// использовать новую модель памяти с `MatrixBufferHandle` без изменения
    /// самих кубиков.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров (мутабельный, CPU).
    /// * `grads`  – дескриптор градиентов (CPU).
    /// * `state`  – дескриптор состояния оптимизатора (CPU). Его размер
    ///   должен быть `num_params * total_state_size_per_param()`.
    ///
    /// # Паника
    /// Паникует, если любой из дескрипторов является GPU‑буфером или
    /// размеры не совпадают.
    pub fn step_buffered_handle(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(!params.is_gpu(), "params handle must be CPU");
        assert!(!grads.is_gpu(), "grads handle must be CPU");
        assert!(!state.is_gpu(), "state handle must be CPU");

        let num_params = params.rows() * params.cols();
        let total_state_per_param = self.chain.total_state_size_per_param();

        // Копируем данные во временные векторы
        let mut p_vec = {
            let guard = params.read();
            guard.as_slice().expect("params must be CPU").to_vec()
        };
        let mut g_vec = {
            let guard = grads.read();
            guard.as_slice().expect("grads must be CPU").to_vec()
        };
        let mut s_vec = {
            let guard = state.read();
            guard.as_slice().expect("state must be CPU").to_vec()
        };

        // Проверяем размеры
        assert_eq!(p_vec.len(), num_params, "params size mismatch");
        assert_eq!(g_vec.len(), num_params, "grads size mismatch");
        assert_eq!(
            s_vec.len(),
            num_params * total_state_per_param,
            "state size mismatch"
        );

        // Применяем цепочку кубиков
        self.chain.apply_all(&mut p_vec, &mut g_vec, &mut s_vec);

        // Записываем результаты обратно в дескрипторы
        {
            let mut p_guard = params.write();
            p_guard.as_slice_mut().expect("params must be CPU").copy_from_slice(&p_vec);
        }
        {
            let mut g_guard = grads.write();
            g_guard.as_slice_mut().expect("grads must be CPU").copy_from_slice(&g_vec);
        }
        {
            let mut s_guard = state.write();
            s_guard.as_slice_mut().expect("state must be CPU").copy_from_slice(&s_vec);
        }

        self.step_counter += 1;
    }

    /// Возвращает номер текущего шага (начиная с 1 после первого вызова `step`).
    pub fn current_step(&self) -> usize {
        self.step_counter
    }
}