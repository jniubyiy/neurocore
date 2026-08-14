// src/plans/optimizer_plan/chain.rs

use super::cube::OptimizerCube;
use crate::compute_manager::matrix_buffer::{MatrixBuffer, MatrixBufferHandle};

/// Цепочка кубиков оптимизации.
///
/// Последовательно применяет каждый кубик, передавая ему
/// соответствующий срез состояния.
pub struct OptimizerChain {
    cubes: Vec<Box<dyn OptimizerCube>>,
}

impl OptimizerChain {
    /// Создаёт пустую цепочку.
    pub fn new() -> Self {
        Self { cubes: Vec::new() }
    }

    /// Добавляет кубик в конец цепочки.
    pub fn add(mut self, cube: Box<dyn OptimizerCube>) -> Self {
        self.cubes.push(cube);
        self
    }

    /// Возвращает срез кубиков.
    pub fn cubes(&self) -> &[Box<dyn OptimizerCube>] {
        &self.cubes
    }

    /// Общий размер состояния на один параметр
    /// (сумма `state_size_per_param` всех кубиков).
    pub fn total_state_size_per_param(&self) -> usize {
        self.cubes.iter().map(|c| c.state_size_per_param()).sum()
    }

    /// Применяет все кубики последовательно, используя обычные срезы `[f32]`.
    ///
    /// # Аргументы
    /// * `params` - все параметры модели (мутабельный срез).
    /// * `grads`  - градиенты (мутабельный срез, изменяется кубиками).
    /// * `state`  - полное состояние цепочки (мутабельный срез).
    ///   Длина должна быть `params.len() * total_state_size_per_param()`.
    pub fn apply_all(&self, params: &mut [f32], grads: &mut [f32], state: &mut [f32]) {
        self.apply_all_slices(params, grads, state);
    }

    /// Внутренний метод, применяющий кубики к переданным срезам.
    /// Выделен для избежания дублирования кода между разными обёртками.
    fn apply_all_slices(&self, params: &mut [f32], grads: &mut [f32], state: &mut [f32]) {
        let num_params = params.len();
        let mut offset = 0;
        for cube in &self.cubes {
            let size_per_param = cube.state_size_per_param();
            let state_len = num_params * size_per_param;
            let state_slice = &mut state[offset..offset + state_len];
            cube.apply(params, grads, state_slice);
            offset += state_len;
        }
    }

    /// Применяет все кубики последовательно, работая с управляемыми буферами
    /// `MatrixBuffer`. Внутри извлекаются обычные слайсы, поэтому метод
    /// эквивалентен `apply_all`, но принимает `MatrixBuffer` для удобства
    /// интеграции с памятью под контролем `MemoryExecutor`.
    ///
    /// # Паника
    /// Паникует, если любой из переданных буферов находится на GPU.
    pub fn apply_all_buffered(
        &self,
        params: &mut MatrixBuffer,
        grads: &mut MatrixBuffer,
        state: &mut MatrixBuffer,
    ) {
        assert!(!params.is_gpu() && !grads.is_gpu() && !state.is_gpu(),
            "apply_all_buffered supports only CPU buffers");

        let p = params.as_slice_mut();
        let g = grads.as_slice_mut();
        let s = state.as_slice_mut();

        self.apply_all_slices(p, g, s);
    }

    /// Применяет все кубики последовательно, работая с дескрипторами
    /// `MatrixBufferHandle`. Данные временно копируются в CPU-векторы,
    /// обрабатываются, затем записываются обратно. Такой подход позволяет
    /// использовать новую модель памяти без изменения самих кубиков.
    ///
    /// # Паника
    /// Паникует, если любой из переданных дескрипторов ссылается на GPU-буфер.
    pub fn apply_all_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(!params.is_gpu(), "params handle must be CPU");
        assert!(!grads.is_gpu(), "grads handle must be CPU");
        assert!(!state.is_gpu(), "state handle must be CPU");

        let num_params = params.rows() * params.cols();

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
            num_params * self.total_state_size_per_param(),
            "state size mismatch"
        );

        // Применяем кубики
        self.apply_all_slices(&mut p_vec, &mut g_vec, &mut s_vec);

        // Записываем обратно
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
    }
}