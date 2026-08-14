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
    #[deprecated(note = "Use apply_all_buffered_handle for MemoryExecutor integration")]
    #[allow(deprecated)]
    pub fn apply_all(&self, params: &mut [f32], grads: &mut [f32], state: &mut [f32]) {
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
    #[deprecated(note = "Use apply_all_buffered_handle for MemoryExecutor integration")]
    #[allow(deprecated)]
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

        self.apply_all(p, g, s);
    }

    /// Применяет все кубики последовательно, работая с дескрипторами
    /// `MatrixBufferHandle`.
    ///
    /// Каждый кубик получает свой собственный дескриптор состояния из
    /// переданного вектора `states`. Вектор должен иметь длину, равную
    /// числу кубиков в цепочке.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров.
    /// * `grads`  – дескриптор градиентов.
    /// * `states` – вектор дескрипторов состояний, по одному на каждый кубик.
    ///   Для кубиков без состояния можно передать пустой дескриптор.
    ///
    /// # Паника
    /// Паникует, если `params`, `grads` или любой из `states` являются GPU-буферами,
    /// либо если длина `states` не совпадает с числом кубиков.
    pub fn apply_all_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        states: &[MatrixBufferHandle],
    ) {
        assert_eq!(states.len(), self.cubes.len(),
            "OptimizerChain::apply_all_buffered_handle: states length must match cubes count");

        for (cube, state) in self.cubes.iter().zip(states.iter()) {
            cube.apply_buffered_handle(params, grads, state);
        }
    }
}