// src/optimizer/chain.rs

use super::cube::OptimizerCube;

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

    /// Общий размер состояния на один параметр
    /// (сумма `state_size_per_param` всех кубиков).
    pub fn total_state_size_per_param(&self) -> usize {
        self.cubes.iter().map(|c| c.state_size_per_param()).sum()
    }

    /// Применяет все кубики последовательно.
    ///
    /// # Аргументы
    /// * `params` - все параметры модели (мутабельный срез).
    /// * `grads`  - градиенты (мутабельный срез, изменяется кубиками).
    /// * `state`  - полное состояние цепочки (мутабельный срез).
    ///   Длина должна быть `params.len() * total_state_size_per_param()`.
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
}