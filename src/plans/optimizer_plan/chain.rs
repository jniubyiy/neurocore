// src/plans/optimizer_plan/chain.rs

use super::cube::OptimizerCube;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;

use crate::optimizers::apply_update::ApplyUpdate;

/// Цепочка кубиков оптимизации.
///
/// Последовательно применяет каждый кубик, передавая ему
/// соответствующий срез состояния.
///
/// # Фазы
///
/// В архитектуре v2 шаг оптимизатора разделён на две фазы
/// (MIGRATION_PLAN.md §7, инвариант I-1):
///
/// * `modify_grads_buffered_handle` — все кубики, кроме `ApplyUpdate`;
///   модифицирует только `grads`, не трогает `params`.
///
/// * `apply_update_buffered_handle` — только `ApplyUpdate`;
///   обновляет `params` (`params -= grads`), не трогает `grads`.
///
/// Разделение гарантирует, что между фазами можно вставить `adapter_pass`
/// (per-layer коррекцию градиента) до применения обновления к весам.
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

    /// Фаза 1: модификация градиента.
    ///
    /// Применяет все кубики, кроме `ApplyUpdate`, к `grads` in-place.
    /// `params` передаётся только для чтения (например, `AddWeightDecay`
    /// читает параметры для расчёта вклада в градиент) и НЕ должен
    /// изменяться.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров.
    /// * `grads`  – дескриптор градиентов.
    /// * `states` – вектор дескрипторов состояний, по одному на каждый кубик.
    ///   Для кубиков без состояния передаётся пустой дескриптор.
    ///
    /// # Паника
    /// Паникует, если длина `states` не совпадает с числом кубиков.
    pub fn modify_grads_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        states: &[MatrixBufferHandle],
    ) {
        assert_eq!(
            states.len(),
            self.cubes.len(),
            "OptimizerChain::modify_grads_buffered_handle: states length must match cubes count"
        );

        for (cube, state) in self.cubes.iter().zip(states.iter()) {
            if cube.as_any().downcast_ref::<ApplyUpdate>().is_some() {
                continue;
            }
            cube.apply_buffered_handle(params, grads, state);
        }
    }

    /// Фаза 2: обновление параметров.
    ///
    /// Применяет только кубики `ApplyUpdate` к `params` in-place.
    /// Градиенты не модифицируются.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров.
    /// * `grads`  – дескриптор градиентов.
    /// * `states` – вектор дескрипторов состояний, по одному на каждый кубик.
    ///
    /// # Паника
    /// Паникует, если длина `states` не совпадает с числом кубиков
    /// или если в цепочке нет `ApplyUpdate`.
    pub fn apply_update_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        states: &[MatrixBufferHandle],
    ) {
        assert_eq!(
            states.len(),
            self.cubes.len(),
            "OptimizerChain::apply_update_buffered_handle: states length must match cubes count"
        );

        let mut found = false;
        for (cube, state) in self.cubes.iter().zip(states.iter()) {
            if cube.as_any().downcast_ref::<ApplyUpdate>().is_some() {
                assert!(
                    !found,
                    "OptimizerChain::apply_update_buffered_handle: multiple ApplyUpdate cubes in chain"
                );
                cube.apply_buffered_handle(params, grads, state);
                found = true;
            }
        }
        assert!(
            found,
            "OptimizerChain::apply_update_buffered_handle: no ApplyUpdate cube in chain"
        );
    }
}