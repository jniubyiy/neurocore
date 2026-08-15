// src/plans/optimizer_plan/cube.rs

use std::any::Any;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

/// Атомарный блок оптимизации.
///
/// Каждый кубик выполняет одно элементарное преобразование над градиентами
/// и/или параметрами, используя своё внутреннее состояние.
/// Цепочка кубиков, завершающаяся кубиком `ApplyUpdate`, образует
/// конкретный алгоритм оптимизации.
///
/// # Пример
/// ```ignore
/// let chain = OptimizerChain::new()
///     .add(Box::new(ScaleGradient::new(0.01)))
///     .add(Box::new(Momentum::new(0.9)))
///     .add(Box::new(ApplyUpdate));
/// ```
pub trait OptimizerCube: Send + Sync + Any {
    /// Размер состояния, выделяемого на каждый оптимизируемый параметр.
    ///
    /// Например:
    /// - `0` для кубиков без памяти (`ScaleGradient`, `ApplyUpdate`)
    /// - `1` для `Momentum` (одна скорость на параметр)
    /// - `2` для `AdamTransform` (первый и второй моменты)
    fn state_size_per_param(&self) -> usize;

    /// Применяет кубик к дескрипторам `MatrixBufferHandle`.
    ///
    /// Этот метод является основным и обязательным для реализации.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров (мутабельный через `write()`).
    /// * `grads`  – дескриптор градиентов (мутабельный).
    /// * `state`  – дескриптор состояния кубика.
    ///   Его размер должен быть `num_params * state_size_per_param()`.
    ///   Для кубиков без состояния можно передавать пустой handle.
    fn apply_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    );

    /// Позволяет downcasting к конкретному типу кубика.
    fn as_any(&self) -> &dyn Any;
}