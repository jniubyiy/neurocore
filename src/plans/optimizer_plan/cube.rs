// src/plans/optimizer_plan/cube.rs

use std::any::Any;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

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

    /// Применяет кубик к срезам параметров, градиентов и состояния.
    ///
    /// # Аргументы
    /// * `params` - мутабельный срез всех параметров модели (длина `N`).
    /// * `grads`  - мутабельный срез градиентов (длина `N`).
    ///   Кубик может изменять градиенты in‑place (например,
    ///   `ScaleGradient` умножает их, `AddWeightDecay` добавляет слагаемое).
    /// * `state`  - мутабельный срез состояния кубика.
    ///   Его длина равна `N * state_size_per_param()`, где `N` — число параметров.
    fn apply(&self, params: &mut [f32], grads: &mut [f32], state: &mut [f32]);

    /// То же, что и `apply`, но принимает управляемые буферы `MatrixBuffer`.
    ///
    /// По умолчанию паникует, если конкретный кубик не переопределил этот метод.
    /// Буферизованная версия должна использоваться для полной интеграции
    /// с `MemoryExecutor` и пулом временных матриц.
    fn apply_buffered(
        &self,
        _params: &mut MatrixBuffer,
        _grads: &mut MatrixBuffer,
        _state: &mut MatrixBuffer,
    ) {
        panic!("apply_buffered not implemented for this cube");
    }

    /// Позволяет downcasting к конкретному типу кубика.
    fn as_any(&self) -> &dyn Any;
}