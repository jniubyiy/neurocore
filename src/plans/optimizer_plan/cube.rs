// src/plans/optimizer_plan/cube.rs

use std::any::Any;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
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

    /// Применяет кубик к дескрипторам `MatrixBufferHandle`.
    ///
    /// Реализация по умолчанию копирует данные из дескрипторов во временные
    /// векторы, вызывает [`apply`], затем записывает результаты обратно.
    /// Это обеспечивает совместимость с новой моделью памяти без необходимости
    /// переопределять метод в каждом кубике. Однако для максимальной
    /// производительности кубики могут предоставить собственную реализацию,
    /// работающую напрямую с дескрипторами.
    ///
    /// # Паника
    /// Паникует, если какой-либо из дескрипторов ссылается на GPU-буфер.
    fn apply_buffered_handle(
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
            num_params * self.state_size_per_param(),
            "state size mismatch"
        );

        // Применяем кубик
        self.apply(&mut p_vec, &mut g_vec, &mut s_vec);

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

    /// Позволяет downcasting к конкретному типу кубика.
    fn as_any(&self) -> &dyn Any;
}