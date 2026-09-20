// src/plans/optimizer_plan/expr.rs

use std::sync::{Arc, RwLock};

use crate::compute_manager::operators_v2::gpu_v2::compute::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::operators_v2::memory_v2::MemoryExecutor;

use super::chain::OptimizerChain;

/// Интерпретатор оптимизатора, объединяющий цепочку кубиков и их состояние.
///
/// Работает с дескрипторами `MatrixBufferHandle`. Поддерживает выполнение
/// шага как для CPU-буферов, так и для GPU-буферов (с автоматическим
/// копированием на CPU, выполнением шага и возвратом на GPU).
///
/// # Фазы (MIGRATION_PLAN.md §7, инвариант I-1)
///
/// Шаг оптимизатора разделён на две независимые операции:
///
/// * `modify_grads_step_*` — модификация градиента (lr, momentum, adam,
///   clip, weight_decay). Не обновляет параметры.
///
/// * `apply_update_step_*` — обновление параметров (`params -= grads`).
///   Не модифицирует градиент.
///
/// Между этими двумя фазами в общий цикл обучения встраивается `adapter_pass`
/// (per-layer коррекция градиента).
pub struct OptimizerExpr {
    chain: OptimizerChain,
    /// Состояния для каждого кубика в буферизованном пути.
    /// Всегда хранятся на CPU для простоты.
    states: Vec<MatrixBufferHandle>,
    /// Счётчик шагов. Инкрементируется один раз за полный шаг обучения,
    /// то есть в фазе `modify_grads`.
    step_counter: usize,
}

impl OptimizerExpr {
    /// Создаёт оптимизатор, который работает полностью на `MatrixBufferHandle`.
    ///
    /// Для каждого кубика выделяется отдельный `MatrixBufferHandle` через
    /// `TempMatrixPool`. Состояния сохраняются между вызовами шага.
    ///
    /// # Аргументы
    /// * `memory_executor` – менеджер памяти (используется косвенно через `pool`).
    /// * `num_params` – количество оптимизируемых параметров.
    /// * `chain` – цепочка кубиков.
    /// * `pool` – пул временных матриц для выделения состояний.
    pub fn new_buffered_handle(
        _memory_executor: Arc<RwLock<MemoryExecutor>>,
        num_params: usize,
        chain: OptimizerChain,
        pool: &mut TempMatrixPool,
    ) -> Self {
        let mut states = Vec::with_capacity(chain.cubes().len());
        for cube in chain.cubes() {
            let state_size = cube.state_size_per_param();
            if state_size > 0 {
                // Храним состояние как вектор размером `num_params * state_size` в столбце.
                let handle = pool.acquire(num_params * state_size, 1);
                states.push(handle);
            } else {
                // Для кубиков без состояния используем пустой handle.
                let empty = pool.acquire(0, 0);
                states.push(empty);
            }
        }

        Self {
            chain,
            states,
            step_counter: 0,
        }
    }

    // ------------------------------------------------------------------------
    // Фаза 1: модификация градиента (CPU-only)
    // ------------------------------------------------------------------------

    /// Модифицирует градиент (CPU-only).
    ///
    /// Применяет все кубики цепочки, кроме `ApplyUpdate`. Обновляет
    /// `step_counter`.
    ///
    /// # Паника
    /// Паникует, если `params` или `grads` являются GPU-буферами.
    /// Используйте `modify_grads_step_buffered_handle_hybrid` для GPU.
    pub fn modify_grads_step_buffered_handle(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
    ) {
        assert!(
            !params.is_gpu() && !grads.is_gpu(),
            "modify_grads_step_buffered_handle supports only CPU handles. \
             Use modify_grads_step_buffered_handle_hybrid for GPU."
        );
        assert_eq!(
            self.states.len(),
            self.chain.cubes().len(),
            "OptimizerExpr was not initialized with new_buffered_handle"
        );

        self.chain.modify_grads_buffered_handle(params, grads, &self.states);
        self.step_counter += 1;
    }

    // ------------------------------------------------------------------------
    // Фаза 2: обновление параметров (CPU-only)
    // ------------------------------------------------------------------------

    /// Обновляет параметры (CPU-only).
    ///
    /// Применяет только `ApplyUpdate`. Не модифицирует градиент и не
    /// инкрементирует `step_counter`.
    ///
    /// # Паника
    /// Паникует, если `params` или `grads` являются GPU-буферами.
    /// Используйте `apply_update_step_buffered_handle_hybrid` для GPU.
    pub fn apply_update_step_buffered_handle(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
    ) {
        assert!(
            !params.is_gpu() && !grads.is_gpu(),
            "apply_update_step_buffered_handle supports only CPU handles. \
             Use apply_update_step_buffered_handle_hybrid for GPU."
        );
        assert_eq!(
            self.states.len(),
            self.chain.cubes().len(),
            "OptimizerExpr was not initialized with new_buffered_handle"
        );

        self.chain.apply_update_buffered_handle(params, grads, &self.states);
    }

    // ------------------------------------------------------------------------
    // Фаза 1: модификация градиента (hybrid)
    // ------------------------------------------------------------------------

    /// Модифицирует градиент для возможно GPU-буферов.
    ///
    /// Если `params`/`grads` на GPU — они временно копируются на CPU, шаг
    /// выполняется на CPU. Обратно на GPU заливаются **только `grads`**:
    /// по инварианту I-1 фаза `modify_grads` не изменяет `params`.
    ///
    /// Состояния оптимизатора всегда находятся на CPU.
    ///
    /// # Паника
    /// Паникует, если один из буферов GPU, а другой CPU, или если
    /// `gpu_compute` не предоставлен для GPU-буферов.
    pub fn modify_grads_step_buffered_handle_hybrid(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        gpu_compute: Option<&GpuCompute>,
    ) {
        let params_is_gpu = params.is_gpu();
        let grads_is_gpu = grads.is_gpu();

        if params_is_gpu || grads_is_gpu {
            assert!(
                params_is_gpu && grads_is_gpu,
                "Mixed CPU/GPU buffers not supported. \
                 params_is_gpu={}, grads_is_gpu={}",
                params_is_gpu,
                grads_is_gpu
            );
            let gpu = gpu_compute.expect("GPU buffers require GpuCompute reference");

            let cpu_params = gpu.download_gpu_handle_to_cpu_handle(params);
            let cpu_grads = gpu.download_gpu_handle_to_cpu_handle(grads);

            self.modify_grads_step_buffered_handle(&cpu_params, &cpu_grads);

            // ФАЗА 1 (MIGRATION_PLAN.md §7, инвариант I-1):
            // modify_grads не модифицирует params, поэтому заливаем
            // обратно только grads. Это экономит одну GPU↔CPU копию.
            gpu.copy_cpu_to_gpu_handle(&cpu_grads, grads);
        } else {
            self.modify_grads_step_buffered_handle(params, grads);
        }
    }

    // ------------------------------------------------------------------------
    // Фаза 2: обновление параметров (hybrid)
    // ------------------------------------------------------------------------

    /// Обновляет параметры для возможно GPU-буферов.
    ///
    /// Если `params`/`grads` на GPU — они временно копируются на CPU, шаг
    /// выполняется на CPU. Обратно на GPU заливаются **только `params`**:
    /// по инварианту I-1 фаза `apply_update` не изменяет `grads`, а после
    /// неё градиенты уже не используются до следующего forward.
    ///
    /// # Паника
    /// Паникует, если один из буферов GPU, а другой CPU, или если
    /// `gpu_compute` не предоставлен для GPU-буферов.
    pub fn apply_update_step_buffered_handle_hybrid(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        gpu_compute: Option<&GpuCompute>,
    ) {
        let params_is_gpu = params.is_gpu();
        let grads_is_gpu = grads.is_gpu();

        if params_is_gpu || grads_is_gpu {
            assert!(
                params_is_gpu && grads_is_gpu,
                "Mixed CPU/GPU buffers not supported. \
                 params_is_gpu={}, grads_is_gpu={}",
                params_is_gpu,
                grads_is_gpu
            );
            let gpu = gpu_compute.expect("GPU buffers require GpuCompute reference");

            let cpu_params = gpu.download_gpu_handle_to_cpu_handle(params);
            let cpu_grads = gpu.download_gpu_handle_to_cpu_handle(grads);

            self.apply_update_step_buffered_handle(&cpu_params, &cpu_grads);

            // ФАЗА 1 (MIGRATION_PLAN.md §7, инвариант I-1):
            // apply_update модифицирует params, grads остаются как есть.
            // Заливаем обратно только params.
            gpu.copy_cpu_to_gpu_handle(&cpu_params, params);
        } else {
            self.apply_update_step_buffered_handle(params, grads);
        }
    }

    /// Возвращает номер текущего шага (начиная с 1 после первого вызова
    /// `modify_grads_step_*`).
    pub fn current_step(&self) -> usize {
        self.step_counter
    }
}