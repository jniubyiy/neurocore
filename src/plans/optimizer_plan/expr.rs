// src/plans/optimizer_plan/expr.rs

use std::sync::{Arc, RwLock};

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::memory_executor::MemoryExecutor;

use super::chain::OptimizerChain;

/// Интерпретатор оптимизатора, объединяющий цепочку кубиков и их состояние.
///
/// Работает с дескрипторами `MatrixBufferHandle`. Поддерживает выполнение
/// шага как для CPU-буферов, так и для GPU-буферов (с автоматическим
/// копированием на CPU, выполнением шага и возвратом на GPU).
pub struct OptimizerExpr {
    chain: OptimizerChain,
    /// Состояния для каждого кубика в буферизованном пути.
    /// Всегда хранятся на CPU для простоты.
    states: Vec<MatrixBufferHandle>,
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

    /// Выполняет один шаг оптимизации, работая полностью с `MatrixBufferHandle`.
    ///
    /// Параметры и градиенты изменяются in‑place через `write()`/`read()`.
    /// Поддерживаются только CPU-буферы.
    ///
    /// # Паника
    /// Паникует, если `params` или `grads` являются GPU‑буферами, или если
    /// буферизованный путь не был инициализирован (состояния отсутствуют).
    pub fn step_buffered_handle(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
    ) {
        assert!(!params.is_gpu() && !grads.is_gpu(),
            "step_buffered_handle supports only CPU handles. Use step_buffered_handle_hybrid for GPU.");
        assert_eq!(self.states.len(), self.chain.cubes().len(),
            "OptimizerExpr was not initialized with new_buffered_handle");

        self.chain.apply_all_buffered_handle(params, grads, &self.states);
        self.step_counter += 1;
    }

    /// Выполняет один шаг оптимизации для возможно GPU-буферов.
    ///
    /// Если параметры и градиенты находятся на GPU, они временно копируются
    /// на CPU, шаг выполняется на CPU, затем обновлённые параметры копируются
    /// обратно на GPU. Состояния оптимизатора всегда находятся на CPU.
    ///
    /// # Аргументы
    /// * `params` – дескриптор параметров (CPU или GPU).
    /// * `grads` – дескриптор градиентов (CPU или GPU).
    /// * `gpu_compute` – ссылка на `GpuCompute`, необходимая если буферы GPU.
    ///
    /// # Паника
    /// Паникует, если один из буферов GPU, а другой CPU, или если `gpu_compute`
    /// не предоставлен для GPU-буферов.
    pub fn step_buffered_handle_hybrid(
        &mut self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        gpu_compute: Option<&GpuCompute>,
    ) {
        let params_is_gpu = params.is_gpu();
        let grads_is_gpu = grads.is_gpu();

        if params_is_gpu || grads_is_gpu {
            // Оба должны быть GPU
            assert!(params_is_gpu && grads_is_gpu,
                "Mixed CPU/GPU buffers not supported. params_is_gpu={}, grads_is_gpu={}",
                params_is_gpu, grads_is_gpu);

            let gpu = gpu_compute.expect("GPU buffers require GpuCompute reference");

            // Скачиваем параметры и градиенты в управляемые CPU-буферы.
            let cpu_params = gpu.download_gpu_handle_to_cpu_handle(params);
            let cpu_grads = gpu.download_gpu_handle_to_cpu_handle(grads);

            // Выполняем шаг на CPU.
            self.step_buffered_handle(&cpu_params, &cpu_grads);

            // Загружаем обновлённые параметры обратно на GPU.
            gpu.copy_cpu_to_gpu_handle(&cpu_params, params);
        } else {
            // Оба CPU — обычный шаг.
            self.step_buffered_handle(params, grads);
        }
    }

    /// Возвращает номер текущего шага (начиная с 1 после первого вызова шага).
    pub fn current_step(&self) -> usize {
        self.step_counter
    }
}