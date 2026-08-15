// src/compute_manager/gpu/executor.rs

use std::thread;
use crate::compute_manager::executor::Executor;
use super::init::GpuContext;

/// Размер стека (в байтах) для потоков, выполняющих GPU-задачи.
/// 32 МБ достаточно для самых глубоких графов вычислений Vulkan.
const GPU_THREAD_STACK_SIZE: usize = 32 * 1024 * 1024;

#[derive(Clone)]
pub struct GpuExecutor;

impl GpuExecutor {
    pub fn new(_context: GpuContext) -> Self {
        Self
    }
}

impl Executor for GpuExecutor {
    fn execute_dyn(&self, f: Box<dyn FnOnce() + Send>) {
        // Запускаем GPU-операцию в отдельном потоке с большим стеком,
        // чтобы избежать переполнения стека на Windows при работе с Vulkan.
        let handle = thread::Builder::new()
            .stack_size(GPU_THREAD_STACK_SIZE)
            .spawn(f)
            .expect("Failed to spawn GPU worker thread");
        // Дожидаемся завершения, сохраняя синхронность интерфейса.
        handle.join().expect("GPU worker thread panicked");
    }

    fn wait_all(&self) {
        // Все задачи выполняются синхронно, поэтому wait_all ничего не делает.
    }

    fn num_workers(&self) -> usize {
        1
    }

    fn plan_chunks_assignment(&self, total_tasks: usize) -> Vec<Vec<(usize, usize)>> {
        if total_tasks == 0 {
            return vec![Vec::new(); 1];
        }
        vec![vec![(0, total_tasks)]]
    }

    fn clone_executor(&self) -> Box<dyn Executor> {
        Box::new(self.clone())
    }
}

