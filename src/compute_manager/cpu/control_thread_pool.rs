// src/compute_manager/cpu/control_thread_pool.rs

use std::sync::Arc;

use crate::compute_manager::cpu::worker_pool::WorkerPool;
use crate::compute_manager::executor::Executor;

/// Пул управляющих потоков.
///
/// Предназначен для выполнения задач, связанных с координацией,
/// планированием, подготовкой данных и запуском GPU-операций.
/// Не используется для ресурсоёмких вычислений, поэтому обычно
/// содержит небольшое число потоков (1–2).
///
/// Потоки этого пула создаются с увеличенным размером стека (32 МБ),
/// так как управляющие потоки могут выполнять вызовы Vulkan, которым
/// требуется значительное пространство стека.
///
/// Реализует [`Executor`], что позволяет унифицировать взаимодействие
/// с другими компонентами системы.
pub struct ControlThreadPool {
    pool: Arc<WorkerPool>,
}

impl ControlThreadPool {
    /// Создаёт новый пул с заданным числом потоков.
    ///
    /// # Аргументы
    /// * `num_threads` – количество управляющих потоков.
    ///
    /// # Паника
    /// Паникует, если `num_threads` равно нулю.
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "ControlThreadPool requires at least one thread");
        // Управляющие потоки могут запускать GPU-операции (Vulkan),
        // поэтому используем увеличенный стек (32 МБ).
        const GPU_THREAD_STACK_SIZE: usize = 32 * 1024 * 1024;
        Self {
            pool: Arc::new(WorkerPool::new_with_stack_size(
                num_threads,
                GPU_THREAD_STACK_SIZE,
            )),
        }
    }
}

impl Clone for ControlThreadPool {
    /// Клонирование пула: все клоны разделяют один и тот же `WorkerPool`,
    /// поэтому задачи, отправленные через любой клон, попадают в общую очередь.
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

impl Executor for ControlThreadPool {
    fn execute_dyn(&self, f: Box<dyn FnOnce() + Send>) {
        self.pool.execute(f);
    }

    fn wait_all(&self) {
        self.pool.wait_all();
    }

    fn num_workers(&self) -> usize {
        self.pool.num_workers()
    }

    fn plan_chunks_assignment(&self, total_tasks: usize) -> Vec<Vec<(usize, usize, usize)>> {
        // Управляющий пул не выполняет параллельных вычислений,
        // поэтому возвращаем одно "назначение" на каждый поток,
        // где все задачи приписаны первому потоку (или всем, если потоков несколько,
        // но реально используется только первый).
        let workers = self.num_workers();
        let mut assignment = vec![Vec::new(); workers];
        if total_tasks > 0 && workers > 0 {
            assignment[0].push((0, total_tasks, total_tasks));
        }
        assignment
    }

    fn clone_executor(&self) -> Box<dyn Executor> {
        Box::new(self.clone())
    }
}