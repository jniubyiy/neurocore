// src/compute_manager/cpu/compute_thread_pool.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::scheduler::Scheduler;
use crate::compute_manager::cpu::worker_pool::WorkerPool;
use crate::compute_manager::executor::Executor;

/// Пул вычислительных потоков.
///
/// Предназначен для выполнения ресурсоёмких операций (прямой и обратный
/// проходы, вычисление потерь, шаги оптимизатора) на CPU. Пул тесно связан
/// с планировщиком [`Scheduler`], который распределяет батчи между потоками
/// с учётом их производительности.
///
/// Реализует [`Executor`] для унификации интерфейса.
pub struct ComputeThreadPool {
    pool: Arc<WorkerPool>,
    scheduler: Arc<Mutex<Scheduler>>,
}

impl ComputeThreadPool {
    /// Создаёт новый пул вычислительных потоков.
    ///
    /// # Аргументы
    /// * `num_threads` – количество потоков в пуле.
    /// * `scheduler` – планировщик задач, который будет использоваться
    ///   для распределения работы.
    ///
    /// # Паника
    /// Паникует, если `num_threads` равно нулю.
    pub fn new(num_threads: usize, scheduler: Arc<Mutex<Scheduler>>) -> Self {
        assert!(num_threads > 0, "ComputeThreadPool requires at least one thread");

        // Убеждаемся, что планировщик настроен на то же число потоков.
        scheduler.lock().unwrap().set_num_workers(num_threads);

        Self {
            pool: Arc::new(WorkerPool::new(num_threads)),
            scheduler,
        }
    }

    /// Отправляет несколько задач в пул и ожидает их завершения.
    ///
    /// Полезно, когда нужно выполнить независимые части работы параллельно
    /// и дождаться всех результатов.
    ///
    /// # Аргументы
    /// * `tasks` – вектор замыканий, которые будут выполнены в пуле.
    pub fn execute_many(&self, tasks: Vec<Box<dyn FnOnce() + Send>>) {
        for task in tasks {
            self.pool.execute(task);
        }
        self.pool.wait_all();
    }

    /// Возвращает ссылку на внутренний `WorkerPool`.
    ///
    /// Может использоваться для низкоуровневого взаимодействия.
    pub fn pool(&self) -> &Arc<WorkerPool> {
        &self.pool
    }
}

impl Clone for ComputeThreadPool {
    /// Клонирование пула: все клоны разделяют тот же `WorkerPool` и
    /// `Scheduler`, поэтому задачи распределяются в общую очередь.
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            scheduler: self.scheduler.clone(),
        }
    }
}

impl Executor for ComputeThreadPool {
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
        self.scheduler.lock().unwrap().plan_chunks_assignment(total_tasks)
    }

    fn clone_executor(&self) -> Box<dyn Executor> {
        Box::new(self.clone())
    }
}