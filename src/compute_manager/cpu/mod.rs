// src/compute_manager/cpu/mod.rs

pub mod worker_pool;
pub mod scheduler;
pub mod cost;
pub mod hardware;
pub mod profiler;
pub mod mini_model;
pub mod send_ptr;
pub mod task;

pub use worker_pool::WorkerPool;
pub use scheduler::Scheduler;
pub use cost::CostModel;
pub use hardware::CpuInfo;
pub use profiler::HardwareProfile;
pub use mini_model::ForwardTimePredictor;

/// Заглушка для будущего универсального CPU‑исполнителя.
/// В текущей версии `MixedModel` напрямую использует `WorkerPool` и `Scheduler`,
/// поэтому `CpuExecutor` пока не несёт логики, но необходим для публичного API.
#[derive(Clone)]
pub struct CpuExecutor {
    #[allow(dead_code)]
    pool: std::sync::Arc<WorkerPool>,
    #[allow(dead_code)]
    scheduler: std::sync::Arc<std::sync::Mutex<Scheduler>>,
}

impl CpuExecutor {
    pub fn new(pool: std::sync::Arc<WorkerPool>, scheduler: std::sync::Arc<std::sync::Mutex<Scheduler>>) -> Self {
        Self { pool, scheduler }
    }
}