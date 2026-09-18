// src/compute_manager/operators_v2/cpu_v2/mod.rs
//
// CPU-оператор v2 и вся его инфраструктура.
//
// Инфраструктура (переехала из compute_manager/cpu/):
//   * worker_pool             — пул воркеров;
//   * scheduler               — планировщик с per-CPU mini-model;
//   * cost / hardware / profiler / mini_model — модели стоимости и железа;
//   * send_ptr / task         — низкоуровневые утилиты;
//   * control_thread_pool / compute_thread_pool — пулы потоков;
//   * parallel                — параллельный forward/backward по чанкам.
//
// Диспетчеры v2:
//   * scheduler_v2            — обёртка над Scheduler;
//   * mini_model_pool_v2      — алиас на Vec<ForwardTimePredictor>;
//   * chunk_dispatch_v2       — forward/backward UniversalProcessor;
//   * loss_dispatch_v2        — CPU-вычисление loss;
//   * opt_dispatch_v2         — шаг оптимизатора.

#![allow(dead_code, unused_imports)]

// Инфраструктура CPU.
pub mod worker_pool;
pub mod scheduler;
pub mod cost;
pub mod hardware;
pub mod profiler;
pub mod mini_model;
pub mod send_ptr;
pub mod task;

pub mod control_thread_pool;
pub mod compute_thread_pool;
pub mod parallel;

// Диспетчеры v2.
pub mod scheduler_v2;
pub mod mini_model_pool_v2;
pub mod chunk_dispatch_v2;
pub mod loss_dispatch_v2;
pub mod opt_dispatch_v2;

// Реэкспорты инфраструктуры.
pub use worker_pool::WorkerPool;
pub use scheduler::Scheduler;
pub use cost::CostModel;
pub use hardware::CpuInfo;
pub use profiler::HardwareProfile;
pub use mini_model::ForwardTimePredictor;

pub use control_thread_pool::ControlThreadPool;
pub use compute_thread_pool::ComputeThreadPool;
