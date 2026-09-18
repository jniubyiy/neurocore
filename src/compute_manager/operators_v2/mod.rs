// src/compute_manager/operators_v2/mod.rs
//
// Корневой модуль операторов v2.
//
// Оператор — исполнитель заданий. Он знает только свой мир (память, CPU
// или GPU) и не знает про граф, распределитель или другие операторы.
//
// Состав модуля:
//   * operator_v2.rs          — трейт OperatorV2
//   * capacity_v2.rs          — OperatorCapacity
//   * memory_operator_v2.rs   — MemoryOperatorV2 (синхронный)
//   * cpu_operator_v2.rs      — CpuOperatorV2 (синхронный, параллелит через пул)
//   * cpu_v2/                 — cpu-диспетчеры (scheduler, chunk, loss, opt)
//   * gpu_operator_v2.rs      — GpuOperatorV2 (асинхронный, свой GPU-тред)
//   * gpu_v2/                 — gpu-подмодули (queue)
//
// После этапа B модуль подключается к crate безусловно.

#![allow(dead_code, unused_imports)]

pub mod operator_v2;
pub mod capacity_v2;

pub mod memory_operator_v2;
pub mod cpu_operator_v2;
pub mod cpu_v2;

pub mod gpu_operator_v2;
pub mod gpu_v2;

pub use operator_v2::OperatorV2;
pub use capacity_v2::OperatorCapacity;

pub use memory_operator_v2::MemoryOperatorV2;
pub use cpu_operator_v2::CpuOperatorV2;
pub use gpu_operator_v2::GpuOperatorV2;