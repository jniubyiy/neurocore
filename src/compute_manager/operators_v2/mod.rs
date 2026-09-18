// src/compute_manager/operators_v2/mod.rs
//
// Операторы v2: Memory / CPU / GPU.
// После реорганизации инфраструктура CPU, GPU и памяти живёт внутри
// соответствующих подмодулей (cpu_v2/, gpu_v2/, memory_v2/).

#![allow(dead_code, unused_imports)]

pub mod operator_v2;
pub mod capacity_v2;

pub mod memory_operator_v2;
pub mod cpu_operator_v2;
pub mod gpu_operator_v2;

pub mod cpu_v2;
pub mod gpu_v2;
pub mod memory_v2;

pub use operator_v2::OperatorV2;
pub use capacity_v2::OperatorCapacity;

pub use memory_operator_v2::MemoryOperatorV2;
pub use cpu_operator_v2::CpuOperatorV2;
pub use gpu_operator_v2::GpuOperatorV2;
