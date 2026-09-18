// src/compute_manager/operators_v2/cpu_v2/mod.rs
//
// Подмодули CPU-оператора v2.
//
// Все компоненты тонкие и изолированные:
//   * scheduler_v2         — обёртка над существующим `Scheduler`;
//   * mini_model_pool_v2   — алиас: mini-model'и уже живут в `Scheduler`;
//   * chunk_dispatch_v2    — forward/backward UniversalProcessor;
//   * loss_dispatch_v2     — CPU-вычисление loss;
//   * opt_dispatch_v2      — шаг оптимизатора по одному буферу.
//
// Логика каждого компонента не имеет состояния: состояние живёт в
// `CpuOperatorV2`, а функции подмодулей получают нужные дескрипторы
// аргументами.

#![allow(dead_code, unused_imports)]

pub mod scheduler_v2;
pub mod mini_model_pool_v2;
pub mod chunk_dispatch_v2;
pub mod loss_dispatch_v2;
pub mod opt_dispatch_v2;