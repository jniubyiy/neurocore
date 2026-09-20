// src/layers/linear/mod.rs

pub mod linear;

/// Пилотный градиентный адаптер (MIGRATION_PLAN.md §7, Фаза 5).
///
/// No-op: проверяет среду адаптеров, не модифицирует градиент.
/// GPU-версия заморожена до готовности инфраструктуры для адаптеров
/// любых слоёв (решение администратора).
pub mod adapter;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use linear::Linear;