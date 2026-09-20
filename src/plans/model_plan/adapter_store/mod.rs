// src/plans/model_plan/adapter_store/mod.rs

//! Модуль хранения персистентного состояния градиентных адаптеров.
//!
//! Инвариант I-6 (MIGRATION_PLAN.md §2): состояние адаптеров живёт в
//! отдельном `AdapterStateStore`, параллельном `ParamStore`. Тот же
//! жизненный цикл:
//!
//!   * **init** — при построении графа (Фаза 5 плана, когда появляется
//!     первый адаптер с состоянием);
//!   * **миграция** — в `GraphV2::observe_epoch` через
//!     `SmartDistributor::on_epoch_boundary`;
//!   * **save/load** — по аналогии с `ParamStore` (при необходимости
//!     добавляется отдельной задачей).
//!
//! Optimizer это состояние **не обновляет** (инвариант I-6).
//! Обновление — исключительно ответственность адаптеров через
//! `GradientAdapter::apply` (Фаза 3+ плана).
//!
//! # Фаза 2 (вхолостую)
//!
//! На текущем этапе адаптеров ещё нет, `AdapterStateStore` создаётся
//! пустым и участвует в lifecycle `GraphV2` без видимых эффектов:
//! `num_buffers() == 0`, миграция ничего не делает. Это позволяет
//! интегрировать store в общий конвейер, не меняя поведение
//! существующих примеров (точка валидации §7).

mod buffer;
mod slice;
mod store;

pub use buffer::AdapterStateBuffer;
pub use slice::AdapterSlice;
pub use store::AdapterStateStore;