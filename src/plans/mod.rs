// src/plans/mod.rs

//! Модуль планирования NeuroCore.
//! Содержит все подсистемы, связанные с описанием и выполнением планов:
//! - model_plan: описание архитектуры модели
//! - loss_plan: функции потерь
//! - optimizer_plan: алгоритмы оптимизации
//! - training_plan: процесс обучения (эпохи, батчи, профилирование)
//! - device_plan: конфигурация вычислительных и запоминающих устройств

pub mod model_plan;
pub mod loss_plan;
pub mod optimizer_plan;
pub mod training_plan;
pub mod device_plan;