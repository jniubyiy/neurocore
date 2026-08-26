// src/lib.rs

pub mod tensor;
pub mod layers;
pub mod logging;
pub mod compute_manager;

// Новый модуль потерь
pub mod losses;

// Новый модуль оптимизаторов (рефакторинг)
pub mod optimizers;

// Модуль plans теперь определён через src/plans/mod.rs
pub mod plans;

// Реэкспорты для обратной совместимости
pub use plans::model_plan as model_plan;
pub use plans::loss_plan as loss_plan;
pub use plans::optimizer_plan as optimizer_plan;
pub use plans::training_plan as training_plan;
pub use plans::device_plan as device_plan;