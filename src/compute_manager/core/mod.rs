// src/compute_manager/core/mod.rs
//
// Базовые типы и трейты, общие для всего compute_manager.
// Раньше эти файлы лежали разрозненно в корне compute_manager,
// теперь собраны в core/.

pub mod device;
pub mod device_spec;
pub mod dim_change;
pub mod dynamic_context;
pub mod executor;
