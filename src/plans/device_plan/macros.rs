// src/device_plan/macros.rs
//
// Макрос create_models! устарел. Модель теперь создаётся исключительно через TrainingPlan.
// Используйте TrainingPlan::new()...build_model(device_plan) для получения готовой модели.

#[macro_export]
macro_rules! create_models {
    ( $( $mod:ident ),+ $(,)? ) => {
        compile_error!("create_models! is deprecated. Use TrainingPlan to build models instead.")
    };
}