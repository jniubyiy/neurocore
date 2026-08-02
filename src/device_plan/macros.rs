// src/device_plan/macros.rs

/// Макрос для создания моделей с автоматическим применением плана устройств.
///
/// Обязательно требует наличия модуля `device_plan` с функцией `plan()` в области вызова.
/// Пример:
/// ```ignore
/// mod device_plan {
///     use neurocore::device_plan::DevicePlan;
///     pub fn plan() -> DevicePlan {
///         DevicePlan::empty().cpu(0, 4).ram(0, 8192)
///     }
/// }
///
/// let (model,) = neurocore::create_models!(models::linear_model);
/// ```
#[macro_export]
macro_rules! create_models {
    ( $( $func:path ),+ $(,)? ) => {
        {
            let plan = device_plan::plan();
            ( $(
                $crate::model_plan::Plan::from_layer_descs($func())
                    .expect("Invalid model description")
                    .build_with_device_plan(plan.clone())
            ,)+ )
        }
    };
}