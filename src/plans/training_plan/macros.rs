// src/training_plan/macros.rs

/// Запускает обучение согласно плану `training_plan::plan()` и конфигурации устройств `device_plan::plan()`.
///
/// Оба модуля должны быть определены в области вызова макроса (как `mod device_plan` и `mod training_plan`).
///
/// # Пример
///
/// ```ignore
/// mod device_plan {
///     use neurocore::device_plan::DevicePlan;
///     pub fn plan() -> DevicePlan { DevicePlan::default() }
/// }
///
/// mod training_plan {
///     use neurocore::training_plan::TrainingPlan;
///     pub fn plan() -> TrainingPlan {
///         TrainingPlan::new()
///             .model(models::my_model)
///             .loss(losses::mse())
///             .optimizer(optimizers::sgd())
///             .epochs(10)
///             .batch_size(32)
///             .train_data(DataSource::from_tensor2d(...))
///             .target_data(DataSource::from_tensor2d(...))
///     }
/// }
///
/// let result = neurocore::run_training!(training_plan::plan);
/// ```
#[macro_export]
macro_rules! run_training {
    ( $training_plan_fn:path ) => {
        {
            let device_plan = device_plan::plan();
            let training_plan = $training_plan_fn();
            $crate::training_plan::execution::execute(&training_plan, &device_plan)
                .expect("Training failed")
        }
    };
    // вариант с явным указанием плана устройств (если нужно переопределить)
    ( $training_plan_fn:path, device = $device_plan_fn:path ) => {
        {
            let device_plan = $device_plan_fn();
            let training_plan = $training_plan_fn();
            $crate::training_plan::execution::execute(&training_plan, &device_plan)
                .expect("Training failed")
        }
    };
}