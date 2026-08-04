// src/device_plan/macros.rs

/// Макрос для создания одной или нескольких моделей с автоматическим применением плана устройств.
///
/// # Важное правило
/// **Каждый переданный модуль должен содержать ровно одну публичную функцию `model()`**,
/// возвращающую `Vec<LayerDesc>`. Это гарантирует, что один модуль описывает ровно одну модель.
/// Попытка использовать модуль, в котором нет функции `model()` или их несколько, приведёт
/// к ошибке компиляции.
///
/// # Обязательные требования
/// - В области вызова макроса должен быть определён модуль `device_plan` с функцией `plan()`,
///   возвращающей `DevicePlan`.
/// - Каждый переданный модуль должен содержать функцию `model()` с сигнатурой `fn() -> Vec<LayerDesc>`.
///   Другие функции внутри модуля игнорируются макросом и не влияют на его работу.
///
/// # Пример правильного использования
/// ```ignore
/// mod device_plan {
///     use neurocore::device_plan::DevicePlan;
///     pub fn plan() -> DevicePlan {
///         DevicePlan::empty().cpu(0, 4).ram(0, 8192)
///     }
/// }
///
/// mod model_encoder {
///     use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
///     pub fn model() -> Vec<LayerDesc> {
///         vec![
///             LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim1)
///                 .input(Dim::Dim1, &[4])
///                 .output(Dim::Dim1, &[2]),
///         ]
///     }
/// }
///
/// mod model_decoder {
///     use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
///     pub fn model() -> Vec<LayerDesc> {
///         vec![
///             LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim1)
///                 .input(Dim::Dim1, &[2])
///                 .output(Dim::Dim1, &[4]),
///         ]
///     }
/// }
///
/// // Создаём две модели из двух разных модулей
/// let (encoder, decoder) = neurocore::create_models!(model_encoder, model_decoder);
/// ```
///
/// # Ошибочное использование (не скомпилируется)
/// ```ignore
/// // В одном модуле определены две модели – это нарушение правила.
/// mod model_wrong {
///     pub fn model1() -> Vec<LayerDesc> { ... }
///     pub fn model2() -> Vec<LayerDesc> { ... }
/// }
///
/// // Это не скомпилируется, так как макрос ожидает функцию с именем `model`.
/// let (m1, m2) = neurocore::create_models!(model_wrong::model1, model_wrong::model2);
/// ```
#[macro_export]
macro_rules! create_models {
    ( $( $mod:ident ),+ $(,)? ) => {
        {
            // План устройств обязательно должен быть определён в модуле `device_plan` с функцией `plan()`.
            let plan = device_plan::plan();
            ( $(
                $crate::model_plan::Plan::from_layer_descs($mod::model())
                    .expect("Invalid model description")
                    .build_with_device_plan(plan.clone())
            ,)+ )
        }
    };
}