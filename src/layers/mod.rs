// src/layers/mod.rs

// ============= Новые модули слоёв =============
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;
pub mod memory;
pub mod splitter;
pub mod combiner;
pub mod splitter_connector;
pub mod combiner_connector;

// ============= Контексты и трейты =============
pub mod context1d;
pub mod context2d;
pub mod context3d;
pub mod context4d;

// ============= Специальные слои =============
pub mod layers_special;

// ============= Универсальный трейт слоя =============
pub trait UniversalLayer: Send + Sync {
    // --- Традиционные методы (полный прямой/обратный проход) ---

    fn forward(
        &self,
        input: &crate::compute_manager::dim_change::DynamicTensor,
        params: &[f32],
        slice: &crate::model_plan::param_store::ParamSlice,
    ) -> (
        crate::compute_manager::dim_change::DynamicTensor,
        crate::compute_manager::graph::types::DynamicContext,
    );

    fn backward(
        &self,
        ctx: &crate::compute_manager::graph::types::DynamicContext,
        delta: &crate::compute_manager::dim_change::DynamicTensor,
        params: &[f32],
        slice: &crate::model_plan::param_store::ParamSlice,
    ) -> (crate::compute_manager::dim_change::DynamicTensor, Vec<f32>);

    fn param_len(&self) -> usize;
    fn input_features(&self) -> usize;
    fn output_features(&self) -> usize;

    // --- Новые методы для динамических чанков задач ---

    /// Общее количество атомарных задач слоя для данного входного тензора.
    fn total_tasks(
        &self,
        input: &crate::compute_manager::dim_change::DynamicTensor,
    ) -> usize;

    /// Выполняет непрерывный диапазон задач (по плоскому индексу) и записывает
    /// результат в выходной тензор `output`. Входной тензор `input` предоставляет
    /// данные всего батча, а `output` уже создан нужной формы (см. `output_tensor_shape`).
    fn execute_tasks(
        &self,
        input: &crate::compute_manager::dim_change::DynamicTensor,
        output: &mut crate::compute_manager::dim_change::DynamicTensor,
        task_offset: usize,
        task_count: usize,
        params: &[f32],
        slice: &crate::model_plan::param_store::ParamSlice,
    );

    /// Создаёт контекст для одного образца после завершения обработки всего батча.
    fn create_sample_context(
        &self,
        input_sample: &crate::compute_manager::dim_change::DynamicTensor,
        output_sample: &crate::compute_manager::dim_change::DynamicTensor,
    ) -> crate::compute_manager::graph::types::DynamicContext;

    /// Возвращает нулевой тензор такой же формы, как выход слоя для заданного
    /// входного тензора `input` (который описывает форму батча). Используется
    /// для выделения выходных буферов перед параллельным выполнением чанков.
    fn output_tensor_shape(
        &self,
        input: &crate::compute_manager::dim_change::DynamicTensor,
    ) -> crate::compute_manager::dim_change::DynamicTensor;
}

// ============= Реэкспорт универсальных слоёв (публичный API) =============
pub use linear::Linear;
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;
pub use memory::Memory;
pub use splitter::Splitter;
pub use combiner::Combiner;
pub use splitter_connector::SplitterConnector;
pub use combiner_connector::CombinerConnector;

// ============= Контексты (оставлены для внутреннего использования и обратной совместимости) =============
pub use context1d::{Layer, LayerContext1D, LayerInfo};
pub use context2d::{Layer2D, LayerContext as LayerContext2D};
pub use context3d::{Layer3D, LayerContext3D};
pub use context4d::{Layer4D, LayerContext4D};

// ============= Специальные слои =============
pub use layers_special::{DimReduce, DimExpand, ReduceMean, Unsqueeze};

