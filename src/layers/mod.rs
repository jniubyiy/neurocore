// src/layers/mod.rs

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
pub mod leaky_relu;
pub mod identity;
pub mod soft_sparse_gate;
pub mod soft_keep_gate;
pub mod dual_anchor;

pub mod mat_context;
pub mod layers_special;

use crate::model_plan::param_store::ParamSlice;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use faer::Mat;

// ---------------------------------------------------------------------------
// Старый трейт UniversalLayer (оставлен для обратной совместимости)
// ---------------------------------------------------------------------------

pub trait UniversalLayer: Send + Sync + 'static {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext);

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>);

    fn param_len(&self) -> usize;
    fn input_features(&self) -> usize;
    fn output_features(&self) -> usize;

    fn total_tasks(&self, batch_size: usize) -> usize { batch_size }

    fn execute_tasks(
        &self,
        input: &Mat<f32>,
        output: &mut Mat<f32>,
        task_offset: usize,
        task_count: usize,
        params: &[f32],
        slice: &ParamSlice,
    );

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        output_sample: &Mat<f32>,
    ) -> DynamicContext;

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.output_features())
    }

    // ----- Методы для GPU-диспетчеризации -----
    fn as_linear(&self) -> Option<&Linear> { None }
    fn as_relu(&self) -> Option<&ReLU> { None }
    fn as_sigmoid(&self) -> Option<&Sigmoid> { None }
    fn as_tanh(&self) -> Option<&Tanh> { None }
    fn as_leaky_relu(&self) -> Option<&LeakyReLU> { None }
    fn as_identity(&self) -> Option<&Identity> { None }
    fn as_softmax(&self) -> Option<&Softmax> { None }
    fn as_dual_anchor(&self) -> Option<&DualAnchor> { None }
    fn as_soft_sparse_gate(&self) -> Option<&SoftSparseGate> { None }
    fn as_soft_keep_gate(&self) -> Option<&SoftKeepGate> { None }
    fn as_reduce_mean(&self) -> Option<&ReduceMean> { None }
    fn as_unsqueeze(&self) -> Option<&Unsqueeze> { None }
    fn as_memory(&self) -> Option<&Memory> { None }
}

// ---------------------------------------------------------------------------
// Новый трейт UniversalLayerBuffered – основа для работы с буферами
// ---------------------------------------------------------------------------

/// Версия слоя, работающая с управляемыми буферами [`MatrixBuffer`].
///
/// Входные и выходные данные передаются через пул временных матриц,
/// что позволяет `MemoryExecutor` отслеживать и переиспользовать память.
pub trait UniversalLayerBuffered: Send + Sync + 'static {
    /// Прямой проход.
    ///
    /// # Аргументы
    /// * `input` – входная матрица (доступна только для чтения).
    /// * `output` – матрица, в которую будет записан результат.
    ///   Должна иметь размер `(batch_size, output_features())`.
    /// * `params` – плоский срез всех параметров модели.
    /// * `slice` – границы параметров, принадлежащих данному слою.
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        params: &[f32],
        slice: &ParamSlice,
    );

    /// Обратный проход.
    ///
    /// # Аргументы
    /// * `ctx` – контекст, сохранённый прямым проходом (пока старая версия,
    ///   в будущем будет заменён на хранение [`MatrixBuffer`]).
    /// * `grad_output` – градиент по выходу слоя.
    /// * `grad_input` – буфер, куда будет записан градиент по входу.
    /// * `params` – плоский срез всех параметров модели.
    /// * `slice` – границы параметров, принадлежащих данному слою.
    ///
    /// # Возвращает
    /// Вектор градиентов по параметрам слоя.
    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBuffer,
        grad_input: &mut MatrixBuffer,
        params: &[f32],
        slice: &ParamSlice,
    ) -> Vec<f32>;

    /// Количество обучаемых параметров слоя.
    fn param_len(&self) -> usize;

    /// Количество входных признаков.
    fn input_features(&self) -> usize;

    /// Количество выходных признаков.
    fn output_features(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Публичные реэкспорты
// ---------------------------------------------------------------------------

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
pub use leaky_relu::LeakyReLU;
pub use identity::Identity;
pub use soft_sparse_gate::SoftSparseGate;
pub use soft_keep_gate::SoftKeepGate;
pub use dual_anchor::DualAnchor;

pub use mat_context::{MatContext, LayerInfo};
pub use layers_special::{DimReduce, DimExpand, ReduceMean, Unsqueeze};
