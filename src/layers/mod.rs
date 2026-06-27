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

pub mod context1d;
pub mod context2d;
pub mod context3d;
pub mod context4d;

pub mod layers_special;

use crate::model_plan::param_store::ParamSlice;
use crate::compute_manager::graph::types::DynamicContext;
use faer::Mat;

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

    // ----- Методы для GPU-диспетчеризации (возвращают Some(self), если слой того же типа) -----
    fn as_linear(&self) -> Option<&Linear> { None }
    fn as_relu(&self) -> Option<&ReLU> { None }
    fn as_sigmoid(&self) -> Option<&Sigmoid> { None }
    fn as_tanh(&self) -> Option<&Tanh> { None }
    fn as_leaky_relu(&self) -> Option<&LeakyReLU> { None }
    fn as_identity(&self) -> Option<&Identity> { None }
    // При необходимости аналогично добавляются as_softmax, as_reduce_mean и т.д.
}

// Реэкспорт слоёв
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

pub use context1d::{Layer, LayerContext1D, LayerInfo};
pub use context2d::{Layer2D, LayerContext as LayerContext2D};
pub use context3d::{Layer3D, LayerContext3D};
pub use context4d::{Layer4D, LayerContext4D};

pub use layers_special::{DimReduce, DimExpand, ReduceMean, Unsqueeze};

