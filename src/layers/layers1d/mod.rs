use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::ParamSlice;

pub mod linear1d;
pub mod relu1d;
pub mod sigmoid1d;
pub mod softmax1d;
pub mod sequential1d;
pub mod builder1d;
pub mod memory1d;
pub mod tanh1d; 

pub use linear1d::LinearLayer;
pub use relu1d::ReLULayer;
pub use sigmoid1d::SigmoidLayer;
pub use softmax1d::SoftmaxLayer;
pub use sequential1d::Sequential;
pub use memory1d::MemoryLayer;
pub use tanh1d::TanhLayer;
pub use builder1d::{LayerBuilder, LinearLayerBuilder, ReLULayerBuilder, SigmoidLayerBuilder, SoftmaxLayerBuilder};

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer_type: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub param_count: usize,
    pub param_start_index: Option<usize>,
}

pub trait Layer {
    fn forward(
        &self,
        input: &Tensor1D,
        j_input: &Jacobian,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor1D, Jacobian);

    fn param_len(&self) -> usize;

    fn layer_info(&self) -> LayerInfo;

    /// Метод для параллельного выполнения (опционально).
    /// Пока оставим пустую реализацию по умолчанию, которая вызывает forward.
    fn execute_range(
        &self,
        input: &Tensor1D,
        j_input: &Jacobian,
        out: &mut [f32],
        j_out: &mut [f32],
        start: usize,
        end: usize,
        total_params: usize,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let (full_val, full_jac) = self.forward(input, j_input, params, slice);
        for i in start..end {
            out[i] = full_val.data[i];
            let base = i * total_params;
            for p in 0..total_params {
                j_out[base + p] = full_jac.data[i][p];
            }
        }
    }
}




