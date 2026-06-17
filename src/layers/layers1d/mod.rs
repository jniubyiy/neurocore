use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};

pub mod linear1d;
pub mod relu1d;
pub mod sigmoid1d;
pub mod softmax1d;
pub mod memory1d;
pub mod tanh1d;

pub use linear1d::LinearLayer;
pub use relu1d::ReLULayer;
pub use sigmoid1d::SigmoidLayer;
pub use softmax1d::SoftmaxLayer;
pub use memory1d::MemoryLayer;
pub use tanh1d::TanhLayer;

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer_type: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub param_count: usize,
    pub param_start_index: Option<usize>,
}

pub trait Layer: Send + Sync {
    fn forward(&mut self, input: &Tensor1D, params: &[f32], slice: &ParamSlice) -> Tensor1D {
        let mut buf = vec![0.0; self.layer_info().output_dim];
        self.forward_into(input, params, slice, &mut buf);
        Tensor1D::new(buf)
    }

    fn forward_into(&mut self, input: &Tensor1D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<f32>);

    fn backward(&mut self, delta: &Tensor1D, params: &[f32], slice: &ParamSlice) -> Tensor1D;

    fn apply_gradients(&mut self, store: &mut ParamStore, lr: f32, slice: &ParamSlice);

    fn param_len(&self) -> usize;

    fn layer_info(&self) -> LayerInfo;
}




