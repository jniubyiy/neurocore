// ============================================================
// Файл: src/layers/layers1d/mod.rs
// ============================================================

use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear1d;
pub mod relu1d;
pub mod sigmoid1d;
pub mod softmax1d;
pub mod memory1d;
pub mod tanh1d;
pub mod combiner_layer1D;
pub mod splitter_layer1D;
pub mod splitter_connector_1d;
pub mod combiner_connector_1d;

pub use linear1d::LinearLayer;
pub use relu1d::ReLULayer;
pub use sigmoid1d::SigmoidLayer;
pub use softmax1d::SoftmaxLayer;
pub use memory1d::MemoryLayer;
pub use tanh1d::TanhLayer;
pub use combiner_layer1D::CombinerLayer1D;
pub use splitter_layer1D::SplitterLayer1D;
pub use splitter_connector_1d::SplitterConnector1D;
pub use combiner_connector_1d::CombinerConnector1D;

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer_type: String,
    pub input_dim1s: Vec<usize>,
    pub output_dim1s: Vec<usize>,
    pub param_count: usize,
    pub param_start_index: Option<usize>,
}

/// Контекст, сохраняемый 1D‑слоем во время прямого прохода.
#[derive(Clone)]
pub enum LayerContext1D {
    Linear   { input: Tensor1D },
    ReLU     { input: Tensor1D },
    Sigmoid  { output: Tensor1D },
    Tanh     { output: Tensor1D },
    Softmax  { output: Tensor1D },
    Memory   { input: Tensor1D },
    Combiner { input_a: Tensor1D, input_b: Tensor1D, pre_act: Vec<f32> },
    Splitter { input: Tensor1D, pre_a: Vec<f32>, pre_b: Vec<f32> },
    SplitterConnector { input: Tensor1D },
    CombinerConnector { inputs: Vec<Tensor1D> },
}

pub trait Layer: Send + Sync {
    fn input_dim1s(&self) -> Vec<usize>;
    fn output_dim1s(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor1D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor1D>, Vec<LayerContext1D>) {
        let out_sizes = self.output_dim1s();
        let mut out_bufs: Vec<Vec<f32>> = out_sizes.iter().map(|&sz| vec![0.0; sz]).collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor1D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D>;

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor1D>, Vec<f32>);

    fn param_len(&self) -> usize;

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Unknown".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}



