// src/layers/context1d.rs

use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;

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
    Linear   { input: Tensor2D },
    ReLU     { input: Tensor2D },
    Sigmoid  { output: Tensor2D },
    Tanh     { output: Tensor2D },
    Softmax  { output: Tensor2D },
    Memory   { input: Tensor2D },
    Combiner { input_a: Tensor2D, input_b: Tensor2D, pre_act: Vec<f32> },
    Splitter { input: Tensor2D, pre_a: Vec<f32>, pre_b: Vec<f32> },
    SplitterConnector { input: Tensor2D },
    CombinerConnector { inputs: Vec<Tensor2D> },

    // Новые слои
    LeakyReLU       { input: Tensor2D },
    SoftSparseGate  { input: Tensor2D },
    SoftKeepGate    { input: Tensor2D },
    DualAnchor1D    { input: Tensor2D },   // <-- добавлено
}

pub trait Layer: Send + Sync {
    fn input_dim1s(&self) -> Vec<usize>;
    fn output_dim1s(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor2D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor2D>, Vec<LayerContext1D>) {
        let out_sizes = self.output_dim1s();
        let mut out_bufs: Vec<Vec<f32>> = out_sizes.iter().map(|&sz| vec![0.0; sz]).collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(|buf| Tensor2D::new(vec![buf])).collect(); // batch=1
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor2D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D>;

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor2D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor2D>, Vec<f32>);

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