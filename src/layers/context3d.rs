// src/layers/context3d.rs
use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;

#[derive(Clone)]
pub enum LayerContext3D {
    Linear3D  { input: Tensor4D },
    ReLU3D    { input: Tensor4D },
    Sigmoid3D { output: Tensor4D },
    Tanh3D    { output: Tensor4D },
    Softmax3D { output: Tensor4D },
    Memory3D  { input: Tensor4D },
    Splitter3D { input: Tensor4D, pre_a: Vec<f32>, pre_b: Vec<f32> },
    Combiner3D { input_a: Tensor4D, input_b: Tensor4D, pre_act: Vec<Vec<Vec<f32>>> },
    SplitterConnector { input: Tensor4D },
    CombinerConnector { inputs: Vec<Tensor4D> },

    // Новые слои
    LeakyReLU3D       { input: Tensor4D },
    SoftSparseGate3D  { input: Tensor4D },
    SoftKeepGate3D    { input: Tensor4D },
    DualAnchor3D      { input: Tensor4D },   // <-- добавлено
}

pub trait Layer3D: Send + Sync {
    fn input_dims(&self) -> Vec<usize>;
    fn output_dims(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor4D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor4D>, Vec<LayerContext3D>) {
        let out_sizes = self.output_dims();
        let dim1 = inputs.first().map(|t| t.dim1).unwrap_or(0);
        let dim2 = inputs.first().map(|t| t.dim2).unwrap_or(0);
        let mut out_bufs: Vec<Vec<Vec<Vec<Vec<f32>>>>> = out_sizes.iter()
            .map(|&sz| vec![vec![vec![vec![0.0; sz]; dim2]; dim1]])
            .collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor4D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor4D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<f32>>>>],
    ) -> Vec<LayerContext3D>;

    fn backward(
        &self,
        ctxs: &[LayerContext3D],
        deltas: &[Tensor4D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor4D>, Vec<f32>);

    fn param_len(&self) -> usize;
}