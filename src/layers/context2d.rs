// src/layers/context2d.rs
use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;

#[derive(Clone)]
pub enum LayerContext {
    Linear2D  { input: Tensor3D },
    ReLU2D    { input: Tensor3D },
    Sigmoid2D { output: Tensor3D },
    Tanh2D    { output: Tensor3D },
    Softmax2D { output: Tensor3D },
    Memory2D  { input: Tensor3D },
    Splitter2D { input: Tensor3D, pre_a: Vec<f32>, pre_b: Vec<f32> },  // <-- изменено
    Combiner2D { input_a: Tensor3D, input_b: Tensor3D, pre_act: Vec<Vec<f32>> },
    SplitterConnector { input: Tensor3D },
    CombinerConnector { inputs: Vec<Tensor3D> },
}

pub trait Layer2D: Send + Sync {
    fn input_dims(&self) -> Vec<usize>;
    fn output_dims(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor3D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor3D>, Vec<LayerContext>) {
        let out_sizes = self.output_dims();
        let dim1 = if let Some(first) = inputs.first() { first.dim1 } else { 0 };
        let dim2 = if let Some(first) = inputs.first() { first.dim2 } else { 0 };
        let mut out_bufs: Vec<Vec<Vec<Vec<f32>>>> = out_sizes.iter().map(|&sz| vec![vec![vec![0.0; sz]; dim2]; dim1]).collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor3D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor3D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<f32>>>],
    ) -> Vec<LayerContext>;

    fn backward(
        &self,
        ctxs: &[LayerContext],
        deltas: &[Tensor3D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor3D>, Vec<f32>);

    fn param_len(&self) -> usize;
}