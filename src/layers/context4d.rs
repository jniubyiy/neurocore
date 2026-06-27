// src/layers/context4d.rs
use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;

pub type Buf5D = Vec<Vec<Vec<Vec<Vec<f32>>>>>;

#[derive(Clone)]
pub enum LayerContext4D {
    Linear4D  { input: Tensor5D },
    ReLU4D    { input: Tensor5D },
    Sigmoid4D { output: Tensor5D },
    Tanh4D    { output: Tensor5D },
    Softmax4D { output: Tensor5D },
    Memory4D  { input: Tensor5D },
    Splitter4D { input: Tensor5D, pre_a: Vec<f32>, pre_b: Vec<f32> },
    Combiner4D { input_a: Tensor5D, input_b: Tensor5D, pre_act: Vec<Vec<Vec<Vec<f32>>>> },
    SplitterConnector { input: Tensor5D },
    CombinerConnector { inputs: Vec<Tensor5D> },

    // Новые слои
    LeakyReLU4D       { input: Tensor5D },
    SoftSparseGate4D  { input: Tensor5D },
    SoftKeepGate4D    { input: Tensor5D },
    DualAnchor4D      { input: Tensor5D },   // <-- добавлено
}

pub trait Layer4D: Send + Sync {
    fn input_dims(&self) -> Vec<usize>;
    fn output_dims(&self) -> Vec<usize>;

    fn forward(
        &self,
        inputs: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor5D>, Vec<LayerContext4D>) {
        let out_sizes = self.output_dims();
        let dim1 = inputs.first().map(|t| t.dim1).unwrap_or(0);
        let dim2 = inputs.first().map(|t| t.dim2).unwrap_or(0);
        let dim3 = inputs.first().map(|t| t.dim3).unwrap_or(0);
        let dim4 = inputs.first().map(|t| t.dim4).unwrap_or(0);

        let mut out_bufs: Vec<Buf5D> = Vec::with_capacity(out_sizes.len());
        for sz in &out_sizes {
            let buf: Buf5D = vec![vec![vec![vec![vec![0.0; *sz]; dim4]; dim3]; dim2]; dim1];
            out_bufs.push(buf);
        }

        let ctxs = <Self as Layer4D>::forward_into(self, inputs, params, slice, &mut out_bufs[..]);
        let tensors = out_bufs.into_iter().map(Tensor5D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Buf5D],
    ) -> Vec<LayerContext4D>;

    fn backward(
        &self,
        ctxs: &[LayerContext4D],
        deltas: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor5D>, Vec<f32>);

    fn param_len(&self) -> usize;
}


