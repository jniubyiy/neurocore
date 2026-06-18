use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer, LayerContext1D, LayerInfo};

pub struct CombinerConnector1D {
    pub input_dims: Vec<usize>,
    pub output_dim: usize,
}

impl CombinerConnector1D {
    pub fn new(input_dims: Vec<usize>) -> Self {
        let output_dim = input_dims.iter().sum();
        Self { input_dims, output_dim }
    }
}

impl Layer for CombinerConnector1D {
    fn input_dim1s(&self) -> Vec<usize> { self.input_dims.clone() }
    fn output_dim1s(&self) -> Vec<usize> { vec![self.output_dim] }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        let out = &mut out_bufs[0];
        let mut offset = 0;
        for input in inputs {
            let len = input.dim1();
            out[offset..offset + len].copy_from_slice(&input.data);
            offset += len;
        }
        vec![LayerContext1D::CombinerConnector { inputs: inputs.to_vec() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor1D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor1D>, Vec<f32>) {
        let delta = &deltas[0].data;
        let inputs = match &ctxs[0] {
            LayerContext1D::CombinerConnector { inputs } => inputs,
            _ => panic!(),
        };
        let mut in_grads = Vec::new();
        let mut offset = 0;
        for input in inputs {
            let len = input.dim1();
            let d_i = delta[offset..offset + len].to_vec();
            in_grads.push(Tensor1D::new(d_i));
            offset += len;
        }
        (in_grads, vec![])
    }

    fn param_len(&self) -> usize { 0 }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "CombinerConnector".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: 0,
            param_start_index: None,
        }
    }
}