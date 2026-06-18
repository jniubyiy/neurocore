use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer, LayerContext1D, LayerInfo};

pub struct SplitterConnector1D {
    pub input_dim: usize,
    pub output_dims: Vec<usize>,
}

impl SplitterConnector1D {
    pub fn new(input_dim: usize, output_dims: Vec<usize>) -> Self {
        Self { input_dim, output_dims }
    }
}

impl Layer for SplitterConnector1D {
    fn input_dim1s(&self) -> Vec<usize> { vec![self.input_dim] }
    fn output_dim1s(&self) -> Vec<usize> { self.output_dims.clone() }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        assert_eq!(inputs.len(), 1);
        let x = &inputs[0].data;
        let mut offset = 0;
        for (i, &dim) in self.output_dims.iter().enumerate() {
            out_bufs[i].copy_from_slice(&x[offset..offset + dim]);
            offset += dim;
        }
        vec![LayerContext1D::SplitterConnector { input: inputs[0].clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor1D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor1D>, Vec<f32>) {
        let input = match &ctxs[0] {
            LayerContext1D::SplitterConnector { input } => input,
            _ => panic!(),
        };
        let mut dx = vec![0.0; input.dim1()];
        let mut offset = 0;
        for delta in deltas {
            let len = delta.dim1();
            for (i, &d) in delta.data.iter().enumerate() {
                dx[offset + i] = d;
            }
            offset += len;
        }
        (vec![Tensor1D::new(dx)], vec![])
    }

    fn param_len(&self) -> usize { 0 }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "SplitterConnector".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: 0,
            param_start_index: None,
        }
    }
}