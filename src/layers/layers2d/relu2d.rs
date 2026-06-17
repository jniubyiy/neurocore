use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::ReLU;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct ReLU2D {
    pub size: usize,
}

impl ReLU2D {
    pub fn new(size: usize) -> Self { Self { size } }
}

impl Layer2D for ReLU2D {
    fn forward_into(&self, input: &Tensor2D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext {
        let mat = linalg::tensor2d_to_faer(input);
        let out = ReLU.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor2d(&out);
        *out_buf = out_t.data;
        LayerContext::ReLU2D { input: input.clone() }
    }

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, _params: &[f32], _slice: &ParamSlice) -> (Tensor2D, Vec<f32>) {
        let input = match ctx { LayerContext::ReLU2D { input } => input, _ => panic!() };
        let rows = input.rows;
        let cols = input.cols;
        let mut dprev = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let grad = if input.data[r][c] > 0.0 { 1.0 } else { 0.0 };
                dprev[r][c] = delta.data[r][c] * grad;
            }
        }
        (Tensor2D::new(dprev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.size }
    fn out_features(&self) -> usize { self.size }
}





