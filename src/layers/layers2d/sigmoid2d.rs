use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Sigmoid;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct Sigmoid2D {
    pub size: usize,
}

impl Sigmoid2D {
    pub fn new(size: usize) -> Self { Self { size } }
}

impl Layer2D for Sigmoid2D {
    fn forward_into(&self, input: &Tensor2D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext {
        let mat = linalg::tensor2d_to_faer(input);
        let out = Sigmoid.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor2d(&out);
        *out_buf = out_t.data.clone();
        LayerContext::Sigmoid2D { output: out_t }
    }

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, _params: &[f32], _slice: &ParamSlice) -> (Tensor2D, Vec<f32>) {
        let output = match ctx { LayerContext::Sigmoid2D { output } => output, _ => panic!() };
        let rows = output.rows;
        let cols = output.cols;
        let mut dprev = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let sig = output.data[r][c];
                dprev[r][c] = delta.data[r][c] * sig * (1.0 - sig);
            }
        }
        (Tensor2D::new(dprev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.size }
    fn out_features(&self) -> usize { self.size }
}