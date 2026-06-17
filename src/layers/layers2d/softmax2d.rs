use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct Softmax2D {
    pub size: usize,
}

impl Softmax2D {
    pub fn new(size: usize) -> Self { Self { size } }
}

impl Layer2D for Softmax2D {
    fn forward_into(&self, input: &Tensor2D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext {
        let mat = linalg::tensor2d_to_faer(input);
        let out = Softmax.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor2d(&out);
        *out_buf = out_t.data.clone();
        LayerContext::Softmax2D { output: out_t }
    }

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, _params: &[f32], _slice: &ParamSlice) -> (Tensor2D, Vec<f32>) {
        let sm = match ctx { LayerContext::Softmax2D { output } => output, _ => panic!() };
        let rows = sm.rows;
        let cols = sm.cols;
        let mut dprev = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for i in 0..cols {
                let mut sum = 0.0;
                for j in 0..cols {
                    let kron = if i == j { 1.0 } else { 0.0 };
                    sum += delta.data[r][j] * sm.data[r][j] * (kron - sm.data[r][i]);
                }
                dprev[r][i] = sum;
            }
        }
        (Tensor2D::new(dprev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.size }
    fn out_features(&self) -> usize { self.size }
}




