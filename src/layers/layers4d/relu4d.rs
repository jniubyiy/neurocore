use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::ReLU;
use crate::neuron::base::Neuron;
use super::{Layer4D, LayerContext4D};

pub struct ReLU4D {
    neuron: ReLU,
    inner_size: usize,
}

impl ReLU4D {
    pub fn new(size: usize) -> Self { Self { neuron: ReLU, inner_size: size } }
}

impl Layer4D for ReLU4D {
    fn forward_into(&self, input: &Tensor4D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D {
        for d1 in 0..input.dim1 {
            for d in 0..input.depth {
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        out_buf[d1][d][r][c] = self.neuron.apply(input.data[d1][d][r][c]);
                    }
                }
            }
        }
        LayerContext4D::ReLU4D { input: input.clone() }
    }

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, params: &[f32], slice: &ParamSlice) -> (Tensor4D, Vec<f32>) {
        let input = match ctx { LayerContext4D::ReLU4D { input } => input, _ => panic!() };
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let mut d_prev = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for c in 0..cols {
                        let grad = if input.data[d1][d][r][c] > 0.0 { 1.0 } else { 0.0 };
                        d_prev[d1][d][r][c] = delta.data[d1][d][r][c] * grad;
                    }
                }
            }
        }
        (Tensor4D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}





