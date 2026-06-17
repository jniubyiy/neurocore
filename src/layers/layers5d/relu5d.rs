use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::ReLU;
use crate::neuron::base::Neuron;
use super::{Layer5D, LayerContext5D};

pub struct ReLU5D {
    neuron: ReLU,
    inner_size: usize,
}

impl ReLU5D {
    pub fn new(size: usize) -> Self { Self { neuron: ReLU, inner_size: size } }
}

impl Layer5D for ReLU5D {
    fn forward_into(&self, input: &Tensor5D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> LayerContext5D {
        for o in 0..input.outer {
            for d1 in 0..input.dim1 {
                for d in 0..input.depth {
                    for r in 0..input.rows {
                        for c in 0..input.cols {
                            out_buf[o][d1][d][r][c] = self.neuron.apply(input.data[o][d1][d][r][c]);
                        }
                    }
                }
            }
        }
        LayerContext5D::ReLU5D { input: input.clone() }
    }

    fn backward(&self, ctx: &LayerContext5D, delta: &Tensor5D, _params: &[f32], _slice: &ParamSlice) -> (Tensor5D, Vec<f32>) {
        let input = match ctx { LayerContext5D::ReLU5D { input } => input, _ => panic!() };
        let outer = input.outer;
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let mut d_prev = vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer];
        for o in 0..outer {
            for d1 in 0..dim1 {
                for d in 0..depth {
                    for r in 0..rows {
                        for c in 0..cols {
                            let grad = if input.data[o][d1][d][r][c] > 0.0 { 1.0 } else { 0.0 };
                            d_prev[o][d1][d][r][c] = delta.data[o][d1][d][r][c] * grad;
                        }
                    }
                }
            }
        }
        (Tensor5D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}





