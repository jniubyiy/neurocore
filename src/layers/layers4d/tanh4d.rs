use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Tanh;
use crate::neuron::base::Neuron;
use super::{Layer4D, LayerContext4D};

pub struct Tanh4D {
    neuron: Tanh,
    inner_size: usize,
}

impl Tanh4D {
    pub fn new(size: usize) -> Self { Self { neuron: Tanh, inner_size: size } }
}

impl Layer4D for Tanh4D {
    fn forward_into(&self, input: &Tensor4D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D {
        let mut output = vec![vec![vec![vec![0.0; input.cols]; input.rows]; input.depth]; input.dim1];
        for d1 in 0..input.dim1 {
            for d in 0..input.depth {
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        let val = self.neuron.apply(input.data[d1][d][r][c]);
                        out_buf[d1][d][r][c] = val;
                        output[d1][d][r][c] = val;
                    }
                }
            }
        }
        LayerContext4D::Tanh4D { output: Tensor4D::new(output) }
    }

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, params: &[f32], slice: &ParamSlice) -> (Tensor4D, Vec<f32>) {
        let output = match ctx { LayerContext4D::Tanh4D { output } => output, _ => panic!() };
        let dim1 = output.dim1;
        let depth = output.depth;
        let rows = output.rows;
        let cols = output.cols;
        let mut d_prev = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for c in 0..cols {
                        let t = output.data[d1][d][r][c];
                        d_prev[d1][d][r][c] = delta.data[d1][d][r][c] * (1.0 - t * t);
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