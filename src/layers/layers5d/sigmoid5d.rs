use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Sigmoid;
use crate::neuron::base::Neuron;
use super::{Layer5D, LayerContext5D};

pub struct Sigmoid5D {
    neuron: Sigmoid,
    inner_size: usize,
}

impl Sigmoid5D {
    pub fn new(size: usize) -> Self { Self { neuron: Sigmoid, inner_size: size } }
}

impl Layer5D for Sigmoid5D {
    fn forward_into(&self, input: &Tensor5D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> LayerContext5D {
        let mut output = vec![vec![vec![vec![vec![0.0; input.cols]; input.rows]; input.depth]; input.dim1]; input.outer];
        for o in 0..input.outer {
            for d1 in 0..input.dim1 {
                for d in 0..input.depth {
                    for r in 0..input.rows {
                        for c in 0..input.cols {
                            let val = self.neuron.apply(input.data[o][d1][d][r][c]);
                            out_buf[o][d1][d][r][c] = val;
                            output[o][d1][d][r][c] = val;
                        }
                    }
                }
            }
        }
        LayerContext5D::Sigmoid5D { output: Tensor5D::new(output) }
    }

    fn backward(&self, ctx: &LayerContext5D, delta: &Tensor5D, _params: &[f32], _slice: &ParamSlice) -> (Tensor5D, Vec<f32>) {
        let output = match ctx { LayerContext5D::Sigmoid5D { output } => output, _ => panic!() };
        let outer = output.outer;
        let dim1 = output.dim1;
        let depth = output.depth;
        let rows = output.rows;
        let cols = output.cols;
        let mut d_prev = vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer];
        for o in 0..outer {
            for d1 in 0..dim1 {
                for d in 0..depth {
                    for r in 0..rows {
                        for c in 0..cols {
                            let sig = output.data[o][d1][d][r][c];
                            d_prev[o][d1][d][r][c] = delta.data[o][d1][d][r][c] * sig * (1.0 - sig);
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





