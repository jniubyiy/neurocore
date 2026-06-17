use crate::tensor::{Tensor5D, Tensor1D};
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use super::{Layer5D, LayerContext5D};

pub struct Softmax5D {
    neuron: Softmax,
    inner_size: usize,
}

impl Softmax5D {
    pub fn new(size: usize) -> Self { Self { neuron: Softmax, inner_size: size } }
}

impl Layer5D for Softmax5D {
    fn forward_into(&self, input: &Tensor5D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> LayerContext5D {
        let mut output = vec![vec![vec![vec![vec![0.0; input.cols]; input.rows]; input.depth]; input.dim1]; input.outer];
        for o in 0..input.outer {
            for d1 in 0..input.dim1 {
                for d in 0..input.depth {
                    for r in 0..input.rows {
                        let row_in = Tensor1D::new(input.data[o][d1][d][r].clone());
                        let row_out = self.neuron.forward(&row_in);
                        for c in 0..input.cols {
                            let val = row_out.data[c];
                            out_buf[o][d1][d][r][c] = val;
                            output[o][d1][d][r][c] = val;
                        }
                    }
                }
            }
        }
        LayerContext5D::Softmax5D { output: Tensor5D::new(output) }
    }

    fn backward(&self, ctx: &LayerContext5D, delta: &Tensor5D, _params: &[f32], _slice: &ParamSlice) -> (Tensor5D, Vec<f32>) {
        let sm = match ctx { LayerContext5D::Softmax5D { output } => output, _ => panic!() };
        let outer = sm.outer;
        let dim1 = sm.dim1;
        let depth = sm.depth;
        let rows = sm.rows;
        let cols = sm.cols;
        let mut d_prev = vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer];
        for o in 0..outer {
            for d1 in 0..dim1 {
                for d in 0..depth {
                    for r in 0..rows {
                        for i in 0..cols {
                            let mut sum = 0.0;
                            for j in 0..cols {
                                let kron = if i == j { 1.0 } else { 0.0 };
                                sum += delta.data[o][d1][d][r][j] * sm.data[o][d1][d][r][j] * (kron - sm.data[o][d1][d][r][i]);
                            }
                            d_prev[o][d1][d][r][i] = sum;
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




