use crate::tensor::{Tensor4D, Tensor1D};
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use super::{Layer4D, LayerContext4D};

pub struct Softmax4D {
    neuron: Softmax,
    inner_size: usize,
}

impl Softmax4D {
    pub fn new(size: usize) -> Self { Self { neuron: Softmax, inner_size: size } }
}

impl Layer4D for Softmax4D {
    fn forward_into(&self, input: &Tensor4D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D {
        let mut output = vec![vec![vec![vec![0.0; input.cols]; input.rows]; input.depth]; input.dim1];
        for d1 in 0..input.dim1 {
            for d in 0..input.depth {
                for r in 0..input.rows {
                    let row_in = Tensor1D::new(input.data[d1][d][r].clone());
                    let row_out = self.neuron.forward(&row_in);
                    for c in 0..input.cols {
                        let val = row_out.data[c];
                        out_buf[d1][d][r][c] = val;
                        output[d1][d][r][c] = val;
                    }
                }
            }
        }
        LayerContext4D::Softmax4D { output: Tensor4D::new(output) }
    }

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, params: &[f32], slice: &ParamSlice) -> (Tensor4D, Vec<f32>) {
        let sm = match ctx { LayerContext4D::Softmax4D { output } => output, _ => panic!() };
        let dim1 = sm.dim1;
        let depth = sm.depth;
        let rows = sm.rows;
        let cols = sm.cols;
        let mut d_prev = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    for i in 0..cols {
                        let mut sum = 0.0;
                        for j in 0..cols {
                            let kron = if i == j { 1.0 } else { 0.0 };
                            sum += delta.data[d1][d][r][j] * sm.data[d1][d][r][j] * (kron - sm.data[d1][d][r][i]);
                        }
                        d_prev[d1][d][r][i] = sum;
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



