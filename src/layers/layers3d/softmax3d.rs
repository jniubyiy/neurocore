use crate::tensor::{Tensor3D, Tensor1D};
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use super::{Layer3D, LayerContext3D};

pub struct Softmax3D {
    neuron: Softmax,
    inner_size: usize,
}

impl Softmax3D {
    pub fn new(size: usize) -> Self {
        Self { neuron: Softmax, inner_size: size }
    }
}

impl Layer3D for Softmax3D {
    fn forward_into(&self, input: &Tensor3D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<f32>>>) -> LayerContext3D {
        let mut output = vec![vec![vec![0.0; input.cols]; input.rows]; input.depth];
        for d in 0..input.depth {
            for r in 0..input.rows {
                let row_in = Tensor1D::new(input.data[d][r].clone());
                let row_out = self.neuron.forward(&row_in);
                for c in 0..input.cols {
                    let val = row_out.data[c];
                    out_buf[d][r][c] = val;
                    output[d][r][c] = val;
                }
            }
        }
        LayerContext3D::Softmax3D { output: Tensor3D::new(output) }
    }

    fn backward(&self, ctx: &LayerContext3D, delta: &Tensor3D, params: &[f32], slice: &ParamSlice) -> (Tensor3D, Vec<f32>) {
        let sm = match ctx { LayerContext3D::Softmax3D { output } => output, _ => panic!() };
        let depth = sm.depth;
        let rows = sm.rows;
        let cols = sm.cols;
        let mut d_prev = vec![vec![vec![0.0; cols]; rows]; depth];
        for d in 0..depth {
            for r in 0..rows {
                for i in 0..cols {
                    let mut sum = 0.0;
                    for j in 0..cols {
                        let kron = if i == j { 1.0 } else { 0.0 };
                        sum += delta.data[d][r][j] * sm.data[d][r][j] * (kron - sm.data[d][r][i]);
                    }
                    d_prev[d][r][i] = sum;
                }
            }
        }
        (Tensor3D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}





