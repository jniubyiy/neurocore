use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Tanh;
use crate::neuron::base::Neuron;
use super::{Layer3D, LayerContext3D};

pub struct Tanh3D {
    neuron: Tanh,
    inner_size: usize,
}

impl Tanh3D {
    pub fn new(size: usize) -> Self {
        Self { neuron: Tanh, inner_size: size }
    }
}

impl Layer3D for Tanh3D {
    fn forward_into(&self, input: &Tensor3D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<f32>>>) -> LayerContext3D {
        let mut output = vec![vec![vec![0.0; input.cols]; input.rows]; input.depth];
        for d in 0..input.depth {
            for r in 0..input.rows {
                for c in 0..input.cols {
                    let val = self.neuron.apply(input.data[d][r][c]);
                    out_buf[d][r][c] = val;
                    output[d][r][c] = val;
                }
            }
        }
        LayerContext3D::Tanh3D { output: Tensor3D::new(output) }
    }

    fn backward(&self, ctx: &LayerContext3D, delta: &Tensor3D, params: &[f32], slice: &ParamSlice) -> (Tensor3D, Vec<f32>) {
        let output = match ctx { LayerContext3D::Tanh3D { output } => output, _ => panic!() };
        let depth = output.depth;
        let mut d_prev = vec![vec![vec![0.0; output.cols]; output.rows]; depth];
        for d in 0..depth {
            for r in 0..output.rows {
                for c in 0..output.cols {
                    let t = output.data[d][r][c];
                    d_prev[d][r][c] = delta.data[d][r][c] * (1.0 - t * t);
                }
            }
        }
        (Tensor3D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}