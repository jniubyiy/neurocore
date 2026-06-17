use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Sigmoid;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer3D, LayerContext3D};

pub struct Sigmoid3D {
    inner_size: usize,
}

impl Sigmoid3D {
    pub fn new(size: usize) -> Self { Self { inner_size: size } }
}

impl Layer3D for Sigmoid3D {
    fn forward_into(&self, input: &Tensor3D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<f32>>>) -> LayerContext3D {
        let mat = linalg::tensor3d_to_faer(input);
        let out = Sigmoid.forward_mat(&mat);
        let t = linalg::faer_to_tensor3d(&out, input.depth, input.rows, input.cols);
        *out_buf = t.data.clone();
        LayerContext3D::Sigmoid3D { output: t }
    }

    fn backward(&self, ctx: &LayerContext3D, delta: &Tensor3D, _params: &[f32], _slice: &ParamSlice) -> (Tensor3D, Vec<f32>) {
        let output = match ctx { LayerContext3D::Sigmoid3D { output } => output, _ => panic!() };
        let depth = output.depth;
        let rows = output.rows;
        let cols = output.cols;
        let mut d_prev = vec![vec![vec![0.0; cols]; rows]; depth];
        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols {
                    let sig = output.data[d][r][c];
                    d_prev[d][r][c] = delta.data[d][r][c] * sig * (1.0 - sig);
                }
            }
        }
        (Tensor3D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}





