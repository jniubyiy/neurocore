use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Tanh;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer4D, LayerContext4D};

pub struct Tanh4D { inner_size: usize }
impl Tanh4D { pub fn new(size: usize) -> Self { Self { inner_size: size } } }

impl Layer4D for Tanh4D {
    fn forward_into(&self, input: &Tensor4D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D {
        let mat = linalg::tensor4d_to_faer(input);
        let out = Tanh.forward_mat(&mat);
        let t = linalg::faer_to_tensor4d(&out, input.dim1, input.depth, input.rows, input.cols);
        *out_buf = t.data.clone();
        LayerContext4D::Tanh4D { output: t }
    }

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, _params: &[f32], _slice: &ParamSlice) -> (Tensor4D, Vec<f32>) {
        let output = match ctx { LayerContext4D::Tanh4D { output } => output, _ => panic!() };
        let (dim1, depth, rows, cols) = (output.dim1, output.depth, output.rows, output.cols);
        let mut d_prev = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        for d1 in 0..dim1 { for d in 0..depth { for r in 0..rows { for c in 0..cols {
            let t = output.data[d1][d][r][c];
            d_prev[d1][d][r][c] = delta.data[d1][d][r][c] * (1.0 - t * t);
        }}}}
        (Tensor4D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}