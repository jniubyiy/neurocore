use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Sigmoid;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer5D, LayerContext5D};

pub struct Sigmoid5D { inner_size: usize }
impl Sigmoid5D { pub fn new(size: usize) -> Self { Self { inner_size: size } } }

impl Layer5D for Sigmoid5D {
    fn forward_into(&self, input: &Tensor5D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> LayerContext5D {
        let mat = linalg::tensor5d_to_faer(input);
        let out = Sigmoid.forward_mat(&mat);
        let t = linalg::faer_to_tensor5d(&out, input.outer, input.dim1, input.depth, input.rows, input.cols);
        *out_buf = t.data.clone();
        LayerContext5D::Sigmoid5D { output: t }
    }

    fn backward(&self, ctx: &LayerContext5D, delta: &Tensor5D, _params: &[f32], _slice: &ParamSlice) -> (Tensor5D, Vec<f32>) {
        let output = match ctx { LayerContext5D::Sigmoid5D { output } => output, _ => panic!() };
        let (outer, dim1, depth, rows, cols) = (output.outer, output.dim1, output.depth, output.rows, output.cols);
        let mut d_prev = vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer];
        for o in 0..outer { for d1 in 0..dim1 { for d in 0..depth { for r in 0..rows { for c in 0..cols {
            let sig = output.data[o][d1][d][r][c];
            d_prev[o][d1][d][r][c] = delta.data[o][d1][d][r][c] * sig * (1.0 - sig);
        }}}}}
        (Tensor5D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}





